import pandas as pd
import numpy as np
import seaborn as sns
import warnings
from dataclasses import dataclass, field
from scipy.cluster.hierarchy import (
    average,
    complete,
    fcluster,
    single,
    ward,
)
from scipy.cluster.hierarchy import ClusterWarning

from .utils import is_sorted, unzip
from .dataset import Observation

warnings.simplefilter("ignore", ClusterWarning)


N_REMAINING_COMPONENTS = 2


@dataclass
class ReturnsCorrelations:
    returns: pd.DataFrame  # returns[t] = df_price[t] / df_price[t - 1] - 1
    cov: pd.DataFrame
    corr: pd.DataFrame
    index: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.index = list(self.returns.columns)
        assert np.all(self.index == self.cov.columns)
        assert np.all(self.index == self.cov.index)
        assert np.all(self.index == self.corr.columns)
        assert np.all(self.index == self.corr.index)


def get_returns_correlations(df_price) -> ReturnsCorrelations:
    df_returns = df_price.pct_change().dropna()
    matrix_corr = df_returns.corr()
    matrix_cov = df_returns.cov()
    assert matrix_corr.shape == matrix_cov.shape == (df_returns.shape[1], df_returns.shape[1])
    return ReturnsCorrelations(returns=df_returns, cov=matrix_cov, corr=matrix_corr)


def sort_corr(corr_matrix: pd.DataFrame) -> list[str]:
    corr_matrix = corr_matrix.copy()
    distance_matrix = np.sqrt(0.5 * (1 - corr_matrix))

    linkage_matrix = _hierarchical_clustering(distance_matrix)

    cluster_labels = fcluster(linkage_matrix, 6, criterion="maxclust")

    sorted_cols_and_orders = sorted(
        list(zip(corr_matrix.columns, cluster_labels)), key=lambda x: x[1]
    )
    sorted_cols, _ = list(map(list, list(zip(*sorted_cols_and_orders))))
    return sorted_cols


def plot_correlations(Sigma, labels, **kwargs):
    return sns.heatmap(Sigma.loc[labels, labels], cmap='coolwarm', xticklabels=True, yticklabels=True, square=True, vmin=-1, vmax=1, **kwargs)


def detone(Sigma: pd.DataFrame, n_removed_components: int = 1):
    assert np.all(np.isclose(Sigma, Sigma.T))
    U, S, VT = np.linalg.svd(Sigma)
    assert np.all(np.isclose(U, VT.T))
    assert np.all(np.isclose(U * S @ VT, Sigma))

    S[range(n_removed_components)] = 0
    Sigma_detoned = _normalize_correlation_matrix(_reconstruct_from_svd(U, S, VT))

    assert np.all(np.isclose(Sigma_detoned, Sigma_detoned.T))
    assert np.all(np.isclose(np.diag(Sigma_detoned), np.ones(Sigma_detoned.shape[0])))
    return pd.DataFrame(Sigma_detoned, columns=Sigma.columns, index=Sigma.index)


def get_marchenko_pastur_lambdas(sigma: float, c: float):
    assert 0 < c < 1

    lambda_minus = sigma ** 2 * (1 - np.sqrt(c)) ** 2
    lambda_plus = sigma ** 2 * (1 + np.sqrt(c)) ** 2

    return lambda_minus, lambda_plus


def marchenko_pastur_pdf(sigma: float, c: float, n_points: int) -> pd.Series:
    """
    # Marcenko-Pastur pdf
    # c = N / T in (0, 1)
    """
    lambda_minus, lambda_plus = get_marchenko_pastur_lambdas(sigma, c)
    lambda_ = np.linspace(lambda_minus, lambda_plus, n_points)

    pdf = np.sqrt((lambda_plus - lambda_) * (lambda_ - lambda_minus)) / (2 * np.pi * sigma ** 2 * c * lambda_)
    pdf = pd.Series(pdf, index=lambda_)

    return pdf


def denoise(df_price: pd.DataFrame, Sigma: pd.DataFrame, n_remaining_components: int | None = None) -> tuple[pd.DataFrame, int]:
    T, N = df_price.shape
    c = N / T
    U, S, VT = np.linalg.svd(Sigma)
    if n_remaining_components is None:
        lambda_minus, lambda_plus = get_marchenko_pastur_lambdas(sigma=1.0, c=c)
        removed_values = (S < lambda_plus)
        n_remaining_components = len(S) - removed_values.sum()
    else:
        assert is_sorted(np.flip(S))
        removed_values = (S <= S[n_remaining_components])
    S[removed_values] = S[removed_values].mean()
    Sigma_denoised = _normalize_correlation_matrix(_reconstruct_from_svd(U, S, VT))
    return pd.DataFrame(Sigma_denoised, columns=Sigma.columns, index=Sigma.index), n_remaining_components


def denoise_and_detone(df_price: pd.DataFrame, Sigma: pd.DataFrame, n_remaining_components=None, n_removed_components=1) -> tuple[pd.DataFrame, int]:
    return denoise(df_price, detone(Sigma,  n_removed_components=n_removed_components), n_remaining_components=n_remaining_components)
    # T, N = df_price.shape
    # c = N / T
    # U, S, VT = np.linalg.svd(Sigma)
    # if n_remaining_components is None:
    #     lambda_minus, lambda_plus = get_marchenko_pastur_lambdas(sigma=1.0, c=c)
    #     removed_values = (S < lambda_plus)
    #     n_remaining_components = len(S) - removed_values.sum() - n_removed_components
    # else:
    #     assert is_sorted(np.flip(S))
    #     removed_values = (S <= S[n_remaining_components + n_removed_components])
    # # denoise
    # S[removed_values] = S[removed_values].mean()
    # # detone
    # S[range(n_removed_components)] = 0
    # Sigma_detoned_denoised = _normalize_correlation_matrix(_reconstruct_from_svd(U, S, VT))
    # return pd.DataFrame(Sigma_detoned_denoised, columns=Sigma.columns, index=Sigma.index), n_remaining_components


@dataclass
class CorrelationMatrices:
    returns: list[pd.DataFrame]
    Sigmas: list[pd.DataFrame]
    singular_values: list[np.array]
    singular_vectors: list[np.array]
    stds: list[np.array]
    Sigmas_detoned: list[pd.DataFrame]
    Sigmas_denoised: list[pd.DataFrame]
    Sigmas_detoned_denoised: list[pd.DataFrame]
    n_remaining_components_denoised: list[int]
    n_remaining_components_detoned_denoised: list[int]

    def __post_init__(self):
        assert len(self.Sigmas) == len(self.Sigmas_detoned) == len(self.Sigmas_denoised) == len(self.Sigmas_detoned_denoised) == len(self.n_remaining_components_denoised) == len(self.n_remaining_components_detoned_denoised)


def get_correlation_matrices(observations: list[Observation], is_train: bool, n_remaining_components_denoised: int | None = None, n_remaining_components_detoned_denoised: int | None = None) -> CorrelationMatrices:
    correlations = [get_returns_correlations(observation.df_price_train if is_train else observation.df_price_test) for observation in observations]
    returns = [c.returns for c in correlations]
    Sigmas = [c.corr for c in correlations]
    Sigmas_detoned = [detone(Sigma) for Sigma in Sigmas]
    Sigmas_denoised, n_remaining_components_denoised = unzip([
        denoise(observation.df_price_train if is_train else observation.df_price_test, Sigma, n_remaining_components_denoised) for observation, Sigma in zip(observations, Sigmas)
    ])
    Sigmas_detoned_denoised, n_remaining_components_detoned_denoised = unzip([
        denoise_and_detone(observation.df_price_train if is_train else observation.df_price_test, Sigma, n_remaining_components=n_remaining_components_detoned_denoised) for observation, Sigma in zip(observations, Sigmas)
    ])
    singular_vectors = []
    singular_values = []
    for Sigma in Sigmas:
        U, S, VT = np.linalg.svd(Sigma)
        singular_values.append(S)
        singular_vectors.append(U)
    stds = [np.sqrt(np.diag(c.cov)) for c in correlations]
    return CorrelationMatrices(
        returns=returns,
        Sigmas=Sigmas,
        singular_values=singular_values,
        singular_vectors=singular_vectors,
        stds=stds,
        Sigmas_detoned=Sigmas_detoned,
        Sigmas_denoised=Sigmas_denoised,
        Sigmas_detoned_denoised=Sigmas_detoned_denoised,
        n_remaining_components_denoised=n_remaining_components_denoised,
        n_remaining_components_detoned_denoised=n_remaining_components_detoned_denoised
    )


def _hierarchical_clustering(distance_matrix, method="complete"):
    # some algorithm from internet

    if method == "complete":
        return complete(distance_matrix)
    if method == "single":
        return single(distance_matrix)
    if method == "average":
        return average(distance_matrix)
    if method == "ward":
        return ward(distance_matrix)
    assert False, 'Unreachable'


def _reconstruct_from_svd(U, S, VT):
    return U * S @ VT


def _normalize_correlation_matrix(Sigma):
    """
    Sigma = diag(Sigma)^{-1/2} * Sigma * diag(Sigma)^{-1/2}
    """
    C = np.diag(Sigma) ** (-1/2)
    Sigma = C.reshape(-1, 1) * Sigma * C.reshape(1, -1)
    assert np.all(np.isclose(np.diag(Sigma), 1.0))
    return Sigma
