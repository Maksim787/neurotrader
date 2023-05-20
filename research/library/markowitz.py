import pandas as pd
import numpy as np
import typing as tp
from enum import Enum, auto
from scipy.optimize import minimize
from tqdm.notebook import tqdm

from .dataset import Observation
from .correlations import get_returns_correlations
from .load import TRADING_DAYS_IN_YEAR


class MarkowitzMethod(Enum):
    # w^T Sigma w -> min_w s. t. w^T 1 = 1 and w^T mean_r = mu, w >= 0
    # Params: mu_year_pct - float
    MinVarianceGivenMu = auto()
    # w^T Sigma w - q * returns^T w -> min_w s. t. w^T 1 = 1, w >= 0
    # Params: q - float
    MinVarianceMaxReturnGivenQ = auto()
    # w^T Sigma w -> min_w s. t. w^T 1 = 1 and w^T mean_r = mu
    # Params: mu_year_pct - float
    MinVarianceGivenMuMaybeNegative = auto()


def get_markowitz_w(observations: list[Observation], method: MarkowitzMethod, parameters: dict[str, tp.Any], Sigmas: list[pd.DataFrame] | None = None) -> pd.DataFrame:
    """
    Return weights allocation at each time
    """
    Sigmas = Sigmas or [None] * len(observations)
    assert len(observations) == len(Sigmas)
    result_dates = []
    result_w = []
    for observation, Sigma in tqdm(list(zip(observations, Sigmas))):
        w = _get_markowitz_w_on_one_observation(observation.df_price_train, method, parameters, Sigma)
        assert np.all(w.index == observation.df_price_train.columns)
        result_dates.append(observation.df_returns_test.index[0])
        result_w.append(w)
    return pd.DataFrame(result_w, index=result_dates)


def _get_markowitz_w_using_method(Sigma: np.array, returns: np.array, columns: list[str], method: MarkowitzMethod, parameters: dict[str, tp.Any]):
    if method == MarkowitzMethod.MinVarianceGivenMu:
        return _get_markowitz_MinVarianceGivenMu(Sigma, returns, columns, **parameters)
    if method == MarkowitzMethod.MinVarianceMaxReturnGivenQ:
        return _get_markowitz_MinVarianceMaxReturnGivenQ(Sigma, returns, columns, **parameters)
    if method == MarkowitzMethod.MinVarianceGivenMuMaybeNegative:
        return _get_markowitz_MinVarianceGivenMuMaybeNegative(Sigma, returns, columns, **parameters)
    assert False, 'Unreachable'


def _get_markowitz_w_on_one_observation(df_price: pd.DataFrame, method: MarkowitzMethod, parameters: dict[str, tp.Any], Sigma: pd.DataFrame | None) -> pd.Series:
    correlations = get_returns_correlations(df_price)
    stds = np.sqrt(np.diag(correlations.cov.values))
    Sigma = correlations.cov.values if Sigma is None else stds.reshape(-1, 1) * Sigma * stds.reshape(1, -1)
    returns = correlations.returns.values.mean(axis=0)
    return _get_markowitz_w_using_method(Sigma, returns, df_price.columns, method, parameters)


def _get_markowitz_MinVarianceGivenMu(Sigma: np.array, returns: np.array, columns: list[str], mu_year_pct: float) -> pd.Series:
    # convert mu to ratio return in 1 day
    mu = mu_year_pct / 100 / TRADING_DAYS_IN_YEAR

    n_assets = len(columns)
    assert Sigma.shape == (n_assets, n_assets)
    assert returns.shape == (n_assets,)

    bounds = [(0, 1)] * n_assets
    w0 = [1 / n_assets] * n_assets
    constraints = [{'type': 'eq', 'fun': lambda w:  w.sum() - 1},
                   {'type': 'eq', 'fun': lambda w: w @ returns - mu}]

    def objective(w): return w.reshape(1, -1) @ Sigma @ w.reshape(-1, 1)
    result = minimize(objective, w0, bounds=bounds, constraints=constraints)

    try:
        assert result.success
        w = result.x
    except AssertionError:
        print(f'Fail to optimize (no success): {returns=}')
        w = np.array(w0)
    try:
        assert np.isclose(w.sum(), 1)
    except AssertionError:
        print(f'Fail to optimize (sum of weights is not 1): {returns=}; {w=}; {w.sum()=}; {w @ returns=}')
        w = np.array(w0)
    assert np.isclose(w.sum(), 1)
    assert np.all(w >= 0)
    return pd.Series(w, index=columns)


def _get_markowitz_MinVarianceMaxReturnGivenQ(Sigma: np.array, returns: np.array, columns: list[str], q: float) -> pd.Series:
    n_assets = len(columns)
    assert Sigma.shape == (n_assets, n_assets)
    assert returns.shape == (n_assets,)

    bounds = [(0, 1)] * n_assets
    w0 = [1 / n_assets] * n_assets
    constraints = [{'type': 'eq', 'fun': lambda w:  w.sum() - 1}]
    def objective(w): return w.reshape(1, -1) @ Sigma @ w.reshape(-1, 1) - q * returns @ w
    result = minimize(objective, w0, bounds=bounds, constraints=constraints)

    assert result.success
    w = result.x
    assert np.isclose(w.sum(), 1)
    assert np.all(w >= 0)
    return pd.Series(w, index=columns)


def _get_markowitz_MinVarianceGivenMuMaybeNegative(Sigma: np.array, returns: np.array, columns: list[str], mu_year_pct: float) -> pd.Series:
    # convert mu to ratio return in 1 day
    mu = mu_year_pct / 100 / TRADING_DAYS_IN_YEAR

    n_assets = len(columns)
    assert Sigma.shape == (n_assets, n_assets)
    assert returns.shape == (n_assets,)

    # construct linear equations
    ones = np.ones((n_assets, 1))
    zeros = np.zeros((1, 1))
    returns = returns.reshape(-1, 1)
    X = np.block([[2 * Sigma, -returns, -ones],
                  [returns.T, zeros, zeros],
                  [ones.T, zeros, zeros]])
    b = np.array([0.0] * n_assets + [mu, 1.0]).reshape(-1, 1)

    assert X.shape[0] == X.shape[1]
    assert X.shape[0] == n_assets + 2
    assert X.shape[0] == b.shape[0]

    # solve linear equations
    x = np.linalg.solve(X, b).reshape(-1)
    assert x.shape[0] == n_assets + 2

    # extract w, lambda_1, lambda_2 from solution
    w = x[:n_assets]
    w = pd.Series(w, columns)
    lambda_1 = x[n_assets]
    lambda_2 = x[n_assets + 1]

    assert np.isclose(w.sum(), 1.0)
    assert len(w.shape) == 1
    return w
