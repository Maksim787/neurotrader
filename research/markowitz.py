import pandas as pd
import numpy as np

from dataset import Observation
from correlations import get_returns_correlations
from load import TRADING_DAYS_IN_YEAR


def get_markowitz_w(observations: list[Observation], mu_year_pct: float, sigmas: list[pd.DataFrame] | None = None) -> pd.DataFrame:
    """
    Return weights allocation at each time
    """
    sigmas = sigmas or [None] * len(observations)
    assert len(observations) == len(sigmas)
    result_dates = []
    result_w = []
    for observation, sigma in zip(observations, sigmas):
        w = _get_markowitz_w_on_one_observation(observation.df_price_train, mu_year_pct, sigma)
        assert np.all(w.index == observation.df_price_train.columns)
        result_dates.append(observation.df_price_train.index[-1])
        result_w.append(w)
    return pd.DataFrame(result_w, index=result_dates)


def _get_markowitz_w_on_one_observation(df_price: pd.DataFrame, mu_year_pct: float, sigma: pd.DataFrame | None) -> pd.Series:
    """
    mu is annual expected annual return in %
    """

    # convert mu to ratio return in 1 day
    mu = mu_year_pct / 100 / TRADING_DAYS_IN_YEAR

    n_assets = len(df_price.columns)
    correlations = get_returns_correlations(df_price)

    # construct linear equations
    returns = correlations.returns.values.mean(axis=0).reshape(-1, 1)
    stds = np.sqrt(np.diag(correlations.cov))
    sigma = correlations.cov.values if sigma is None else stds.reshape(-1, 1) * sigma * stds.reshape(1, -1)
    ones = np.ones((n_assets, 1))
    zeros = np.zeros((1, 1))
    X = np.block([[2 * sigma, -returns, -ones],
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
    w = pd.Series(w, df_price.columns)
    lambda_1 = x[n_assets]
    lambda_2 = x[n_assets + 1]

    assert np.isclose(w.sum(), 1.0)
    assert len(w.shape) == 1
    return w
