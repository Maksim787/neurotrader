"""
This file contains implementation of Ornstein-Uhlenbeck process
https://www.sciencedirect.com/topics/engineering/ornstein-uhlenbeck-process
d P_t = alpha (gamma - P_t) dt + beta d W
Here W - Brownian motion
This process has analytic solution
P_t = y0 exp(- alpha t) + gamma (1 - exp(- alpha t)) + beta exp(-alpha t) * int_0^t exp(alpha s) d Ws
"""

import numpy as np
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression


@dataclass
class OUParams:
    """
    Params for Ornstein-Uhlenbeck process
    """
    alpha: float
    beta: float
    gamma: float


def calculate_integral(t: np.array, dw: np.array, params: OUParams) -> np.ndarray:
    """
    Calculate integral for OU process
    :param t: time steps
    :param dw: steps of Brownian motion
    :param params: parameters of OU process
    :return: approximated integral int_0^t exp(alpha * s) d Ws
    """
    exps: np.ndarray = np.exp(params.alpha * t)
    integral: np.ndarray = np.cumsum(exps * dw)
    return np.insert(integral, 0, 0)[:-1]


def simulate_ou_process(n_periods: int, params: OUParams, x0: float = 0.) -> np.ndarray:
    """
    Implement simulation of OU process
    :param n_periods: number of steps in process
    :param params: parameters for OU params
    :param x0: start condition for OU process
    :return: array with steps of simulated process
    """
    t: np.ndarray = np.arange(n_periods)
    dw: np.ndarray = np.random.normal(0, 1, size=n_periods)
    integral: np.ndarray = calculate_integral(t, dw, params)
    minus_exps: np.ndarray = np.exp(- params.alpha * t)
    return x0 * minus_exps + params.gamma * (1 - minus_exps) + params.beta * minus_exps * integral


def estimate_params(x: np.array) -> OUParams:
    """
    Find optimal approximation of time series with OU process
    :param x: given time series
    :return: parameters for closest solution
    """
    y: np.ndarray = np.diff(x)
    model = LinearRegression().fit(x[:-1].reshape(-1, 1), y)
    alpha: float = - model.coef_[0]
    gamma: float = model.intercept_ / alpha
    y_pred: np.ndarray = model.predict(x[:-1].reshape(-1, 1))
    beta: float = float(np.std(y - y_pred))
    return OUParams(alpha, beta, gamma)
