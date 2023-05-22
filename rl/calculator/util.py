import datetime

import numpy as np
import pandas as pd
import typing as tp

import torch

from .stats import exp_moving_average, gk_std
from .algo.reinforce import ReinforceEnv, ReinforceAgent
from .algo.a2c import A2CEnv, A2CAgent


def preprocess_table(table: pd.DataFrame,
                     eps: float = 0.1,
                     ema_lengths: tp.Tuple[int] = (2, 5, 10, 25, 30, 90, 100))\
        -> tp.Tuple[pd.DataFrame, tp.List[str], pd.Series]:
    """
    Make preprocessing for table with required columns `volume`, `value`, `close`, `open`, `high`, `low`
    All these columns can be found in candles
    For all numerical columns pd.Series.pct_change() is applied (to compute returns of securities)
    :param table: pd.DataFrame with features (time series, dates, etc.)
    :param eps: float parameter to lower bound in gk_std (see `stats.gk_std`)
    :param ema_lengths: periods for computing EMA, as additional features
    :return:
        1. modified table with dropped nans and normalized volumes
            additionally, volatility is computed
        2. numerical columns of table to use them as features
        3. time series of close prices, normalized by first value
    """
    for required_column in ["volume", "value", "close", "open", "high", "low"]:
        assert required_column in table.columns, f'{required_column} was not found in {table.columns}'
    table[['volume', 'value']] /= np.array([np.nanmax(table.volume), np.nanmax(table.value)])
    series: pd.Series = table['close'].copy()
    num_columns: tp.List[str] = table.select_dtypes(include=[np.number]).columns

    table.loc[:, num_columns] = table[num_columns].pct_change()
    table['vol'] = gk_std(table, eps=eps)
    table.replace([np.inf, -np.inf], np.nan, inplace=True); table.dropna(inplace=True)
    series = series[table.index]; series /= series.iloc[0]

    for length in ema_lengths:
        for column in num_columns:
            table[f'exp-{length}-{column}'] = exp_moving_average(np.array(table[column]), length)

    table.index = pd.to_datetime(table['begin']).dt.date
    series.index = table.index
    table.drop(columns=['end', 'begin'], inplace=True)
    return table, num_columns, series


def return_revenue(env: ReinforceEnv, agent: ReinforceAgent) -> tp.Tuple[np.ndarray, np.ndarray]:
    """
    Calculate time series of portfolio during all trading time
    Reinforce is specified, because argmax need to be calculated for probabilities
    :param env: Reinforce environment
    :param agent: Reinforce agent
    :return:
        1. array of cumulative portfolio return
        2. array of actions from range(0, 3)
    """
    # resetting from the first trading interval
    env.index = 0
    state: torch.Tensor = env._state(env.index)
    results: tp.List[float] = [0.]
    actions: tp.List[int] = []

    while True:
        action: int = int(np.argmax(agent.get_action(state)['probs'].detach().numpy()))
        actions.append(action)
        res: float = env.get_revenue(action)
        next_state, _, done = env.step(action)
        results.append((1 + results[-1]) * (1 + res) - 1)
        state = next_state
        if done:
            break

    return np.array(results), np.array(actions)


def make_date_hour(date_column: pd.Series) -> pd.Series:
    """
    Make datetime column with date & hour from standard datetime column
    :param date_column: input datetime column
    :return: datetime column with date and hour
    """
    return pd.to_datetime(date_column.dt.date.astype(str) + ' ' + date_column.dt.hour.astype(str) + ':00:00')


def preprocess_several(table: pd.DataFrame,
                       set_indexes: tp.Set[datetime.datetime],
                       use_vol: bool = True,
                       eps: float = 0.1,
                       ema_lengths: tp.Tuple[int] = (2, 5, 10, 30, 90)) \
        -> tp.Tuple[pd.DataFrame, tp.List[str], tp.Set[datetime.datetime]]:
    """
    Make preprocessing for one of selected tables
    For numerical columns `pd.Series.pct_change()` is applied
    Additionally, volatility is computed
    :param table: DataFrame with required `volume`, `value`, `begin`, `close`, `open`, `high`, `low` columns
        (all required columns can be found in candles)
    :param set_indexes: global variable, where intersection of indexes is computed (common dates for all tables)
    :param use_vol: if True standard deviation with eps bound is computed, else volatility is ignored
        (all volatilises are constant)
    :param eps: lower bound for standard deviation values
    :param ema_lengths: periods used for calculating EMA
    :return: 1. DataFrame with normalized columns
             2. list of numerical column names
    """
    table[['volume', 'value']] /= np.array([1e6, 1e9])
    num_columns = table.select_dtypes(include=[np.number]).columns
    table.loc[:, num_columns] = table[num_columns].pct_change()
    table['vol'] = gk_std(table, eps=eps) if use_vol else np.ones(len(table))
    table.replace([np.inf, -np.inf], np.nan, inplace=True)
    table.dropna(inplace=True)
    for length in ema_lengths:
        for column in num_columns:
            table[f'exp-{length}-{column}'] = exp_moving_average(np.array(table[column]), length)
    date_hour = make_date_hour(pd.to_datetime(table['begin']))
    set_indexes = set(date_hour) if set_indexes is None else set(date_hour) & set_indexes
    table.index = date_hour
    table.drop(columns=['begin', 'end'])
    return table, num_columns, set_indexes


def several_revenue(env: A2CEnv, agent: A2CAgent) -> tp.Tuple[np.ndarray, np.ndarray]:
    """
    Calculate time series of portfolio during all trading time
    A2C is specified, because action is portfolio weights or probabilities
    :param env: A2C environment
    :param agent: A2C agent
    :return:
    """
    # resetting from the first trading interval
    env.index = 0
    state: torch.Tensor = env._state(env.index)
    results: tp.List[float] = [0.]
    means: tp.List[float] = [0.]

    while True:
        action: torch.Tensor = agent.get_action(state)['probs'].detach()
        mean_action: np.ndarray = np.ones_like(action) / len(action)
        next_state, _, done = env.step(action)

        mean_res: float = env.get_revenue(torch.tensor(mean_action, dtype=torch.float32))
        res: float = env.get_revenue(action.numpy())

        # calculating cumulative return
        results.append((1 + results[-1]) * (1 + res) - 1)
        means.append((1 + means[-1]) * (1 + mean_res) - 1)
        state = next_state
        if done:
            break

    return np.array(results), np.array(means)
