import numpy as np
import pandas as pd


def moving_average(data_column: np.array, length: int) -> np.array:
    """
    Calculate moving average for time series
    MA_n(t) = (series_{t} + ... + series_{t + n - 1}) / n
    :param data_column: given series
    :param length: period of moving average (n)
    :return: time series with calculated moving averages
    """
    return np.array([np.sum(data_column[i:i + length], axis=0) / len(data_column[i:i + length])
                     for i in range(len(data_column))])


def exp_moving_average(data_column: np.array, length: int) -> np.array:
    """
    Calculate exponential moving average for time series
    EMA_n(t) = (1 - alpha) * EMA_n(t - 1) + alpha * series_t, where alpha = 1/n
    :param data_column: given series
    :param length: period of exponential moving average (n)
    :return: time series with calculated moving averages
    """
    res_column = data_column.copy()
    alpha = 1 / length
    for index in range(1, len(data_column)):
        res_column[index] = alpha * data_column[index] + (1 - alpha) * res_column[index - 1]
    return res_column


def replace_lower(result: np.array, eps: float) -> np.array:
    """
    Replace elements, which are lower `eps`, with minimal element, greater `eps`
    :param result: array to replace elements
    :param eps: lower bound for elements
    :return: modified array
    """
    result = np.nan_to_num(result)
    result[result < eps] += result[result >= eps].min()
    return result


def moving_index_std(data_column: np.array, index: int, length: int) -> np.array:
    """
    Calculate standard deviation in given point
    :param data_column: time series
    :param index: index where to calculate std
    :param length: length for std
    :return: calculated std
    """
    return np.std(data_column[index - length:index], axis=0)


def moving_std(data_column: np.array, length: int, eps: float) -> np.array:
    """
    Calculate moving standard deviation
    :param data_column: time series
    :param length: period for moving std
    :param eps: lower bound to replace minimal values
    :return: calculated moving std
    """
    result = np.array([float(moving_index_std(data_column, index, length)) for index in range(len(data_column))])
    return replace_lower(result, eps)


def gk_std(data: pd.DataFrame, eps: float) -> np.array:
    """
    Calculate standard deviation for
    :param data:
    :param eps:
    :return:
    """
    moved_close = np.concatenate([[0], data.close[:-1]])
    result = np.sqrt(
        (data.open - moved_close)**2 + (data.high - data.low)**2 +
        (2 * np.log(2) - 1) * (data.close - data.open)**2
    )
    return replace_lower(result, eps)
