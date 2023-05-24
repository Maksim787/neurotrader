import numpy as np
import typing as tp


def Quantile(time_series: np.ndarray, border: float = 0.95) -> bool:
    norm_series = (time_series - time_series.mean(axis=0)) / time_series.std(axis=0)
    abs_series = np.abs(norm_series)
    right = np.quantile(abs_series, border, axis=0)
    left = np.quantile(abs_series, 1 - border, axis=0)
    last = norm_series[-1]
    return not (np.linalg.norm(left) <= np.linalg.norm(last) <= np.linalg.norm(right))


def create_sequence_of_quantiles(time_series: np.ndarray, border: float) -> tp.Sequence:
    for stop in range(5, len(time_series)):
        yield Quantile(time_series[:stop], border)
