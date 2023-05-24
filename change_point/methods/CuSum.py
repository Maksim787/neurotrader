import numpy as np
import typing as tp
from tqdm import tqdm

def CUsUM(time_series: np.ndarray, border: float) -> float:
    """
    Method from https://sixsigmastudyguide.com/cumulative-sum-chart-cusum/
    :param time_series:
    :param border:
    :return:
    """
    mean = np.mean(time_series, axis=0)
    cov_inv = np.linalg.inv(np.cov(time_series, rowvar=False))
    mcusum = np.zeros(time_series.shape)
    norms = []
    for i in range(1, time_series.shape[0]):
        mcusum[i] = np.maximum(0, mcusum[i - 1] + np.dot(cov_inv, (time_series[i] - mean) - border))
        norms.append(np.linalg.norm(mcusum[i]))
    return max(norms)


def create_sequence_of_cusums(time_series: np.ndarray, border: float = 2.) -> tp.Sequence:
    for stop in tqdm(range(5, len(time_series))):
        yield CUsUM(time_series[:stop], border)
