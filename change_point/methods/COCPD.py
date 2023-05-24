from joblib import Parallel, delayed
import numpy as np
import typing as tp
from scipy.optimize import minimize
from tqdm import tqdm

def COCPD(time_series: np.ndarray) -> float:
    """
    Method from https://proceedings.mlr.press/v206/puchkin23a/puchkin23a.pdf
    :param params:
    :param time_series:
    :return:
    """

    def compute_value(params: np.ndarray, prefix: tp.Sequence, suffix: tp.Sequence):
        score = 0.
        for elem in prefix:
            value = np.concatenate([elem, elem ** 2, [1]]) @ params / (np.linalg.norm(params) + 0.0001)
            score += (len(suffix) / (len(prefix) + len(suffix))) * (value - np.logaddexp(0, value) - np.log(2))
        for elem in suffix:
            value = np.concatenate([elem, elem ** 2, [1]]) @ params / (np.linalg.norm(params) + 0.0001)
            score -= (len(prefix) / (len(prefix) + len(suffix))) * (np.logaddexp(0, value) - np.log(2))
        return score

    def iteration(params: np.ndarray, num_of_iter: int) -> float:
        prefix = time_series[:num_of_iter]
        suffix = time_series[num_of_iter:]
        return -compute_value(params, prefix, suffix)

    def target_function(params: tp.Sequence) -> float:
        result = Parallel(n_jobs=5)(delayed(iteration)(params, t) for t in range(1, len(time_series)))
        return -min(result)

    x0 = np.zeros(2 * time_series.shape[1] + 1)
    return -minimize(target_function, x0)['fun']


def get_sequence_of_COCPD(time_series: np.ndarray) -> tp.Sequence:
    for stop in tqdm(range(5, len(time_series))):
        yield COCPD(time_series[:stop])
