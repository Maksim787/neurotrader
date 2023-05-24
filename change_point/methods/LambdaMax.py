import typing as tp

import numpy.linalg as la
import numpy as np
from tqdm import tqdm


def create_sequence_of_norms(sequence: tp.Sequence[tp.Any], windows_size: int = 10) -> tp.Sequence[float]:
    """
    return list of lambda-max observation
    :param sequence: [a_i]_1^n, a_i -- i-th time series.
    :return:
    """
    matrix = np.array(sequence)
    for stop in tqdm(range(windows_size, len(sequence))):
        yield la.norm(np.corrcoef(matrix[stop - windows_size: stop]), 2)
