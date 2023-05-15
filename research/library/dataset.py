import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from .load import load_data, train_test_split_our, TEST_RATIO
from .utils import is_sorted

TRAIN_SIZE_DAYS = 91
TEST_SIZE_DAYS = 91


@dataclass
class Observation:
    df_price_train: pd.DataFrame  # len = TRAIN_SIZE_DAYS
    df_price_test: pd.DataFrame  # len = TEST_SIZE_DAYS, first element is last element in df_price_train
    df_returns_test: pd.DataFrame = None  # p(t + 1) / p(t) - 1
    next_returns: pd.Series = None  # p(t + 1) / p(t) - 1

    def __post_init__(self):
        assert np.all(self.df_price_train.columns == self.df_price_test.columns)
        assert self.df_price_train.index[-1] == self.df_price_test.index[0]
        assert is_sorted(self.df_price_train.index) and is_sorted(self.df_price_test.index)
        self.df_returns_test = self.df_price_test.pct_change().dropna()
        assert len(self.df_returns_test) == len(self.df_price_test) - 1
        self.df_returns_test.index = self.df_price_test.index[0:-1]
        assert self.df_returns_test.index[0] == self.df_price_train.index[-1]
        self.next_returns = self.df_returns_test.iloc[0]


def create_train_test_dataset(
    df_price_train: pd.DataFrame, df_price_test: pd.DataFrame, test_ratio: float = TEST_RATIO,
    train_size_days: int = TRAIN_SIZE_DAYS, test_size_days: int = TEST_SIZE_DAYS
) -> tuple[list[Observation], list[Observation]]:
    assert (df_price_test.index[0] - df_price_train.index[-1]) < pd.Timedelta(days=10)
    df_price = pd.concat((df_price_train, df_price_test), axis=0)
    observations = _create_dataset(df_price, train_size_days=train_size_days, test_size_days=test_size_days)
    observations_train, observations_test = train_test_split(observations, test_size=test_ratio, shuffle=False)
    print(f'Train: {len(observations_train)}. Test: {len(observations_test)}. Test ratio: {len(observations_test) / (len(observations_train) + len(observations_test))}')
    return observations_train, observations_test


def load_train_test_dataset(verbose=False) -> tuple[list[Observation], list[Observation]]:
    df_price = load_data(verbose=verbose)
    df_price_train, df_price_test = train_test_split_our(df_price, verbose=verbose)
    observations_train, observations_test = create_train_test_dataset(df_price_train, df_price_test)
    return observations_train, observations_test


def _create_dataset(df_price: pd.DataFrame, train_size_days: int, test_size_days: int) -> list[Observation]:
    observations = []
    for i in range(len(df_price)):
        start_train = df_price.index[i]
        df_price_train = df_price.iloc[i:i + train_size_days]
        df_price_test = df_price.iloc[i + train_size_days - 1:i + train_size_days + test_size_days - 1]
        if len(df_price_test) < test_size_days:
            break
        assert len(df_price_train) == train_size_days and len(df_price_test) == test_size_days
        observations.append(Observation(df_price_train=df_price_train, df_price_test=df_price_test))
    return observations
