import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from .load import load_data, TEST_RATIO
from .utils import is_sorted

TRAIN_SIZE_DAYS = 91
TEST_SIZE_DAYS = 91


@dataclass
class Observation:
    df_price_train: pd.DataFrame  # len = TRAIN_SIZE_DAYS
    df_price_test: pd.DataFrame  # len = TEST_SIZE_DAYS, first element is last element in df_price_train
    df_returns_test: pd.DataFrame = None  # p(t + 1) / p(t) - 1
    next_returns: pd.Series = None  # p(t + 1) / p(t) - 1, first element in df_returns_test

    def __post_init__(self):
        assert np.all(self.df_price_train.columns == self.df_price_test.columns)
        assert self.df_price_train.index[-1] == self.df_price_test.index[0]
        assert is_sorted(self.df_price_train.index) and is_sorted(self.df_price_test.index)
        self.df_returns_test = self.df_price_test.pct_change().dropna()
        assert len(self.df_returns_test) == len(self.df_price_test) - 1
        self.df_returns_test.index = self.df_price_test.index[0:-1]
        assert self.df_returns_test.index[0] == self.df_price_train.index[-1]
        self.next_returns = self.df_returns_test.iloc[0]


def _create_train_test_dataset(
    df_price: pd.DataFrame, test_ratio: float = TEST_RATIO,
    train_size_days: int = TRAIN_SIZE_DAYS, test_size_days: int = TEST_SIZE_DAYS
) -> tuple[list[Observation], list[Observation]]:
    observations = _create_dataset(df_price, train_size_days=train_size_days, test_size_days=test_size_days)
    observations_train, observations_test = train_test_split(observations, test_size=test_ratio, shuffle=False)
    observations_train = observations_train[:-test_size_days]  # train and test should not intersect
    print(f'Train: {len(observations_train)}. Test: {len(observations_test)}. Test ratio: {len(observations_test) / (len(observations_train) + len(observations_test))}')
    return observations_train, observations_test


def load_train_test_dataset(verbose=False) -> tuple[list[Observation], list[Observation], pd.DataFrame, pd.DataFrame]:
    df_price = load_data(verbose=verbose)
    observations_train, observations_test = _create_train_test_dataset(df_price)
    train_dates = np.unique(np.concatenate([o.df_price_train.index for o in observations_train]))
    test_dates = np.unique(np.concatenate([o.df_price_train.index for o in observations_test] + [observations_test[-1].df_price_test.index]))
    df_price_train = df_price[df_price.index.isin(train_dates)]
    df_price_test = df_price[df_price.index.isin(test_dates)]
    assert np.max(df_price_train.index) < np.min(df_price_test.index)
    assert is_sorted(df_price_train.index) and is_sorted(df_price_test.index)

    if verbose:
        plt.title('Train Test split')
        _plot_ticker('SBER', df_price_train, df_price_test)
        _plot_ticker('SBERP', df_price_train, df_price_test, plot_train_test_split_line=True)
        plt.show()
    return observations_train, observations_test, df_price_train, df_price_test


def _create_dataset(df_price: pd.DataFrame, train_size_days: int, test_size_days: int) -> list[Observation]:
    observations = []
    for i in range(len(df_price)):
        df_price_train = df_price.iloc[i:i + train_size_days]
        df_price_test = df_price.iloc[i + train_size_days - 1:i + train_size_days + test_size_days - 1]
        if len(df_price_test) < test_size_days:
            break
        assert len(df_price_train) == train_size_days and len(df_price_test) == test_size_days
        observations.append(Observation(df_price_train=df_price_train, df_price_test=df_price_test))
    return observations


def _plot_ticker(ticker: str, df_price_train: pd.DataFrame, df_price_test: pd.DataFrame, plot_train_test_split_line: bool = False):
    plt.plot(df_price_train[ticker], label=f'{ticker} train')
    plt.plot(df_price_test[ticker], label=f'{ticker} test')
    if plot_train_test_split_line:
        plt.axvline(df_price_test.index[0], linestyle='dashed', label=f'border: {df_price_test.index[0].strftime("%Y-%m-%d")}')
    plt.ylabel('Price')
    plt.legend(loc='upper left')
