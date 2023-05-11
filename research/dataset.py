import numpy as np
import pandas as pd
from dataclasses import dataclass

from load import load_data, train_test_split


TRAIN_SIZE_MONTHS = 2
TEST_SIZE_MONTHS = 1


@dataclass
class Observation:
    df_price_train: pd.DataFrame
    df_price_test: pd.DataFrame

    def __post_init__(self):
        assert np.all(self.df_price_train.columns == self.df_price_test.columns)
        assert self.df_price_train.index[-1] < self.df_price_test.index[0]


def create_train_test_dataset(
    df_price_train: pd.DataFrame, df_price_test: pd.DataFrame,
    train_size_months: int = TRAIN_SIZE_MONTHS, test_size_months: int = TEST_SIZE_MONTHS
) -> tuple[list[Observation], list[Observation]]:
    observations_train = _create_dataset(df_price_train, train_size_months=train_size_months, test_size_months=test_size_months)
    observations_test = _create_dataset(df_price_test, train_size_months=train_size_months, test_size_months=test_size_months)
    print(f'Train: {len(observations_train)}. Test: {len(observations_test)}. Test ratio: {len(observations_test) / (len(observations_train) + len(observations_test))}')
    return observations_train, observations_test


def load_train_test_dataset(verbose=False) -> tuple[list[Observation], list[Observation]]:
    df_price = load_data(verbose=verbose)
    df_price_train, df_price_test = train_test_split(df_price, verbose=verbose)
    observations_train, observations_test = create_train_test_dataset(df_price_train, df_price_test)
    return observations_train, observations_test


def _create_dataset(df_price: pd.DataFrame, train_size_months: int, test_size_months: int) -> list[Observation]:
    observations = []
    for i in range(len(df_price)):
        start_train = df_price.index[i]
        finish_train = start_train + pd.DateOffset(months=train_size_months)
        finish_test = finish_train + pd.DateOffset(months=test_size_months)
        if finish_test > df_price.index[-1] + pd.Timedelta(days=1):
            break
        df_price_train = df_price[(df_price.index >= start_train) & (df_price.index < finish_train)]
        df_price_test = df_price[(df_price.index >= finish_train) & (df_price.index < finish_test)]
        observations.append(Observation(df_price_train=df_price_train, df_price_test=df_price_test))
    return observations
