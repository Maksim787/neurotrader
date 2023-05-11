import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from dataclasses import dataclass

from dataset import Observation
from load import TRADING_DAYS_IN_YEAR


@dataclass
class PortfolioStats:
    observations: list[Observation]

    portfolio_w: pd.DataFrame
    portfolio_label: str

    # __post_init__ attributes
    columns: list[str] = None
    returns: pd.DataFrame = None  # returns[t] are returns of investment at time t

    baseline_w: pd.DataFrame = None
    baseline_label: str = None

    portfolio_returns_pct: pd.Series = None
    baseline_returns_pct: pd.Series = None

    portfolio_mean: float = np.nan
    portfolio_std: float = np.nan
    portfolio_sharpe_ratio: float = np.nan

    baseline_mean: float = np.nan
    baseline_std: float = np.nan
    baseline_sharpe_ratio = np.nan

    difference_mean: float = np.nan
    difference_std: float = np.nan
    portfolio_information_ratio: float = np.nan

    def __post_init__(self):
        # compute returns
        self.columns = list(self.observations[0].df_price_train.columns)
        self.returns = self._compute_returns()

        # get baseline portfolio allocation
        self.baseline_label, self.baseline_w = self.get_baseline()

        # compute returns in %
        self.portfolio_returns_pct = self._get_portfolio_returns_pct(self.portfolio_w)
        self.baseline_returns_pct = self._get_portfolio_returns_pct(self.baseline_w)

        # compute metrics
        self.portfolio_mean, self.portfolio_std, self.portfolio_sharpe_ratio = self._get_annual_mean_std_ratio(self.portfolio_returns_pct)
        self.baseline_mean, self.baseline_std, self.baseline_sharpe_ratio = self._get_annual_mean_std_ratio(self.baseline_returns_pct)
        self.difference_mean, self.difference_std, self.portfolio_information_ratio = self._get_annual_mean_std_ratio(
            self.portfolio_returns_pct - self.baseline_returns_pct
        )

    # Computation functions
    def _compute_returns(self) -> pd.DataFrame:
        returns = []
        dates = []
        for observation in self.observations:
            assert observation.df_price_test.index[0] > observation.df_price_train.index[-1]
            returns.append(observation.df_price_test.iloc[0] / observation.df_price_train.iloc[-1] - 1)
            dates.append(observation.df_price_train.index[-1])
        return pd.DataFrame(returns, index=dates, columns=self.columns)

    def _get_portfolio_returns_pct(self, weights: pd.DataFrame) -> pd.Series:
        assert np.all(weights.columns == self.columns)
        assert np.all(self.returns.index == weights.index)
        # at t we decide to allocate w rubles to each asset
        # then we get the return p(t + 1) / p(t) - 1
        # returns(t) = p(t + 1) / p(t) - 1
        portfolio_returns = (self.returns * weights).sum(axis=1)
        return portfolio_returns * 100

    @staticmethod
    def _get_annual_mean_std_ratio(returns: pd.Series) -> tuple[float, float]:
        mean = returns.mean() * TRADING_DAYS_IN_YEAR
        std = returns.std() * np.sqrt(TRADING_DAYS_IN_YEAR)
        return mean, std, mean / std

    # Plot functions
    def plot_cumulative_return(self, color: str, baseline: bool, plot_label: bool):
        plt.gcf().set_figwidth(15)
        plt.gcf().set_figheight(7)
        cumulative_returns_pct = np.cumsum(self.portfolio_returns_pct if not baseline else self.baseline_returns_pct)
        label = self.portfolio_label if not baseline else self.baseline_label
        plt.plot(cumulative_returns_pct, label=label if plot_label else None, color=color)
        plt.title('Portfolio cumulative return')
        plt.ylabel('Cumulative return, %')
        plt.legend(loc='upper left')

    def plot_returns_hist(self, label: str, quantile: float = 0.01, alpha=0.7):
        plt.gcf().set_figwidth(15)
        plt.gcf().set_figheight(7)
        portfolio_returns = self.portfolio_returns_pct.clip(
            lower=self.portfolio_returns_pct.quantile(quantile),
            upper=self.portfolio_returns_pct.quantile(1 - quantile)
        )
        sns.histplot(portfolio_returns, stat='density', label=label, alpha=alpha)
        plt.xlabel('Daily return, %')
        plt.legend(loc='upper left')

    # Format functions
    @staticmethod
    def format_mean_std(mean: float, std: float):
        return f'{mean:.2f}% ± {std:.2f}%'

    def __repr__(self):
        return f'{self.portfolio_label}\n{self.get_pandas().to_string()}\n'

    def get_pandas(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                'portfolio': [
                    self.format_mean_std(self.portfolio_mean, self.portfolio_std),
                    self.portfolio_sharpe_ratio,
                    self.format_mean_std(self.difference_mean, self.difference_std),
                    self.portfolio_information_ratio
                ],
                'baseline': [
                    self.format_mean_std(self.baseline_mean, self.baseline_std),
                    self.baseline_sharpe_ratio,
                    self.format_mean_std(0.0, 0.0),
                    np.nan
                ]
            },
            index=['return', 'sharpe ratio', 'difference', 'information ratio']
        )

    # Baseline functions
    def get_baseline(self) -> tuple[str, pd.DataFrame]:
        n_assets = len(self.columns)
        w = pd.DataFrame(np.full((len(self.observations), n_assets), fill_value=1 / n_assets), columns=self.columns, index=self.returns.index)
        return 'Equal', w


@dataclass
class PortfolioTrainTestStats:
    train_stats: PortfolioStats = None
    test_stats: PortfolioStats = None

    def __init__(self, portfolio_label: str, train_w: pd.DataFrame, test_w: pd.DataFrame, observations_train: pd.DataFrame, observations_test: pd.DataFrame):
        self.train_stats = PortfolioStats(observations=observations_train, portfolio_w=train_w, portfolio_label=portfolio_label)
        self.test_stats = PortfolioStats(observations=observations_test, portfolio_w=test_w, portfolio_label=portfolio_label)

    def plot_weights(self):
        w = self.train_stats.portfolio_w.iloc[0]
        sns.barplot(x=w.index, y=w)
        plt.ylabel('Weight')
        plt.xticks(rotation=90)
        plt.title(self.train_stats.portfolio_label)

    def plot_cumulative_return(self, color: str, baseline: bool = False):
        self.train_stats.plot_cumulative_return(color, baseline, plot_label=True)
        self.test_stats.plot_cumulative_return(color, baseline, plot_label=False)

    def plot_train_test_split(self):
        plt.axvline(self.test_stats.observations[0].df_price_test.index[0], color='black', label='train test split', linestyle='dashed')
        plt.axvline(self.train_stats.observations[-1].df_price_test.index[0], color='black', linestyle='dashed')
        plt.axhline(0.0, color='black', linestyle='dashed')

    def plot_returns_hist(self):
        label = self.train_stats.portfolio_label
        self.train_stats.plot_returns_hist(label=f'{label}: train')
        self.test_stats.plot_returns_hist(label=f'{label}: test')


@dataclass
class Strategy:
    name: str
    train_w: pd.DataFrame
    test_w: pd.DataFrame


def get_stats(strategies: list[Strategy], observations_train: list[Observation], observations_test: list[Observation]) -> list[PortfolioTrainTestStats]:
    return [PortfolioTrainTestStats(
        portfolio_label=strategy.name,
        train_w=strategy.train_w,
        test_w=strategy.test_w,
        observations_train=observations_train,
        observations_test=observations_test
    ) for strategy in strategies]


def print_stats(stats: list[PortfolioTrainTestStats]):
    assert len(stats) >= 1

    for stat in stats:
        train_df = stat.train_stats.get_pandas()
        train_df.columns = [f'{col}_train' for col in train_df.columns]
        test_df = stat.test_stats.get_pandas()
        test_df.columns = [f'{col}_test' for col in test_df.columns]
        print(stat.train_stats.portfolio_label)
        print(pd.concat((train_df, test_df), axis=1))
        print()


def plot_cumulative_returns(stats: list[PortfolioTrainTestStats]):
    assert len(stats) >= 1

    stats[0].plot_train_test_split()
    stats[0].plot_cumulative_return(color='C0', baseline=True)

    for i, stat in enumerate(stats, start=1):
        stat.plot_cumulative_return(color=f'C{i}')


def plot_weights(stats: list[PortfolioTrainTestStats]):
    n_strats = len(stats)

    plt.subplots(n_strats, 1, figsize=(10, 12))
    for i, stat in enumerate(stats, start=1):
        plt.subplot(n_strats, 1, i)
        stat.plot_weights()

    plt.tight_layout()
    plt.show()


def compare_strategies(strategies: list[Strategy], observations_train: list[Observation], observations_test: list[Observation],
                       print_stats_: bool = True,
                       plot_cumulative_returns_: bool = True,
                       plot_weights_: bool = True):
    assert len(strategies) >= 1

    # compute stats
    stats = get_stats(strategies, observations_train, observations_test)

    if print_stats_:
        print_stats(stats)
    if plot_cumulative_returns_:
        plot_cumulative_returns(stats)
    if plot_weights_:
        plot_weights(stats)
