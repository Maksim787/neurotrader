# Library

## `load.py`

1. `load_data` — загружает `pd.DataFrame`, в котором столбцы — названия акций, индекс - дни, значения — цены. Фильтрует те акции, по которым есть больше всего истории. Удаляет пропуски.
_Параметры:_
    - `day_close_folder`: `str` — папка с дневными ценами закрытия.
    - `n_target_tickers`: `int` — количество акций, которые нужно загрузить. Будут выбраны те, торги которыми начались раньше всех.
    - `remove_tickers`: `list[str]` — какие тикеры нужно удалить из выборки.
    - `min_observations_per_year`: `int` — минимальное число наблюдений в году, чтобы год вошел в выборку. Нужно для того, чтобы удалить первый и последний года из выборки.
    - `verbose`: `bool` — нужно ли рисовать графики, визуализирующие выборку.
1. `train_test_split_our` — делит выборку на Train и Test. Train идет строго до Test
_Параметры:_
    - `df_price`: `pd.DataFrame` — выборка из цен из функции `load_data`.
    - `test_start_date`: `pd.Timestamp` — дата начала Test выборки.
    - `verbose`: `bool` — нужно ли нарисовать график цен по акциям `SBER` и `SBERP`.

## `dataset.py`

Класс наблюдения. Нужен для того, чтобы считать признаки на `df_price_train`, и считать `loss` по `df_price_test`. Можно использовать как 1 первую строчку из `df_price_test`, чтобы посчитать доходность, так и все строчки (если мы будем держать этот портфель всю тестовую выборку).

```
@dataclass
class Observation:
    df_price_train: pd.DataFrame  # len = TRAIN_SIZE_DAYS
    df_price_test: pd.DataFrame  # len = TEST_SIZE_DAYS, first element is last element in df_price_train
    df_returns_test: pd.DataFrame = None  # p(t + 1) / p(t) - 1
    next_returns: pd.Series = None  # p(t + 1) / p(t) - 1, first element in df_returns_test
```

1. `load_train_test_dataset` — загружает данные и составляет две выборки из `list[Observation]`. Для удобства возвращает для каждой выборки еще и цены всех активов по днями в виде `pd.DataFrame`.
_Параметры:_
    - `verbose`: `bool` — нужно ли нарисовать график цен по акциям `SBER` и `SBERP`, отражающий даты тренировочной и тестовой выборок.

## `correlations.py`

1. `get_returns_correlations` —  по ценам активов возвращает объект типа `ReturnsCorrelations`, в котором хранятся доходности активов, их матрица ковариаций и матрица корреляций, а также названия активов.
_Параметры:_
    - `df_price`: `pd.DataFrame` — цены активов в каждый момент времени
1. `sort_corr` — делает иерархическую кластеризацию матрицы корреляций, чтобы лучше визуализировать. Возвращает отсортированный список активов.
_Параметры:_
    - `corr_matrix`: `pd.DataFrame` — матрица корреляций доходностей.
1. `plot_correlations` — визуализирует матрицу корреляций.
 _Параметры:_
    - `Sigma`: `pd.DataFrame` — матрица корреляций доходностей.
    - `labels`: `list[str]` — порядок, в котором нужно отсортировать активы в матрице. Обычно берется из выхода функции `sort_corr`.
1. `detone` — удаляет рыночную компоненту из матрицы корреляций. Возвращает модифицированную матрицу корреляций.
 _Параметры:_
    - `Sigma`: `pd.DataFrame` — матрица корреляций доходностей.
    - `n_removed_components`: `int` — количество первых компонент для удаления, по умолчанию удаляется одна компонента.
1. `denoise` — удаляет шум из матрицы корреляций. Возвращает модифицированную матрицу корреляций и число удаленных компонент.
 _Параметры:_
    - `df_price`: `pd.DataFrame` — цены активов в каждый момент времени.
    - `Sigma`: `pd.DataFrame` — матрица корреляций доходностей.
    - `n_remaining_components`: `int | None` — количество первых компонент, которые не будут удалены. Если не указан, то параметр будет получен из распределения Марченко-Пастура.
1. `denoise_and_detone` — сначала удаляет рыночную компоненту, а затем шум. Возвращает модифицированную матрицу и количество компонент, которые остались после удаления шума.
 _Параметры:_ объединение параметров функций `detone` и `denoise`.
1. `get_distance_matrix` — по матрице корреляций возвращает матрицу расстояний между активами.
 _Параметры:_
    - `Sigma`: `pd.DataFrame` — матрица корреляций доходностей.
1. `get_correlation_matrices` — для списка наблюдений считает разные матрицы корреляций и дополнительные статистики.
 _Параметры:_
    - `observations`: `list[Observation]` — список наблюдений.
    - `is_train`: `bool` — нужно ли считать статистики для тренировочных или тестовых наблюдений.
    - `n_remaining_components_denoised`: `int | None` — количество первых компонент, которые не будут удалены при удалении шума.
    - `n_remaining_components_detoned_denoised`: `int | None`— количество первых компонент, которые не будут удалены при удалении шума и рыночной компоненты.

 Возвращает следующий объект:
```
@dataclass
class CorrelationMatrices:
    returns: list[pd.DataFrame]  # returns
    Sigmas: list[pd.DataFrame]  # original correlation matrices
    singular_values: list[np.array]  # singular values of correlation matrices
    singular_vectors: list[np.array]  # singular vectors of correlation matrices
    stds: list[np.array]  # standard deviations of returns
    Sigmas_detoned: list[pd.DataFrame]  # detoned correlation matrices
    Sigmas_denoised: list[pd.DataFrame]  # denoised correlation matrices
    Sigmas_detoned_denoised: list[pd.DataFrame]  # detoned and denoised correlation matrices
    n_remaining_components_denoised: list[int]  # number of remaining components after denoising
    n_remaining_components_detoned_denoised: list[int]  # number of remaining components after detoning and denoising
```

## `markowitz.py`
1. `get_markowitz_w` — для списка наблюдений считает оптимальные веса на тренировочной выборке с помощью выбранного метода.
 _Параметры:_
    - `observations`: `list[Observation]` — список наблюдений.
    - `method`: `MarkowitzMethod` — метод, по которому нужно считать статистики. Возможные значения:
    ```
    class MarkowitzMethod(Enum):
    # w^T Sigma w -> min_w s. t. w^T 1 = 1 and w^T mean_r = mu, w >= 0
    # Params: mu_year_pct - float
    MinVarianceGivenMu = auto()
    # w^T Sigma w - q * returns^T w -> min_w s. t. w^T 1 = 1, w >= 0
    # Params: q - float
    MinVarianceMaxReturnGivenQ = auto()
    # w^T Sigma w -> min_w s. t. w^T 1 = 1 and w^T mean_r = mu
    # Params: mu_year_pct - float
    MinVarianceGivenMuMaybeNegative = auto()
    ```
    - `parameters`: `dict[str, Any]` — параметры метода: $\mu$ или $q$.
    - `Sigmas`: `list[pd.DataFrame] | None` — матрица корреляций, которые нужно использовать в оптимизации по Марковицу. Если не указаны, берутся обычные матрицы корреляций.

## `backtest.py`

Стратегия выглядит следующим образом:

```
@dataclass
class Strategy:
    name: str
    get_train_w: tp.Callable  # function to obtain train_w
    get_test_w: tp.Callable  # function to obtain test_w
    train_w: pd.DataFrame = None
    test_w: pd.DataFrame = None
```

`get_train_w` и `get_test_w` функции нужны для того, чтобы посчитать веса активов для стратегии на тренировочной и тестовой выборке, если они до этого не были посчитаны и сохранены на диск.

1. `get_stats` — по стратегиям загружает из памяти или считает заново веса активов и вычисляет статистики по портфелю. Возвращает `list[PortfolioTrainTestStats]`, которые далее можно визуализировать.
 _Параметры:_
    - `strategies`: `list[Strategy]` — список стратегий.
    - `observations_train`: `list[Observation]`, `observations_test`: `list[Observation]` — тренировочные и тестовые наблюдения.
    - `cache_folder` — папка с сохраненными весами стратегий.
    - `use_cache` — если `True`, то использовать веса из файла, если он есть. Иначе, веса принудительно пересчитываются.
1. `print_stats` — печатает метрики качества стратегий на тренировочной и тестовой выборках.
 _Параметры:_
    - `strategies`: `list[Strategy]` — список стратегий.
1. `plot_cumulative_returns` — визуализирует кумулятивную доходность стратегий на тренировочной и тестовой выборках.
 _Параметры:_
    - `strategies`: `list[Strategy]` — список стратегий.
1. `plot_weights` — визуализирует первые веса на тренировочной выборке.
 _Параметры:_
    - `strategies`: `list[Strategy]` — список стратегий.
1. `compare_strategies` — функция, которая использует все предыдущие. Получает результаты стратегий из `get_stats` и выдает результаты с помощью `print_stats`, `plot_cumulative_returns`, `plot_weights`.
 _Параметры:_ объединение параметров предыдущих функций + 3 булевых флага, использовать ли `print_stats`, `plot_cumulative_returns`, `plot_weights` при визуализации.

## `clustering.py`

1. `plot_correlation_matrix_clusters` — рисует кластеры на матрице корреляций.
 _Параметры:_
    - `Sigma`: `pd.DataFrame` — матрица корреляций.
    - `labels`: `np.array` — распределение активов по кластерам.
    - `print_clusters`: `bool` — стоит ли дополнительно вывести кластеры.

1. `find_optimal_clusters_number` — перебирает число кластеров и выбирает оптимальное по `Silhouette Score`.
 _Параметры:_
    - `Sigma_train`: `pd.DataFrame` — матрица корреляций для кластеризации.
    - `Sigma_test`: `pd.DataFrame` — матрица корреляций, на которой нужно считать `Silhouette Score`.
    - `n_clusters_list`: `list[int]` — количество кластеров, которые нужно перебирать.

## `utils.py`

Небольшие технические функции.