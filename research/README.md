# Research

## load.py

1. `load_data` — загружает `pd.DataFrame`, в котором столбцы — названия акций, индекс - дни, значения — цены.
_Параметры:_
- `day_close_folder`: `str` — папка с дневными ценами закрытия.
- `n_target_tickers`: `int` — количество акций, которые нужно загрузить. Будут выбраны те, торги которыми начались раньше всех.
- `remove_tickers`: `list[str]` — какие тикеры нужно удалить из выборки.
- `min_observations_per_year`: `int` — минимальное число наблюдений в году, чтобы год вошел в выборку. Нужно для того, чтобы удалить первый и последний года из выборки.
- `verbose`: `bool` — нужно ли рисовать графики, визуализирующие выборку.
1. `train_test_split` — делит выборку на Train и Test. Train идет строго до Test
_Параметры:_
- `df_price`: `pd.DataFrame` — выборка из цен из функции `load_data`.
- `test_start_date`: `pd.Timestamp` — дата начала Test выборки.
- `verbose`: `bool` — нужно ли нарисовать график цен по акциям `SBER` и `SBERP`.

## dataset.py

Класс наблюдения. Нужен для того, чтобы считать признаки на `df_price_train`, и считать `loss` по `df_price_test`. Можно использовать как 1 первую строчку из `df_price_test`, чтобы посчитать доходность, так и все строчки (если мы будем держать этот портфель всю тестовую выборку).

```
@dataclass
class Observation:
    df_price_train: pd.DataFrame
    df_price_test: pd.DataFrame
```


1. `create_train_test_dataset` — развивает две выборки на `list[Observation]`.
_Параметры:_
- `df_price_train`,  `df_price_test`: `pd.DataFrame` — тренировочная и тестовая выборка.
- `train_size_months`, `test_size_months`: `int` — количество месяцев в тренировочных и тестовых данных в каждом `Observation`.

## correlations.py

