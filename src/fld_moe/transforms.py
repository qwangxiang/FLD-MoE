"""与市场无关的十五分钟时间序列特征变换。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import chinese_calendar as calendar
import numpy as np
import pandas as pd

POINTS_PER_DAY = 96
POINTS_PER_WEEK = POINTS_PER_DAY * 7
ZSCORE_CLIP = 10.0
LOG_LOAD_RENEWABLE_COLUMN = "log负荷比新能源"
NET_LOAD_SEASON_COLUMNS: tuple[str, ...] = (
    "净负荷x采暖季",
    "净负荷x制冷季",
)


def build_time_features(
    frame: pd.DataFrame,
    time_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """构造周期时间编码和状态标签，并返回日内时刻编号。"""
    times = frame[time_column].dt
    quarter = times.hour * 4 + times.minute // 15
    day = times.day
    month = times.month
    days_in_month = times.days_in_month
    day_of_week = times.dayofweek
    dates = times.date
    features = pd.DataFrame(
        {
            "时刻正弦": np.sin(quarter / POINTS_PER_DAY * 2 * np.pi),
            "时刻余弦": np.cos(quarter / POINTS_PER_DAY * 2 * np.pi),
            "星期正弦": np.sin(day_of_week / 7 * 2 * np.pi),
            "星期余弦": np.cos(day_of_week / 7 * 2 * np.pi),
            "月内日期正弦": np.sin(day / days_in_month * 2 * np.pi),
            "月内日期余弦": np.cos(day / days_in_month * 2 * np.pi),
            "月份正弦": np.sin(month / 12 * 2 * np.pi),
            "月份余弦": np.cos(month / 12 * 2 * np.pi),
            "是否周末": (day_of_week >= 5).astype(int),
            "是否节假日": dates.apply(lambda value: int(calendar.is_holiday(value))),
            "是否采暖季": month.isin([11, 12, 1, 2, 3]).astype(int),
            "是否制冷季": month.isin([6, 7, 8, 9]).astype(int),
        },
        index=frame.index,
    )
    return features, quarter


def build_lag_frame(
    frame: pd.DataFrame,
    columns: Sequence[str],
    days: Sequence[int] = tuple(range(1, 8)),
) -> pd.DataFrame:
    """为每列构造指定天数的同刻滞后值。"""
    data = {
        f"{column}_滞后{day}天同刻": frame[column].shift(POINTS_PER_DAY * day)
        for column in columns
        for day in days
    }
    return pd.DataFrame(data, index=frame.index)


def _stabilize_divisor(
    denominator: pd.Series,
    epsilon: float = 1e-8,
) -> pd.Series:
    """给分母补充带符号微量，并保留原空值。"""
    values = denominator.to_numpy(dtype=np.float64, copy=True)
    values += np.where(values >= 0, epsilon, -epsilon)
    values[denominator.isna().to_numpy()] = np.nan
    return pd.Series(values, index=denominator.index)


def _calculate_zscore(
    current: pd.Series,
    mean: pd.Series,
    std: pd.Series,
) -> pd.Series:
    """计算并截断 Z-Score，避免零方差产生极端值。"""
    zscore = (current - mean) / _stabilize_divisor(std)
    return zscore.clip(-ZSCORE_CLIP, ZSCORE_CLIP)


def _same_time_history(series: pd.Series, window: int) -> pd.DataFrame:
    """返回过去若干天同刻值组成的临时数据帧。"""
    return pd.concat(
        [series.shift(POINTS_PER_DAY * day) for day in range(1, window + 1)],
        axis=1,
    )


def build_same_time_statistics(
    series: pd.Series,
    name: str,
    windows: Sequence[int],
    *,
    include_zscore: bool,
) -> pd.DataFrame:
    """根据严格历史同刻值构造滚动统计。"""
    result: dict[str, pd.Series] = {}
    for window in windows:
        history = _same_time_history(series, window)
        valid = history.notna().all(axis=1)
        mean = history.mean(axis=1).where(valid)
        std = history.std(axis=1).where(valid)
        result[f"过去{window}天同刻{name}均值"] = mean
        result[f"过去{window}天同刻{name}标准差"] = std
        result[f"过去{window}天同刻{name}最大值"] = history.max(axis=1).where(valid)
        result[f"过去{window}天同刻{name}最小值"] = history.min(axis=1).where(valid)
        if include_zscore:
            result[f"过去{window}天同刻{name}Z-Score"] = _calculate_zscore(
                series,
                mean,
                std,
            ).where(valid)
    return pd.DataFrame(result, index=series.index)


def build_momentum_acceleration(
    frame: pd.DataFrame,
    columns: Sequence[str],
) -> pd.DataFrame:
    """为原始边界序列构造日周动量与加速度。"""
    result: dict[str, pd.Series] = {}
    for column in columns:
        day_momentum = frame[column] - frame[column].shift(POINTS_PER_DAY)
        week_momentum = frame[column] - frame[column].shift(POINTS_PER_WEEK)
        result[f"{column}_一天动量"] = day_momentum
        result[f"{column}_一周动量"] = week_momentum
        result[f"{column}_一天加速度"] = day_momentum - day_momentum.shift(
            POINTS_PER_DAY
        )
        result[f"{column}_一周加速度"] = week_momentum - week_momentum.shift(
            POINTS_PER_WEEK
        )
    return pd.DataFrame(result, index=frame.index)


def build_peak_valley_features(
    frame: pd.DataFrame,
    columns: Sequence[str],
    time_column: str,
) -> pd.DataFrame:
    """构造各边界序列的日内峰谷比例及其日变化。"""
    result: dict[str, pd.Series] = {}
    grouped = frame.groupby(frame[time_column].dt.date)
    for column in columns:
        daily_max = grouped[column].transform("max")
        daily_min = grouped[column].transform("min")
        denominator = np.maximum(daily_max.abs(), daily_min.abs())
        zero_day = denominator == 0
        ratio = ((daily_max - daily_min) / denominator.mask(zero_day, 1.0)).mask(
            zero_day,
            0.0,
        )
        result[f"{column}_峰谷差比例"] = ratio
        result[f"{column}_峰谷差比例动量"] = ratio - ratio.shift(POINTS_PER_DAY)
    return pd.DataFrame(result, index=frame.index)


def _stable_difference(series: pd.Series) -> pd.Series:
    """计算日内相邻点绝对差分。"""
    difference = series.diff().astype(np.float64)
    if not difference.empty:
        difference.iloc[0] = np.nan
    return difference


def _realized_volatility(differences: pd.Series) -> float:
    """计算有效差分平方和的平方根。"""
    values = differences.to_numpy(dtype=np.float64, copy=False)
    finite = values[np.isfinite(values)]
    return float(np.sqrt(np.square(finite).sum())) if finite.size else 0.0


def _skewness_and_kurtosis(differences: pd.Series) -> tuple[float, float]:
    """计算标准化偏度和超额峰度。"""
    values = differences.to_numpy(dtype=np.float64, copy=False)
    finite = values[np.isfinite(values)]
    if not finite.size:
        return np.nan, np.nan
    centered = finite - float(finite.mean())
    second = float(np.square(centered).mean())
    if np.isclose(second, 0.0):
        return 0.0, 0.0
    skewness = float(np.power(centered, 3).mean()) / second**1.5
    kurtosis = float(np.power(centered, 4).mean()) / second**2 - 3.0
    return skewness, kurtosis


def build_daily_distribution_features(
    frame: pd.DataFrame,
    columns: Sequence[str],
    time_column: str,
) -> pd.DataFrame:
    """构造各边界序列的日内波动率、偏度和峰度。"""
    dates = frame[time_column].dt.date
    result: dict[str, pd.Series] = {}
    for column in columns:
        volatility: dict[Any, float] = {}
        skewness: dict[Any, float] = {}
        kurtosis: dict[Any, float] = {}
        for date, group in frame.groupby(dates, sort=True):
            differences = _stable_difference(group[column])
            volatility[date] = _realized_volatility(differences)
            skewness[date], kurtosis[date] = _skewness_and_kurtosis(differences)
        result[f"{column}_日内波动率"] = dates.map(volatility).astype(np.float64)
        result[f"{column}_差分偏度"] = dates.map(skewness).astype(np.float64)
        result[f"{column}_差分峰度"] = dates.map(kurtosis).astype(np.float64)
    return pd.DataFrame(result, index=frame.index)


def _build_long_term_statistics(
    series: pd.Series,
    name: str,
    time_slot: pd.Series,
) -> tuple[pd.DataFrame, str]:
    """构造排除当前值的扩张同刻统计。"""
    grouped = series.groupby(time_slot)
    shifted = grouped.shift(1)
    mean = grouped.transform(lambda values: values.shift(1).expanding().mean())
    std = grouped.transform(
        lambda values: values.shift(1).expanding(min_periods=2).std()
    )
    maximum = shifted.groupby(time_slot).transform(lambda values: values.cummax())
    minimum = shifted.groupby(time_slot).transform(lambda values: values.cummin())
    zscore_name = f"长期同刻{name}Z-Score"
    result = pd.DataFrame(
        {
            f"长期同刻{name}均值": mean,
            f"长期同刻{name}标准差": std,
            f"长期同刻{name}最大值": maximum,
            f"长期同刻{name}最小值": minimum,
            zscore_name: _calculate_zscore(series, mean, std),
        },
        index=series.index,
    )
    return result, zscore_name


def _estimate_ou_speed(
    zscore: pd.Series,
    time_slot: pd.Series,
    min_pairs: int = 3,
) -> pd.Series:
    """用当前行之前的同刻历史对估计 AR(1) 型 OU 回复速度。"""
    lagged = zscore.groupby(time_slot).shift(1)
    valid = (lagged.notna() & zscore.notna()).astype(int)
    pair_sum = (
        (lagged * zscore)
        .groupby(time_slot)
        .transform(lambda values: values.cumsum().shift(1))
    )
    square_sum = (
        lagged.pow(2)
        .groupby(time_slot)
        .transform(lambda values: values.cumsum().shift(1))
    )
    pair_count = valid.groupby(time_slot).transform(
        lambda values: values.cumsum().shift(1).fillna(0)
    )
    phi = pair_sum / _stabilize_divisor(square_sum)
    result = pd.Series(np.nan, index=zscore.index, dtype=np.float64)
    enough = pair_count >= min_pairs
    valid_phi = enough & phi.gt(0) & phi.lt(1)
    result.loc[enough] = 0.0
    result.loc[valid_phi] = -np.log(phi.loc[valid_phi])
    return result


def build_mean_reversion_features(
    frame: pd.DataFrame,
    columns: Sequence[str],
    time_slot: pd.Series,
) -> pd.DataFrame:
    """构造原始边界序列的短长期同刻统计与 OU 回复速度。"""
    blocks: list[pd.DataFrame] = []
    for column in columns:
        short = build_same_time_statistics(
            frame[column],
            column,
            (3, 5, 7),
            include_zscore=True,
        )
        long, long_zscore = _build_long_term_statistics(
            frame[column],
            column,
            time_slot,
        )
        zscore_columns = [f"过去{window}天同刻{column}Z-Score" for window in (3, 5, 7)]
        zscores = pd.concat(
            [short[zscore_columns], long[[long_zscore]]],
            axis=1,
        )
        ou = pd.DataFrame(
            {
                name.replace("Z-Score", "OU回复速度"): _estimate_ou_speed(
                    zscores[name],
                    time_slot,
                )
                for name in zscores.columns
            },
            index=frame.index,
        )
        blocks.extend([short, long, ou])
    return pd.concat(blocks, axis=1) if blocks else pd.DataFrame(index=frame.index)


def _build_price_volatility(
    frame: pd.DataFrame,
    target_column: str,
    time_column: str,
    long_term_days: int = 7,
) -> pd.DataFrame:
    """只用此前完整自然日构造价格短长期波动率。"""
    dates = frame[time_column].dt.date
    daily = {
        date: group[target_column].reset_index(drop=True)
        for date, group in frame.groupby(dates, sort=True)
    }
    incomplete = [
        date for date, series in daily.items() if len(series) != POINTS_PER_DAY
    ]
    if incomplete:
        raise ValueError(f"价格波动率要求完整自然日，异常日期: {incomplete[:5]}")
    ordered_dates = list(daily)
    short_values: dict[Any, float] = {}
    long_values: dict[Any, float] = {}
    ratios: dict[Any, float] = {}
    for index, date in enumerate(ordered_dates):
        if index == 0:
            short_values[date] = np.nan
            long_values[date] = np.nan
            ratios[date] = np.nan
            continue
        short = _realized_volatility(daily[ordered_dates[index - 1]].diff())
        history = pd.concat(
            [
                daily[history_date]
                for history_date in ordered_dates[
                    max(0, index - long_term_days) : index
                ]
            ],
            ignore_index=True,
        )
        long = _realized_volatility(history.diff())
        short_values[date] = short
        long_values[date] = long
        ratios[date] = float(
            short / _stabilize_divisor(pd.Series([long], dtype=np.float64)).iloc[0]
        )
    return pd.DataFrame(
        {
            f"{target_column}_短期波动率": dates.map(short_values).astype(np.float64),
            f"{target_column}_长期波动率": dates.map(long_values).astype(np.float64),
            f"{target_column}_短期长期波动率比率": dates.map(ratios).astype(np.float64),
        },
        index=frame.index,
    )


def build_price_history_features(
    frame: pd.DataFrame,
    target_column: str,
    time_column: str,
) -> pd.DataFrame:
    """构造严格基于历史可见价格的统计、动量和波动率。"""
    price = frame[target_column]
    lagged_day = price.shift(POINTS_PER_DAY)
    momentum = pd.DataFrame(
        {
            f"{target_column}_滞后一天逐点动量": lagged_day
            - price.shift(POINTS_PER_DAY + 1),
            f"{target_column}_滞后一天同刻动量": lagged_day
            - price.shift(POINTS_PER_DAY * 2),
        },
        index=frame.index,
    )
    statistics = build_same_time_statistics(
        price,
        target_column,
        (3, 5, 7),
        include_zscore=False,
    )
    volatility = _build_price_volatility(frame, target_column, time_column)
    return pd.concat([statistics, momentum, volatility], axis=1)


def build_expression_history(
    frame: pd.DataFrame,
    columns: Sequence[str],
) -> tuple[pd.DataFrame, set[str]]:
    """为跨列 level 序列追加固定且不递归的时序衍生。"""
    lagged = build_lag_frame(frame, columns)
    result: dict[str, pd.Series] = {}
    for column in columns:
        result[f"{column}_一天动量"] = frame[column] - frame[column].shift(
            POINTS_PER_DAY
        )
        result[f"{column}_一周动量"] = frame[column] - frame[column].shift(
            POINTS_PER_WEEK
        )
        for window in (3, 7):
            history = _same_time_history(frame[column], window)
            valid = history.notna().all(axis=1)
            mean = history.mean(axis=1).where(valid)
            std = history.std(axis=1).where(valid)
            result[f"过去{window}天同刻{column}Z-Score"] = _calculate_zscore(
                frame[column],
                mean,
                std,
            ).where(valid)
    selected = pd.DataFrame(result, index=frame.index)
    return pd.concat([lagged, selected], axis=1), set(lagged.columns)


def build_common_super_abundant_features(
    frame: pd.DataFrame,
    load_column: str,
    renewable_column: str,
    net_load_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame, set[str]]:
    """构造陕西所需的净负荷季节交互项及对数比时序衍生。"""
    required = {
        load_column,
        renewable_column,
        net_load_column,
        "是否采暖季",
        "是否制冷季",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"公共超丰富特征缺少输入列: {missing}")

    load = np.log(np.clip(frame[load_column].to_numpy(dtype=np.float64), 1e-8, None))
    renewable = np.log(
        np.clip(frame[renewable_column].to_numpy(dtype=np.float64), 1e-8, None)
    )
    levels = pd.DataFrame(
        {
            LOG_LOAD_RENEWABLE_COLUMN: load - renewable,
            NET_LOAD_SEASON_COLUMNS[0]: (frame[net_load_column] * frame["是否采暖季"]),
            NET_LOAD_SEASON_COLUMNS[1]: (frame[net_load_column] * frame["是否制冷季"]),
        },
        index=frame.index,
    )
    selected, lag_columns = build_expression_history(
        levels,
        [LOG_LOAD_RENEWABLE_COLUMN],
    )
    return levels, selected, lag_columns
