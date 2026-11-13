"""构造陕西专家原始特征和未经筛选的完整门控特征。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fld_moe.transforms import (
    build_common_super_abundant_features,
    build_daily_distribution_features,
    build_expression_history,
    build_lag_frame,
    build_mean_reversion_features,
    build_momentum_acceleration,
    build_peak_valley_features,
    build_price_history_features,
    build_time_features,
)

TIME_COLUMN = "时间"
TARGET_COLUMN = "日前价格"
RAW_TARGET_COLUMN = "日前电价(元/MWh)"
BOUNDARY_COLUMNS = (
    "新能源预测(MW)",
    "风电预测(MW)",
    "光伏预测(MW)",
    "负荷预测(MW)",
    "联络线预测(MW)",
    "发电总出力预测(MW)",
    "非市场机组预测出力(MW)",
    "水电预测(MW)",
    "2米气温_01",
    "2米气温_02",
    "2米气温_03",
    "2米气温_04",
    "2米露点温度_01",
    "2米露点温度_02",
    "2米露点温度_03",
    "2米露点温度_04",
    "100米风速_01",
    "100米风速_02",
    "100米风速_03",
    "100米风速_04",
    "短波总辐射_01",
    "短波总辐射_02",
    "短波总辐射_03",
    "短波总辐射_04",
)


def _ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """保持原陕西表达式的除法口径，拒绝近零分母。"""
    if (denominator.abs() <= 1e-8).any():
        raise ValueError("陕西特征表达式存在近零分母")
    return numerator / denominator


def _market_expressions(frame: pd.DataFrame) -> pd.DataFrame:
    """仅实现陕西 super_abundant 档实际使用的七个跨列表达式。"""
    load = frame["负荷预测(MW)"]
    renewable = frame["新能源预测(MW)"]
    generation = frame["发电总出力预测(MW)"]
    interchange = frame["联络线预测(MW)"]
    temperature = frame[[f"2米气温_{i:02d}" for i in range(1, 5)]].mean(axis=1)
    return pd.DataFrame(
        {
            "平均气温": temperature,
            "净负荷": load - renewable,
            "可再生能源占比": _ratio(renewable, load),
            "供需裕度": generation + interchange - load,
            "供需比": _ratio(load, generation + interchange),
            "非市场机组占比": _ratio(frame["非市场机组预测出力(MW)"], generation),
            "平均气温x负荷": temperature * load,
        },
        index=frame.index,
    )


def build_features(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[str, ...]]:
    """保留原始列顺序生成专家与完整门控特征，不进行筛选。"""
    time, slot = build_time_features(frame, TIME_COLUMN)
    main = pd.concat([frame.loc[:, BOUNDARY_COLUMNS], time], axis=1)
    source = pd.concat([frame, time], axis=1)
    price_lags = build_lag_frame(source, [TARGET_COLUMN])
    boundary_lags = build_lag_frame(source, BOUNDARY_COLUMNS)
    price_history = build_price_history_features(source, TARGET_COLUMN, TIME_COLUMN)
    expressions = _market_expressions(source)
    expression_source = pd.concat([source, expressions], axis=1)
    expression_history = pd.concat(
        [
            build_expression_history(expression_source, [column])[0]
            for column in expressions.columns
            if column != "平均气温"
        ],
        axis=1,
    )
    common, log_history, _ = build_common_super_abundant_features(
        expression_source,
        "负荷预测(MW)",
        "新能源预测(MW)",
        "净负荷",
    )
    gate = pd.concat(
        [
            main,
            price_lags,
            boundary_lags,
            price_history,
            build_momentum_acceleration(source, BOUNDARY_COLUMNS),
            build_peak_valley_features(source, BOUNDARY_COLUMNS, TIME_COLUMN),
            build_daily_distribution_features(source, BOUNDARY_COLUMNS, TIME_COLUMN),
            build_mean_reversion_features(source, BOUNDARY_COLUMNS, slot),
            expressions,
            expression_history,
            common,
            log_history,
        ],
        axis=1,
    ).replace([np.inf, -np.inf], np.nan)
    if not gate.columns.is_unique:
        raise ValueError("门控特征包含重复列名")
    # 保留原角色规则：时间编码、二元列与价格衍生列不再标准化。
    excluded = set(time.columns) | set(price_lags.columns) | set(price_history.columns)
    rolling = tuple(column for column in gate.columns if column not in excluded)
    return main, gate, rolling
