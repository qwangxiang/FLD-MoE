"""准备陕西固定双流特征及严格按时间划分的自然日样本。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from fld_moe.config import FLDMoEConfig
from fld_moe.features import (
    BOUNDARY_COLUMNS,
    RAW_TARGET_COLUMN,
    TARGET_COLUMN,
    TIME_COLUMN,
    build_features,
)
from fld_moe.gate import build_price_segment_responsibilities
from fld_moe.schema import FLDMoESplit, PreparedFLDMoEData


def load_data(path: str | Path) -> pd.DataFrame:
    """读取仅含陕西的 CSV，拒绝缺失、重复和不连续的自然日数据。"""
    frame = pd.read_csv(path, encoding="utf-8-sig")
    expected = [TIME_COLUMN, RAW_TARGET_COLUMN, *BOUNDARY_COLUMNS]
    if list(frame.columns) != expected:
        raise ValueError("CSV 列及顺序必须与 data/shaanxi.csv 一致")
    frame = frame.rename(columns={RAW_TARGET_COLUMN: TARGET_COLUMN})
    timestamps = pd.DatetimeIndex(pd.to_datetime(frame[TIME_COLUMN], errors="raise"))
    if timestamps.empty or timestamps.tz is not None or timestamps.isna().any():
        raise ValueError("时间轴必须为非空的中国本地无时区时间")
    if len(timestamps) % 96 or timestamps[0] != timestamps[0].normalize():
        raise ValueError("数据必须从零点开始并由完整 96 点自然日组成")
    if not timestamps.equals(
        pd.date_range(timestamps[0], periods=len(timestamps), freq="15min")
    ):
        raise ValueError("数据时间轴必须唯一且以 15 分钟连续递增")
    frame[TIME_COLUMN] = timestamps
    numeric = frame.columns.drop(TIME_COLUMN)
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(frame[numeric].to_numpy(dtype=np.float64)).all():
        raise ValueError("数据包含缺失值或非有限值")
    return frame


def _safe_scale(values: np.ndarray) -> np.ndarray:
    """计算样本标准差，常量列使用尺度一。"""
    scale = values.std(axis=0, ddof=1)
    return np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)


def _build_split(
    name: str,
    days: range,
    main: pd.DataFrame,
    gate: pd.DataFrame,
    rolling_columns: tuple[str, ...],
    timestamps: pd.DatetimeIndex,
    prices: np.ndarray,
    bounds: np.ndarray,
    config: FLDMoEConfig,
) -> FLDMoESplit:
    """仅用每个目标日之前的历史标准化，并保持原实验暖机口径。"""
    if not days:
        raise ValueError(f"{name} 切分为空，数据不足以完成暖机和三段切分")
    main_values = main.to_numpy(dtype=np.float64)
    gate_values = gate.to_numpy(dtype=np.float64)
    window = config.data.feature_normalization_window_days * 96
    main_start = config.dataset.main_history_start_day * 96
    gate_start = config.dataset.gate_history_start_day * 96
    positions = {column: index for index, column in enumerate(gate.columns)}
    gate_indices = [positions[column] for column in rolling_columns]
    main_positions = {column: index for index, column in enumerate(main.columns)}
    main_indices = [main_positions[column] for column in BOUNDARY_COLUMNS]
    shared_columns = set(BOUNDARY_COLUMNS)
    shared_slots = [
        i for i, column in enumerate(rolling_columns) if column in shared_columns
    ]
    shared_indices = [BOUNDARY_COLUMNS.index(rolling_columns[i]) for i in shared_slots]
    gate_only_slots = [
        i for i, column in enumerate(rolling_columns) if column not in shared_columns
    ]
    gate_only_indices = [positions[rolling_columns[i]] for i in gate_only_slots]
    expert_samples, gate_samples, loss_scales, sample_indices = [], [], [], []
    for day in days:
        start, stop = day * 96, (day + 1) * 96
        history = main_values[max(main_start, start - window) : start, main_indices]
        mean, scale = history.mean(axis=0), _safe_scale(history)
        expert_day = main_values[start:stop].copy()
        # 与原实现一致：目标日特征先转 float32，再做因果标准化赋值。
        expert_day = expert_day.astype(np.float32)
        expert_day[:, main_indices] = (expert_day[:, main_indices] - mean) / scale
        gate_day = gate_values[start:stop].astype(np.float32)
        means = np.empty(len(rolling_columns), dtype=np.float64)
        scales = np.empty_like(means)
        means[shared_slots], scales[shared_slots] = (
            mean[shared_indices],
            scale[shared_indices],
        )
        if gate_only_slots:
            gate_history = gate_values[
                max(gate_start, start - window) : start, gate_only_indices
            ]
            means[gate_only_slots] = gate_history.mean(axis=0)
            scales[gate_only_slots] = _safe_scale(gate_history)
        gate_day[:, gate_indices] = (gate_day[:, gate_indices] - means) / scales
        price_history = prices[start - 7 * 96 : start].reshape(7, 96)
        pooled = float(price_history.std(ddof=1))
        if not np.isfinite(pooled) or pooled <= 0:
            pooled = 1.0
        slot = price_history.std(axis=0, ddof=1)
        loss_scales.append(
            np.maximum(slot, config.data.loss_scale_floor_ratio * pooled)
        )
        expert_samples.append(expert_day)
        gate_samples.append(gate_day)
        sample_indices.append(np.arange(start, stop))
    indices = np.stack(sample_indices)
    targets = prices[indices]
    return FLDMoESplit(
        name=name,
        expert_features=torch.from_numpy(np.stack(expert_samples)),
        target=torch.from_numpy(targets.astype(np.float32)),
        targets_physical=targets,
        loss_scale=torch.from_numpy(np.stack(loss_scales).astype(np.float32)),
        bounds=torch.from_numpy(bounds[indices].astype(np.float32)),
        gate_features=torch.from_numpy(np.stack(gate_samples)),
        timestamps_ns=timestamps.to_numpy(dtype="datetime64[ns]").astype(np.int64)[
            indices
        ],
    )


def prepare_data(config: FLDMoEConfig) -> PreparedFLDMoEData:
    """生成固定特征、382/30/45 个日样本及训练价格段责任度。"""
    frame = load_data(config.data_path)
    timestamps = pd.DatetimeIndex(frame[TIME_COLUMN])
    bounds = config.dataset.bounds_for(timestamps)
    prices = frame[TARGET_COLUMN].to_numpy(dtype=np.float64)
    if np.any(prices < bounds[:, 0]) or np.any(prices > bounds[:, 1]):
        raise ValueError("原始电价超出观测当时的行政限价")
    normalized_frame = frame.copy()
    normalized_frame[TARGET_COLUMN] = (prices - bounds[:, 0]) / np.diff(bounds, axis=1)[
        :, 0
    ]
    main, gate, rolling = build_features(normalized_frame)
    if (
        not np.isfinite(main.to_numpy()).all()
        or not np.isfinite(
            gate.iloc[config.dataset.gate_history_start_day * 96 :].to_numpy()
        ).all()
    ):
        raise ValueError("固定特征在原实验暖机后包含非有限值")
    total_days = len(frame) // 96
    train_end = total_days - config.dataset.test_days - config.dataset.validation_days
    validation_end = total_days - config.dataset.test_days
    arguments = (main, gate, rolling, timestamps, prices, bounds, config)
    train = _build_split(
        "train", range(config.dataset.warmup_days, train_end), *arguments
    )
    validation = _build_split(
        "validation", range(train_end, validation_end), *arguments
    )
    test = _build_split("test", range(validation_end, total_days), *arguments)
    if train.target.shape[0] <= config.xgboost_gate.inner_validation_days:
        raise ValueError("训练数据不足以划分树门控内部验证集")
    responsibility, metadata = build_price_segment_responsibilities(
        train.target.numpy(),
        train.bounds.numpy(),
        config.model.num_experts,
        config.model.cut_quantiles,
    )
    return PreparedFLDMoEData(
        feature_columns=tuple(main.columns),
        gate_feature_columns=tuple(gate.columns),
        gate_rolling_columns=rolling,
        train=train,
        validation=validation,
        test=test,
        price_segment_responsibility=torch.from_numpy(responsibility),
        price_segment_metadata=metadata,
    )
