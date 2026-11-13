"""集中加载陕西固定参数并验证训练契约。"""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass, fields
from datetime import date
from pathlib import Path
from typing import get_type_hints

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DatasetConfig:
    """保存输入路径、时间切分和陕西行政限价。"""

    path: str
    validation_days: int
    test_days: int
    warmup_days: int
    main_history_start_day: int
    gate_history_start_day: int
    tail_threshold: float
    price_bounds: list[dict]

    def bounds_for(self, timestamps: pd.DatetimeIndex) -> np.ndarray:
        """按观测所属日期返回逐点的行政价格下界与上界。"""
        result = np.full((len(timestamps), 2), np.nan, dtype=np.float64)
        for rule in self.price_bounds:
            mask = timestamps >= pd.Timestamp(rule["from"])
            result[mask] = [rule["floor"], rule["cap"]]
        if not np.isfinite(result).all():
            raise ValueError("部分时间戳没有对应的行政限价")
        return result


@dataclass(frozen=True)
class DataConfig:
    """保存日样本步长、历史标准化和损失尺度参数。"""

    train_stride: int
    feature_normalization_window_days: int
    loss_scale_floor_ratio: float


@dataclass(frozen=True)
class ModelConfig:
    """保存独立 TCN 专家结构和初始价格段分位切点。"""

    width: int
    num_layers: int
    kernel_size: int
    dropout: float
    num_experts: int
    cut_quantiles: list[float]


@dataclass(frozen=True)
class DistributionConfig:
    """保存等距节点数、经验初始化和优化参数。"""

    grid_points: int
    initialization_split_price: float
    backbone_learning_rate: float
    backbone_weight_decay: float
    head_learning_rate: float
    head_weight_decay: float
    gradient_clip_norm: float
    epsilon: float


@dataclass(frozen=True)
class ScheduleConfig:
    """保存线性退火起点、终点及持续轮数。"""

    start: float
    end: float
    warmup_epochs: int
    anneal_start_epoch: int


@dataclass(frozen=True)
class ExpertTrainingConfig:
    """保存专家训练、学习率调度和验证早停参数。"""

    max_epochs: int
    patience: int
    min_delta: float
    mass_floor_ratio: float
    scheduler_factor: float
    scheduler_patience: int
    scheduler_min_learning_rate: float
    price_segment_weight_schedule: ScheduleConfig
    responsibility_temperature_schedule: ScheduleConfig


@dataclass(frozen=True)
class XGBoostGateConfig:
    """保存软目标树门控和部署温度参数。"""

    max_depth: int
    max_boost_rounds: int
    early_stopping_rounds: int
    inner_validation_days: int
    prior_uniform_epsilon: float
    gate_temperature: float
    learning_rate: float
    subsample: float
    colsample_bytree: float
    min_child_weight: float
    max_delta_step: float
    nthread: int


@dataclass(frozen=True)
class RuntimeConfig:
    """保存随机种子、设备和批量配置。"""

    seed: int
    device: str
    batch_size: int
    inference_batch_size: int
    num_workers: int


@dataclass(frozen=True)
class FLDMoEConfig:
    """组合算法配置及可追溯的输入路径。"""

    dataset: DatasetConfig
    data: DataConfig
    model: ModelConfig
    distribution: DistributionConfig
    expert_training: ExpertTrainingConfig
    xgboost_gate: XGBoostGateConfig
    runtime: RuntimeConfig
    source_path: Path
    source_text: str
    data_path: Path

    @property
    def protection_epochs(self) -> int:
        """返回两个退火调度结束后完成保护阶段所需的轮数。"""
        schedules = (
            self.expert_training.price_segment_weight_schedule,
            self.expert_training.responsibility_temperature_schedule,
        )
        return max(s.anneal_start_epoch + s.warmup_epochs for s in schedules) + 1


def _section(cls: type, payload: dict):
    """拒绝缺失、多余、类型错误和非有限的配置字段。"""
    expected = {f.name for f in fields(cls)}
    if set(payload) != expected:
        raise ValueError(
            f"{cls.__name__}: missing={expected - set(payload)}, "
            f"unknown={set(payload) - expected}"
        )
    hints = get_type_hints(cls)
    for name, value in payload.items():
        kind = hints[name]
        if kind in (int, float):
            valid = type(value) is int if kind is int else type(value) in (int, float)
            if not valid or not math.isfinite(value):
                raise ValueError(f"{cls.__name__}.{name} 类型错误或非有限")
        elif kind is str and not isinstance(value, str):
            raise ValueError(f"{cls.__name__}.{name} 必须为字符串")
    return cls(**payload)


def load_config(path: str | Path = "configs/shaanxi.toml") -> FLDMoEConfig:
    """读取相对配置文件定位的路径，并验证最小实现支持的参数。"""
    path = Path(path).resolve()
    source = path.read_text(encoding="utf-8")
    payload = tomllib.loads(source)
    sections = {
        "dataset": DatasetConfig,
        "data": DataConfig,
        "model": ModelConfig,
        "distribution": DistributionConfig,
        "expert_training": ExpertTrainingConfig,
        "xgboost_gate": XGBoostGateConfig,
        "runtime": RuntimeConfig,
    }
    if set(payload) != set(sections):
        raise ValueError(
            "配置必须且只能包含 dataset/data/model/distribution/"
            "expert_training/xgboost_gate/runtime"
        )
    for name in (
        "price_segment_weight_schedule",
        "responsibility_temperature_schedule",
    ):
        payload["expert_training"][name] = _section(
            ScheduleConfig, payload["expert_training"][name]
        )
    values = {name: _section(cls, payload[name]) for name, cls in sections.items()}
    dataset = values["dataset"]
    config = FLDMoEConfig(
        **values,
        source_path=path,
        source_text=source,
        data_path=(path.parent / dataset.path).resolve(),
    )
    _validate(config)
    return config


def _validate(c: FLDMoEConfig) -> None:
    """校验取值范围以及样本、网格和调度之间的约束。"""
    positive = [
        c.dataset.validation_days,
        c.dataset.test_days,
        c.data.train_stride,
        c.data.feature_normalization_window_days,
        c.data.loss_scale_floor_ratio,
        c.model.width,
        c.model.num_layers,
        c.model.kernel_size,
        c.distribution.backbone_learning_rate,
        c.distribution.head_learning_rate,
        c.distribution.gradient_clip_norm,
        c.distribution.epsilon,
        c.expert_training.patience,
        c.expert_training.mass_floor_ratio,
        c.xgboost_gate.max_depth,
        c.xgboost_gate.max_boost_rounds,
        c.xgboost_gate.early_stopping_rounds,
        c.xgboost_gate.inner_validation_days,
        c.xgboost_gate.gate_temperature,
        c.xgboost_gate.learning_rate,
        c.runtime.batch_size,
        c.runtime.inference_batch_size,
    ]
    if any(v <= 0 for v in positive):
        raise ValueError("窗口、批量、学习率及必要的正数参数必须大于零")
    if c.data.train_stride != 96:
        raise ValueError("最小实现仅支持 train_stride=96 的完整自然日样本")
    if c.model.kernel_size % 2 != 1 or c.model.num_experts < 2:
        raise ValueError("TCN 要求奇数卷积核及至少两个专家")
    q = np.asarray(c.model.cut_quantiles, dtype=np.float64)
    if (
        q.shape != (c.model.num_experts - 1,)
        or not np.isfinite(q).all()
        or np.any(q <= 0)
        or np.any(q >= 1)
        or np.any(np.diff(q) <= 0)
    ):
        raise ValueError("价格段分位切点必须严格递增并位于 (0,1)")
    if not 0 <= c.model.dropout < 1 or c.distribution.grid_points < 5:
        raise ValueError("dropout 必须位于 [0,1)，网格至少五个节点")
    if not 0 < c.expert_training.scheduler_factor < 1:
        raise ValueError("学习率衰减因子必须位于 (0,1)")
    for v in (c.xgboost_gate.subsample, c.xgboost_gate.colsample_bytree):
        if not 0 < v <= 1:
            raise ValueError("树采样比例必须位于 (0,1]")
    if not 0 <= c.xgboost_gate.prior_uniform_epsilon <= 1:
        raise ValueError("门控先验混合比例必须位于 [0,1]")
    for schedule in (
        c.expert_training.price_segment_weight_schedule,
        c.expert_training.responsibility_temperature_schedule,
    ):
        if schedule.warmup_epochs < 1 or schedule.anneal_start_epoch < 0:
            raise ValueError("退火长度必须为正数，起点必须非负")
    s = c.expert_training.price_segment_weight_schedule
    if not (0 <= s.start <= 1 and 0 <= s.end <= 1):
        raise ValueError("价格段责任混合权重必须位于 [0,1]")
    s = c.expert_training.responsibility_temperature_schedule
    if min(s.start, s.end) <= 0:
        raise ValueError("责任温度必须大于零")
    if c.expert_training.max_epochs <= c.protection_epochs:
        raise ValueError("max_epochs 必须大于保护期轮数")
    if c.runtime.device not in ("auto", "cpu", "cuda"):
        raise ValueError("device 必须是 auto/cpu/cuda")
    if c.runtime.num_workers < 0 or not 0 <= c.runtime.seed < 2**32:
        raise ValueError("num_workers 必须非负，seed 必须位于 [0,2**32)")
    if c.xgboost_gate.nthread == 0 or c.xgboost_gate.nthread < -1:
        raise ValueError("nthread 必须为 -1 或正整数")
    nonnegative = (
        c.distribution.backbone_weight_decay,
        c.distribution.head_weight_decay,
        c.expert_training.min_delta,
        c.expert_training.scheduler_patience,
        c.expert_training.scheduler_min_learning_rate,
        c.xgboost_gate.min_child_weight,
        c.xgboost_gate.max_delta_step,
    )
    if min(nonnegative) < 0:
        raise ValueError("正则、早停容差和调度器下限等参数不得为负")
    if (
        c.dataset.main_history_start_day != 7
        or c.dataset.gate_history_start_day != 14
        or c.dataset.warmup_days < 28
    ):
        raise ValueError("保留原始暖机口径：专家第8天、门控第15天、标签第29天")
    previous = None
    for rule in c.dataset.price_bounds:
        if set(rule) != {"from", "floor", "cap"}:
            raise ValueError("限价规则必须包含 from/floor/cap")
        current = date.fromisoformat(rule["from"])
        lo, hi = rule["floor"], rule["cap"]
        if previous is not None and current <= previous:
            raise ValueError("限价规则必须按生效日期严格递增")
        if not all(math.isfinite(v) for v in (lo, hi)) or lo >= hi:
            raise ValueError("行政限价必须有限且下界小于上界")
        if not lo < c.distribution.initialization_split_price < hi:
            raise ValueError("经验初始化分割价必须位于行政限价内部")
        if not lo <= c.dataset.tail_threshold < hi:
            raise ValueError("twCRPS 阈值必须位于行政限价范围内")
        previous = current
    if previous is None:
        raise ValueError("行政限价规则不能为空")
