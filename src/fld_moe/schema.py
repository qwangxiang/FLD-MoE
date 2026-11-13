"""定义训练数据与预测结果的本地契约。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

POINTS_PER_DAY = 96


@dataclass(frozen=True)
class FLDMoESplit:
    """保存一个严格对齐的 FLDMoE 数据切分。"""

    name: str
    expert_features: torch.Tensor
    target: torch.Tensor
    targets_physical: np.ndarray
    loss_scale: torch.Tensor
    bounds: torch.Tensor
    gate_features: torch.Tensor
    timestamps_ns: np.ndarray

    def __post_init__(self) -> None:
        """校验样本、时段、数值和时间戳严格对齐。"""
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("数据切分名称不能为空")
        if (
            self.expert_features.ndim != 3
            or self.expert_features.shape[1] != POINTS_PER_DAY
        ):
            raise ValueError(f"{self.name} expert_features 形状错误")
        samples = int(self.expert_features.shape[0])
        if samples < 1 or self.expert_features.shape[2] < 1:
            raise ValueError(f"{self.name} expert_features 必须非空")
        expected = {
            "target": (samples, POINTS_PER_DAY),
            "targets_physical": (samples, POINTS_PER_DAY),
            "loss_scale": (samples, POINTS_PER_DAY),
            "bounds": (samples, POINTS_PER_DAY, 2),
            "gate_features": (
                samples,
                POINTS_PER_DAY,
                int(self.gate_features.shape[-1]),
            ),
            "timestamps_ns": (samples, POINTS_PER_DAY),
        }
        for name, shape in expected.items():
            if getattr(self, name).shape != shape:
                raise ValueError(f"{self.name} {name} 形状错误")
        tensors = (
            self.expert_features,
            self.target,
            self.loss_scale,
            self.bounds,
            self.gate_features,
        )
        if any(tensor.dtype != torch.float32 for tensor in tensors):
            raise ValueError(f"{self.name} 数值张量必须是 float32")
        if any(not bool(torch.isfinite(tensor).all()) for tensor in tensors):
            raise ValueError(f"{self.name} 数值张量包含非有限值")
        if self.targets_physical.dtype != np.float64:
            raise ValueError(f"{self.name} 原始物理真值必须是 float64")
        if not np.isfinite(self.targets_physical).all():
            raise ValueError(f"{self.name} 原始物理真值包含非有限值")
        if not np.array_equal(
            self.target.detach().cpu().numpy(),
            self.targets_physical.astype(np.float32),
        ):
            raise ValueError(f"{self.name} 训练真值与原始物理真值不一致")
        if self.gate_features.shape[-1] < 1:
            raise ValueError(f"{self.name} 门控特征不能为空")
        if bool((self.loss_scale <= 0.0).any()):
            raise ValueError(f"{self.name} 尺度必须为正数")
        if bool((self.bounds[..., 0] >= self.bounds[..., 1]).any()):
            raise ValueError(f"{self.name} 价格边界必须满足下界小于上界")
        if self.timestamps_ns.dtype != np.int64:
            raise ValueError(f"{self.name} 时间戳必须是 int64 纳秒")
        if np.any(np.diff(self.timestamps_ns, axis=1) <= 0):
            raise ValueError(f"{self.name} 单样本时间戳必须严格递增")
        if samples > 1 and np.any(np.diff(self.timestamps_ns[:, 0]) <= 0):
            raise ValueError(f"{self.name} 样本起点必须严格递增")


@dataclass(frozen=True)
class PreparedFLDMoEData:
    """保存固定双流特征、三段数据和训练集价格段初始化。"""

    feature_columns: tuple[str, ...]
    gate_feature_columns: tuple[str, ...]
    gate_rolling_columns: tuple[str, ...]
    train: FLDMoESplit
    validation: FLDMoESplit
    test: FLDMoESplit
    price_segment_responsibility: torch.Tensor
    price_segment_metadata: dict[str, object]


@dataclass(frozen=True)
class DiscreteForecast:
    """保存原生价格节点及概率质量，不投影为分位数分布。"""

    split: str
    timestamps_ns: np.ndarray
    y_true: np.ndarray
    price_grid: np.ndarray
    probability_mass: np.ndarray

    def __post_init__(self) -> None:
        """拒绝空预测、非法概率、无序时间或不对齐的物理量。"""
        n = self.y_true.size
        if n < 1 or self.y_true.shape != (n,) or self.timestamps_ns.shape != (n,):
            raise ValueError("预测时间戳与真值必须为非空同长向量")
        if self.price_grid.ndim != 2 or self.price_grid.shape[0] != n:
            raise ValueError("价格坐标必须为 (N,G)")
        if self.price_grid.shape != self.probability_mass.shape:
            raise ValueError("概率质量必须与价格坐标逐项对齐")
        if self.price_grid.shape[1] < 2 or any(
            not np.isfinite(v).all()
            for v in (self.y_true, self.price_grid, self.probability_mass)
        ):
            raise ValueError("预测价格、概率和真值必须有限，网格至少两个节点")
        if self.timestamps_ns.dtype != np.int64 or np.any(
            np.diff(self.timestamps_ns) <= 0
        ):
            raise ValueError("时间戳必须为严格递增的 int64 纳秒")
        if np.any(self.timestamps_ns == np.iinfo(np.int64).min):
            raise ValueError("时间戳不得包含 NaT")
        if (
            np.any(np.diff(self.price_grid, axis=1) <= 0)
            or np.any(self.probability_mass < 0)
            or not np.allclose(self.probability_mass.sum(axis=1), 1, atol=1e-5, rtol=0)
        ):
            raise ValueError("网格必须严格递增，概率必须非负且总和为一")
        if np.any(self.y_true < self.price_grid[:, 0]) or np.any(
            self.y_true > self.price_grid[:, -1]
        ):
            raise ValueError("真值超出对应行政限价")

    def quantile_prediction(self, level: float) -> np.ndarray:
        """返回累计质量首次达到指定概率水平的价格节点。"""
        if not np.isfinite(level) or not 0 < level < 1:
            raise ValueError("分位水平必须位于 (0,1)")
        indices = (self.probability_mass.cumsum(axis=1) < level).sum(axis=1)
        indices = np.minimum(indices, self.price_grid.shape[1] - 1)
        return self.price_grid[np.arange(self.y_true.size), indices]
