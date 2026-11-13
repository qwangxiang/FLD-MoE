"""在行政限价间构造等距价格节点并混合专家概率。"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from fld_moe.config import DistributionConfig
from fld_moe.model import FLDMoE


@dataclass(frozen=True)
class DistributionOutput:
    """保存专家及部署混合的原生离散分布。"""

    pmf: torch.Tensor
    expert_pmf: torch.Tensor
    price_grid: torch.Tensor


def build_uniform_price_grid(
    bounds: torch.Tensor, config: DistributionConfig
) -> torch.Tensor:
    """在逐时行政上下限之间构造包含端点的严格等距坐标。"""
    if bounds.ndim != 3 or bounds.shape[1:] != (96, 2):
        raise ValueError("价格边界必须是 (B,96,2)")
    if not bool(torch.isfinite(bounds).all()):
        raise ValueError("行政边界包含非有限值")
    floor, cap = bounds.unbind(dim=-1)
    if bool((floor >= cap).any()):
        raise ValueError("行政下界必须严格小于上界")
    fraction = torch.linspace(
        0, 1, config.grid_points, device=bounds.device, dtype=bounds.dtype
    )
    grid = floor.unsqueeze(-1) + (cap - floor).unsqueeze(-1) * fraction
    grid[..., 0], grid[..., -1] = floor, cap
    if not bool(torch.isfinite(grid).all()) or bool((grid.diff(dim=-1) <= 0).any()):
        raise ValueError("价格网格必须有限且严格递增")
    return grid


def forward_distribution(
    model: FLDMoE,
    expert_features: torch.Tensor,
    gate_logits: torch.Tensor,
    bounds: torch.Tensor,
    gate_temperature: float,
    config: DistributionConfig,
) -> DistributionOutput:
    """以温度归一化门控分数，混合相同节点上的完整专家 PMF。"""
    price_grid = build_uniform_price_grid(bounds, config)
    expert_logits = model(expert_features)
    if gate_logits.shape != expert_logits.shape[:3]:
        raise ValueError("门控分数与专家输出的前三维不一致")
    expert_pmf = torch.softmax(expert_logits, dim=-1)
    weights = torch.softmax(gate_logits.detach() / gate_temperature, dim=-1)
    pmf = (weights.unsqueeze(-1) * expert_pmf).sum(dim=2)
    if not bool(torch.isfinite(pmf).all()) or not bool(
        torch.allclose(
            pmf.sum(dim=-1),
            torch.ones_like(pmf[..., 0]),
            atol=1e-5,
            rtol=0,
        )
    ):
        raise ValueError("专家混合概率必须有限且总质量为一")
    return DistributionOutput(pmf, expert_pmf, price_grid)
