"""实现等间距价格网格上的专家责任训练损失与混合 CRPS。"""

from __future__ import annotations

import torch
from torch import nn

from fld_moe.distribution import DistributionOutput


def _loss_scale_per_slot(loss_scale: torch.Tensor, steps: int) -> torch.Tensor:
    """把逐样本或逐时段尺度统一为 ``(B,T)``。"""
    values = loss_scale.view(loss_scale.shape[0], -1)
    if values.shape[1] == 1:
        values = values.expand(-1, steps)
    if values.shape[1] != steps or not bool(torch.isfinite(values).all()):
        raise ValueError("损失尺度无法对齐目标时段或包含非有限值")
    return values.clamp_min(1e-6)


def physical_crps_from_pmf(
    pmf: torch.Tensor,
    price_grid: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """在非等距物理价格网格上积分离散 CDF 的 CRPS。"""
    if pmf.shape != price_grid.shape or target.shape != pmf.shape[:-1]:
        raise ValueError("物理 CRPS 的 PMF、支撑或目标形状不一致")
    if not bool(
        torch.isfinite(pmf).all()
        and torch.isfinite(price_grid).all()
        and torch.isfinite(target).all()
    ):
        raise ValueError("物理 CRPS 输入包含非有限值")
    interval = price_grid[..., 1:] - price_grid[..., :-1]
    if bool((interval < 0.0).any()):
        raise ValueError("物理价格网格必须保持非降顺序")
    cdf = pmf.cumsum(dim=-1)[..., :-1]
    left_width = torch.minimum(
        torch.clamp(target.unsqueeze(-1) - price_grid[..., :-1], min=0.0),
        interval,
    )
    right_width = interval - left_width
    return (left_width * cdf.square() + right_width * (cdf - 1.0).square()).sum(dim=-1)


def per_expert_physical_crps(
    output: DistributionOutput,
    target: torch.Tensor,
    loss_scale: torch.Tensor,
) -> torch.Tensor:
    """返回形状为 ``(B,T,K)`` 的尺度归一完整专家 CRPS。"""
    if output.expert_pmf.ndim != 4:
        raise ValueError("完整专家 PMF 必须是 (B,T,K,G)")
    experts = output.expert_pmf.shape[2]
    price_grid = output.price_grid.unsqueeze(2).expand(-1, -1, experts, -1)
    expert_target = target.unsqueeze(-1).expand(-1, -1, experts)
    crps = physical_crps_from_pmf(output.expert_pmf, price_grid, expert_target)
    return crps / _loss_scale_per_slot(loss_scale, target.shape[1]).unsqueeze(-1)


def mixture_physical_crps(
    output: DistributionOutput,
    target: torch.Tensor,
    loss_scale: torch.Tensor,
) -> torch.Tensor:
    """返回部署混合分布的平均尺度归一物理 CRPS。"""
    crps = physical_crps_from_pmf(output.pmf, output.price_grid, target)
    return (crps / _loss_scale_per_slot(loss_scale, target.shape[1])).mean()


def _responsibility_average(
    values: torch.Tensor,
    responsibility: torch.Tensor,
    mass_floor_ratio: float,
) -> torch.Tensor:
    """按各专家责任质量归一后对专家等权平均。"""
    if values.shape != responsibility.shape:
        raise ValueError("逐专家损失与责任形状不一致")
    mass = responsibility.sum(dim=(0, 1))
    points = float(responsibility.shape[0] * responsibility.shape[1])
    floor = mass_floor_ratio * points / responsibility.shape[2]
    return ((responsibility * values).sum(dim=(0, 1)) / mass.clamp_min(floor)).mean()


class DistributionLoss(nn.Module):
    """仅保留陕西配置中启用的责任加权、质量归一化完整 CRPS。"""

    def forward(
        self,
        output: DistributionOutput,
        target: torch.Tensor,
        loss_scale: torch.Tensor,
        responsibility: torch.Tensor,
        mass_floor_ratio: float,
        per_expert_crps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """停止责任梯度并对各专家归一化 CRPS 等权平均。"""
        responsibility = responsibility.detach()
        if not bool(torch.isfinite(responsibility).all()) or bool(
            (responsibility < 0).any()
        ):
            raise ValueError("责任度必须为有限非负值")
        if not bool(
            torch.allclose(
                responsibility.sum(dim=-1),
                torch.ones_like(responsibility[..., 0]),
                atol=1e-6,
                rtol=0,
            )
        ):
            raise ValueError("责任度必须逐行归一")
        values = (
            per_expert_physical_crps(output, target, loss_scale)
            if per_expert_crps is None
            else per_expert_crps
        )
        return _responsibility_average(values, responsibility, mass_floor_ratio)
