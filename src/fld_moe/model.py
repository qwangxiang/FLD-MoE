"""定义多个独立双向膨胀 TCN 专家及其完整 PMF 单头。"""

from __future__ import annotations

import torch
from torch import nn


class _TemporalBlock(nn.Module):
    """实现长度不变的双层双向膨胀卷积残差块。"""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        """构造卷积主路和必要的 1×1 残差投影。"""
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(
            input_channels,
            output_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.conv2 = nn.Conv1d(
            output_channels,
            output_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.residual = (
            nn.Conv1d(input_channels, output_channels, 1)
            if input_channels != output_channels
            else nn.Identity()
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """把 ``(B,C,T)`` 输入映射为同长度残差表示。"""
        hidden = self.dropout(self.activation(self.conv1(values)))
        hidden = self.dropout(self.activation(self.conv2(hidden)))
        return self.activation(hidden + self.residual(values))


class ExpertTCN(nn.Module):
    """沿目标日 96 时段提取上下文的独立双向 TCN 专家。"""

    def __init__(
        self,
        input_dim: int,
        width: int,
        num_layers: int,
        kernel_size: int,
        dropout: float,
    ) -> None:
        """按逐层翻倍的膨胀率构造等宽残差块。"""
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("双向对称 TCN 的卷积核必须为奇数")
        blocks: list[nn.Module] = []
        input_channels = input_dim
        for layer in range(num_layers):
            blocks.append(
                _TemporalBlock(
                    input_channels,
                    width,
                    kernel_size,
                    2**layer,
                    dropout,
                )
            )
            input_channels = width
        self.network = nn.Sequential(*blocks)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """把 ``(B,T,F)`` 目标日特征映射为 ``(B,T,W)``。"""
        if values.ndim != 3:
            raise ValueError("TCN 专家输入必须是 (B,T,F)")
        return self.network(values.transpose(1, 2)).transpose(1, 2)


class ProbabilityMassHead(nn.Module):
    """用唯一线性层输出一个专家的完整等间距价格网格分数。"""

    def __init__(self, input_dim: int, initial_pmf: torch.Tensor) -> None:
        """以训练集经验 PMF 初始化偏置并将输入权重置零。"""
        super().__init__()
        if initial_pmf.ndim != 1 or initial_pmf.numel() < 5:
            raise ValueError("专家初始 PMF 必须是至少五槽的一维张量")
        if not bool(torch.isfinite(initial_pmf).all()) or bool(
            (initial_pmf <= 0.0).any()
        ):
            raise ValueError("专家初始 PMF 必须是有限正数")
        if not bool(
            torch.allclose(
                initial_pmf.sum(),
                initial_pmf.new_tensor(1.0),
                atol=1e-6,
                rtol=0.0,
            )
        ):
            raise ValueError("专家初始 PMF 概率质量必须为 1")
        self.linear = nn.Linear(input_dim, int(initial_pmf.numel()))
        nn.init.zeros_(self.linear.weight)
        with torch.no_grad():
            self.linear.bias.copy_(initial_pmf.log())

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """把专家隐藏表示映射为完整分布原始分数。"""
        return self.linear(hidden)


class FLDMoE(nn.Module):
    """组合多个参数独立的 TCN 专家与各自的概率质量输出头。"""

    def __init__(
        self,
        *,
        input_dim: int,
        width: int,
        num_layers: int,
        kernel_size: int,
        dropout: float,
        num_experts: int,
        grid_points: int,
        initial_expert_pmf: torch.Tensor,
    ) -> None:
        """构造固定专家数的独立骨干和分布头。"""
        super().__init__()
        expected = (num_experts, grid_points)
        if tuple(initial_expert_pmf.shape) != expected:
            raise ValueError(
                f"专家初始 PMF 形状必须是 {expected}，当前 "
                f"{tuple(initial_expert_pmf.shape)}"
            )
        self.num_experts = num_experts
        self.input_dim = input_dim
        self.width = width
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.experts = nn.ModuleList(
            [
                ExpertTCN(
                    input_dim,
                    width,
                    num_layers,
                    kernel_size,
                    dropout,
                )
                for _ in range(num_experts)
            ]
        )
        self.heads = nn.ModuleList(
            [
                ProbabilityMassHead(width, initial_expert_pmf[index])
                for index in range(num_experts)
            ]
        )

    @property
    def receptive_field(self) -> int:
        """返回无边界截断时 TCN 的理论时序感受野。"""
        return 1 + 2 * (self.kernel_size - 1) * (2**self.num_layers - 1)

    @property
    def grid_points(self) -> int:
        """返回所有专家一致的完整分布槽位数。"""
        dimensions = {head.linear.out_features for head in self.heads}
        if len(dimensions) != 1:
            raise RuntimeError("专家完整分布头的输出维度不一致")
        return dimensions.pop()

    def encode_experts(self, expert_features: torch.Tensor) -> torch.Tensor:
        """返回形状为 ``(B,T,K,W)`` 的全部专家表示。"""
        return torch.stack([expert(expert_features) for expert in self.experts], dim=2)

    def distribution_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """返回形状为 ``(B,T,K,G)`` 的全部完整分布分数。"""
        if hidden.ndim != 4 or hidden.shape[2] != self.num_experts:
            raise ValueError("专家隐藏表示形状错误")
        return torch.stack(
            [head(hidden[:, :, index]) for index, head in enumerate(self.heads)],
            dim=2,
        )

    def forward(self, expert_features: torch.Tensor) -> torch.Tensor:
        """由目标日特征直接返回全部专家的完整分布分数。"""
        return self.distribution_logits(self.encode_experts(expert_features))


__all__ = ["FLDMoE"]
