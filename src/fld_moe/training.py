"""训练 FLD-MoE 的神经专家、软责任与 XGBoost 树门控。"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from fld_moe.config import FLDMoEConfig, ScheduleConfig
from fld_moe.distribution import (
    build_uniform_price_grid,
    forward_distribution,
)
from fld_moe.gate import (
    GateDataSplit,
    XGBoostGateFitResult,
    XGBoostGateModel,
    fit_xgboost_gate,
    gate_log_prior,
    split_gate_training_data,
)
from fld_moe.loss import (
    DistributionLoss,
    mixture_physical_crps,
    per_expert_physical_crps,
)
from fld_moe.model import FLDMoE
from fld_moe.schema import DiscreteForecast, FLDMoESplit, PreparedFLDMoEData


class FLDMoETrainingError(RuntimeError):
    """表示设备、联合训练、树门控或预测过程失败。"""


@dataclass(frozen=True)
class FLDMoETrainingHistory:
    """保存配对早停结果和逐 epoch 的必要训练轨迹。"""

    epochs: tuple[dict[str, Any], ...]
    protection_epochs: int
    best_epoch: int
    best_validation_crps: float
    stopped_epoch: int
    early_stopped: bool
    gate_split: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        """返回可直接写入 JSON 的训练历史。"""
        return {
            "protection_epochs": self.protection_epochs,
            "best_epoch": self.best_epoch,
            "best_validation_crps": self.best_validation_crps,
            "stopped_epoch": self.stopped_epoch,
            "early_stopped": self.early_stopped,
            "gate_split": self.gate_split,
            "epochs": list(self.epochs),
            "optimizer_contract": {
                "optimizer_count": 1,
                "forward_per_batch": 1,
                "backward_per_batch": 1,
                "step_per_batch": 1,
                "parameter_groups": ["tcn", "head"],
            },
        }


@dataclass(frozen=True)
class FLDMoEFitResult:
    """保存最佳神经专家、同 epoch 树门控和训练元数据。"""

    model: FLDMoE
    xgboost_gate: XGBoostGateModel
    history: FLDMoETrainingHistory
    data: PreparedFLDMoEData
    model_init: dict[str, int | float]
    initial_expert_pmf: torch.Tensor


@dataclass(frozen=True)
class _PairedCheckpoint:
    """保存同一联合 epoch 的网络权重和树门控。"""

    model_state: dict[str, torch.Tensor]
    xgboost_gate: XGBoostGateModel
    epoch: int
    validation_crps: float


@dataclass(frozen=True)
class _ExpertTrainingState:
    """保存保护阶段与联合阶段共用的专家训练对象。"""

    model: FLDMoE
    model_init: dict[str, int | float]
    initial_pmf: torch.Tensor
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau


def set_seed(seed: int) -> None:
    """固定 Python、NumPy、CPU 和 CUDA 的随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(name: str) -> torch.device:
    """按参数解析训练设备并拒绝不可用的显式 CUDA。"""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise FLDMoETrainingError("参数要求 CUDA，但当前环境不可用")
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    raise FLDMoETrainingError(f"不支持的训练设备: {name!r}")


def _loader_options(
    config: FLDMoEConfig,
    device: torch.device,
) -> dict[str, int | bool]:
    """返回与设备一致的 DataLoader 并行和页锁内存参数。"""
    workers = config.runtime.num_workers
    return {
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": workers > 0,
    }


def _to_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """使用页锁内存条件下的异步设备迁移。"""
    return tensor.to(device, non_blocking=device.type == "cuda")


def scheduled_value(epoch: int, schedule: ScheduleConfig) -> float:
    """按保护起点和线性退火长度计算当前 epoch 的调度值。"""
    if epoch < schedule.anneal_start_epoch:
        return schedule.start
    progress = min(
        1.0,
        (epoch - schedule.anneal_start_epoch) / schedule.warmup_epochs,
    )
    return schedule.start + (schedule.end - schedule.start) * progress


def _linear_price_grid_mass(
    target: torch.Tensor,
    price_grid: torch.Tensor,
    config: FLDMoEConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """在真值所属物理价格段内分配初始化质量并处理段端点。"""
    if target.shape != price_grid.shape[:2]:
        raise ValueError("初始化目标与价格坐标形状不一致")
    if not bool(torch.isfinite(target).all()):
        raise ValueError("初始化目标包含非有限值")
    split_price = config.distribution.initialization_split_price
    bulk_slots = price_grid < split_price
    segment_mask = torch.where(
        (target < split_price).unsqueeze(-1),
        bulk_slots,
        ~bulk_slots,
    )
    allowed = segment_mask
    if not bool(allowed.any(dim=-1).all()):
        raise ValueError("初始化目标所属价格段没有有效坐标")
    nearest = (
        (target.unsqueeze(-1) - price_grid)
        .abs()
        .masked_fill(
            ~allowed,
            torch.inf,
        )
        .argmin(dim=-1)
    )
    difference = target.unsqueeze(-1) - price_grid
    left_distance = difference.masked_fill((difference < 0.0) | ~allowed, torch.inf)
    right_distance = (-difference).masked_fill(
        (difference > 0.0) | ~allowed,
        torch.inf,
    )
    left = torch.where(
        torch.isfinite(left_distance).any(dim=-1),
        left_distance.argmin(dim=-1),
        nearest,
    )
    right = torch.where(
        torch.isfinite(right_distance).any(dim=-1),
        right_distance.argmin(dim=-1),
        nearest,
    )
    left_value = price_grid.gather(-1, left.unsqueeze(-1)).squeeze(-1)
    right_value = price_grid.gather(-1, right.unsqueeze(-1)).squeeze(-1)
    interval = right_value - left_value
    right_weight = torch.where(
        interval > 0.0,
        ((target - left_value) / interval.clamp_min(1e-12)).clamp(0.0, 1.0),
        torch.zeros_like(target),
    )
    return left, right, 1.0 - right_weight, right_weight


def build_initial_expert_pmf(
    split: FLDMoESplit,
    price_segment_responsibility: torch.Tensor,
    config: FLDMoEConfig,
) -> torch.Tensor:
    """只用训练真值和 价格分段责任度构造各专家完整经验 PMF。"""
    target = split.target.to(torch.float32)
    if price_segment_responsibility.shape[:2] != target.shape:
        raise ValueError("价格分段责任度与训练目标不对齐")
    experts = price_segment_responsibility.shape[-1]
    price_segment_mass = price_segment_responsibility.sum(dim=(0, 1))
    if bool((price_segment_mass <= 0.0).any()) or not bool(
        torch.isfinite(price_segment_mass).all()
    ):
        raise ValueError("至少一个专家没有 价格分段责任度质量")
    histograms = torch.zeros(
        experts,
        config.distribution.grid_points,
        dtype=torch.float32,
    )
    batch_size = config.runtime.batch_size
    for start in range(0, target.shape[0], batch_size):
        stop = min(target.shape[0], start + batch_size)
        price_grid = build_uniform_price_grid(
            split.bounds[start:stop],
            config.distribution,
        )
        left, right, left_weight, right_weight = _linear_price_grid_mass(
            target[start:stop],
            price_grid,
            config,
        )
        responsibility = price_segment_responsibility[start:stop]
        for expert in range(experts):
            expert_weight = responsibility[..., expert]
            histograms[expert].scatter_add_(
                0,
                left.reshape(-1),
                (left_weight * expert_weight).reshape(-1),
            )
            histograms[expert].scatter_add_(
                0,
                right.reshape(-1),
                (right_weight * expert_weight).reshape(-1),
            )
    histograms += config.distribution.epsilon
    mass = histograms.sum(dim=-1, keepdim=True)
    if bool((mass <= 0.0).any()) or not bool(torch.isfinite(mass).all()):
        raise ValueError("经验 PMF 初始化产生非法专家质量")
    return histograms / mass


def _build_model(
    data: PreparedFLDMoEData,
    config: FLDMoEConfig,
    initial_pmf: torch.Tensor,
    device: torch.device,
) -> tuple[FLDMoE, dict[str, int | float]]:
    """由数据维度、结构参数和经验 PMF 构造神经专家。"""
    model_init: dict[str, int | float] = {
        "input_dim": len(data.feature_columns),
        "width": config.model.width,
        "num_layers": config.model.num_layers,
        "kernel_size": config.model.kernel_size,
        "dropout": config.model.dropout,
        "num_experts": config.model.num_experts,
        "grid_points": config.distribution.grid_points,
    }
    model = FLDMoE(
        **model_init,
        initial_expert_pmf=initial_pmf,
    ).to(device)
    return model, model_init


def _initialize_expert_training(
    data: PreparedFLDMoEData,
    config: FLDMoEConfig,
    device: torch.device,
) -> _ExpertTrainingState:
    """集中构造专家初始 PMF、模型、优化器和调度器。"""
    initial_pmf = build_initial_expert_pmf(
        data.train,
        data.price_segment_responsibility,
        config,
    )
    model, model_init = _build_model(data, config, initial_pmf, device)
    tcn_parameters = [
        parameter for expert in model.experts for parameter in expert.parameters()
    ]
    head_parameters = [
        parameter for head in model.heads for parameter in head.parameters()
    ]
    optimizer = torch.optim.Adam(
        [
            {
                "name": "tcn",
                "params": tcn_parameters,
                "lr": config.distribution.backbone_learning_rate,
                "weight_decay": config.distribution.backbone_weight_decay,
            },
            {
                "name": "head",
                "params": head_parameters,
                "lr": config.distribution.head_learning_rate,
                "weight_decay": config.distribution.head_weight_decay,
            },
        ]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.expert_training.scheduler_factor,
        patience=config.expert_training.scheduler_patience,
        min_lr=config.expert_training.scheduler_min_learning_rate,
    )
    return _ExpertTrainingState(
        model=model,
        model_init=model_init,
        initial_pmf=initial_pmf,
        optimizer=optimizer,
        scheduler=scheduler,
    )


def _train_epoch(
    model: FLDMoE,
    split: FLDMoESplit,
    price_segment_responsibility: torch.Tensor,
    log_prior: torch.Tensor,
    config: FLDMoEConfig,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    epoch: int,
) -> dict[str, Any]:
    """以一次前向、反向和更新联合训练全部 TCN 与完整 PMF 头。"""
    expected = (*split.target.shape, config.model.num_experts)
    if price_segment_responsibility.shape != expected or log_prior.shape != expected:
        raise ValueError("价格分段责任度或树先验与训练目标形状不一致")
    generator = torch.Generator().manual_seed(config.runtime.seed + epoch)
    loader = DataLoader(
        TensorDataset(
            split.expert_features,
            split.target,
            split.loss_scale,
            split.bounds,
            price_segment_responsibility,
            log_prior,
        ),
        batch_size=config.runtime.batch_size,
        shuffle=True,
        generator=generator,
        **_loader_options(config, device),
    )
    criterion = DistributionLoss().to(device)
    price_segment_weight = scheduled_value(
        epoch,
        config.expert_training.price_segment_weight_schedule,
    )
    temperature = scheduled_value(
        epoch,
        config.expert_training.responsibility_temperature_schedule,
    )
    model.train()
    totals = {"total": 0.0, "full_crps": 0.0}
    sample_count = 0
    for batch in loader:
        (
            expert_features,
            target,
            loss_scale,
            bounds,
            segment,
            prior,
        ) = (_to_device(tensor, device) for tensor in batch)
        optimizer.zero_grad(set_to_none=True)
        output = forward_distribution(
            model,
            expert_features,
            prior,
            bounds,
            config.xgboost_gate.gate_temperature,
            config.distribution,
        )
        expert_crps = per_expert_physical_crps(output, target, loss_scale)
        with torch.no_grad():
            loss_responsibility = torch.softmax(
                prior - expert_crps.detach() / temperature,
                dim=-1,
            )
            responsibility = (
                price_segment_weight * segment
                + (1.0 - price_segment_weight) * loss_responsibility
            ).detach()
        loss = criterion(
            output,
            target,
            loss_scale,
            responsibility,
            config.expert_training.mass_floor_ratio,
            per_expert_crps=expert_crps,
        )
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config.distribution.gradient_clip_norm,
        )
        if not torch.isfinite(gradient_norm):
            raise FLDMoETrainingError("神经专家梯度范数包含非有限值")
        optimizer.step()
        current = int(expert_features.shape[0])
        sample_count += current
        for name in totals:
            totals[name] += float(loss.detach()) * current
    if sample_count == 0:
        raise FLDMoETrainingError("神经专家训练 DataLoader 为空")
    return {
        **{name: value / sample_count for name, value in totals.items()},
        "lambda_segment": price_segment_weight,
        "responsibility_temperature": temperature,
        "learning_rates": {
            str(group["name"]): float(group["lr"]) for group in optimizer.param_groups
        },
    }


@torch.no_grad()
def _precompute_responsibility(
    model: FLDMoE,
    split: FLDMoESplit,
    log_prior: torch.Tensor,
    config: FLDMoEConfig,
    device: torch.device,
) -> torch.Tensor:
    """冻结专家并按训练顺序计算供树门控蒸馏的纯软责任。"""
    loader = DataLoader(
        TensorDataset(
            split.expert_features,
            split.target,
            split.loss_scale,
            split.bounds,
            log_prior,
        ),
        batch_size=config.runtime.inference_batch_size,
        shuffle=False,
        **_loader_options(config, device),
    )
    temperature = config.expert_training.responsibility_temperature_schedule.end
    model.eval()
    chunks: list[torch.Tensor] = []
    for expert_features, target, loss_scale, bounds, prior in loader:
        output = forward_distribution(
            model,
            _to_device(expert_features, device),
            _to_device(prior, device),
            _to_device(bounds, device),
            config.xgboost_gate.gate_temperature,
            config.distribution,
        )
        expert_crps = per_expert_physical_crps(
            output,
            _to_device(target, device),
            _to_device(loss_scale, device),
        )
        chunks.append(
            torch.softmax(
                _to_device(prior, device) - expert_crps / temperature,
                dim=-1,
            ).cpu()
        )
    if not chunks:
        raise FLDMoETrainingError("训练集为空，无法计算软责任")
    return torch.cat(chunks)


def _run_protection_stage(
    state: _ExpertTrainingState,
    data: PreparedFLDMoEData,
    config: FLDMoEConfig,
    device: torch.device,
) -> tuple[list[dict[str, Any]], torch.Tensor]:
    """使用零树先验运行完整保护期并返回训练记录。"""
    zero_prior = torch.zeros_like(data.price_segment_responsibility)
    records: list[dict[str, Any]] = []
    for epoch in range(config.protection_epochs):
        train_metrics = _train_epoch(
            state.model,
            data.train,
            data.price_segment_responsibility,
            zero_prior,
            config,
            device,
            state.optimizer,
            epoch,
        )
        logging.info(
            "epoch=%d phase=protection loss=%.6f", epoch + 1, train_metrics["total"]
        )
        records.append(
            {
                "epoch": epoch + 1,
                "phase": "protection",
                "train": train_metrics,
                "validation_crps": None,
                "gate_validation_soft_ce": None,
                "gate_boost_rounds": None,
            }
        )
    return records, zero_prior


def _gate_prior(
    xgboost_gate: XGBoostGateModel,
    gate_features: torch.Tensor,
    config: FLDMoEConfig,
) -> torch.Tensor:
    """把当前树门控转换成下一专家 epoch 使用的平滑对数先验。"""
    samples, steps, features = gate_features.shape
    values = gate_features.reshape(-1, features).numpy()
    prior = gate_log_prior(
        xgboost_gate,
        values,
        config.xgboost_gate.prior_uniform_epsilon,
    )
    return torch.from_numpy(prior.astype(np.float32)).reshape(
        samples,
        steps,
        config.model.num_experts,
    )


def _fit_gate(
    data: PreparedFLDMoEData,
    responsibility: torch.Tensor,
    config: FLDMoEConfig,
    device: torch.device,
) -> tuple[GateDataSplit, XGBoostGateFitResult]:
    """按固定末端自然日切分并蒸馏当前专家软责任。"""
    split = split_gate_training_data(
        data.train.gate_features.numpy(),
        responsibility.numpy(),
        data.train.timestamps_ns,
        config.xgboost_gate.inner_validation_days,
    )
    result = fit_xgboost_gate(
        split,
        config.xgboost_gate,
        config.runtime.seed,
        device,
    )
    return split, result


@torch.no_grad()
def score_validation_crps(
    model: FLDMoE,
    xgboost_gate: XGBoostGateModel,
    split: FLDMoESplit,
    config: FLDMoEConfig,
    device: torch.device,
) -> float:
    """按原生等间距价格坐标计算正式验证集混合 CRPS。"""
    loader = DataLoader(
        TensorDataset(
            split.expert_features,
            split.target,
            split.loss_scale,
            split.bounds,
            split.gate_features,
        ),
        batch_size=config.runtime.inference_batch_size,
        shuffle=False,
        **_loader_options(config, device),
    )
    model.eval()
    weighted_sum = 0.0
    point_count = 0
    for (
        expert_features,
        target,
        loss_scale,
        bounds,
        gate_features,
    ) in loader:
        flat_gate = gate_features.reshape(-1, gate_features.shape[-1]).numpy()
        logits = _to_device(
            torch.from_numpy(xgboost_gate.predict_margin(flat_gate)).to(torch.float32),
            device,
        )
        logits = logits.reshape(*gate_features.shape[:2], config.model.num_experts)
        output = forward_distribution(
            model,
            _to_device(expert_features, device),
            logits,
            _to_device(bounds, device),
            config.xgboost_gate.gate_temperature,
            config.distribution,
        )
        loss = mixture_physical_crps(
            output,
            _to_device(target, device),
            _to_device(loss_scale, device),
        )
        current_points = int(target.numel())
        weighted_sum += float(loss) * current_points
        point_count += current_points
    if point_count == 0:
        raise FLDMoETrainingError("正式验证集没有可评分时段")
    result = weighted_sum / point_count
    if not np.isfinite(result):
        raise FLDMoETrainingError("正式验证 CRPS 不是有限数值")
    return result


def train_model(
    data: PreparedFLDMoEData,
    config: FLDMoEConfig,
    device: torch.device,
) -> FLDMoEFitResult:
    """使用全部门控特征联合训练专家与门控并恢复最优配对点。"""
    state = _initialize_expert_training(data, config, device)
    model = state.model
    protection_epochs = config.protection_epochs
    records, zero_prior = _run_protection_stage(
        state,
        data,
        config,
        device,
    )
    responsibility = _precompute_responsibility(
        model,
        data.train,
        zero_prior,
        config,
        device,
    )
    gate_split, gate_fit = _fit_gate(
        data,
        responsibility,
        config,
        device,
    )
    current_gate = gate_fit.model
    validation_crps = score_validation_crps(
        model,
        current_gate,
        data.validation,
        config,
        device,
    )
    state.scheduler.step(validation_crps)
    records[-1].update(
        {
            "validation_crps": validation_crps,
            "gate_validation_soft_ce": gate_fit.best_validation_soft_ce,
            "gate_boost_rounds": gate_fit.boost_rounds,
        }
    )
    best = _PairedCheckpoint(
        model_state={
            name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()
        },
        xgboost_gate=current_gate,
        epoch=protection_epochs,
        validation_crps=validation_crps,
    )
    bad_epochs = 0
    early_stopped = False
    for epoch in range(protection_epochs, config.expert_training.max_epochs):
        log_prior = _gate_prior(
            current_gate,
            data.train.gate_features,
            config,
        )
        train_metrics = _train_epoch(
            model,
            data.train,
            data.price_segment_responsibility,
            log_prior,
            config,
            device,
            state.optimizer,
            epoch,
        )
        record: dict[str, Any] = {
            "epoch": epoch + 1,
            "phase": "joint_gate",
            "train": train_metrics,
            "validation_crps": None,
            "gate_validation_soft_ce": None,
            "gate_boost_rounds": None,
        }
        responsibility = _precompute_responsibility(
            model,
            data.train,
            log_prior,
            config,
            device,
        )
        gate_split, gate_fit = _fit_gate(
            data,
            responsibility,
            config,
            device,
        )
        current_gate = gate_fit.model
        validation_crps = score_validation_crps(
            model,
            current_gate,
            data.validation,
            config,
            device,
        )
        state.scheduler.step(validation_crps)
        logging.info(
            "epoch=%d phase=joint_gate validation_scaled_crps=%.6f",
            epoch + 1,
            validation_crps,
        )
        record.update(
            {
                "validation_crps": validation_crps,
                "gate_validation_soft_ce": gate_fit.best_validation_soft_ce,
                "gate_boost_rounds": gate_fit.boost_rounds,
            }
        )
        if validation_crps < best.validation_crps - config.expert_training.min_delta:
            best = _PairedCheckpoint(
                model_state={
                    name: tensor.detach().cpu().clone()
                    for name, tensor in model.state_dict().items()
                },
                xgboost_gate=current_gate,
                epoch=epoch + 1,
                validation_crps=validation_crps,
            )
            bad_epochs = 0
        else:
            bad_epochs += 1
        records.append(record)
        if bad_epochs >= config.expert_training.patience:
            early_stopped = True
            break
    model.load_state_dict(best.model_state, strict=True)
    history = FLDMoETrainingHistory(
        epochs=tuple(records),
        protection_epochs=protection_epochs,
        best_epoch=best.epoch,
        best_validation_crps=best.validation_crps,
        stopped_epoch=len(records),
        early_stopped=early_stopped,
        gate_split={
            "train_days": gate_split.train_days,
            "validation_days": gate_split.validation_days,
            "train_rows": int(gate_split.train_features.shape[0]),
            "validation_rows": int(gate_split.validation_features.shape[0]),
        },
    )
    return FLDMoEFitResult(
        model=model,
        xgboost_gate=best.xgboost_gate,
        history=history,
        data=data,
        model_init=state.model_init,
        initial_expert_pmf=state.initial_pmf,
    )


@torch.no_grad()
def predict_distribution(
    fit: FLDMoEFitResult,
    split: FLDMoESplit,
    config: FLDMoEConfig,
    device: torch.device,
) -> DiscreteForecast:
    """按树门控推理并保留原生价格网格、混合概率质量。"""
    loader = DataLoader(
        TensorDataset(
            split.expert_features,
            split.bounds,
            split.gate_features,
        ),
        batch_size=config.runtime.inference_batch_size,
        shuffle=False,
        **_loader_options(config, device),
    )
    fit.model.eval()
    grids: list[torch.Tensor] = []
    masses: list[torch.Tensor] = []
    for expert_features, bounds, gate_features in loader:
        flat_gate = gate_features.reshape(-1, gate_features.shape[-1]).numpy()
        logits = _to_device(
            torch.from_numpy(fit.xgboost_gate.predict_margin(flat_gate)).to(
                torch.float32
            ),
            device,
        )
        logits = logits.reshape(*gate_features.shape[:2], config.model.num_experts)
        output = forward_distribution(
            fit.model,
            _to_device(expert_features, device),
            logits,
            _to_device(bounds, device),
            config.xgboost_gate.gate_temperature,
            config.distribution,
        )
        grids.append(output.price_grid.flatten(0, 1).cpu())
        masses.append(output.pmf.flatten(0, 1).cpu())
    if not grids:
        raise FLDMoETrainingError("推理切分为空")
    return DiscreteForecast(
        split=split.name,
        timestamps_ns=split.timestamps_ns.reshape(-1),
        y_true=split.targets_physical.reshape(-1),
        price_grid=torch.cat(grids).numpy().astype(np.float64),
        probability_mass=torch.cat(masses).numpy().astype(np.float64),
    )


__all__ = [
    "FLDMoEFitResult",
    "FLDMoETrainingError",
    "FLDMoETrainingHistory",
    "build_initial_expert_pmf",
    "predict_distribution",
    "resolve_device",
    "scheduled_value",
    "set_seed",
    "train_model",
]
