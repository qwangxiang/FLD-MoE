"""实现 FLD-MoE 的价格分段责任度与软目标树门控。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from fld_moe.config import XGBoostGateConfig


class PriceSegmentPartitionError(ValueError):
    """表示候选分位切分无法为每个专家建立有效初始价格分段。"""


def _softmax_rows(margin: np.ndarray) -> np.ndarray:
    """把二维原始分数稳定归一为逐行专家概率。"""
    shifted = margin - margin.max(axis=1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / exponential.sum(axis=1, keepdims=True)


def _validate_features(features: np.ndarray, context: str) -> np.ndarray:
    """校验并返回非空、有限的二维门控特征。"""
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError(f"{context}门控特征必须是非空二维数组")
    if not np.isfinite(values).all():
        raise ValueError(f"{context}门控特征包含非有限值")
    return values


def _validate_responsibilities(
    responsibilities: np.ndarray,
    expected_rows: int,
    context: str,
) -> np.ndarray:
    """校验并返回逐行归一的二维软责任。"""
    values = np.asarray(responsibilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != expected_rows:
        raise ValueError(f"{context}软责任行数或维度错误: {values.shape}")
    if values.shape[1] < 2:
        raise ValueError(f"{context}软责任至少需要两个专家")
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError(f"{context}软责任包含非有限值或负值")
    if not np.allclose(values.sum(axis=1), 1.0, atol=1e-6, rtol=1e-5):
        raise ValueError(f"{context}软责任未逐行归一")
    return values


def build_price_segment_responsibilities(
    targets: np.ndarray,
    bounds: np.ndarray,
    num_experts: int,
    cut_quantiles: tuple[float, ...],
) -> tuple[np.ndarray, dict[str, object]]:
    """在训练集归一化价格上拟合分位切点并生成独热价格分段责任度。"""
    target_values = np.asarray(targets, dtype=np.float64)
    bound_values = np.asarray(bounds, dtype=np.float64)
    if target_values.ndim != 2 or bound_values.shape != (*target_values.shape, 2):
        raise ValueError("价格分段责任度的目标或价格边界形状错误")
    if num_experts < 2 or len(cut_quantiles) != num_experts - 1:
        raise ValueError("价格分段责任度的专家数与分位切点数量不一致")
    floor = bound_values[..., 0]
    span = bound_values[..., 1] - floor
    if np.any(span <= 0.0):
        raise ValueError("价格分段责任度价格边界跨度必须为正数")
    normalized_price = ((target_values - floor) / span).reshape(-1)
    normalized_price_cutpoints = np.quantile(
        normalized_price, np.asarray(cut_quantiles)
    )
    if np.any(np.diff(normalized_price_cutpoints) <= 0.0):
        raise PriceSegmentPartitionError("训练数据产生重合分位切点，无法初始化独立专家")
    segments = np.searchsorted(
        normalized_price_cutpoints, normalized_price, side="left"
    )
    segment_sample_counts = [
        int(np.count_nonzero(segments == index)) for index in range(num_experts)
    ]
    if any(count == 0 for count in segment_sample_counts):
        raise PriceSegmentPartitionError(
            f"价格分段责任度存在空专家段: {segment_sample_counts}"
        )
    responsibility = np.zeros((segments.size, num_experts), dtype=np.float32)
    responsibility[np.arange(segments.size), segments] = 1.0
    reshaped = responsibility.reshape(*target_values.shape, num_experts)
    metadata: dict[str, object] = {
        "num_experts": num_experts,
        "cut_quantiles": list(cut_quantiles),
        "normalized_price_cutpoints": normalized_price_cutpoints.tolist(),
        "segment_sample_counts": segment_sample_counts,
        "assignment": "normalized_price_quantile_right_closed",
    }
    return reshaped, metadata


def soft_target_cross_entropy(
    margin: np.ndarray,
    responsibilities: np.ndarray,
) -> float:
    """计算多分类原始分数相对软责任的平均交叉熵。"""
    prediction = np.asarray(margin, dtype=np.float64)
    target = np.asarray(responsibilities, dtype=np.float64)
    if prediction.ndim != 2 or target.shape != prediction.shape:
        raise ValueError("门控分数与软责任形状不一致")
    probability = np.clip(_softmax_rows(prediction), 1e-12, 1.0)
    return float(-(target * np.log(probability)).sum(axis=1).mean())


def soft_target_grad_hess(
    margin: np.ndarray,
    responsibilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """返回软标签交叉熵对多分类分数的梯度和对角海森。"""
    prediction = np.asarray(margin, dtype=np.float64)
    target = np.asarray(responsibilities, dtype=np.float64)
    if prediction.ndim != 2 or target.shape != prediction.shape:
        raise ValueError("门控分数与软责任形状不一致")
    probability = _softmax_rows(prediction)
    gradient = probability - target
    hessian = np.maximum(2.0 * probability * (1.0 - probability), 1e-6)
    return gradient, hessian


@dataclass(frozen=True)
class GateDataSplit:
    """保存按自然日切分的树门控训练与内部验证数据。"""

    train_features: np.ndarray
    train_responsibilities: np.ndarray
    validation_features: np.ndarray
    validation_responsibilities: np.ndarray
    train_days: int
    validation_days: int


def split_gate_training_data(
    features: np.ndarray,
    responsibilities: np.ndarray,
    timestamps_ns: np.ndarray,
    inner_validation_days: int,
) -> GateDataSplit:
    """按最后若干自然日构造树门控的前向内部验证集。"""
    if features.ndim != 3 or responsibilities.ndim != 3:
        raise ValueError("联合训练门控特征和软责任必须是三维数组")
    if features.shape[:2] != responsibilities.shape[:2]:
        raise ValueError("联合训练门控特征和软责任样本、时段不对齐")
    if timestamps_ns.ndim != 2 or timestamps_ns.shape[:2] != features.shape[:2]:
        raise ValueError("门控时间戳与特征样本、时段不对齐")
    flat_features = _validate_features(
        np.asarray(features).reshape(-1, features.shape[-1]),
        "联合训练",
    )
    flat_responsibilities = _validate_responsibilities(
        np.asarray(responsibilities).reshape(-1, responsibilities.shape[-1]),
        flat_features.shape[0],
        "联合训练",
    )
    flat_timestamps = timestamps_ns.astype("datetime64[ns]").reshape(-1)
    if np.isnat(flat_timestamps).any():
        raise ValueError("树门控时间戳包含缺失值")
    flat_dates = flat_timestamps.astype("datetime64[D]")
    unique_dates = np.unique(flat_dates)
    if unique_dates.size <= inner_validation_days:
        raise ValueError("训练日数不足以构造树门控内部验证集")
    validation_dates = unique_dates[-inner_validation_days:]
    for date in validation_dates:
        actual = np.unique(flat_timestamps[flat_dates == date])
        expected = date.astype("datetime64[ns]") + np.arange(96) * np.timedelta64(
            15,
            "m",
        )
        if not np.array_equal(actual, expected):
            raise ValueError(f"树门控内部验证日期 {date} 不是完整 96 点自然日")
    validation_mask = np.isin(flat_dates, validation_dates)
    train_mask = ~validation_mask
    if not bool(train_mask.any()) or not bool(validation_mask.any()):
        raise ValueError("树门控内部切分产生空数据集")
    if np.intersect1d(
        np.unique(flat_timestamps[train_mask]),
        np.unique(flat_timestamps[validation_mask]),
    ).size:
        raise ValueError("树门控训练与内部验证存在重复原始时间戳")
    return GateDataSplit(
        train_features=flat_features[train_mask],
        train_responsibilities=flat_responsibilities[train_mask],
        validation_features=flat_features[validation_mask],
        validation_responsibilities=flat_responsibilities[validation_mask],
        train_days=int(unique_dates.size - inner_validation_days),
        validation_days=int(inner_validation_days),
    )


class XGBoostGateModel:
    """封装软责任蒸馏后的 XGBoost Booster。"""

    def __init__(self, booster: Any, num_experts: int) -> None:
        """保存 Booster 和固定专家数。"""
        self.booster = booster
        self.num_experts = int(num_experts)

    def predict_margin(self, features: np.ndarray) -> np.ndarray:
        """返回形状为 ``(N,K)`` 的有限原始分数。"""
        import xgboost as xgb

        values = _validate_features(features, "预测")
        margin = np.asarray(
            self.booster.predict(xgb.DMatrix(values), output_margin=True),
            dtype=np.float64,
        ).reshape(values.shape[0], self.num_experts)
        if not np.isfinite(margin).all():
            raise ValueError("树门控输出包含非有限值")
        return margin


@dataclass(frozen=True)
class XGBoostGateFitResult:
    """保存树门控最佳轮次、模型和内部验证信息。"""

    model: XGBoostGateModel
    best_iteration: int
    boost_rounds: int
    best_validation_soft_ce: float
    evaluation_history: dict[str, dict[str, list[float]]]
    train_days: int
    validation_days: int


def fit_xgboost_gate(
    split: GateDataSplit,
    config: XGBoostGateConfig,
    random_state: int,
    device: torch.device,
) -> XGBoostGateFitResult:
    """用软责任训练 XGBoost，并按同一软交叉熵执行早停。"""
    import xgboost as xgb

    train_x = _validate_features(split.train_features, "训练")
    validation_x = _validate_features(split.validation_features, "内部验证")
    train_r = _validate_responsibilities(
        split.train_responsibilities,
        train_x.shape[0],
        "训练",
    )
    validation_r = _validate_responsibilities(
        split.validation_responsibilities,
        validation_x.shape[0],
        "内部验证",
    )
    if train_x.shape[1] != validation_x.shape[1]:
        raise ValueError("树门控训练与内部验证特征数不一致")
    if train_r.shape[1] != validation_r.shape[1]:
        raise ValueError("树门控训练与内部验证专家数不一致")
    experts = int(train_r.shape[1])
    train_matrix = xgb.DMatrix(
        train_x,
        label=np.zeros(train_x.shape[0]),
        nthread=config.nthread,
    )
    validation_matrix = xgb.DMatrix(
        validation_x,
        label=np.zeros(validation_x.shape[0]),
        nthread=config.nthread,
    )
    targets = {id(train_matrix): train_r, id(validation_matrix): validation_r}

    def _target(matrix: xgb.DMatrix) -> np.ndarray:
        """按 DMatrix 身份读取对应软责任。"""
        try:
            return targets[id(matrix)]
        except KeyError as exc:
            raise RuntimeError("XGBoost 回调收到未知数据集") from exc

    def _objective(
        prediction: np.ndarray,
        matrix: xgb.DMatrix,
    ) -> tuple[np.ndarray, np.ndarray]:
        """向 XGBoost 返回软交叉熵梯度与对角海森。"""
        margin = np.asarray(prediction, dtype=np.float64).reshape(-1, experts)
        return soft_target_grad_hess(margin, _target(matrix))

    def _metric(
        prediction: np.ndarray,
        matrix: xgb.DMatrix,
    ) -> tuple[str, float]:
        """向 XGBoost 返回软交叉熵验证指标。"""
        margin = np.asarray(prediction, dtype=np.float64).reshape(-1, experts)
        return "soft_ce", soft_target_cross_entropy(margin, _target(matrix))

    parameters = {
        "num_class": experts,
        "max_depth": config.max_depth,
        "eta": config.learning_rate,
        "subsample": config.subsample,
        "colsample_bytree": config.colsample_bytree,
        "min_child_weight": config.min_child_weight,
        "max_delta_step": config.max_delta_step,
        "lambda": 1.0,
        "seed": random_state,
        "tree_method": "hist",
        "device": "cuda" if device.type == "cuda" else "cpu",
        "disable_default_eval_metric": 1,
    }
    if config.nthread > 0:
        parameters["nthread"] = config.nthread
    history: dict[str, dict[str, list[float]]] = {}
    booster = xgb.train(
        parameters,
        train_matrix,
        num_boost_round=config.max_boost_rounds,
        evals=[(train_matrix, "train"), (validation_matrix, "validation")],
        obj=_objective,
        custom_metric=_metric,
        early_stopping_rounds=config.early_stopping_rounds,
        evals_result=history,
        verbose_eval=False,
    )
    best_iteration = int(getattr(booster, "best_iteration", -1))
    validation_history = history.get("validation", {}).get("soft_ce", [])
    if not 0 <= best_iteration < len(validation_history):
        raise RuntimeError("XGBoost 未返回有效的门控最佳轮次")
    boost_rounds = best_iteration + 1
    best_booster = booster[:boost_rounds]
    if int(best_booster.num_boosted_rounds()) != boost_rounds:
        raise RuntimeError("XGBoost 最佳门控模型切片失败")
    return XGBoostGateFitResult(
        model=XGBoostGateModel(best_booster, experts),
        best_iteration=best_iteration,
        boost_rounds=boost_rounds,
        best_validation_soft_ce=float(validation_history[best_iteration]),
        evaluation_history=history,
        train_days=split.train_days,
        validation_days=split.validation_days,
    )


def gate_log_prior(
    model: XGBoostGateModel,
    features: np.ndarray,
    uniform_epsilon: float,
) -> np.ndarray:
    """返回加均匀先验平滑后的树门控对数概率。"""
    margin = model.predict_margin(features)
    probability = _softmax_rows(margin)
    smoothed = (
        1.0 - uniform_epsilon
    ) * probability + uniform_epsilon / model.num_experts
    return np.log(np.clip(smoothed, 1e-12, 1.0))


__all__ = [
    "GateDataSplit",
    "PriceSegmentPartitionError",
    "XGBoostGateFitResult",
    "XGBoostGateModel",
    "build_price_segment_responsibilities",
    "fit_xgboost_gate",
    "soft_target_cross_entropy",
    "soft_target_grad_hess",
    "split_gate_training_data",
    "gate_log_prior",
]
