"""独立计算原生离散分布的论文指标，不依赖训练器。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fld_moe.schema import DiscreteForecast


def integrated_crps(
    forecast: DiscreteForecast, lower: float | None = None
) -> np.ndarray:
    """精确积分阶梯 CDF；指定下界时得到阈值加权 CRPS。"""
    grid = forecast.price_grid
    left, right = grid[:, :-1], grid[:, 1:]
    if lower is not None:
        if not np.isfinite(lower):
            raise ValueError("积分下界必须有限")
        left = np.maximum(left, lower)
    target = forecast.y_true[:, None]
    below = np.maximum(np.minimum(right, target) - left, 0.0)
    above = np.maximum(right - np.maximum(left, target), 0.0)
    cdf = forecast.probability_mass.cumsum(axis=1)[:, :-1]
    return (below * cdf**2 + above * (cdf - 1.0) ** 2).sum(axis=1)


def score_samples(
    forecast: DiscreteForecast, tail_threshold: float = 400
) -> pd.DataFrame:
    """导出中位数、80% 区间、逐样本误差与概率评分。"""
    median = forecast.quantile_prediction(0.5)
    lower = forecast.quantile_prediction(0.1)
    upper = forecast.quantile_prediction(0.9)
    y = forecast.y_true
    width = upper - lower
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(forecast.timestamps_ns, unit="ns"),
            "y_true": y,
            "median": median,
            "q10": lower,
            "q90": upper,
            "absolute_error": np.abs(y - median),
            "squared_error": (y - median) ** 2,
            "crps": integrated_crps(forecast),
            "twcrps": integrated_crps(forecast, tail_threshold),
            "covered80": (lower <= y) & (y <= upper),
            "interval_width80": width,
            "interval_score80": width
            + 10 * np.maximum(lower - y, 0)
            + 10 * np.maximum(y - upper, 0),
        }
    )


def evaluate(forecast: DiscreteForecast, tail_threshold: float = 400) -> dict:
    """汇总八项指标；常量真值的 R² 显式记为 null。"""
    scores = score_samples(forecast, tail_threshold)
    denominator = float(np.sum((forecast.y_true - forecast.y_true.mean()) ** 2))
    return {
        "mae": float(scores.absolute_error.mean()),
        "rmse": float(np.sqrt(scores.squared_error.mean())),
        "r2": (
            float(1 - scores.squared_error.sum() / denominator)
            if denominator > np.finfo(np.float64).eps
            else None
        ),
        "crps": float(scores.crps.mean()),
        "twcrps": float(scores.twcrps.mean()),
        "picp80": float(scores.covered80.mean()),
        "mpiw80": float(scores.interval_width80.mean()),
        "is80": float(scores.interval_score80.mean()),
    }


def save_forecast(path: Path, forecast: DiscreteForecast) -> None:
    """以无 pickle 的压缩 NPZ 保存完整原生分布和评价真值。"""
    np.savez_compressed(
        path,
        schema_version=np.array(1),
        algorithm=np.array("FLD-MoE"),
        price_unit=np.array("CNY/MWh"),
        split=np.array(forecast.split),
        timestamps_ns=forecast.timestamps_ns,
        y_true=forecast.y_true,
        price_grid=forecast.price_grid,
        probability_mass=forecast.probability_mass,
    )


def load_forecast(path: str | Path) -> DiscreteForecast:
    """校验文件格式并回读分布，支持训练后独立复算指标。"""
    with np.load(path, allow_pickle=False) as saved:
        if (
            saved["schema_version"].item() != 1
            or saved["algorithm"].item() != "FLD-MoE"
            or saved["price_unit"].item() != "CNY/MWh"
        ):
            raise ValueError("不支持的预测文件格式、算法或单位")
        return DiscreteForecast(
            split=str(saved["split"].item()),
            timestamps_ns=saved["timestamps_ns"],
            y_true=saved["y_true"],
            price_grid=saved["price_grid"],
            probability_mass=saved["probability_mass"],
        )
