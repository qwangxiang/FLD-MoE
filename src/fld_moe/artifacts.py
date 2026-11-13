"""将配对模型、完整预测、指标和复现元数据发布到独立目录。"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from fld_moe.config import FLDMoEConfig
from fld_moe.evaluation import evaluate, save_forecast, score_samples
from fld_moe.features import BOUNDARY_COLUMNS
from fld_moe.schema import DiscreteForecast, PreparedFLDMoEData
from fld_moe.training import FLDMoEFitResult


def write_json(path: Path, value) -> None:
    """写入严格 JSON，保留中文且拒绝 NaN/Infinity。"""
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    """分块计算输入和产物的内容哈希。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe_data(data: PreparedFLDMoEData) -> dict:
    """记录实际进入训练与预测的日期、样本数和维度。"""
    splits = {}
    for name in ("train", "validation", "test"):
        split = getattr(data, name)
        splits[name] = {
            "days": int(split.target.shape[0]),
            "points": int(split.target.numel()),
            "first_timestamp": str(np.datetime64(int(split.timestamps_ns[0, 0]), "ns")),
            "last_timestamp": str(
                np.datetime64(int(split.timestamps_ns[-1, -1]), "ns")
            ),
        }
    return {
        "splits": splits,
        "expert_features": len(data.feature_columns),
        "gate_features": len(data.gate_feature_columns),
    }


def publish_artifacts(
    output: Path,
    fit: FLDMoEFitResult,
    forecasts: dict[str, DiscreteForecast],
    config: FLDMoEConfig,
    runtime: dict,
) -> dict:
    """先在临时目录完整写入，再发布到尚不存在的目标目录。"""
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"输出目录已存在，请指定新目录：{output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".fld-moe-", dir=output.parent))
    try:
        metrics = {
            name: evaluate(forecast, config.dataset.tail_threshold)
            for name, forecast in forecasts.items()
        }
        for name, forecast in forecasts.items():
            save_forecast(staging / f"{name}_distribution.npz", forecast)
            score_samples(forecast, config.dataset.tail_threshold).to_csv(
                staging / f"{name}_predictions.csv",
                index=False,
            )
        write_json(
            staging / "metrics.json",
            {
                "algorithm": "FLD-MoE",
                "price_unit": "CNY/MWh",
                "interval_level": 0.8,
                "tail_threshold": config.dataset.tail_threshold,
                **metrics,
            },
        )
        write_json(staging / "history.json", fit.history.to_dict())
        torch.save(
            {
                "model_init": fit.model_init,
                "initial_expert_pmf": fit.initial_expert_pmf.cpu(),
                "model_state": {
                    name: tensor.detach().cpu()
                    for name, tensor in fit.model.state_dict().items()
                },
                "best_epoch": fit.history.best_epoch,
            },
            staging / "experts.pt",
        )
        fit.xgboost_gate.booster.save_model(staging / "gate.ubj")
        effective = asdict(config)
        for key in ("source_path", "source_text", "data_path"):
            effective.pop(key)
        write_json(staging / "effective_config.json", effective)
        (staging / "input_config.toml").write_text(config.source_text, encoding="utf-8")
        write_json(
            staging / "data_contract.json",
            {
                **describe_data(fit.data),
                "expert_feature_columns": fit.data.feature_columns,
                "gate_feature_columns": fit.data.gate_feature_columns,
                "normalization": {
                    "mode": "rolling",
                    "scope": "strictly_before_target_day",
                    "window_days": config.data.feature_normalization_window_days,
                    "ddof": 1,
                    "constant_scale": 1,
                    "expert_history_start_day": config.dataset.main_history_start_day,
                    "gate_history_start_day": config.dataset.gate_history_start_day,
                    "expert_rolling_columns": BOUNDARY_COLUMNS,
                    "gate_rolling_columns": fit.data.gate_rolling_columns,
                    "price": "(price - contemporaneous_floor) / (cap - floor)",
                },
                "price_segments": fit.data.price_segment_metadata,
            },
        )
        versions = {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "torch", "xgboost", "chinese-calendar")
        }
        write_json(
            staging / "run.json",
            {
                "algorithm": "FLD-MoE",
                "runtime": runtime,
                "python": platform.python_version(),
                "dependencies": versions,
                "cuda_runtime": torch.version.cuda,
                "device_name": (
                    torch.cuda.get_device_name()
                    if runtime["device"].startswith("cuda")
                    else platform.machine()
                ),
                "data_sha256": sha256(config.data_path),
                "input_config_sha256": sha256(config.source_path),
                "source_sha256": {
                    p.name: sha256(p)
                    for p in sorted(Path(__file__).parent.glob("*.py"))
                },
            },
        )
        write_json(
            staging / "manifest.json",
            {p.name: sha256(p) for p in sorted(staging.iterdir()) if p.is_file()},
        )
        if output.exists():
            raise FileExistsError(f"输出目录已存在：{output}")
        staging.rename(output)
        return metrics
    finally:
        if staging.exists():
            shutil.rmtree(staging)
