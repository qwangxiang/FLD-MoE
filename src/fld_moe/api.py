"""组织 FLD-MoE 训练和交付流程，不依赖命令行解析。"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from time import perf_counter

import torch

from fld_moe.artifacts import describe_data, publish_artifacts
from fld_moe.config import FLDMoEConfig
from fld_moe.data import prepare_data
from fld_moe.training import predict_distribution, resolve_device, set_seed, train_model


def _synchronize(device: torch.device) -> None:
    """在运行计时边界等待 CUDA 工作完成。"""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_training(config: FLDMoEConfig, output: Path) -> dict:
    """完成数据准备、最佳配对训练、原生预测和指标发布。"""
    if output.exists():
        raise FileExistsError(f"输出目录已存在，请指定新目录：{output}")
    logging.info("Preparing Shaanxi data")
    data = prepare_data(config)
    logging.info("Data: %s", json.dumps(describe_data(data), ensure_ascii=False))
    set_seed(config.runtime.seed)
    device = resolve_device(config.runtime.device)
    _synchronize(device)
    start = perf_counter()
    fit = train_model(data, config, device)
    _synchronize(device)
    fit_seconds = perf_counter() - start
    validation = predict_distribution(fit, data.validation, config, device)
    _synchronize(device)
    start = perf_counter()
    test = predict_distribution(fit, data.test, config, device)
    _synchronize(device)
    runtime = {
        "device": str(device),
        "fit_seconds": fit_seconds,
        "test_inference_seconds": perf_counter() - start,
    }
    return publish_artifacts(
        output,
        fit,
        {"validation": validation, "test": test},
        config,
        runtime,
    )
