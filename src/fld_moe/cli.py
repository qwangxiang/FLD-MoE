"""提供一键训练、数据检查与独立评价入口。"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from fld_moe.api import run_training
from fld_moe.artifacts import describe_data, write_json
from fld_moe.config import load_config
from fld_moe.data import prepare_data
from fld_moe.evaluation import evaluate, load_forecast


def main() -> None:
    """解析命令；无子命令时执行正式训练。"""
    parser = argparse.ArgumentParser(description="FLD-MoE: Shaanxi price forecasting")
    parser.add_argument(
        "command",
        nargs="?",
        default="train",
        choices=("train", "check-data", "evaluate"),
    )
    parser.add_argument("--config", type=Path, default=Path("configs/shaanxi.toml"))
    parser.add_argument(
        "--output", type=Path, help="New run directory; JSON file for evaluate"
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument(
        "--forecast", type=Path, help="NPZ file for independent evaluation"
    )
    parser.add_argument(
        "--tail-threshold", type=float, help="twCRPS threshold (default: 400)"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.command == "evaluate":
        if args.forecast is None:
            parser.error("evaluate requires --forecast")
        threshold = 400.0 if args.tail_threshold is None else args.tail_threshold
        metrics = evaluate(load_forecast(args.forecast), threshold)
        if args.output:
            if args.output.exists():
                raise FileExistsError(f"输出文件已存在：{args.output}")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            write_json(args.output, metrics)
        print(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False))
        return
    if args.forecast is not None or args.tail_threshold is not None:
        parser.error("--forecast/--tail-threshold are only valid for evaluate")
    config = load_config(args.config)
    if args.device:
        config = replace(config, runtime=replace(config.runtime, device=args.device))
    if args.command == "check-data":
        print(
            json.dumps(
                describe_data(prepare_data(config)), ensure_ascii=False, indent=2
            )
        )
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output or Path("results") / f"{args.command}-{stamp}"
    metrics = run_training(config, output)
    print(
        json.dumps(
            {"output": str(output.resolve()), **metrics},
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )
