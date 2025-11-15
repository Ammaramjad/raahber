from __future__ import annotations

import argparse
from pathlib import Path
import math

from topo_fer.evaluation import evaluate_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate TOPO-FER checkpoint.")
    parser.add_argument("--config", type=Path, required=True, help="Path to config used for training.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint file path.")
    parser.add_argument(
        "--overrides",
        type=str,
        nargs="*",
        help="Optional KEY=VALUE overrides (same semantics as training).",
    )
    parser.add_argument("--experiment_name", type=str, default="topo-fer-eval")
    parser.add_argument(
        "--stage",
        type=str,
        default="test",
        help="Dataloader split to evaluate (e.g., 'test' or 'val').",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    override_cfg = {}
    if args.overrides:
        from topo_fer.scripts.train import parse_overrides

        override_cfg = parse_overrides(args.overrides)

    metrics = evaluate_model(
        config=args.config,
        checkpoint=args.checkpoint,
        overrides=override_cfg,
        experiment_name=args.experiment_name,
        stage=args.stage,
    )

    def _format(value: object) -> str:
        if isinstance(value, (int, float)):
            if isinstance(value, float) and not math.isfinite(value):
                return "nan"
            return f"{value:.4f}"
        return str(value)

    print(f"Evaluation stage: {metrics['stage']}")
    print(f"Samples evaluated: {metrics['num_samples']}")
    for name, value in metrics["classification"].items():
        print(f"[classification] {name}: {_format(value)}")
    if metrics["open_set"]:
        for name, value in metrics["open_set"].items():
            print(f"[open-set] {name}: {_format(value)}")
    print(f"Log file: {metrics['log_path']}")


if __name__ == "__main__":
    main()
