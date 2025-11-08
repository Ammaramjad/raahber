from __future__ import annotations

import argparse
from pathlib import Path

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
    )

    if metrics:
        for name, value in metrics.items():
            print(f"{name}: {value}")


if __name__ == "__main__":
    main()
