TOPO-FER: Open-World Expression Manifold Topology
==============================================

TOPO-FER introduces a new paradigm for open-world facial expression recognition by modeling expressions as points on a continuous manifold. The approach couples a **Geometric Scaffold Network** based on Neural Ordinary Differential Equations with a **Topological Discovery Module** that leverages persistent homology to surface novel affective categories.

## Key Features
- **Geometric Scaffold Network (GSN):** ResNet backbone with latent Neural ODE dynamics that learn smooth expression transitions resilient to distribution shifts.
- **Topological Discovery Module:** Persistent homology and HDBSCAN clustering discover high-persistence structures and emergent expression categories in unlabeled data.
- **Unified Training Pipeline:** Multi-objective loss balances supervised accuracy, manifold smoothness, reconstruction fidelity, contrastive alignment, and topological stability.
- **Comprehensive Toolkit:** Configuration-driven training, evaluation, visualization, and discovery scripts designed for RAF-DB, FERPlus, AffectNet, and OV-MERD workflows.

## Repository Layout
```
topo_fer/
  config.py              # Dataclasses and YAML loader for experiment configs
  data.py                # Dataset utilities and dataloaders
  evaluation.py          # Evaluation helpers and visualization
  losses.py              # Core loss functions
  models/                # Geometric Scaffold Network and Neural ODE components
  modules/topology.py    # Persistent homology-based discovery module
  trainer.py             # Training loop and checkpoint management
  utils/                 # Logging, metrics, randomness utilities
scripts/
  train.py               # CLI entry point for training
  evaluate.py            # CLI entry point for evaluation, discovery, visualization
configs/
  default.yaml           # Sample configuration template
requirements.txt         # Python package dependencies
pyproject.toml           # Project metadata
```

## Installation
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Dependencies include PyTorch, TorchDiffEq, Giotto-TDA, and HDBSCAN. GPU support is recommended for practical training.

## Dataset Preparation
Each dataset should provide a `metadata.csv` under `dataset.root` with columns:

| path               | label | split |
|--------------------|-------|-------|
| images/00001.jpg   | 0     | train |
| images/00002.jpg   |       | train |
| images/00003.jpg   | 4     | val   |

- `path` is relative to `dataset.root`.
- `label` can be empty for unlabeled samples (treated as open-world candidates).
- `split` must be `train`, `val`, or `test`.
- Configure known category IDs in `configs/default.yaml`.

## Training
```bash
python scripts/train.py \
  --config configs/default.yaml \
  --known-classes 7 \
  --output-dir outputs/rafdb
```

Training artifacts (checkpoints, logs, discovery reports) are written to the output directory. Adjust hyperparameters in the YAML configuration file.

## Evaluation & Discovery
```bash
# Known-class accuracy
python scripts/evaluate.py \
  --config configs/default.yaml \
  --checkpoint outputs/rafdb/best.pth \
  --known-classes 7 \
  --mode known \
  --output outputs/eval_known

# Novel category discovery
python scripts/evaluate.py \
  --config configs/default.yaml \
  --checkpoint outputs/rafdb/best.pth \
  --known-classes 7 \
  --mode discover \
  --output outputs/eval_discover

# Manifold visualization
python scripts/evaluate.py \
  --config configs/default.yaml \
  --checkpoint outputs/rafdb/best.pth \
  --known-classes 7 \
  --mode visualize \
  --output outputs/eval_viz
```

## Extending TOPO-FER
- Swap backbones (ResNet-50, Vision Transformers) by editing the `model.backbone` field.
- Tune topological sensitivity via `topology.persistence_threshold` and HDBSCAN settings.
- Integrate new datasets by providing compatible metadata and updating configuration paths.

## License
Released under the MIT License. Pre-trained models will be published upon paper release.
