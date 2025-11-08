## TOPO-FER: Open-World Expression Manifold Topology

This repository provides a reference implementation of **TOPO-FER**, a facial expression recognition framework that treats affective behaviour as points on a continuous manifold. The system integrates:

- **Geometric Scaffold Network** powered by Neural Ordinary Differential Equations to model dynamic expression transitions.
- **Topological Discovery Module** leveraging persistent homology and density-based clustering to surface previously unseen affective categories.

The codebase includes training, evaluation, and discovery pipelines spanning RAF-DB, FERPlus, AffectNet, and OV-MERD benchmarks.

### Project Structure

- `topo_fer/configs/` – Hydra/OmegaConf configuration files.
- `topo_fer/data/` – Dataset wrappers, augmentations, and Lightning `DataModule`.
- `topo_fer/models/` – Backbone selection, geometric scaffold, and topology discovery modules.
- `topo_fer/training/` – Lightning module assembling model, losses, and metrics.
- `topo_fer/utils/` – Logging, metrics, visualization, and helper utilities.
- `topo_fer/scripts/` – Entry points for training and evaluation.

### Getting Started

```bash
pip install -e .
python topo_fer/scripts/train.py --config topo_fer/configs/default.yaml --experiment_name topo-fer-raf
```

After training, topological discovery over unlabeled splits can be triggered with:

```bash
python topo_fer/scripts/train.py --discover
```

To evaluate a checkpoint:

```bash
python topo_fer/scripts/evaluate.py --config topo_fer/configs/default.yaml --checkpoint outputs/checkpoints/last.ckpt
```

Programmatic evaluation is also supported:

```python
from topo_fer.evaluation import evaluate_model

metrics = evaluate_model(
    config="topo_fer/configs/default.yaml",
    checkpoint="outputs/checkpoints/last.ckpt",
)
print(metrics)
```

### Topological Artifacts

Discovery artifacts (cluster assignments, persistence summaries, mapper graphs) land under `outputs/topology/<experiment_name>/`. Visualization helpers are provided in `topo_fer/utils/visualization.py`.
