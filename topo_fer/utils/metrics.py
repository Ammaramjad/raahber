 from __future__ import annotations

 from typing import Dict, Iterable, Tuple

 import numpy as np
 from sklearn import metrics


 def classification_metrics(
     y_true: np.ndarray,
     y_pred: np.ndarray,
     y_prob: np.ndarray | None = None,
 ) -> Dict[str, float]:
     """Compute standard closed-set classification metrics."""
     results: Dict[str, float] = {}
     results["accuracy"] = metrics.accuracy_score(y_true, y_pred)
     results["f1_macro"] = metrics.f1_score(y_true, y_pred, average="macro")
     if y_prob is not None and y_prob.ndim == 2 and y_prob.shape[1] > 1:
         try:
             results["auc"] = metrics.roc_auc_score(y_true, y_prob, multi_class="ovr")
         except ValueError:
             results["auc"] = float("nan")
     return results


 def open_set_metrics(
     known_scores: Iterable[float],
     novel_scores: Iterable[float],
 ) -> Dict[str, float]:
     """Compute open-set detection metrics given confidence scores.

     Args:
         known_scores: Scores (larger means more known) for known-class samples.
         novel_scores: Scores (larger means more known) for novel-class samples.
     """
     labels = np.concatenate(
         [
             np.ones(len(list(known_scores)), dtype=np.int32),
             np.zeros(len(list(novel_scores)), dtype=np.int32),
         ]
     )
     scores = np.concatenate([np.array(list(known_scores)), np.array(list(novel_scores))])

     fpr, tpr, thresholds = metrics.roc_curve(labels, scores)
     auroc = metrics.auc(fpr, tpr)
     precision, recall, _ = metrics.precision_recall_curve(labels, scores)
     aupr = metrics.auc(recall, precision)

     # FPR at 95% TPR
     target_tpr = 0.95
     idx = np.argmin(np.abs(tpr - target_tpr))
     fpr95 = float(fpr[idx])

     return {"auroc": auroc, "aupr": aupr, "fpr95": fpr95}


 def entropy(scores: np.ndarray, axis: int = -1) -> np.ndarray:
     """Compute normalized entropy of probability distributions."""
     scores = np.clip(scores, 1e-8, 1.0)
     ent = -np.sum(scores * np.log(scores), axis=axis)
     max_ent = np.log(scores.shape[axis])
     return ent / max_ent


 def calibration_error(
     y_true: np.ndarray,
     y_prob: np.ndarray,
     n_bins: int = 15,
 ) -> Tuple[float, np.ndarray, np.ndarray]:
     """Expected calibration error (ECE) and bin statistics."""
     confidences = np.max(y_prob, axis=1)
     predictions = np.argmax(y_prob, axis=1)
     accuracies = predictions == y_true

     bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
     ece = 0.0
     bin_accs = []
     bin_confs = []
     for lower, upper in zip(bin_edges[:-1], bin_edges[1:]):
         mask = (confidences > lower) & (confidences <= upper)
         if not np.any(mask):
             bin_accs.append(0.0)
             bin_confs.append((lower + upper) / 2.0)
             continue
         bin_accuracy = accuracies[mask].mean()
         bin_confidence = confidences[mask].mean()
         ece += np.abs(bin_confidence - bin_accuracy) * mask.mean()
         bin_accs.append(bin_accuracy)
         bin_confs.append(bin_confidence)
     return ece, np.array(bin_accs), np.array(bin_confs)

