"""All scores increase toward class 1 (synthetic speech)."""
import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score, roc_curve)


def equal_error_rate(labels, scores):
    """Linear interpolation of the ROC crossing; a descriptive evaluation metric.

    This does not select an operating threshold to deploy on the evaluation set.
    """
    labels, scores = np.asarray(labels), np.asarray(scores)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("EER requires both genuine and synthetic recordings")
    fpr, tpr, _ = roc_curve(labels, scores, drop_intermediate=False)
    difference = fpr - (1 - tpr)
    index = int(np.flatnonzero(difference >= 0)[0])
    if index == 0:
        return float(fpr[0])
    weight = -difference[index - 1] / (difference[index] - difference[index - 1])
    return float(fpr[index - 1] + weight * (fpr[index] - fpr[index - 1]))


def compute_metrics(labels, predictions, scores):
    labels, predictions, scores = map(np.asarray, (labels, predictions, scores))
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Metrics require both classes")
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    return {
        "n": len(labels), "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "synthetic_precision": float(precision_score(labels, predictions, zero_division=0)),
        "synthetic_recall": float(recall_score(labels, predictions, zero_division=0)),
        "synthetic_f1": float(f1_score(labels, predictions, zero_division=0)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "eer": equal_error_rate(labels, scores),
        "genuine_false_alarm_rate": float(fp / (tn + fp)),
        "synthetic_miss_rate": float(fn / (tp + fn)),
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_order": ["bonafide", "spoof"],
    }
