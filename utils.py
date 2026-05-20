# utils.py
import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, matthews_corrcoef, cohen_kappa_score,
                             confusion_matrix)


def compute_metrics(y_true, y_pred, num_classes=5) -> dict:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return {
        "acc":       round(accuracy_score(y_true, y_pred) * 100, 2),
        "f1":        round(f1_score(y_true, y_pred, average="weighted",
                                    zero_division=0) * 100, 2),
        "precision": round(precision_score(y_true, y_pred, average="weighted",
                                            zero_division=0) * 100, 2),
        "recall":    round(recall_score(y_true, y_pred, average="weighted",
                                         zero_division=0) * 100, 2),
        "mcc":       round(matthews_corrcoef(y_true, y_pred), 4),
        "kappa":     round(cohen_kappa_score(y_true, y_pred), 4),
    }


def per_class_acc(y_true, y_pred, num_classes=5) -> dict:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return {
        f"DR{c}": round((y_pred[y_true == c] == c).mean() * 100, 2)
        for c in range(num_classes) if (y_true == c).sum() > 0
    }


def print_metrics(metrics: dict, per_class: dict = None):
    print("\n" + "─" * 48)
    for k, v in metrics.items():
        print(f"  {k.upper():10s} {v}")
    if per_class:
        print("\n  Per-class accuracy:")
        for k, v in per_class.items():
            print(f"    {k}: {v:.2f}%")
    print("─" * 48)
