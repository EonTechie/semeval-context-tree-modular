"""
Evaluation metrics and reporting
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    cohen_kappa_score,
    matthews_corrcoef,
    hamming_loss,
    jaccard_score
)
from typing import Dict, List, Any, Union

Label = Union[int, str]


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_list: List[Label],
    task_name: str = ""
) -> Dict[str, Any]:
    """
    Compute all metrics (same format as Ihsan's metrics.py)
    
    Returns:
        Dictionary with accuracy, macro/weighted metrics, per-class metrics
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "weighted_precision": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "weighted_recall": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
    }
    
    report = classification_report(
        y_true,
        y_pred,
        labels=label_list,
        digits=4,
        output_dict=True,
        zero_division=0,
    )
    
    # Per-class metrics
    per_class = {}
    for label in label_list:
        label_key = str(label)
        per_class[label] = {
            "precision": float(report[label_key]["precision"]),
            "recall": float(report[label_key]["recall"]),
            "f1": float(report[label_key]["f1-score"]),
            "support": int(report[label_key]["support"]),
        }
        metrics[f"{label}_f1"] = float(report[label_key]["f1-score"])
    
    metrics["per_class"] = per_class
    metrics["macro_avg_detail"] = {
        "precision": float(report["macro avg"]["precision"]),
        "recall": float(report["macro avg"]["recall"]),
        "f1": float(report["macro avg"]["f1-score"]),
        "support": int(report["macro avg"]["support"]),
    }
    metrics["weighted_avg_detail"] = {
        "precision": float(report["weighted avg"]["precision"]),
        "recall": float(report["weighted avg"]["recall"]),
        "f1": float(report["weighted avg"]["f1-score"]),
        "support": int(report["weighted avg"]["support"]),
    }
    
    # Confusion matrix
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=label_list).tolist()
    
    # Specialized metrics
    metrics["cohen_kappa"] = float(cohen_kappa_score(y_true, y_pred))
    metrics["matthews_corrcoef"] = float(matthews_corrcoef(y_true, y_pred))
    metrics["hamming_loss"] = float(hamming_loss(y_true, y_pred))
    
    # Jaccard score (IoU) - average across classes
    metrics["jaccard_score"] = float(jaccard_score(
        y_true, y_pred, labels=label_list, average='macro', zero_division=0
    ))
    
    return metrics


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_list: List[Label],
    task_name: str = ""
) -> None:
    """
    Print classification report (same format as sklearn)
    """
    print(f"\n{'='*60}")
    if task_name:
        print(f"Classification Report: {task_name}")
    else:
        print("Classification Report")
    print(f"{'='*60}")
    print(classification_report(
        y_true,
        y_pred,
        labels=label_list,
        digits=4,
        zero_division=0
    ))
    print(f"{'='*60}\n")

