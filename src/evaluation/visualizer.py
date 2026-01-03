"""
Complete evaluation visualization: all plots and metrics
"""
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path
from .plots import (
    plot_confusion_matrix,
    plot_precision_recall_curves,
    plot_roc_curves,
    plot_metrics_comparison
)
from .metrics import compute_all_metrics


def visualize_all_evaluation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
    label_list: List[Any],
    task_name: str = "",
    classifier_name: str = "",
    save_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create all evaluation visualizations and compute all metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional, for curves)
        label_list: List of label names
        task_name: Task name
        classifier_name: Classifier name
        save_dir: Directory to save plots (optional)
    
    Returns:
        Dictionary with all metrics
    """
    full_name = f"{task_name} - {classifier_name}" if task_name and classifier_name else (task_name or classifier_name)
    
    # Compute all metrics
    metrics = compute_all_metrics(y_true, y_pred, label_list, task_name=full_name)
    
    # Create save paths if save_dir provided
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        cm_path = str(save_dir / f"confusion_matrix_{classifier_name}_{task_name}.png")
        pr_path = str(save_dir / f"precision_recall_{classifier_name}_{task_name}.png")
        roc_path = str(save_dir / f"roc_{classifier_name}_{task_name}.png")
    else:
        cm_path = None
        pr_path = None
        roc_path = None
    
    # Plot confusion matrix
    plot_confusion_matrix(
        y_true, y_pred, label_list,
        task_name=full_name,
        save_path=cm_path
    )
    
    # Plot precision-recall curves (if probabilities available)
    if y_proba is not None:
        plot_precision_recall_curves(
            y_true, y_proba, label_list,
            task_name=full_name,
            save_path=pr_path
        )
        
        plot_roc_curves(
            y_true, y_proba, label_list,
            task_name=full_name,
            save_path=roc_path
        )
    
    return metrics


def visualize_comparison(
    results_dict: Dict[str, Dict[str, Any]],
    label_list: List[Any],
    task_name: str = "",
    save_dir: Optional[str] = None
) -> None:
    """
    Create comparison plots across classifiers
    
    Args:
        results_dict: Dict mapping classifier_name -> results
        label_list: List of label names
        task_name: Task name
        save_dir: Directory to save plots (optional)
    """
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot comparison for key metrics
    key_metrics = ['macro_f1', 'weighted_f1', 'accuracy', 'macro_precision', 'macro_recall']
    
    for metric in key_metrics:
        save_path = str(save_dir / f"comparison_{metric}_{task_name}.png") if save_dir else None
        plot_metrics_comparison(
            results_dict,
            metric_name=metric,
            task_name=task_name,
            save_path=save_path
        )

