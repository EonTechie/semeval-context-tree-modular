"""
Results table creation and printing (like siparismaili01)
"""
import pandas as pd
from typing import Dict, List, Any
import numpy as np


def create_results_table(
    results_dict: Dict[str, Dict[str, Any]],
    task_name: str = ""
) -> pd.DataFrame:
    """
    Create results table from multiple classifier results
    
    Args:
        results_dict: Dict mapping classifier_name -> {
            'dev_pred': predictions,
            'dev_proba': probabilities (optional),
            'metrics': metrics dict (optional)
        }
        task_name: Task name for display
    
    Returns:
        DataFrame with results
    """
    rows = []
    
    for classifier_name, result in results_dict.items():
        if 'metrics' in result:
            metrics = result['metrics']
            rows.append({
                'Classifier': classifier_name,
                'Task': task_name,
                'Accuracy': metrics.get('accuracy', 0.0),
                'Macro F1': metrics.get('macro_f1', 0.0),
                'Weighted F1': metrics.get('weighted_f1', 0.0),
                'Macro Precision': metrics.get('macro_precision', 0.0),
                'Macro Recall': metrics.get('macro_recall', 0.0),
                'Cohen Kappa': metrics.get('cohen_kappa', 0.0),
                'Matthews CorrCoef': metrics.get('matthews_corrcoef', 0.0),
            })
        else:
            # If metrics not computed, add placeholder
            rows.append({
                'Classifier': classifier_name,
                'Task': task_name,
                'Accuracy': None,
                'Macro F1': None,
                'Weighted F1': None,
                'Macro Precision': None,
                'Macro Recall': None,
            })
    
    df = pd.DataFrame(rows)
    return df


def print_results_table(
    results_dict: Dict[str, Dict[str, Any]],
    task_name: str = "",
    sort_by: str = "Macro F1"
) -> pd.DataFrame:
    """
    Print formatted results table
    
    Args:
        results_dict: Results dictionary
        task_name: Task name
        sort_by: Column to sort by
    
    Returns:
        DataFrame (for further use)
    """
    df = create_results_table(results_dict, task_name)
    
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=False)
    
    print(f"\n{'='*80}")
    if task_name:
        print(f"Results Table: {task_name}")
    else:
        print("Results Table")
    print(f"{'='*80}")
    print(df.to_string(index=False))
    print(f"{'='*80}\n")
    
    return df


def create_model_wise_summary(
    all_results: Dict[str, Dict[str, Dict[str, Any]]],
    models: List[str],
    classifiers: List[str],
    tasks: List[str]
) -> pd.DataFrame:
    """
    Create model-wise summary table (like siparismaili01)
    
    Args:
        all_results: Dict[model_name][classifier_name][task_name] -> results
        models: List of model names
        classifiers: List of classifier names
        tasks: List of task names
    
    Returns:
        DataFrame with model-wise summary
    """
    rows = []
    
    for model in models:
        for classifier in classifiers:
            for task in tasks:
                if model in all_results and classifier in all_results[model] and task in all_results[model][classifier]:
                    result = all_results[model][classifier][task]
                    if 'metrics' in result:
                        metrics = result['metrics']
                        rows.append({
                            'Model': model,
                            'Classifier': classifier,
                            'Task': task,
                            'Accuracy': metrics.get('accuracy', 0.0),
                            'Macro F1': metrics.get('macro_f1', 0.0),
                            'Weighted F1': metrics.get('weighted_f1', 0.0),
                        })
    
    df = pd.DataFrame(rows)
    return df

