"""
Data splitting utilities - splits HuggingFace train split into Train/Dev
"""
from sklearn.model_selection import train_test_split
import numpy as np
from typing import Tuple, Any
from collections import Counter


def split_train_into_train_dev(
    train_dataset,
    dev_ratio: float = 0.20,
    seed: int = 42
) -> Tuple[Any, Any]:
    """
    Split HuggingFace train split into Train and Dev (80-20 split)
    
    Args:
        train_dataset: HuggingFace train split (from dataset['train'])
        dev_ratio: Ratio for dev set (0.20 = 20% of train data becomes dev)
        seed: Random seed for reproducibility
    
    Returns:
        train_ds, dev_ds
    """
    # Convert to indices for splitting
    if hasattr(train_dataset, 'select'):
        indices = np.arange(len(train_dataset))
    else:
        indices = np.arange(len(train_dataset))
    
    # Split train into train and dev (80-20)
    train_indices, dev_indices = train_test_split(
        indices,
        test_size=dev_ratio,  # 20% becomes dev
        random_state=seed,
        shuffle=True
    )
    
    # Create splits
    if hasattr(train_dataset, 'select'):
        train_ds = train_dataset.select(train_indices.tolist())
        dev_ds = train_dataset.select(dev_indices.tolist())
    else:
        train_ds = [train_dataset[i] for i in train_indices]
        dev_ds = [train_dataset[i] for i in dev_indices]
    
    print(f"Dataset split:")
    print(f"   Train: {len(train_ds)} samples ({len(train_indices)/len(indices)*100:.1f}%)")
    print(f"   Dev: {len(dev_ds)} samples ({len(dev_indices)/len(indices)*100:.1f}%)")
    
    return train_ds, dev_ds


def build_evasion_majority_dataset(
    dataset,
    label_column: str = "evasion_label",
    annotator_columns: tuple = ("annotator1", "annotator2", "annotator3"),
    verbose: bool = True
):
    """
    Build evasion dataset with majority voting from annotators.
    Drops samples without strict majority (2/3 or 3/3).
    
    This function is idempotent: if evasion_label already exists and is non-empty,
    the dataset is returned as-is without modification.
    
    Args:
        dataset: HuggingFace dataset
        label_column: Name of the label column to create/use
        annotator_columns: Tuple of annotator column names
        verbose: Print progress messages
    
    Returns:
        Filtered dataset with majority-voted evasion labels
    """
    # If dataset already has a usable evasion_label, return as-is
    # Get column names safely (works for both HuggingFace Dataset and SimpleDataset)
    try:
        column_names = dataset.column_names
    except AttributeError:
        # Fallback: get keys from first sample (for SimpleDataset or dict-based datasets)
        if len(dataset) > 0:
            column_names = list(dataset[0].keys()) if isinstance(dataset[0], dict) else []
        else:
            column_names = []
    
    if label_column in column_names:
        sample_value = dataset[0].get(label_column, None)
        if sample_value is not None and str(sample_value).strip() != "":
            if verbose:
                print(f"[EVASION MAJORITY] Existing {label_column} found → using dataset as-is.")
            return dataset
    
    # Majority vote helper
    def majority_vote(labels):
        """Returns majority label if >= 2 votes, else None"""
        counts = Counter(labels)
        top_label, top_count = counts.most_common(1)[0]
        if top_count >= 2:
            return top_label
        return None
    
    # Collect majority labels
    keep_indices = []
    majority_labels = []
    
    for idx in range(len(dataset)):
        votes = []
        for col in annotator_columns:
            val = dataset[idx].get(col, None)
            if val is not None and str(val).strip() != "":
                votes.append(str(val).strip())
        
        # Need at least 2 votes for majority
        if len(votes) < 2:
            continue
        
        maj = majority_vote(votes)
        if maj is not None:
            keep_indices.append(idx)
            majority_labels.append(maj)
    
    # Build new dataset with majority labels
    if hasattr(dataset, 'select'):
        filtered_dataset = dataset.select(keep_indices)
    else:
        filtered_dataset = [dataset[i] for i in keep_indices]
    
    # Add/update evasion_label column
    # Get column names safely
    try:
        filtered_column_names = filtered_dataset.column_names
    except AttributeError:
        # Fallback: get keys from first sample
        if len(filtered_dataset) > 0:
            filtered_column_names = list(filtered_dataset[0].keys()) if isinstance(filtered_dataset[0], dict) else []
        else:
            filtered_column_names = []
    
    if hasattr(filtered_dataset, 'remove_columns') and label_column in filtered_column_names:
        filtered_dataset = filtered_dataset.remove_columns([label_column])
    
    if hasattr(filtered_dataset, 'add_column'):
        filtered_dataset = filtered_dataset.add_column(label_column, majority_labels)
    else:
        # For list-based datasets, update in place
        for i, label in enumerate(majority_labels):
            filtered_dataset[i][label_column] = label
    
    if verbose:
        print(f"[EVASION MAJORITY] Original size: {len(dataset)}")
        print(f"[EVASION MAJORITY] Kept (majority): {len(filtered_dataset)}")
        print(f"[EVASION MAJORITY] Dropped (no majority): {len(dataset) - len(filtered_dataset)}")
    
    return filtered_dataset

