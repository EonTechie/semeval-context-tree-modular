"""
Data splitting utilities - ensures TEST set is never used for training/selection
"""
from sklearn.model_selection import train_test_split
import numpy as np
from typing import Tuple, Any


def split_dataset(
    dataset,
    test_ratio: float = 0.15,
    dev_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[Any, Any, Any]:
    """
    Split dataset into Train/Dev/Test
    
    IMPORTANT: TEST set is separated FIRST and will ONLY be used in final evaluation!
    
    Args:
        dataset: HuggingFace dataset (train split)
        test_ratio: Ratio for test set (final evaluation only)
        dev_ratio: Ratio for dev set (model selection, feature selection)
        seed: Random seed for reproducibility
    
    Returns:
        train_ds, dev_ds, test_ds
    """
    # Convert to list for splitting (if needed)
    if hasattr(dataset, 'select'):
        # HuggingFace dataset
        indices = np.arange(len(dataset))
    else:
        indices = np.arange(len(dataset))
    
    # First split: separate TEST (never touch until final eval)
    train_dev_indices, test_indices = train_test_split(
        indices,
        test_size=test_ratio,
        random_state=seed,
        shuffle=True
    )
    
    # Second split: Train vs Dev (for model/feature selection)
    train_indices, dev_indices = train_test_split(
        train_dev_indices,
        test_size=dev_ratio / (1 - test_ratio),  # Adjust ratio
        random_state=seed,
        shuffle=True
    )
    
    # Create splits
    if hasattr(dataset, 'select'):
        train_ds = dataset.select(train_indices.tolist())
        dev_ds = dataset.select(dev_indices.tolist())
        test_ds = dataset.select(test_indices.tolist())
    else:
        train_ds = [dataset[i] for i in train_indices]
        dev_ds = [dataset[i] for i in dev_indices]
        test_ds = [dataset[i] for i in test_indices]
    
    print(f"✅ Dataset split:")
    print(f"   Train: {len(train_ds)} samples ({len(train_indices)/len(indices)*100:.1f}%)")
    print(f"   Dev: {len(dev_ds)} samples ({len(dev_indices)/len(indices)*100:.1f}%)")
    print(f"   Test: {len(test_ds)} samples ({len(test_indices)/len(indices)*100:.1f}%)")
    print(f"   ⚠️  TEST set will ONLY be used in final evaluation notebook!")
    
    return train_ds, dev_ds, test_ds

