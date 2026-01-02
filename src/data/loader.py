"""
Dataset loading utilities
"""
from datasets import load_dataset as hf_load_dataset
from typing import Dict, Any


def load_dataset(dataset_name: str = "ailsntua/QEvasion", cache_dir: str = None) -> Dict[str, Any]:
    """
    Load QEvasion dataset from HuggingFace
    
    Args:
        dataset_name: HuggingFace dataset identifier
        cache_dir: Optional cache directory
    
    Returns:
        Dictionary with 'train' and 'test' splits
    """
    ds = hf_load_dataset(dataset_name, cache_dir=cache_dir)
    
    print(f"✅ Dataset loaded: {dataset_name}")
    print(f"   Train: {len(ds['train'])} samples")
    print(f"   Test: {len(ds['test'])} samples")
    
    return ds

