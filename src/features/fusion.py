"""
Feature fusion utilities: Early fusion (concatenate attention features from multiple models)
"""
import numpy as np
from typing import List, Dict, Tuple


def fuse_attention_features(
    feature_dict: Dict[str, np.ndarray],
    feature_names_dict: Dict[str, List[str]]
) -> Tuple[np.ndarray, List[str]]:
    """
    Fuse attention features from multiple models by concatenation
    
    Args:
        feature_dict: Dictionary mapping model_name -> (N, F) feature matrix
        feature_names_dict: Dictionary mapping model_name -> list of feature names
    
    Returns:
        fused_features: (N, total_features) concatenated features
        fused_feature_names: List of all feature names with model prefix
    """
    model_names = sorted(feature_dict.keys())
    
    # Verify all have same number of samples
    n_samples = feature_dict[model_names[0]].shape[0]
    for model_name in model_names:
        assert feature_dict[model_name].shape[0] == n_samples, \
            f"Model {model_name} has {feature_dict[model_name].shape[0]} samples, expected {n_samples}"
    
    # Concatenate features
    fused_features_list = []
    fused_feature_names = []
    
    for model_name in model_names:
        features = feature_dict[model_name]
        feature_names = feature_names_dict[model_name]
        
        fused_features_list.append(features)
        fused_feature_names.extend([f"{model_name}_{name}" for name in feature_names])
    
    fused_features = np.hstack(fused_features_list)
    
    return fused_features, fused_feature_names


def create_fused_features(
    model_features: Dict[str, np.ndarray],
    model_feature_names: Dict[str, List[str]],
    fusion_method: str = "concat"
) -> Tuple[np.ndarray, List[str]]:
    """
    Create fused features from multiple model features
    
    Args:
        model_features: Dict of model_name -> (N, F) features
        model_feature_names: Dict of model_name -> feature names
        fusion_method: "concat" (default) or "mean" (average)
    
    Returns:
        fused_features: (N, F_fused) fused feature matrix
        fused_feature_names: List of feature names
    """
    if fusion_method == "concat":
        return fuse_attention_features(model_features, model_feature_names)
    elif fusion_method == "mean":
        # Average features (requires same number of features per model)
        model_names = sorted(model_features.keys())
        n_features = model_features[model_names[0]].shape[1]
        for model_name in model_names:
            assert model_features[model_name].shape[1] == n_features, \
                f"Model {model_name} has {model_features[model_name].shape[1]} features, expected {n_features}"
        
        stacked = np.stack([model_features[name] for name in model_names], axis=0)  # (M, N, F)
        fused = np.mean(stacked, axis=0)  # (N, F)
        fused_names = model_feature_names[model_names[0]]  # Use first model's names
        return fused, fused_names
    else:
        raise ValueError(f"Unknown fusion_method: {fusion_method}")

