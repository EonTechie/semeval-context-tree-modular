# Migration Guide: From siparismaili01 to Modular Structure

## 🔄 What's Different?

### ✅ What's the Same:
- **19 Context Tree Features**: Same feature extraction (`extract_batch_features_v2`)
- **Multiple Models**: BERT, RoBERTa, DeBERTa, XLNet
- **Multiple Classifiers**: LogisticRegression, LinearSVC, RandomForest, XGBoost, LightGBM
- **Multiple Tasks**: Clarity, Evasion
- **Results Format**: Same metrics (accuracy, macro F1, weighted F1, etc.)

### 🔄 What's Changed:
- **TEST → DEV**: All experiments now use DEV set (TEST is only for final evaluation)
- **Modular Structure**: Code split into modules (data, features, models, evaluation, storage)
- **Storage Strategy**: Large files → Drive, Metadata → GitHub
- **Load Functions**: All saved data can be reloaded

## Storage Locations

### What Gets Saved Where:

| Item | Location | Format | Size |
|------|----------|--------|------|
| **Features** | Drive: `features/raw/X_{split}_{model}_{task}.npy` | .npy | Large |
| **Feature Metadata** | GitHub: `metadata/features_{split}_{model}_{task}.json` | JSON | Small |
| **Predictions** | Drive: `predictions/pred_{split}_{model}_{classifier}_{task}.npy` | .npy | Medium |
| **Probabilities** | Drive: `features/probabilities/probs_{split}_{model}_{classifier}_{task}.npy` | .npy | Medium |
| **Results** | GitHub: `results/{experiment_id}.json` | JSON | Small |
| **Splits** | Drive: `splits/dataset_splits.pkl` | .pkl | Small |

### What Gets Printed:
- ✅ Classification reports (per classifier)
- ✅ Results tables (sorted by Macro F1)
- ✅ Model-wise summaries
- ✅ Progress bars during feature extraction

## Load Functions

All saved data can be reloaded:

```python
from src.storage.manager import StorageManager

storage = StorageManager(...)

# Load features
X_train = storage.load_features('bert', 'clarity', 'train')

# Load predictions
preds = storage.load_predictions('bert', 'LogisticRegression', 'clarity', 'dev')

# Load probabilities
probs = storage.load_probabilities('bert', 'LogisticRegression', 'clarity', 'dev')

# Load splits
train_ds = storage.load_split('train')
```

## 📚 Reference Other Implementations

Since this is in the same repo as Paper authors', Ihsan's, and Ece's code:
- **Paper authors' code**: `../../` (root) - Original baseline implementations
- **Ihsan's approach**: `../../clarity-semeval-2026-ihsan/` - Check `src/representation/` for representation-level fusion
- **Ece's approach**: `../../clarity-semeval-2026-ece/` - Check `src/evaluation/late_fusion.py` for decision-level fusion

You can adapt their code if needed for late fusion or representation fusion.

## 🚀 Next Steps

1. **Run setup notebook** (`00_setup.ipynb`) - TODO: Create notebooks
2. **Split data** (`01_data_split.ipynb`) - Creates Train/Dev/Test
3. **Extract features** (`02_feature_extraction_separate.ipynb`) - For each model
4. **Train & evaluate** (`03_train_evaluate.ipynb`) - Train classifiers, get results
5. **Early fusion** (`04_early_fusion.ipynb`) - Fuse attention features
6. **Final evaluation** (`05_final_evaluation.ipynb`) - Evaluate on TEST set only

