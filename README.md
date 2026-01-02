# SemEval 2026 Task 6: Modular Context Tree Feature Extraction

## 📋 Overview

This repository implements **modular Context Tree feature extraction** for SemEval 2026 Task 6 (CLARITY).

**Key Features:**
- ✅ 19 Context Tree features (attention-based, pattern-based, lexicon-based)
- ✅ Multiple models: BERT, RoBERTa, DeBERTa, XLNet
- ✅ Multiple classifiers: LogisticRegression, LinearSVC, RandomForest, XGBoost, LightGBM
- ✅ Early fusion support (concatenate attention features)
- ✅ Complete evaluation (metrics, plots, confusion matrix, PR/ROC curves)
- ✅ Storage manager (GitHub for metadata, Drive for large data)
- ✅ TEST leakage prevention (Train/Dev/Test split)

**Note:** This is a standalone repository for modular Context Tree feature extraction. You can reference other implementations (Paper, Ihsan, Ece) from the main Question-Evasion repository if needed.

## 🚀 Quick Start (Colab)

### Option 1: GitHub Only (Recommended for small experiments)

1. Open [Google Colab](https://colab.research.google.com/)
2. Create a new notebook
3. Run the setup cell:

```python
# Clone repository
!git clone https://github.com/EonTechie/semeval-context-tree-modular.git
%cd semeval-context-tree-modular

# Install dependencies
!pip install -r requirements.txt

# Add to path
import sys
sys.path.append('/content/semeval-context-tree-modular')

# Import modules
from src.features.extraction import extract_features_for_model
from src.storage.manager import StorageManager

print("✅ Setup complete!")
```

4. Open any notebook from `notebooks/` folder
5. Run cells sequentially

### Option 2: GitHub + Google Drive (For large features/models)

1. Open [Google Colab](https://colab.research.google.com/)
2. Create a new notebook
3. Run the setup cell:

```python
# Clone repository
!git clone https://github.com/EonTechie/semeval-context-tree-modular.git
%cd semeval-context-tree-modular

# Install dependencies
!pip install -r requirements.txt

# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Setup paths
BASE_PATH = '/content/semeval-context-tree-modular'
DATA_PATH = '/content/drive/MyDrive/semeval_data'

# Add to path
import sys
sys.path.append(BASE_PATH)

# Import modules
from src.features.extraction import extract_features_for_model
from src.storage.manager import StorageManager

# Initialize storage
storage = StorageManager(
    base_path=BASE_PATH,
    data_path=DATA_PATH,
    github_path=BASE_PATH
)

print("✅ Setup complete!")
print(f"📁 Code/Metadata: {BASE_PATH}")
print(f"📁 Large Data: {DATA_PATH}")
```

4. Open any notebook from `notebooks/` folder
5. Update `DATA_PATH` in notebook to save large files to Drive

## 📁 Repository Structure

```
semeval-modular/
├── src/              # Python modules (importable)
│   ├── data/        # Data loading, splitting
│   ├── features/    # Feature extraction, fusion
│   ├── models/      # Classifiers, fusion models
│   ├── evaluation/  # Metrics, reporting
│   ├── storage/     # Save/load utilities
│   └── utils/       # Helper functions
├── configs/         # Configuration files
├── notebooks/       # Colab notebooks (run these)
├── metadata/        # Metadata JSONs (GitHub)
└── results/         # Results JSONs (GitHub)
```

## 🔧 Requirements

See `requirements.txt` for full list. Main dependencies:
- torch
- transformers
- scikit-learn
- pandas
- numpy

## 📦 Installation from GitHub

```bash
# Clone the repository
git clone https://github.com/EonTechie/semeval-context-tree-modular.git
cd semeval-context-tree-modular

# Install dependencies
pip install -r requirements.txt
```

## 🔗 Related Implementations

For reference, you can check other implementations in the main Question-Evasion repository:
- **Paper authors' code**: https://github.com/konstantinosftw/Question-Evasion (root directory)
- **Ihsan's implementation**: Representation-level fusion
- **Ece's implementation**: Decision-level fusion

## 📊 Experiments

This repository implements:
1. **Separate Models Approach**: Each model (BERT, RoBERTa, DeBERTa, XLNet) trained separately
2. **Early Fusion Approach**: Attention features from all models fused together
3. **Late Fusion Approach**: Probability-level fusion (Ece style) - TODO
4. **Representation Fusion**: Representation-level fusion (Ihsan style) - TODO

This is a standalone repository focused on modular Context Tree feature extraction.

