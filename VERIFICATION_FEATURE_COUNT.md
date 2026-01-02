# Feature Extraction Count Verification

## Siparismaili01'de:
- **4 model**: BERT, RoBERTa, DeBERTa, XLNet
- **2 task**: Clarity, Evasion
- **2 split**: Train, Test
- **Total**: 4 × 2 × 2 = **16 feature extraction**

## Notebook 02'de:
- **6 model**: bert, bert_political, bert_ambiguity, roberta, deberta, xlnet
- **2 task**: Clarity, Evasion
- **2 split**: Train, Dev
- **Total**: 6 × 2 × 2 = **24 feature extraction**

## Neden 24, 16 değil?
1. **2 model daha eklendi**: bert_political, bert_ambiguity
2. **Test split ayrı notebook'ta**: Test features `05_final_evaluation.ipynb`'de extract ediliyor
3. **Dev split eklendi**: Train/Dev split'i ayrıldı (data leakage önleme)

## Feature Storage Verification

### Dosya Adlandırma Formatı:
```
X_{split}_{model_name}_{task}.npy
```

### Örnek Dosya Adları:
- `X_train_bert_clarity.npy`
- `X_train_bert_evasion.npy`
- `X_train_bert_political_clarity.npy`
- `X_train_bert_political_evasion.npy`
- `X_train_roberta_clarity.npy`
- `X_train_roberta_evasion.npy`
- `X_dev_bert_clarity.npy`
- `X_dev_bert_evasion.npy`
- ... (toplam 24 dosya)

### Unique Olma Garantisi:
- `split`: 'train', 'dev', 'test' (3 değer)
- `model_name`: 'bert', 'bert_political', 'bert_ambiguity', 'roberta', 'deberta', 'xlnet' (6 değer)
- `task`: 'clarity', 'evasion' (2 değer)
- **Kombinasyon sayısı**: 3 × 6 × 2 = 36 (unique kombinasyon garantisi)

### Persistence Garantisi:
1. **Google Drive'a kaydediliyor**: `self.data_path / 'features/raw/X_{split}_{model_name}_{task}.npy'`
2. **HuggingFace cache'e bağımlı değil**: Sadece numpy array kaydediliyor
3. **Metadata GitHub'da**: `metadata/features_{split}_{model_name}_{task}.json`
4. **Load fonksiyonu**: `load_features(model_name, task, split)` → aynı format ile yüklüyor

### Verification Script:
```python
# Tüm feature dosyalarını listele
import os
from pathlib import Path

data_path = Path('/content/drive/MyDrive/semeval_data/features/raw')
feature_files = list(data_path.glob('X_*.npy'))
print(f"Total feature files: {len(feature_files)}")

# Her dosyayı parse et
for f in sorted(feature_files):
    # Format: X_{split}_{model}_{task}.npy
    parts = f.stem.split('_')
    split = parts[1]
    task = parts[-1]
    model = '_'.join(parts[2:-1])
    print(f"  {f.name}: split={split}, model={model}, task={task}")
```

