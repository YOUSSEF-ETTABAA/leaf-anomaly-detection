# 🌿 Leaf Anomaly Detection

Anomaly detection system for tomato leaf disease using **DINOv2** + **MLP Autoencoder**.

The model is trained **only on healthy leaves** and detects any disease by measuring
how badly it reconstructs an unseen leaf image — a healthy leaf reconstructs well,
a diseased leaf reconstructs badly.

---

## Results

| Metric | Score |
|---|---|
| AUC | 0.9882 |
| Accuracy | ~94% |
| Precision | ~99% |
| Recall | ~94% |
| Specificity | ~95% |

---

## How it works

```
Leaf image
    ↓
DINOv2 (Vision Transformer) → 384-dim feature vector
    ↓
PCA → 100-dim feature vector
    ↓
Autoencoder (MLP) → reconstruction error
    ↓
Error > threshold → ANOMALY
Error ≤ threshold → HEALTHY
```

---

## Project Structure

```
leaf-anomaly-detection/
├── src/
│   ├── data/
│   │   ├── loader.py          ← image loading + transforms
│   │   └── prepare_data.py    ← dataset organization
│   ├── features/
│   │   ├── extractor.py       ← DINOv2 feature extraction
│   │   └── pca.py             ← PCA dimensionality reduction
│   ├── models/
│   │   └── autoencoder.py     ← Encoder + Decoder + Autoencoder
│   ├── training/
│   │   └── train.py           ← training loop + checkpoints
│   ├── evaluation/
│   │   ├── metrics.py         ← all evaluation metrics
│   │   └── plots.py           ← visualization dashboard
│   └── utils/
│       ├── cache.py           ← smart caching system
│       └── config.py          ← config loader
├── notebooks/
│   └── 01_full_pipeline.ipynb ← run everything here
├── config.yaml                ← all settings
├── requirements.txt
└── .gitignore
```

---

## Quick Start (Google Colab)

1. Open `notebooks/01_full_pipeline.ipynb` in Google Colab
2. Make sure GPU is enabled: Runtime → Change runtime type → GPU
3. Run all cells: Runtime → Run all
4. The notebook will:
   - Clone this repo automatically
   - Download the PlantVillage dataset from Kaggle
   - Extract DINOv2 features (cached after first run)
   - Train the autoencoder
   - Show full evaluation dashboard

---

## Dataset

PlantVillage — Tomato classes only:

| Split | Content | Count |
|---|---|---|
| Train | Healthy tomato leaves | ~2,037 |
| Test healthy | Healthy tomato leaves | ~1,145 |
| Test anomaly | 9 different diseases | ~33,000+ |

---

## Architecture

**Autoencoder (Vanilla MLP):**
```
Input (100) → Linear(256) + BN + ReLU + Dropout
           → Linear(128) + BN + ReLU        ← latent space
           → Linear(256) + BN + ReLU + Dropout
           → Linear(100)                    ← reconstruction
```

**Training:**
- Loss: Mean Squared Error (MSE)
- Optimizer: Adam (lr=0.001)
- Epochs: 30
- Batch size: 64

---

## Requirements

```bash
pip install -r requirements.txt
```

---

## Configuration

All settings are in `config.yaml`. Change values there without touching the code:

```yaml
autoencoder:
  epochs: 30
  latent_dim: 128

threshold:
  percentile: 90   # higher = fewer false alarms
```
