# ENDG 511 — Team 11: Workload-Aware AI Alarm Prioritization
Final Project - ICU Alarm Prioritization System using Self-Supervised Learning & Model Optimization

## Project Overview

This project develops an **intelligent ICU alarm prioritization system** that classifies cardiac arrhythmias into three severity levels using deep learning:
- **Low** (Class 0): Normal rhythm — no immediate threat
- **Medium** (Class 1): Atrial/supraventricular — monitor required  
- **Critical** (Class 2): Ventricular/life-threatening — immediate action

The system trains on the **MIT-BIH Arrhythmia Database** and addresses real challenges:
- **Class imbalance** (200x more "Low" than "Medium" samples)
- **Limited labeled data** (SSL pre-training to learn ECG structure)
- **Deployment constraints** (model compression via pruning & early-exit branching)

**Key ML Techniques:**
1. Self-Supervised Learning (SSL) with NTXentLoss for pre-training
2. Transfer learning with labeled fine-tuning
3. Class-weighted loss for imbalanced data
4. Structured pruning (30% parameter reduction)
5. Early-exit cascaded inference (2x latency reduction)
6. Comprehensive evaluation with confusion matrices & trade-off analysis

---

## Quick Start

### Step 1: Clone & Install
```bash
git clone https://github.com/sejalbharadia/ENDG-511---Alarm-Prioritization-.git
cd ENDG-511---Alarm-Prioritization-
pip install -r requirements.txt
```

### Step 2: Run the Main Notebook
Open and run the main analysis:
```bash
jupyter notebook notebooks/Team11_AP.ipynb
```
This notebook contains the **complete end-to-end pipeline** with all models, training, and evaluation.

### Step 3: (Optional) Explore Data
For data exploration and visualization:
```bash
jupyter notebook notebooks/data_exploration.ipynb
```

---

## Project Structure

```
.
├── notebooks/
│   ├── Team11_AP.ipynb                        # ⭐ MAIN: Complete end-to-end pipeline
│   │   └── All 4 models + results + visualizations
│   └── data_exploration.ipynb                 # Data EDA & signal visualization                         
│
├── src/                                       # Python modules for reusable code
│   ├── model.py                               # CNN architectures (BaselineCNN, EarlyExitCNN)
│   ├── train.py                               # Training functions (supervised, SSL)
│   ├── preprocess.py                          # Data loading & preprocessing pipeline
│   ├── evaluate.py                            # Metrics & evaluation utilities
│   ├── ssl_train.py                           # Self-supervised learning setup
│   ├── prune.py                               # Model pruning utilities
│   ├── quantize.py                            # INT8 quantization (optional)
│   ├── utils.py                               # Helper functions
│   └── __init__.py                            # Package initialization
│
├── data/
│   ├── raw/                                   # MIT-BIH files (not in Git)
│   └── processed/                             # Preprocessed .npy arrays
│
├── results/                                   # Saved models, plots, metrics
│   ├── baseline_model.pt                      # Baseline CNN weights
│   ├── pruned_model.pt                        # Pruned CNN weights
│   ├── confusion_matrices.png                 # 4-model comparison heatmaps
│   ├── model_comparison.png                   # Accuracy/F1/Recall/Latency bars
│   ├── early_exit_tradeoff.png               # Threshold sweep analysis
│   └── pruning_tradeoff.png                  # Pruning ratio analysis
│
├── requirements.txt                           # Dependencies
├── README.md                                  # This file
└── LICENSE
```

---

## Notebooks Guide

### 1. **Team1_AP.ipynb**  

**What it does:** 
Complete end-to-end machine learning pipeline with all 4 models and extensive comments.

**Contents:**
- Data loading from MIT-BIH database
- Data preprocessing (normalization, stratified split)
- **4 Models trained & evaluated:**
  1. **Baseline CNN** — Supervised learning from scratch
  2. **SSL CNN** — Self-supervised pre-training + fine-tuning
  3. **Early Exit CNN** — Cascaded inference for latency reduction
  4. **Pruned CNN** — 30% parameter reduction + recovery training
- Results: Confusion matrices, accuracy metrics, trade-off plots

**Who should use this:** Anyone wanting to understand the full pipeline, researchers comparing strategies, best for learning & reproducibility.

---

### 2. **data_exploration.ipynb**

**What it does:** 
Data visualization and exploratory data analysis (EDA).

**Contents:**
- Load MIT-BIH records using `wfdb`
- Plot raw ECG signals
- Visualize 360-sample beat windows
- Inspect class distribution & label imbalance
- Signal statistics & annotation counts

**Who should use this:** Anyone new to the MIT-BIH dataset, understanding ECG morphology, debugging data loading.

---

### 3. **ENDG511_Team11_AlarmPrioritization_MITBIH.ipynb**



** Use **Team11_AP.ipynb** 

---


## Python Modules (src/) - For Production/Reusability

All modules are designed as **reusable building blocks** for experiments and production systems. These py files are the framework from githubs refrenced in the course D2L. They are used and implemented in the final notebook - Team11_AP.

### **model.py** — Neural Network Architectures

```python
from src.model import BaselineCNN, EarlyExitCNN

# Simple 1-D CNN (360 samples → 3 classes)
model = BaselineCNN()
output = model(ecg_batch)  # (B, 360) → (B, 3) logits

# Cascaded inference model
early_exit_model = EarlyExitCNN()
exit1_logits, final_logits = early_exit_model(ecg_batch)
```

**Architectures:**
- `BaselineCNN`: 2 conv blocks + FC layer (lightweight & efficient)
- `EarlyExitCNN`: Same backbone + early-exit head (enables cascaded inference)
- Both designed for pruning & quantization compatibility

---

### **train.py** — Training Functions

```python
from src.train import train_supervised

model, loss_history = train_supervised(
    model, train_loader, optimizer, device='cuda', epochs=10
)
```

Supports:
- Supervised training with optional weighted loss
- Learning rate scheduling
- Progress bars (tqdm)
- Loss tracking

---

### **preprocess.py** — Data Loading & Preprocessing

```python
from src.preprocess import build_dataset, create_dataloaders

# Load MIT-BIH and extract 360-sample beat windows
X, y = build_dataset('data/raw', 'data/processed')

# Create PyTorch DataLoaders
train_loader, test_loader = create_dataloaders(X, y, test_size=0.2)
```

**Pipeline steps:**
1. Read WFDB `.hea` and `.atr` files from MIT-BIH database
2. Extract beat annotations
3. Center windows (180 samples before/after beat)
4. Normalize per-sample (zero-mean, unit variance)
5. Save as `X.npy` and `y.npy` for caching
6. Create stratified train/test split

---

### **evaluate.py** — Metrics & Evaluation Utilities

```python
from src.evaluate import evaluate_model, early_exit_evaluate

# Standard evaluation
cm, acc, f1, critical_recall = evaluate_model(model, test_loader)

# Early-exit evaluation with threshold sweeping
cm, acc, f1, critical_recall, exit_rate = early_exit_evaluate(
    model, test_loader, threshold=0.75
)
```

**Metrics computed:**
- Accuracy, Macro-F1, **Critical Recall** (most important for medical applications)
- Confusion matrix (3×3)
- Early exit rate (% of samples exiting early)
- Inference latency (ms/sample)

---

### **ssl_train.py** — Self-Supervised Learning

```python
from src.ssl_train import train_ssl, augment

# Pre-train backbone on unlabeled data using contrastive learning
ssl_model = train_ssl(model, train_loader, epochs=10, device='cuda')

# Later: Transfer to supervised task with labels
fine_tuned_model = transfer_learn(ssl_model, train_loader_labeled, epochs=10)
```

**Key Features:**
- **NTXentLoss**: Contrastive learning (SimCLR-style)
- **Data Augmentations**: Noise, amplitude scaling, baseline shift, time masking
- **No labels required** during pre-training → leverages unlabeled ECG data
- Significantly improves downstream performance with limited labeled data

---

### **prune.py** — Structured Model Pruning

```python
from src.prune import prune_model, recover_accuracy

# Remove 30% of convolutional layer parameters
pruned_model = prune_model(baseline_model, ratio=0.3)

# Fine-tune to recover accuracy (critical step!)
pruned_model = recover_accuracy(pruned_model, train_loader, epochs=10)
```

**What it does:**
- Uses magnitude-based structured pruning (torch-pruning)
- Removes entire channels/filters (not individual weights)
- Preserves fully connected layers
- Recovery fine-tuning recovers ~95% of original accuracy


### **utils.py** — Helper Functions

Common utilities:
- Device management (GPU/CPU detection)
- Model checkpointing (save/load)
- Random seed setting (reproducibility)
- Data utilities

---

## End-to-End Workflow Examples

### **For Learning & Experimentation:**

```
1. Open Team11_AP.ipynb in Jupyter
2. Run cells sequentially (RUN ALL) 
3. Examine plots, confusion matrices, trade-off curves
4. Modify hyperparameters (SSL epochs, pruning ratio, etc.) and re-run
5. Compare results between different models
```


**Key Insights:**
- SSL pre-training improves Critical Recall (0.98 → 0.99)
- Early-exit achieves latency reduction with minimal accuracy loss
- Pruning achieves 47% parameter reduction with recovery fine-tuning

---

## Dataset

**MIT-BIH Arrhythmia Database**
- 47 recordings of 30 minutes each
- 2 channels (lead II, V1)
- Sampling rate: 360 Hz
- 110,000+ annotated beats
- 16-bit resolution
- Publicly available (no approval required)

**Download:**
```bash
python -c "import wfdb; wfdb.dl_database('mitdb', './data/raw/')"
```

---

## Dependencies

Core packages:
```
PyTorch 2.0+
NumPy
Pandas
Scikit-learn
Matplotlib / Seaborn
wfdb (MIT-BIH reader)
lightly (SSL framework)
torch-pruning (model compression)
```

Install all:
```bash
pip install -r requirements.txt
```

---

