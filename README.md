# ENDG-511---Alarm-Prioritization-
Final Project - ICU Alarm Prioritization System

## Project Overview
This project develops a machine learning system to prioritize ICU alarms based on severity and clinical significance.

## Setup Instructions

### Step 1: Clone the Repository
```bash
git clone https://github.com/sejalbharadia/ENDG-511---Alarm-Prioritization-.git
cd ENDG-511---Alarm-Prioritization-
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Download Datasets

#### PhysioNet Challenge 2015 (No approval required)
```bash
python -c "import wfdb; wfdb.dl_database('challenge-2015', './data/raw/')"
```


## Folder Structure
```
.
├── data/
│   ├── raw/                 # Downloaded PhysioNet files (not pushed to GitHub)
│   └── processed/           # Cleaned numpy arrays after preprocessing
├── src/                     # All Python source code files
├── notebooks/               # Jupyter notebooks for experiments
├── results/                 # Saved models, plots, metrics
├── requirements.txt         # List of Python packages (torch, lightly, wfdb, numpy, sklearn)
├── .gitignore              # Git ignore rules
└── README.md               # This file
```

## License
See LICENSE for details. 

## Run Experiments

Example runner script (modes kept minimal):

```bash
# SSL pretraining (Lightly)
python src/run_experiment.py --mode ssl

# Structured pruning (torch-pruning)
python src/run_experiment.py --mode prune --amount 0.3

# Early-exit inference (use thresholds in evaluate)
python src/run_experiment.py --mode early_exit

# INT8 quantization (PyTorch)
python src/run_experiment.py --mode quant
```

## Preprocessing pipeline

A helper module provides a pipeline from raw WFDB records to numpy arrays. Example usage:

```python
from src import preprocess
from pathlib import Path

raw = Path('data/raw')
out = Path('data/processed')
X, y = preprocess.build_dataset(raw, out, target_fs=125.0)

# optional stratified split
train_idx, val_idx, test_idx = preprocess.stratified_splits(X, y)
```

The pipeline:

* resamples all signals to 125 Hz
* segments into 30 s windows (50 % overlap)
* normalizes each channel (zero-mean, unit variance)
* labels windows using the CSV provided by Challenge 2015
* saves `X.npy` and `y.npy` in the output folder

Training/validation/test split is stratified by label (70/15/15).
