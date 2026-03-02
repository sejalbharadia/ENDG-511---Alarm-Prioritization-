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

#### MIMIC-III Dataset (Approval required)
1. Go to [physionet.org](https://physionet.org/)
2. Create a free account
3. Sign the data use agreement
4. Approval typically takes 1-3 days
5. Download data to `./data/raw/` once approved

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
