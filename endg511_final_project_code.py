# ==============================
# ENDG 511 — Team 11 (JETSON CLEAN VERSION)
# Workload-Aware AI Alarm Prioritization
# ==============================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

import wfdb

# Optional pruning (safe fallback if missing)
try:
    import torch_pruning as tp
    PRUNING_AVAILABLE = True
except:
    PRUNING_AVAILABLE = False
    print("WARNING: torch_pruning not installed — pruning will be skipped")

from lightly.loss import NTXentLoss


# ==============================
# SETTINGS
# ==============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

RUN_SSL = True
RUN_PRUNING = PRUNING_AVAILABLE


# ==============================
# LABEL MAP
# ==============================
label_map = {
    'N':0, 'L':0, 'R':0, '/':0,
    'A':1, 'S':1, 'J':1, 'a':1,
    'V':2, 'E':2, '!':2, 'W':2
}

CLASS_NAMES = ['Low', 'Medium', 'Critical']


# ==============================
# DATA LOADING
# ==============================
records = [
    '100','101','102','103','104','105',
    '106','107','108','109','111','112',
    '113','114','115','116','117','118','119'
]

def extract_beats(record):
    signal, _ = wfdb.rdsamp(record, pn_dir='mitdb')
    annotation = wfdb.rdann(record, 'atr', pn_dir='mitdb')
    ecg = signal[:, 0]

    beats, labels = [], []
    window = 180

    for i, sample in enumerate(annotation.sample):
        symbol = annotation.symbol[i]
        if symbol not in label_map:
            continue

        start = sample - window
        end = sample + window

        if start < 0 or end > len(ecg):
            continue

        beats.append(ecg[start:end])
        labels.append(label_map[symbol])

    return beats, labels


X_list, y_list = [], []

for r in records:
    b, l = extract_beats(r)
    X_list.extend(b)
    y_list.extend(l)

X = np.array(X_list)
y = np.array(y_list)

print("Dataset shape:", X.shape)


# ==============================
# PREPROCESS
# ==============================
X = (X - np.mean(X, axis=1, keepdims=True)) / (np.std(X, axis=1, keepdims=True) + 1e-8)
X = X[:, None, :]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

train_loader = DataLoader(
    TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                  torch.tensor(y_train, dtype=torch.long)),
    batch_size=64, shuffle=True
)

test_loader = DataLoader(
    TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                  torch.tensor(y_test, dtype=torch.long)),
    batch_size=64
)

print("Data ready")


# ==============================
# CLASS WEIGHTS
# ==============================
counts = np.bincount(y_train, minlength=3).astype(np.float32)
weights = 1.0 / (counts + 1e-6)
weights = weights / weights.sum()
class_weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)


# ==============================
# MODEL
# ==============================
class BaselineCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, 5)
        self.conv2 = nn.Conv1d(16, 32, 5)
        self.pool = nn.MaxPool1d(2)
        self.fc = nn.Linear(32 * 87, 3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    def embed(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        return x.view(x.size(0), -1)


# ==============================
# SSL AUGMENT
# ==============================
def augment(x):
    B, C, T = x.shape
    x = x.clone()

    x += 0.05 * torch.randn_like(x)
    x *= torch.empty(B,1,1).uniform_(0.8,1.2)
    x += torch.empty(B,1,1).uniform_(-0.1,0.1)

    return x


# ==============================
# SSL TRAINING
# ==============================
ssl_model = BaselineCNN().to(DEVICE)

if RUN_SSL:
    print("SSL training...")
    optimizer = torch.optim.Adam(ssl_model.parameters(), lr=3e-4)
    criterion = NTXentLoss(temperature=0.5)

    for epoch in range(5):
        total = 0
        ssl_model.train()

        for x, _ in train_loader:
            x = x.to(DEVICE)

            z1 = ssl_model.embed(augment(x))
            z2 = ssl_model.embed(augment(x))

            z1 = F.normalize(z1, dim=1)
            z2 = F.normalize(z2, dim=1)

            loss = criterion(z1, z2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total += loss.item()

        print("SSL epoch", epoch+1, total / len(train_loader))


# ==============================
# FINE TUNE SSL MODEL
# ==============================
ssl_ft = BaselineCNN().to(DEVICE)
ssl_ft.load_state_dict(ssl_model.state_dict())

opt = torch.optim.Adam(ssl_ft.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss(weight=class_weights)

for epoch in range(5):
    ssl_ft.train()
    for x,y in train_loader:
        x,y = x.to(DEVICE), y.to(DEVICE)

        loss = loss_fn(ssl_ft(x), y)

        opt.zero_grad()
        loss.backward()
        opt.step()


# ==============================
# BASELINE MODEL
# ==============================
baseline = BaselineCNN().to(DEVICE)
opt = torch.optim.Adam(baseline.parameters(), lr=1e-3)

for epoch in range(5):
    baseline.train()
    for x,y in train_loader:
        x,y = x.to(DEVICE), y.to(DEVICE)

        loss = loss_fn(baseline(x), y)

        opt.zero_grad()
        loss.backward()
        opt.step()


# ==============================
# EVALUATION (FIXED)
# ==============================
def evaluate(model):
    model.eval()
    y_t, y_p = [], []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            pred = torch.argmax(model(x), 1)

            y_t.extend(y.cpu().numpy())
            y_p.extend(pred.cpu().numpy())

    return (
        accuracy_score(y_t, y_p),
        f1_score(y_t, y_p, average='macro'),
        y_t,
        y_p
    )


acc, f1, y_t, y_p = evaluate(baseline)

print("\nRESULTS")
print("Accuracy:", acc)
print("F1:", f1)


# ==============================
# PLOTS
# ==============================
cm = confusion_matrix(y_t, y_p)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.close()

print("Saved plots")PS C:\Users\hirat> scp "C:\Users\hirat\Desktop\endg511_final_project_code.py" hira@172.21.102.249:~/