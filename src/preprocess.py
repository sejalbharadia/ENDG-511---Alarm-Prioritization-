import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.model_selection import StratifiedShuffleSplit
import wfdb


def find_record_stems(path: Path, max_files=200):
    """Locate WFDB record stems by scanning for .hea or .dat files."""
    stems = []
    for ext in ['*.hea', '*.dat', '*.hea']:
        for p in path.rglob(ext):
            stem = p.with_suffix('')
            stems.append(stem)
            if len(stems) >= max_files:
                return stems
    return stems


def resample_signal(sig: np.ndarray, orig_fs: float, target_fs: float = 125.0) -> np.ndarray:
    """Resample each channel of a signal to target sampling frequency.

    sig: shape (T, C) or (T,) or (T, 1)
    returns array of shape (T_new, C)
    """
    if orig_fs == target_fs:
        return sig
    # scipy.signal.resample works along axis 0
    num_samples = int(sig.shape[0] * target_fs / orig_fs)
    resampled = signal.resample(sig, num_samples, axis=0)
    return resampled


def segment_signal(sig: np.ndarray, fs: float, window_s: float = 30.0, overlap: float = 0.5) -> np.ndarray:
    """Segment signal into overlapping windows.

    Returns array shape (num_windows, C, T_window)
    """
    step = int(window_s * fs * (1 - overlap))
    wlen = int(window_s * fs)
    segments = []
    for start in range(0, sig.shape[0] - wlen + 1, step):
        seg = sig[start:start + wlen]
        segments.append(seg.T)  # transpose to (C, T)
    if len(segments) == 0 and sig.shape[0] >= wlen:
        segments.append(sig[:wlen].T)
    return np.stack(segments, axis=0) if segments else np.zeros((0, sig.shape[1], wlen))


def normalize_channels(seg: np.ndarray) -> np.ndarray:
    """Zero-mean unit variance per channel. seg shape (C, T) or (N, C, T)."""
    if seg.ndim == 2:
        mean = seg.mean(axis=1, keepdims=True)
        std = seg.std(axis=1, keepdims=True) + 1e-6
        return (seg - mean) / std
    elif seg.ndim == 3:
        mean = seg.mean(axis=2, keepdims=True)
        std = seg.std(axis=2, keepdims=True) + 1e-6
        return (seg - mean) / std
    else:
        return seg


def load_labels(data_dir: Path) -> pd.DataFrame:
    """Attempt to load label CSV from data_dir. Return dataframe with at least 'record' and 'label' columns."""
    candidates = list(data_dir.rglob('*REFERENCE*.csv')) + list(data_dir.rglob('*.csv'))
    if not candidates:
        raise FileNotFoundError('No CSV label files found in ' + str(data_dir))
    df = pd.read_csv(candidates[0])
    # normalize column names
    df.columns = [c.strip().lower() for c in df.columns]
    # find record & label columns
    rec_col = next((c for c in df.columns if 'record' in c), df.columns[0])
    label_col = next((c for c in df.columns if 'label' in c or 'alarm' in c), df.columns[-1])
    return df[[rec_col, label_col]].rename(columns={rec_col: 'record', label_col: 'label'})


def assign_window_labels(record_name: str, n_windows: int, labels_df: pd.DataFrame) -> np.ndarray:
    """Return labels for each window of the given record. If record missing, default to zeros."""
    subset = labels_df[labels_df['record'].astype(str) == str(record_name)]
    if subset.empty:
        return np.zeros(n_windows, dtype=np.int64)
    # assume one label per record; broadcast
    lbl = subset['label'].iloc[0]
    return np.full(n_windows, lbl, dtype=np.int64)


def build_dataset(data_dir: Path, output_dir: Path, target_fs: float = 125.0):
    """Run full preprocessing pipeline and save X.npy/y.npy in output_dir."""
    labels = load_labels(data_dir)
    X_list = []
    y_list = []
    for stem in find_record_stems(data_dir):
        try:
            rec = wfdb.rdrecord(str(stem))
            sig = rec.p_signal if hasattr(rec, 'p_signal') else np.asarray(rec.d_signal)
            orig_fs = getattr(rec, 'fs', None)
        except Exception:
            continue
        if orig_fs is None:
            continue
        sig = np.asarray(sig)
        # ensure shape (T, C)
        if sig.ndim == 1:
            sig = sig[:, None]
        sig = resample_signal(sig, orig_fs, target_fs)
        windows = segment_signal(sig, target_fs)
        if windows.size == 0:
            continue
        windows = normalize_channels(windows)
        nwin = windows.shape[0]
        lbls = assign_window_labels(stem.name, nwin, labels)
        X_list.append(windows)
        y_list.append(lbls)
    if not X_list:
        raise RuntimeError('No data processed; check data_dir contents')
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / 'X.npy', X)
    np.save(output_dir / 'y.npy', y)
    print(f'Saved dataset: X {X.shape}, y {y.shape} to {output_dir}')
    return X, y


def stratified_splits(X: np.ndarray, y: np.ndarray, train_frac=0.7, val_frac=0.15, test_frac=0.15, random_state=42):
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=(1 - train_frac), random_state=random_state)
    train_idx, rest_idx = next(sss1.split(X, y))
    rest_y = y[rest_idx]
    val_rel = val_frac / (val_frac + test_frac)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=(1 - val_rel), random_state=random_state)
    val_idx_rel, test_idx_rel = next(sss2.split(rest_idx, rest_y))
    val_idx = rest_idx[val_idx_rel]
    test_idx = rest_idx[test_idx_rel]
    return train_idx, val_idx, test_idx


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Preprocess PhysioNet data to numpy arrays.')
    parser.add_argument('--raw', type=str, default='data/raw', help='Path to raw WFDB data')
    parser.add_argument('--out', type=str, default='data/processed', help='Output directory')
    parser.add_argument('--fs', type=float, default=125.0, help='Target sampling frequency')
    args = parser.parse_args()
    X, y = build_dataset(Path(args.raw), Path(args.out), target_fs=args.fs)
    print('Saved processed data; X shape', X.shape, 'y shape', y.shape)
    # optionally demonstrate splits
    ti, vi, ui = stratified_splits(X, y)
    print('Train/val/test sizes', len(ti), len(vi), len(ui))

