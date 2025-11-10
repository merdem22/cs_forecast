import numpy as np
from typing import Tuple
from scipy.signal import welch
import torch

def seq_features_meanonly(seq_WCT: np.ndarray, fs: float = 100.0) -> np.ndarray:
    """
    Simple features averaged over channels and windows.
    For each window: Welch bandpowers (delta..low-gamma) + RMS, VAR, Line Length.
    Then mean over channels -> mean over windows -> 1D vector of length 8.
    """
    bands: Tuple[Tuple[float, float], ...] = ((1,4),(4,8),(8,12),(12,30),(30,40))
    W, C, T = seq_WCT.shape
    feats_w = []

    for w in range(W):
        win = seq_WCT[w]             # (C,T)
        per_ch = []
        for c in range(C):
            sig = win[c]
            f, Pxx = welch(sig, fs=fs, nperseg=128, noverlap=64)
            bp = []
            for lo, hi in bands:
                m = (f >= lo) & (f < hi)
                p = np.trapz(Pxx[m], f[m]) if np.any(m) else 0.0
                bp.append(np.log(p + 1e-12))
            rms = float(np.sqrt(np.mean(sig**2)))
            var = float(np.var(sig))
            ll  = float(np.sum(np.abs(np.diff(sig))))
            per_ch.append(bp + [rms, var, ll])
        per_ch = np.asarray(per_ch)  # (C, 8)
        feats_w.append(per_ch.mean(axis=0))  # mean over channels -> (8,)
    feats_w = np.asarray(feats_w)            # (W, 8)
    return feats_w.mean(axis=0)              # mean over windows -> (8,)

def build_feature_matrix(dataset):
    """
    dataset: returns (x,y,pid) with x:(W_in,C,T), y:(L,), pid:str
    Returns X:(N,8), Y:(N,L), G:(N,)
    """
    X, Y, G = [], [], []
    for i in range(len(dataset)):
        x, y, pid = dataset[i]
        x_np = x.numpy() if isinstance(x, torch.Tensor) else x
        y_np = y.numpy() if isinstance(y, torch.Tensor) else y
        X.append(seq_features_meanonly(x_np))
        Y.append(y_np.reshape(1, -1))
        G.append(pid)
    X = np.stack(X, axis=0)
    Y = np.concatenate(Y, axis=0)
    G = np.asarray(G)
    return X, Y, G
