# utils/metrics.py
from __future__ import annotations
import numpy as np
import torch 

def mean_normalize(baseline_vals, model_vals):
    b = float(np.mean(baseline_vals))
    m = float(np.mean(model_vals))
    return m / max(b, 1e-12)

def binary_cls_metrics(probs: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, eps: float = 1e-8):
    if target.dim() == 1:
        target = target.unsqueeze(1)
    if probs.dim() == 1:
        probs = probs.unsqueeze(1)

    target = target.float()
    preds  = (probs >= threshold).float()

    acc = (preds == target).float().mean()

    return acc
