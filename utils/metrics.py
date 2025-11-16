# utils/metrics.py
from __future__ import annotations

import numpy as np
import torch
from typing import Dict


def mean_normalize(baseline_vals, model_vals):
    b = float(np.mean(baseline_vals))
    m = float(np.mean(model_vals))
    return m / max(b, 1e-12)


def binary_cls_metrics(
    probs: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-8,
):
    if target.dim() == 1:
        target = target.unsqueeze(1)
    if probs.dim() == 1:
        probs = probs.unsqueeze(1)

    target = target.float()
    preds = (probs >= threshold).float()

    acc = (preds == target).float().mean()

    return acc


def _to_numpy(arr) -> np.ndarray:
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


def binary_confusion_counts(
    probs,
    target,
    threshold: float = 0.5,
) -> Dict[str, float]:
    y_true = _to_numpy(target).astype(np.int32).reshape(-1)
    y_prob = _to_numpy(probs).astype(np.float64).reshape(-1)
    y_pred = (y_prob >= threshold).astype(np.int32)
    tp = float(np.sum((y_pred == 1) & (y_true == 1)))
    tn = float(np.sum((y_pred == 0) & (y_true == 0)))
    fp = float(np.sum((y_pred == 1) & (y_true == 0)))
    fn = float(np.sum((y_pred == 0) & (y_true == 1)))
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def binary_auc(probs, target) -> float:
    y_true = _to_numpy(target).astype(np.int32).reshape(-1)
    y_score = _to_numpy(probs).astype(np.float64).reshape(-1)
    pos = y_true == 1
    neg = y_true == 0
    n_pos = np.sum(pos)
    n_neg = np.sum(neg)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = np.argsort(np.argsort(y_score, kind="mergesort"), kind="mergesort") + 1
    sum_pos = np.sum(ranks[pos])
    auc = (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def binary_classification_report(
    probs,
    target,
    threshold: float = 0.5,
    eps: float = 1e-8,
) -> Dict[str, float]:
    counts = binary_confusion_counts(probs, target, threshold)
    tp = counts["tp"]
    tn = counts["tn"]
    fp = counts["fp"]
    fn = counts["fn"]
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / max(total, eps)
    precision = tp / max(tp + fp, eps)
    recall = tp / max(tp + fn, eps)
    specificity = tn / max(tn + fp, eps)
    f1 = 2.0 * precision * recall / max(precision + recall, eps)
    balanced_acc = (recall + specificity) / 2.0
    auc = binary_auc(probs, target)
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "balanced_accuracy": float(balanced_acc),
        "auc": float(auc),
    }
