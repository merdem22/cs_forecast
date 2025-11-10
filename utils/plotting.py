# utils/plotting.py
import os
import numpy as np
import matplotlib.pyplot as plt

__all__ = ["plot_subject_timeline"]

def _stitch_timeline(arr: np.ndarray, strategy: str, W_in: int) -> np.ndarray:
    """
    Map (N,L) horizons into one 1D series by averaging overlaps.
      multi      : sample i covers t = i .. i+L-1
      multi_next : sample i covers t = i+W_in .. i+W_in+L-1
    """
    N, L = arr.shape
    offset = 0 if strategy == "multi" else int(W_in)
    T = N + offset + L - 1
    acc = np.zeros(T, dtype=np.float64)
    cnt = np.zeros(T, dtype=np.float64)
    for i in range(N):
        base = i + offset
        acc[base:base+L] += arr[i]
        cnt[base:base+L] += 1.0
    mask = cnt > 0
    return (acc[mask] / np.maximum(cnt[mask], 1.0)).astype(np.float32)

def plot_subject_timeline(
    *,
    test_pid: int,
    Y_true: np.ndarray,   # (N, L)
    Y_pred: np.ndarray,   # (N, L)
    ybar: np.ndarray,     # (L,) train-mean baseline per horizon
    strategy: str,        # "multi" | "multi_next"
    W_in: int,
    out_dir: str,
    title_suffix: str = "",
    mae_model=None,                 
    mae_mean=None,
    ) -> str:
    """Save one PNG overlaying GT, Model, Mean-baseline for the test subject."""
    os.makedirs(out_dir, exist_ok=True)
    fold_dir = os.path.join(out_dir, f"pid_{int(test_pid)}")
    os.makedirs(fold_dir, exist_ok=True)

    # stitch GT and model
    gt_t = _stitch_timeline(Y_true, strategy, W_in)
    pr_t = _stitch_timeline(Y_pred, strategy, W_in)

    # stitch baseline by broadcasting ybar to (N,L)
    N, L = Y_true.shape
    base_mat = np.broadcast_to(ybar.reshape(1, -1), (N, L))
    mean_t = _stitch_timeline(base_mat, strategy, W_in)

    # align lengths just in case
    T = min(len(gt_t), len(pr_t), len(mean_t))
    gt_t, pr_t, mean_t = gt_t[:T], pr_t[:T], mean_t[:T]

    title_bits = []
    if mae_model is not None: title_bits.append(f"Model MAE={mae_model:.3f}")
    if mae_mean  is not None: title_bits.append(f"Mean MAE={mae_mean:.3f}")
    subtitle = ("  ·  ".join(title_bits)) if title_bits else ""

    x = np.arange(T)
    plt.figure(figsize=(14, 3.6))
    plt.plot(x, gt_t,   label="GT", linewidth=1.5)
    plt.plot(x, pr_t,   label=f"Model (MAE={mae_model:.3f})", alpha=0.95)
    plt.plot(x, mean_t, label=f"Mean baseline (MAE={mae_mean:.3f})", alpha=0.9, linestyle="--")
    plt.title(f"PID {int(test_pid)} · stitched timeline {title_suffix} {subtitle}")
    plt.xlabel("Relative time index")
    plt.ylabel("CS")
    plt.legend(loc="best")
    plt.tight_layout()

    path = os.path.join(fold_dir, f"pid{int(test_pid)}_overlay.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path
