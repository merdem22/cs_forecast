"""
Visualize subject-balanced folds for a dataset NPZ.

Examples:
    python scripts/plot_balanced_folds.py --npz data/npz/cdms_1s_x8.npz --k 5 --threshold 0.5 --seed 20170629
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def _subject_stats(groups: np.ndarray, labels: np.ndarray, threshold: float) -> list[dict]:
    uniq = np.unique(groups)
    labels_bin = (labels >= threshold).astype(int)
    stats = []
    for pid in uniq:
        mask = groups == pid
        total = int(mask.sum())
        pos = int(labels_bin[mask].sum())
        stats.append({"pid": pid, "total": total, "pos": pos})
    return stats


def _balanced_assign(groups: np.ndarray, labels: np.ndarray, k: int, seed: int, threshold: float):
    stats = _subject_stats(groups, labels, threshold)
    if k <= 0 or k > len(stats):
        raise ValueError(f"Invalid num_folds={k} for {len(stats)} participants.")

    rng = np.random.default_rng(seed)
    rng.shuffle(stats)
    stats.sort(key=lambda r: (r["pos"], r["total"]), reverse=True)

    seeds, remaining = stats[:k], stats[k:]
    folds = [{"subjects": [s["pid"]], "total": s["total"], "pos": s["pos"]} for s in seeds]

    global_total = sum(s["total"] for s in stats)
    global_pos = sum(s["pos"] for s in stats)
    global_ratio = (global_pos / global_total) if global_total else 0.0
    target_total = global_total / k if k else 0.0
    target_pos = global_pos / k if k else 0.0

    for subj in remaining:
        best_idx, best_score = None, float("inf")
        for idx, fold in enumerate(folds):
            total = fold["total"] + subj["total"]
            pos = fold["pos"] + subj["pos"]
            ratio_pen = abs((pos / total) - global_ratio) if total else 0.0
            size_pen = abs(total - target_total) / target_total if target_total else 0.0
            pos_pen = abs(pos - target_pos) / target_pos if target_pos else 0.0
            score = ratio_pen + 0.5 * size_pen + 0.25 * pos_pen
            if score < best_score:
                best_idx, best_score = idx, score
        folds[best_idx]["subjects"].append(subj["pid"])
        folds[best_idx]["total"] += subj["total"]
        folds[best_idx]["pos"] += subj["pos"]

    for fold in folds:
        fold["neg"] = fold["total"] - fold["pos"]
        fold["ratio"] = (fold["pos"] / fold["total"]) if fold["total"] else 0.0
    return folds, global_ratio


def plot_folds(folds: list[dict], global_ratio: float, out_path: Path):
    ks = [f"Fold {i+1}" for i in range(len(folds))]
    ratios = [f["ratio"] for f in folds]
    pos = [f["pos"] for f in folds]
    neg = [f["neg"] for f in folds]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    ax.bar(ks, ratios, color="#4c72b0", alpha=0.8)
    ax.axhline(global_ratio, color="red", linestyle="--", label=f"Global ratio={global_ratio:.3f}")
    ax.set_ylabel("Positive ratio")
    ax.set_ylim(0, 1)
    ax.set_title("Per-fold class ratio")
    ax.legend()
    ax.tick_params(axis="x", rotation=30)

    ax2 = axes[1]
    ax2.bar(ks, neg, label="Negatives", color="#b0c4de")
    ax2.bar(ks, pos, bottom=neg, label="Positives", color="#dd8452")
    ax2.set_ylabel("Sample count")
    ax2.set_title("Per-fold sample counts")
    ax2.tick_params(axis="x", rotation=30)
    ax2.legend()

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot balanced fold distribution for a dataset.")
    parser.add_argument("--npz", required=True, help="Path to dataset npz.")
    parser.add_argument("--k", type=int, default=5, help="Number of folds.")
    parser.add_argument("--seed", type=int, default=20170629, help="RNG seed.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Positive threshold.")
    parser.add_argument("--out", type=Path, default=Path("reports/plots/balanced_folds.png"))
    args = parser.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    groups = data["participant_ids"].astype(str)
    labels = data["cs_labels"].astype(float)

    folds, global_ratio = _balanced_assign(groups, labels, k=args.k, seed=args.seed, threshold=args.threshold)
    for i, f in enumerate(folds, start=1):
        print(f"Fold {i}: subjects={f['subjects']} total={f['total']} pos={f['pos']} ratio={f['ratio']:.3f}")

    plot_folds(folds, global_ratio, args.out)
    print(f"Saved plot to {args.out}")


if __name__ == "__main__":
    main()
