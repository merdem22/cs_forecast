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


def _participant_kfold(groups: np.ndarray, k: int, seed: int):
    unique_groups = np.unique(groups)
    if k <= 1 or k > len(unique_groups):
        raise ValueError(f"Invalid num_folds={k} for {len(unique_groups)} participants.")
    rng = np.random.default_rng(seed)
    shuffled = unique_groups.copy()
    rng.shuffle(shuffled)
    chunks = np.array_split(shuffled, k)
    for chunk in chunks:
        mask = np.isin(groups, chunk)
        yield chunk.tolist(), np.where(~mask)[0], np.where(mask)[0]


def _balanced_assign(groups: np.ndarray, labels: np.ndarray, k: int, seed: int, threshold: float, min_subjects: int):
    stats = _subject_stats(groups, labels, threshold)
    if k <= 0 or k > len(stats):
        raise ValueError(f"Invalid num_folds={k} for {len(stats)} participants.")

    rng = np.random.default_rng(seed)

    def _build(k_use: int):
        rng.shuffle(stats)
        ordered = sorted(stats, key=lambda r: (r["pos"], r["total"]), reverse=True)
        seeds, remaining = ordered[:k_use], ordered[k_use:]
        folds_local = [{"subjects": [s["pid"]], "total": s["total"], "pos": s["pos"]} for s in seeds]

        global_total = sum(s["total"] for s in stats)
        global_pos = sum(s["pos"] for s in stats)
        global_ratio = (global_pos / global_total) if global_total else 0.0
        target_total = global_total / k_use if k_use else 0.0
        target_pos = global_pos / k_use if k_use else 0.0

        for subj in remaining:
            best_idx, best_score = None, float("inf")
            for idx, fold in enumerate(folds_local):
                total = fold["total"] + subj["total"]
                pos = fold["pos"] + subj["pos"]
                ratio_pen = abs((pos / total) - global_ratio) if total else 0.0
                size_pen = abs(total - target_total) / target_total if target_total else 0.0
                pos_pen = abs(pos - target_pos) / target_pos if target_pos else 0.0
                score = ratio_pen + 0.5 * size_pen + 0.25 * pos_pen
                if score < best_score:
                    best_idx, best_score = idx, score
            folds_local[best_idx]["subjects"].append(subj["pid"])
            folds_local[best_idx]["total"] += subj["total"]
            folds_local[best_idx]["pos"] += subj["pos"]

        for f in folds_local:
            f["neg"] = f["total"] - f["pos"]
            f["ratio"] = (f["pos"] / f["total"]) if f["total"] else 0.0

        max_ratio_dev = max(abs(f["ratio"] - global_ratio) for f in folds_local if f["total"] > 0)
        size_span = max(f["total"] for f in folds_local) - min(f["total"] for f in folds_local)
        return folds_local, global_ratio, max_ratio_dev, size_span

    k_min = 3
    k_max = min(k, len(stats))
    best = None
    for k_use in range(k_min, k_max + 1):
        folds_local, global_ratio, ratio_dev, size_span = _build(k_use)
        cand = dict(k=k_use, folds=folds_local, global_ratio=global_ratio, ratio_dev=ratio_dev, size_span=size_span)
        if best is None or ratio_dev < best["ratio_dev"] or (ratio_dev == best["ratio_dev"] and size_span < best["size_span"]):
            best = cand

    if best is None:
        raise RuntimeError("Balanced fold construction failed.")

    return best["folds"], best["global_ratio"]


def plot_folds(folds: list[dict], groups: np.ndarray, labels: np.ndarray, global_ratio: float, out_path: Path):
    ks = [f"Fold {i+1}" for i in range(len(folds))]
    # Test stats
    test_ratios = [f["ratio"] for f in folds]
    test_pos = [f["pos"] for f in folds]
    test_neg = [f["neg"] for f in folds]
    # Train stats (complement)
    train_ratios, train_pos, train_neg = [], [], []
    for f in folds:
        mask = np.isin(groups, f["subjects"])
        train_mask = ~mask
        y_train = labels[train_mask]
        pos = float(np.sum(y_train))
        total = float(y_train.size)
        train_pos.append(pos)
        train_neg.append(total - pos)
        train_ratios.append((pos / total) if total else 0.0)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))

    ax = axes[0, 0]
    ax.bar(ks, test_ratios, color="#4c72b0", alpha=0.8, label="Test")
    ax.axhline(global_ratio, color="red", linestyle="--", label=f"Global={global_ratio:.3f}")
    ax.set_ylabel("Positive ratio")
    ax.set_ylim(0, 1)
    ax.set_title("Test fold class ratio")
    ax.legend()
    ax.tick_params(axis="x", rotation=30)

    axb = axes[0, 1]
    axb.bar(ks, train_ratios, color="#55a868", alpha=0.8, label="Train")
    axb.axhline(global_ratio, color="red", linestyle="--", label=f"Global={global_ratio:.3f}")
    axb.set_ylabel("Positive ratio")
    axb.set_ylim(0, 1)
    axb.set_title("Train (complement) class ratio")
    axb.legend()
    axb.tick_params(axis="x", rotation=30)

    ax2 = axes[1, 0]
    ax2.bar(ks, test_neg, label="Negatives", color="#b0c4de")
    ax2.bar(ks, test_pos, bottom=test_neg, label="Positives", color="#dd8452")
    ax2.set_ylabel("Sample count")
    ax2.set_title("Test fold sample counts")
    ax2.tick_params(axis="x", rotation=30)
    ax2.legend()

    ax3 = axes[1, 1]
    ax3.bar(ks, train_neg, label="Negatives", color="#c3e0c3")
    ax3.bar(ks, train_pos, bottom=train_neg, label="Positives", color="#88c999")
    ax3.set_ylabel("Sample count")
    ax3.set_title("Train (complement) sample counts")
    ax3.tick_params(axis="x", rotation=30)
    ax3.legend()

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot fold distribution for a dataset.")
    parser.add_argument("--npz", required=True, help="Path to dataset npz.")
    parser.add_argument("--k", type=int, default=5, help="Number of folds.")
    parser.add_argument("--seed", type=int, default=20170629, help="RNG seed.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Positive threshold.")
    parser.add_argument("--out", type=Path, default=Path("reports/plots/balanced_folds.png"))
    parser.add_argument("--min-subjects", type=int, default=0, help="Unused placeholder for compatibility.")
    parser.add_argument("--mode", choices=["balanced", "kfold"], default="balanced", help="Fold construction mode.")
    args = parser.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    groups = data["participant_ids"].astype(str)
    labels = data["cs_labels"].astype(float)

    if args.mode == "balanced":
        folds, global_ratio = _balanced_assign(
            groups,
            labels,
            k=args.k,
            seed=args.seed,
            threshold=args.threshold,
            min_subjects=args.min_subjects,
        )
    else:
        global_ratio = (labels >= args.threshold).mean()
        folds = []
        for subj_list, train_idx, test_idx in _participant_kfold(groups, args.k, args.seed):
            y_test = labels[test_idx]
            pos = float(np.sum(y_test >= args.threshold))
            total = float(y_test.size)
            folds.append(
                {
                    "subjects": subj_list,
                    "total": total,
                    "pos": pos,
                    "neg": total - pos,
                    "ratio": (pos / total) if total else 0.0,
                }
            )
    for i, f in enumerate(folds, start=1):
        print(f"Fold {i}: subjects={f['subjects']} total={f['total']} pos={f['pos']} ratio={f['ratio']:.3f}")

    plot_folds(folds, groups, labels, global_ratio, args.out)
    print(f"Saved plot to {args.out}")


if __name__ == "__main__":
    main()
