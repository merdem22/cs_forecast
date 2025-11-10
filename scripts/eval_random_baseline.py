"""
Evaluate a random-guessing baseline using Leave-One-Participant-Out splits.

Usage:
    python scripts/eval_random_baseline.py --npz data/npz/cdms_first4s.npz data/npz/cdms_2s_x4.npz
"""

import argparse
import os
import sys
from pathlib import Path

import csv
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MPL_CACHE = ROOT / ".matplotlib-cache"
MPL_CACHE.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))
os.environ.setdefault("XDG_CACHE_HOME", str(MPL_CACHE))


def logo_split(groups: np.ndarray):
    unique = np.unique(groups)
    for gid in unique:
        mask = groups == gid
        yield [gid], np.where(~mask)[0], np.where(mask)[0]


def participant_kfold(groups: np.ndarray, k: int, seed: int):
    unique = np.unique(groups)
    if k <= 1 or k > len(unique):
        raise ValueError(f"Invalid num_folds={k} for {len(unique)} participants.")
    rng = np.random.default_rng(seed)
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    chunks = np.array_split(shuffled, k)
    for chunk in chunks:
        mask = np.isin(groups, chunk)
        yield chunk.tolist(), np.where(~mask)[0], np.where(mask)[0]


def random_baseline(labels: np.ndarray, groups: np.ndarray, repeats: int, seed: int, fold_mode: str, num_folds: int):
    rng = np.random.default_rng(seed)
    pos_prob = labels.mean()
    fold_metrics = []
    if fold_mode == "logo":
        iterator = logo_split(groups)
    else:
        iterator = participant_kfold(groups, num_folds, seed)
    for gids, _, test_idx in iterator:
        y_true = labels[test_idx]
        accs = []
        for r in range(repeats):
            preds = (rng.random(len(y_true)) < pos_prob).astype(int)
            accs.append((preds == y_true).mean())
        fold_metrics.append((gids, np.mean(accs), np.std(accs)))
    return fold_metrics


def evaluate_dataset(npz_path: Path, repeats: int, seed: int, fold_mode: str, num_folds: int):
    data = np.load(npz_path, allow_pickle=True)
    labels = data["cs_labels"].astype(int)
    groups = data["participant_ids"]
    fold_stats = random_baseline(labels, groups, repeats=repeats, seed=seed, fold_mode=fold_mode, num_folds=num_folds)
    fold_acc = [m[1] for m in fold_stats]
    return {
        "dataset": npz_path.name,
        "path": str(npz_path),
        "mean_accuracy": float(np.mean(fold_acc)),
        "std_accuracy": float(np.std(fold_acc)),
        "fold_count": len(fold_stats),
    }, fold_stats


def parse_args():
    parser = argparse.ArgumentParser(description="Random guessing baseline for multiple datasets.")
    parser.add_argument("--npz", nargs="+", required=True, help="List of dataset npz files.")
    parser.add_argument("--repeats", type=int, default=100, help="Random restarts per fold.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed.")
    parser.add_argument("--fold-mode", choices=["logo", "kfold"], default="kfold", help="Match training folds.")
    parser.add_argument("--num-folds", type=int, default=5, help="Number of participant folds when using kfold.")
    parser.add_argument("--out-csv", type=Path, default=Path("reports/random_baseline.csv"))
    parser.add_argument("--out-plot", type=Path, default=Path("reports/plots/random_baseline.png"))
    parser.add_argument("--no-plot", action="store_true", help="Skip plot generation.")
    return parser.parse_args()


def main():
    args = parse_args()
    summaries = []
    for npz_path in args.npz:
        npz_path = Path(npz_path)
        summary, fold_stats = evaluate_dataset(
            npz_path,
            repeats=args.repeats,
            seed=args.seed,
            fold_mode=args.fold_mode,
            num_folds=args.num_folds,
        )
        summaries.append(summary)
        print(f"[random] {npz_path.name}: acc={summary['mean_accuracy']:.3f} ± {summary['std_accuracy']:.3f}")

    fieldnames = ["dataset", "path", "mean_accuracy", "std_accuracy", "fold_count"]
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)

    if not args.no_plot:
        plt.style.use("seaborn-v0_8")
        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(len(summaries))
        heights = [s["mean_accuracy"] for s in summaries]
        yerr = [s["std_accuracy"] for s in summaries]
        labels = [s["dataset"] for s in summaries]
        ax.bar(x, heights, yerr=yerr, color="#9999ff", alpha=0.8)
        ax.set_ylabel("Random baseline accuracy")
        ax.set_ylim(0, 1)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.3)
        args.out_plot.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(args.out_plot, dpi=300)
        plt.close(fig)
        print(f"[random] Saved plot to {args.out_plot}")

    print(f"[random] Saved summary to {args.out_csv}")


if __name__ == "__main__":
    main()
