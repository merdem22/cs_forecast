"""
Plot cross-validation summaries produced by train_cnn_cls.py.

Usage:
    python scripts/plot_cv_summary.py --summary outputs/job_1234/cross_val_summary.json
"""

import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

MPL_CACHE = Path(__file__).resolve().parents[1] / ".matplotlib-cache"
MPL_CACHE.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))
os.environ.setdefault("XDG_CACHE_HOME", str(MPL_CACHE))

BASELINE_CSV = Path("reports/random_baseline.csv")


def parse_args():
    parser = argparse.ArgumentParser(description="Plot LOGO accuracies from summary json files.")
    parser.add_argument("--summary", nargs="+", required=True, help="Paths to cross_val_summary.json files.")
    parser.add_argument("--out", type=Path, default=Path("reports/plots/cv_summary.png"))
    return parser.parse_args()


def load_summary(path: Path):
    with open(path, "r") as f:
        data = json.load(f)
    return data


def load_baselines():
    baseline_map = {}
    if not BASELINE_CSV.exists():
        return baseline_map
    with BASELINE_CSV.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            dataset = row.get("dataset")
            if not dataset:
                continue
            baseline_map[dataset] = float(row.get("mean_accuracy", 0.0) or 0.0)
    return baseline_map


def extract_dataset_label(data: dict):
    cfg = data.get("config", {})
    dataset_npz = cfg.get("dataset", {}).get("npz", "unknown")
    return Path(dataset_npz).name


def plot_single(summary_path: Path, data: dict, out_path: Path, baseline_map: dict):
    fold_acc = data.get("individual_fold_accuracies", [])
    fold_acc = [float(a) for a in fold_acc if a is not None]
    if not fold_acc:
        print(f"[cv_summary] No fold accuracies in {summary_path}")
        return
    mean_acc = data.get("mean_accuracy", np.mean(fold_acc))
    std_acc = data.get("std_accuracy", np.std(fold_acc))
    dataset_label = extract_dataset_label(data)
    baseline_label = Path(dataset_label).name
    baseline = baseline_map.get(baseline_label)

    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots(figsize=(max(6, len(fold_acc) * 1.2), 4))
    x = np.arange(1, len(fold_acc) + 1)
    bars = ax.bar(x, fold_acc, color="#66c2a5", alpha=0.9)
    ax.axhline(mean_acc, color="#d55e00", linestyle="--", linewidth=1.5, label=f"Model mean={mean_acc:.3f}±{std_acc:.3f}")
    if baseline is not None:
        ax.axhline(baseline, color="#0072B2", linestyle=":", linewidth=1.5, label=f"Baseline={baseline:.3f}")
    title_ds = dataset_label if dataset_label != "unknown" else ""
    ax.set_title(f"{summary_path.parent.name} · {title_ds}")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, fold_acc):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{val:.3f}",
                ha="center", va="bottom", fontsize=9)
    ax.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[cv_summary] Saved fold plot to {out_path}")


def plot_multiple(labels, means, stds, datasets, baseline_map, out_path):
    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.2), 4))
    pos = np.arange(len(labels))
    bars = ax.bar(pos, means, yerr=stds, color="#66c2a5", alpha=0.8, capsize=4)
    xticklabels = [f"{label}\n{Path(ds).name}" for label, ds in zip(labels, datasets)]
    ax.set_xticks(pos)
    ax.set_xticklabels(xticklabels, rotation=20, ha="right")
    ax.set_ylabel("Mean Accuracy")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{val:.3f}",
                ha="center", va="bottom", fontsize=9)
    # Optionally annotate baselines
    for idx, (label, ds) in enumerate(zip(labels, datasets)):
        baseline = baseline_map.get(Path(ds).name)
        if baseline is not None:
            ax.axhline(baseline, xmin=idx / len(labels), xmax=(idx + 1) / len(labels),
                       color="#0072B2", linestyle=":", linewidth=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[cv_summary] Saved aggregate plot to {out_path}")


def main():
    args = parse_args()
    baseline_map = load_baselines()
    summaries = [(Path(p), load_summary(Path(p))) for p in args.summary]

    if len(summaries) == 1:
        plot_single(summaries[0][0], summaries[0][1], args.out, baseline_map)
        return

    labels, means, stds, datasets = [], [], [], []
    for path, data in summaries:
        datasets.append(data.get("config", {}).get("dataset", {}).get("npz", "unknown"))
        labels.append(path.parent.name)
        means.append(float(data.get("mean_accuracy", 0.0) or 0.0))
        stds.append(float(data.get("std_accuracy", 0.0) or 0.0))
    plot_multiple(labels, means, stds, datasets, baseline_map, args.out)


if __name__ == "__main__":
    main()
