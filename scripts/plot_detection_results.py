"""
Create comparison plots for detection experiments vs random baseline.

Usage:
    MPLCONFIGDIR=.matplotlib-cache python scripts/plot_detection_results.py
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

ROOT = Path(__file__).resolve().parents[1]
MPL_CACHE = ROOT / ".matplotlib-cache"
MPL_CACHE.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


DETECTION_DATASETS: Dict[str, Dict[str, str | float | int]] = {
    "cdms_1s_x8": dict(label="1s × 8", windows=8, notes="non-overlap"),
    "cdms_2s_x4": dict(label="2s × 4", windows=4, notes="non-overlap"),
    "cdms_3s_stride1": dict(label="3s × 6", windows=6, notes="stride 1s"),
    "cdms_4s_x2": dict(label="4s × 2", windows=2, notes="non-overlap"),
    "cdms_8s_x1": dict(label="8s × 1", windows=1, notes="full trial"),
}


def _gather_latest_summaries(outputs_root: Path) -> Dict[str, dict]:
    """Return latest cross_val_summary per dataset (keyed by dataset stem)."""
    summaries: Dict[str, dict] = {}
    for summary_path in outputs_root.glob("job_*/cross_val_summary.json"):
        with summary_path.open("r") as f:
            summary = json.load(f)
        cfg = summary.get("config", {})
        dataset_npz = cfg.get("dataset", {}).get("npz")
        if not dataset_npz:
            continue
        dataset_stem = Path(dataset_npz).stem
        if dataset_stem not in DETECTION_DATASETS:
            continue
        mtime = summary_path.stat().st_mtime
        current = summaries.get(dataset_stem)
        if current is None or mtime > current["mtime"]:
            summaries[dataset_stem] = dict(summary=summary, path=summary_path, mtime=mtime)
    return summaries


def _load_dataset_stats(dataset_path: Path) -> dict:
    data = np.load(dataset_path, allow_pickle=True)
    labels = data["cs_labels"].astype(np.int32)
    pos_rate = labels.mean()
    # Random baseline (analytic fallback) when simulation CSV missing.
    baseline = pos_rate ** 2 + (1.0 - pos_rate) ** 2
    return {
        "num_samples": int(labels.shape[0]),
        "positive_rate": float(pos_rate),
        "random_baseline": float(baseline),
    }


def _load_random_baseline_csv(csv_path: Path) -> Dict[str, dict]:
    if not csv_path.exists():
        return {}
    table: Dict[str, dict] = {}
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dataset_name = row.get("dataset")
            if not dataset_name:
                continue
            stem = Path(dataset_name).stem
            try:
                mean_acc = float(row["mean_accuracy"])
                std_acc = float(row["std_accuracy"])
            except (KeyError, TypeError, ValueError):
                continue
            table[stem] = dict(mean=mean_acc, std=std_acc, source=str(csv_path))
    return table


def main():
    outputs_root = ROOT / "outputs"
    summaries = _gather_latest_summaries(outputs_root)
    if not summaries:
        raise SystemExit("No detection summaries found under outputs/job_*/cross_val_summary.json")

    order = [k for k in DETECTION_DATASETS if k in summaries]
    if not order:
        raise SystemExit("Detection dataset summaries missing. Run detection experiments first.")

    baseline_table = _load_random_baseline_csv(ROOT / "reports" / "random_baseline_detection.csv")

    records: List[dict] = []
    for key in order:
        summary_entry = summaries[key]["summary"]
        dataset_rel = summary_entry["config"]["dataset"]["npz"]
        dataset_path = (ROOT / dataset_rel).resolve()
        stats = _load_dataset_stats(dataset_path)
        baseline_entry: Optional[dict] = baseline_table.get(Path(dataset_rel).stem)
        baseline_mean = (
            baseline_entry["mean"] if baseline_entry is not None else stats["random_baseline"]
        )
        baseline_std = baseline_entry["std"] if baseline_entry is not None else 0.0
        records.append(
            dict(
                key=key,
                label=DETECTION_DATASETS[key]["label"],
                notes=DETECTION_DATASETS[key]["notes"],
                windows=DETECTION_DATASETS[key]["windows"],
                dataset=dataset_rel,
                num_samples=stats["num_samples"],
                baseline=baseline_mean,
                baseline_std=baseline_std,
                positive_rate=stats["positive_rate"],
                mean_acc=float(summary_entry["mean_accuracy"]),
                std_acc=float(summary_entry["std_accuracy"]),
                baseline_source=(
                    baseline_entry["source"] if baseline_entry is not None else "analytic"
                ),
            )
        )

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(records))
    width = 0.35
    baseline_vals = [rec["baseline"] for rec in records]
    baseline_err = [rec["baseline_std"] for rec in records]
    model_vals = [rec["mean_acc"] for rec in records]
    model_err = [rec["std_acc"] for rec in records]

    ax.bar(
        x - width / 2,
        baseline_vals,
        width,
        yerr=baseline_err,
        color="#b0bec5",
        label="Random baseline (simulated)",
        capsize=6,
    )
    ax.bar(
        x + width / 2,
        model_vals,
        width,
        yerr=model_err,
        color="#1976d2",
        label="CNN (5-fold mean ± std)",
        capsize=6,
    )

    tick_labels = [f"{rec['label']}\n{rec['notes']}" for rec in records]
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.45, 0.7)
    ax.set_title("Detection accuracy vs random baseline (different window setups)")
    ax.legend()

    for idx, rec in enumerate(records):
        ax.text(
            idx + width / 2,
            model_vals[idx] + 0.01,
            f"{model_vals[idx]*100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#0d47a1",
        )
        ax.text(
            idx - width / 2,
            baseline_vals[idx] + 0.01,
            f"{baseline_vals[idx]*100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#37474f",
        )

    fig.tight_layout()
    out_dir = ROOT / "reports" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "detection_vs_baseline.png"
    fig.savefig(out_path, dpi=300)
    print(f"[plot_detection_results] Saved figure to {out_path}")
    for rec in records:
        print(
            f"{rec['label']}: acc={rec['mean_acc']:.3f}±{rec['std_acc']:.3f}, "
            f"baseline={rec['baseline']:.3f}, N={rec['num_samples']}, "
            f"pos_rate={rec['positive_rate']:.3f}, dataset={rec['dataset']}, "
            f"baseline_source={rec['baseline_source']}"
        )


if __name__ == "__main__":
    main()
