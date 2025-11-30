"""
Visualize model prediction behavior: does it over/under-predict the positive class?

For each dataset, we:
  - load the latest cross_val_summary.json under outputs/job_*/
  - grab aggregated recall/specificity/precision/F1, etc.
  - estimate the predicted positive rate using recall/specificity and the dataset's true positive rate
    (p_pred = recall * p_true + (1 - specificity) * (1 - p_true))
  - plot actual vs predicted positive rates, plus precision/recall/F1 bars.

Usage:
    MPLCONFIGDIR=.matplotlib-cache python scripts/plot_prediction_behavior.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List

import matplotlib

ROOT = Path(__file__).resolve().parents[1]
MPL_CACHE = ROOT / ".matplotlib-cache"
MPL_CACHE.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


DATASET_LABELS: Dict[str, str] = {
    # Detection
    "cdms_1s_x8": "1s × 8",
    "cdms_2s_x4": "2s × 4",
    "cdms_3s_stride1": "3s × 6 (stride 1s)",
    "cdms_4s_x2": "4s × 2",
    "cdms_8s_x1": "8s × 1",
    # Forecast
    "cdms_fc_2s_first4": "2s × 2 (first 4s)",
    "cdms_fc_2s_stride1x3": "2s × 3 (stride 1s)",
    "cdms_fc_3s_stride1x2": "3s × 2 (stride 1s)",
}


def _latest_summaries(outputs_root: Path) -> Dict[str, dict]:
    summaries: Dict[str, dict] = {}
    for path in outputs_root.glob("job_*/cross_val_summary.json"):
        with path.open("r") as f:
            data = json.load(f)
        dataset_npz = data.get("config", {}).get("dataset", {}).get("npz")
        if not dataset_npz:
            continue
        stem = Path(dataset_npz).stem
        if stem not in DATASET_LABELS:
            continue
        mtime = path.stat().st_mtime
        if stem not in summaries or mtime > summaries[stem]["mtime"]:
            summaries[stem] = {"summary": data, "mtime": mtime, "path": path}
    return summaries


def _pos_rate(npz_path: Path) -> float:
    import numpy as np

    data = np.load(npz_path, allow_pickle=True)
    labels = data["cs_labels"]
    return float(labels.mean())


def main():
    outputs_root = ROOT / "outputs"
    summaries = _latest_summaries(outputs_root)
    if not summaries:
        raise SystemExit("No cross_val_summary.json files found for known datasets.")

    records: List[dict] = []
    for stem, entry in summaries.items():
        summary = entry["summary"]
        dataset_rel = summary["config"]["dataset"]["npz"]
        npz_path = (ROOT / dataset_rel).resolve()
        p_true = _pos_rate(npz_path)
        agg = summary.get("aggregate_metrics", {})
        recall = agg.get("recall", {}).get("mean")
        specificity = agg.get("specificity", {}).get("mean")
        precision = agg.get("precision", {}).get("mean")
        f1 = agg.get("f1", {}).get("mean")
        auc = agg.get("auc", {}).get("mean")
        balanced_acc = agg.get("balanced_accuracy", {}).get("mean")
        acc = agg.get("accuracy", {}).get("mean", summary.get("mean_accuracy"))

        if recall is None or specificity is None:
            # Skip datasets without detailed metrics
            continue

        p_pred = recall * p_true + (1.0 - specificity) * (1.0 - p_true)

        records.append(
            dict(
                stem=stem,
                label=DATASET_LABELS.get(stem, stem),
                p_true=p_true,
                p_pred=p_pred,
                recall=recall,
                specificity=specificity,
                precision=precision,
                f1=f1,
                auc=auc,
                balanced_acc=balanced_acc,
                acc=acc,
            )
        )

    if not records:
        raise SystemExit("No datasets with detailed metrics available; rerun training with updated code.")

    # Sort for consistent display (detection first, then forecast)
    order = [
        k
        for k in [
            "cdms_1s_x8",
            "cdms_2s_x4",
            "cdms_3s_stride1",
            "cdms_4s_x2",
            "cdms_8s_x1",
            "cdms_fc_2s_first4",
            "cdms_fc_2s_stride1x3",
            "cdms_fc_3s_stride1x2",
        ]
        if any(rec["stem"] == k for rec in records)
    ]
    records = [next(rec for rec in records if rec["stem"] == stem) for stem in order]

    x = np.arange(len(records))
    labels = [rec["label"] for rec in records]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Panel 1: predicted vs actual positive rate
    axes[0].bar(x - 0.15, [rec["p_true"] for rec in records], 0.3, label="True pos rate", color="#90a4ae")
    axes[0].bar(x + 0.15, [rec["p_pred"] for rec in records], 0.3, label="Predicted pos rate (est.)", color="#00796b")
    axes[0].set_ylabel("Rate")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Are models over/under-predicting the positive class?")
    axes[0].legend()
    for idx, rec in enumerate(records):
        axes[0].text(x=idx - 0.15, y=rec["p_true"] + 0.02, s=f"{rec['p_true']:.2f}", ha="center", va="bottom", fontsize=9)
        axes[0].text(x=idx + 0.15, y=rec["p_pred"] + 0.02, s=f"{rec['p_pred']:.2f}", ha="center", va="bottom", fontsize=9, color="#004d40")

    # Panel 2: precision/recall/F1 (bar cluster)
    width = 0.2
    axes[1].bar(x - width, [rec["precision"] for rec in records], width, label="Precision", color="#8e24aa")
    axes[1].bar(x, [rec["recall"] for rec in records], width, label="Recall", color="#3949ab")
    axes[1].bar(x + width, [rec["f1"] for rec in records], width, label="F1", color="#1e88e5")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0, 1)
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].legend()
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=0, ha="center")

    fig.tight_layout()
    out_dir = ROOT / "reports" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "prediction_behavior.png"
    fig.savefig(out_path, dpi=300)
    print(f"[plot_prediction_behavior] Saved figure to {out_path}")

    for rec in records:
        print(
            f"{rec['label']}: p_true={rec['p_true']:.3f}, p_pred_est={rec['p_pred']:.3f}, "
            f"precision={rec['precision']:.3f}, recall={rec['recall']:.3f}, f1={rec['f1']:.3f}"
        )


if __name__ == "__main__":
    main()
