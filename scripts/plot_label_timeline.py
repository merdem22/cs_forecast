import argparse
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, List, Optional

MPL_CACHE = Path(".matplotlib-cache")
if "MPLCONFIGDIR" not in os.environ:
    MPL_CACHE.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(MPL_CACHE.resolve())
os.environ.setdefault("XDG_CACHE_HOME", str(MPL_CACHE.resolve()))

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import csv


FACTOR_NAMES = {"Speed", "Complexity", "Stereo"}


def natural_key(text: str) -> List:
    """Sort helper that keeps embedded numbers in order (Speed_2 < Speed_10)."""
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", text)]


def iter_trial_files(
    data_dir: Path,
    participants: Optional[Iterable[str]] = None,
    sessions: Optional[Iterable[str]] = None,
) -> Iterable[tuple[Path, str, str, int]]:
    participants = {p.upper() for p in participants} if participants else None
    sessions = {s.upper() for s in sessions} if sessions else None

    for participant_dir in sorted(data_dir.iterdir(), key=lambda p: natural_key(p.name)):
        if not participant_dir.is_dir() or not participant_dir.name.upper().startswith("P"):
            continue
        if participants and participant_dir.name.upper() not in participants:
            continue

        for session_dir in sorted(participant_dir.iterdir(), key=lambda p: natural_key(p.name)):
            if not session_dir.is_dir() or not session_dir.name.upper().startswith("S"):
                continue
            if sessions and session_dir.name.upper() not in sessions:
                continue

            mat_files = sorted(session_dir.glob("*.mat"), key=lambda p: natural_key(p.name))
            for order, mat_file in enumerate(mat_files, start=1):
                yield mat_file, participant_dir.name.upper(), session_dir.name.upper(), order


def parse_level_from_name(stem: str) -> Optional[int]:
    match = re.search(r"_(\d+)$", stem)
    return int(match.group(1)) if match else None


def extract_trial_metadata(mat_path: Path) -> dict:
    mat = loadmat(mat_path, squeeze_me=True)

    eeg = mat["EEGData"]
    if eeg.ndim != 2 or eeg.shape[0] < 4:
        raise ValueError(f"Unexpected EEGData shape {eeg.shape} in {mat_path}")

    # Last 4 rows store Speed, Complexity, InterAxialD (stereo), ZPMSSQScore.
    speed_value = float(np.mean(eeg[-4]))
    complexity_value = float(np.mean(eeg[-3]))
    stereo_value = float(np.mean(eeg[-2]))
    zpm_score = float(np.mean(eeg[-1]))

    trial_type = str(mat.get("type", "Unknown"))
    trial_label = int(mat.get("label", 0))

    return {
        "trial_type": trial_type,
        "label": trial_label,
        "speed_value": speed_value,
        "complexity_value": complexity_value,
        "stereo_value": stereo_value,
        "zpm_score": zpm_score,
    }


def collect_trials(
    data_dir: Path,
    participants: Optional[Iterable[str]] = None,
    sessions: Optional[Iterable[str]] = None,
) -> List[dict]:
    rows = []
    for mat_path, participant, session, trial_idx in iter_trial_files(data_dir, participants, sessions):
        trial_name = mat_path.stem
        level = parse_level_from_name(trial_name)
        metadata = extract_trial_metadata(mat_path)
        rows.append(
            {
                "participant": participant,
                "session": session,
                "session_num": int(session.replace("S", "")),
                "trial_idx": trial_idx,
                "trial_name": trial_name,
                "factor_level": level,
                **metadata,
            }
        )

    if not rows:
        raise RuntimeError(f"No .mat files found under {data_dir}")

    rows.sort(key=lambda r: (r["participant"], r["session_num"], r["trial_idx"]))
    return rows


def plot_timeline(rows: List[dict], out_path: Path) -> None:
    agg = defaultdict(lambda: {"label": [], "speed": [], "complexity": [], "stereo": []})
    session_order = {}

    for row in rows:
        key = (row["session"], row["trial_idx"])
        agg[key]["label"].append(row["label"])
        agg[key]["speed"].append(row["speed_value"])
        agg[key]["complexity"].append(row["complexity_value"])
        agg[key]["stereo"].append(row["stereo_value"])
        session_order[row["session"]] = row["session_num"]

    agg_rows = []
    for (session, trial_idx), metrics in agg.items():
        agg_rows.append(
            {
                "session": session,
                "session_num": session_order.get(session, 0),
                "trial_idx": trial_idx,
                "label_mean": float(np.mean(metrics["label"])) if metrics["label"] else np.nan,
                "speed_mean": float(np.mean(metrics["speed"])) if metrics["speed"] else np.nan,
                "complexity_mean": float(np.mean(metrics["complexity"])) if metrics["complexity"] else np.nan,
                "stereo_mean": float(np.mean(metrics["stereo"])) if metrics["stereo"] else np.nan,
            }
        )

    agg_rows.sort(key=lambda r: (r["session_num"], r["trial_idx"]))
    sessions = sorted({row["session"] for row in agg_rows}, key=lambda s: session_order.get(s, 0))

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, len(sessions), figsize=(5 * len(sessions), 8), sharex=True, constrained_layout=True)

    if len(sessions) == 1:
        axes = np.array(axes).reshape(2, 1)

    for col, session in enumerate(sessions):
        session_data = [row for row in agg_rows if row["session"] == session]
        if not session_data:
            continue
        session_data.sort(key=lambda r: r["trial_idx"])
        x = [row["trial_idx"] for row in session_data]
        ax_label = axes[0, col]
        ax_factors = axes[1, col]

        ax_label.plot(x, [row["label_mean"] for row in session_data], marker="o", color="#0072B2")
        ax_label.set_title(f"{session} · mean label vs. trial order")
        ax_label.set_ylabel("Mean cybersickness label")
        ax_label.set_ylim(-0.05, 1.05)

        ax_factors.plot(x, [row["speed_mean"] for row in session_data], label="Speed", color="#E69F00")
        ax_factors.plot(x, [row["complexity_mean"] for row in session_data], label="Complexity", color="#009E73")
        ax_factors.plot(x, [row["stereo_mean"] for row in session_data], label="Stereo (InterAxialD)", color="#D55E00")
        ax_factors.set_xlabel("Trial order within session")
        ax_factors.set_ylabel("Mean normalized factor value")
        ax_factors.set_title(f"{session} · stimulus factors over time")
        ax_factors.legend(loc="upper right")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot cybersickness labels and stimulus factors over session timelines.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/Recordings/SplittedDataWithAssumption-EliminatedPerSession"),
        help="Root directory that holds participant folders.",
    )
    parser.add_argument(
        "--participants",
        nargs="+",
        help="Optional list of participant IDs to include (e.g., P1 P5 P12).",
    )
    parser.add_argument(
        "--sessions",
        nargs="+",
        help="Optional list of session IDs to include (e.g., S1 S2).",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("reports/trial_timeline.csv"),
        help="Where to save the per-trial metadata table.",
    )
    parser.add_argument(
        "--out-plot",
        type=Path,
        default=Path("reports/plots/label_factor_timeline.png"),
        help="Where to save the generated figure.",
    )
    parser.add_argument(
        "--stage-csv",
        type=Path,
        default=Path("reports/mean_label_by_stage.csv"),
        help="Where to save the per-factor-stage mean label table.",
    )
    parser.add_argument(
        "--stage-plot",
        type=Path,
        default=Path("reports/plots/label_by_stage.png"),
        help="Where to save the factor-specific label plot.",
    )
    parser.add_argument(
        "--first-positive-csv",
        type=Path,
        default=Path("reports/first_positive_levels.csv"),
        help="Table that records the first level where each participant/session became sick per factor.",
    )
    parser.add_argument(
        "--first-positive-plot",
        type=Path,
        default=Path("reports/plots/first_positive_distribution.png"),
        help="Plot showing how the first positive label distributes across levels for each factor.",
    )
    return parser.parse_args()


def build_stage_panel(rows: List[dict]) -> List[dict]:
    """
    Builds a dense panel of stage labels where missing levels inherit the previous
    label (carry-forward) as requested.
    """
    per_group = defaultdict(lambda: defaultdict(list))
    max_level = defaultdict(int)

    for row in rows:
        trial_type = row["trial_type"]
        level = row["factor_level"]
        if trial_type not in FACTOR_NAMES or level is None:
            continue
        level = int(level)
        per_group[(row["participant"], row["session"], trial_type)][level].append(int(row["label"]))
        max_level[trial_type] = max(max_level[trial_type], level)

    panel = []
    for (participant, session, trial_type), level_map in per_group.items():
        last_label = None
        seen = False
        for level in range(1, max_level[trial_type] + 1):
            labels = level_map.get(level)
            if labels:
                val = int(round(sum(labels) / len(labels)))
                last_label = val
                seen = True
                panel.append(
                    {
                        "participant": participant,
                        "session": session,
                        "trial_type": trial_type,
                        "factor_level": level,
                        "label": val,
                        "is_imputed": False,
                    }
                )
            elif seen and last_label is not None:
                panel.append(
                    {
                        "participant": participant,
                        "session": session,
                        "trial_type": trial_type,
                        "factor_level": level,
                        "label": last_label,
                        "is_imputed": True,
                    }
                )
            else:
                continue
    return panel


def summarize_by_stage(panel: List[dict]) -> List[dict]:
    buckets = defaultdict(list)
    for row in panel:
        buckets[(row["trial_type"], row["factor_level"])].append(row["label"])

    summary = []
    for (trial_type, level), labels in buckets.items():
        summary.append(
            {
                "trial_type": trial_type,
                "factor_level": level,
                "mean_label": float(np.mean(labels)),
                "count": len(labels),
            }
        )
    summary.sort(key=lambda r: (r["trial_type"], r["factor_level"]))
    return summary


def plot_stage_summary(summary: List[dict], out_path: Path) -> None:
    if not summary:
        raise RuntimeError("Stage summary is empty; nothing to plot.")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True, sharey=True)
    stage_names = ["Speed", "Complexity", "Stereo"]
    colors = {"Speed": "#E69F00", "Complexity": "#009E73", "Stereo": "#D55E00"}

    for ax, stage in zip(axes, stage_names):
        stage_rows = [row for row in summary if row["trial_type"] == stage]
        if not stage_rows:
            ax.set_title(f"{stage} (no data)")
            ax.set_xlabel("Level")
            ax.set_ylabel("Mean cybersickness label")
            continue
        stage_rows.sort(key=lambda r: r["factor_level"])
        ax.plot(
            [row["factor_level"] for row in stage_rows],
            [row["mean_label"] for row in stage_rows],
            marker="o",
            color=colors[stage],
        )
        ax.set_title(f"{stage} stage")
        ax.set_xlabel("Level")
        ax.set_ylabel("Mean cybersickness label")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, which="both", axis="y", alpha=0.3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def summarize_first_positive(rows: List[dict]) -> List[dict]:
    grouped = defaultdict(list)
    for row in rows:
        trial_type = row["trial_type"]
        level = row["factor_level"]
        if trial_type not in FACTOR_NAMES or level is None:
            continue
        key = (row["participant"], row["session"], trial_type)
        grouped[key].append((int(level), int(row["label"])) )

    summary = []
    for (participant, session, trial_type), levels in grouped.items():
        levels.sort(key=lambda x: x[0])
        first_positive = next((lvl for lvl, lbl in levels if lbl == 1), None)
        summary.append(
            {
                "participant": participant,
                "session": session,
                "trial_type": trial_type,
                "first_positive_level": first_positive,
            }
        )
    summary.sort(key=lambda r: (r["trial_type"], r["participant"], r["session"]))
    return summary


def plot_first_positive(summary: List[dict], out_path: Path) -> None:
    if not summary:
        raise RuntimeError("First-positive summary is empty; nothing to plot.")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True, sharey=True)
    stage_names = ["Speed", "Complexity", "Stereo"]
    colors = {"Speed": "#E69F00", "Complexity": "#009E73", "Stereo": "#D55E00"}

    for ax, stage in zip(axes, stage_names):
        stage_rows = [row for row in summary if row["trial_type"] == stage]
        counts = Counter(row["first_positive_level"] for row in stage_rows)
        levels = sorted([lvl for lvl in counts.keys() if lvl is not None])
        heights = [counts[lvl] for lvl in levels]
        ax.bar(levels, heights, color=colors[stage], alpha=0.85)
        ax.set_title(f"{stage} · first sick level (N={sum(heights)})")
        ax.set_xlabel("Level")
        ax.set_ylabel("# participant/session pairs")
        if counts.get(None):
            ax.text(0.02, 0.95, f"Never sick: {counts[None]}", transform=ax.transAxes, ha="left", va="top", fontsize=9)
        ax.set_xticks(levels if levels else [0])
        ax.grid(axis="y", alpha=0.3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main():
    args = parse_args()
    rows = collect_trials(
        data_dir=args.data_dir,
        participants=args.participants,
        sessions=args.sessions,
    )
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "participant",
        "session",
        "session_num",
        "trial_idx",
        "trial_name",
        "factor_level",
        "trial_type",
        "label",
        "speed_value",
        "complexity_value",
        "stereo_value",
        "zpm_score",
    ]
    with args.out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    plot_timeline(rows, args.out_plot)

    stage_panel = build_stage_panel(rows)
    stage_summary = summarize_by_stage(stage_panel)
    args.stage_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.stage_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["trial_type", "factor_level", "mean_label", "count"])
        writer.writeheader()
        writer.writerows(stage_summary)
    plot_stage_summary(stage_summary, args.stage_plot)

    first_positive = summarize_first_positive(rows)
    args.first_positive_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.first_positive_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["participant", "session", "trial_type", "first_positive_level"])
        writer.writeheader()
        writer.writerows(first_positive)
    plot_first_positive(first_positive, args.first_positive_plot)

    print(f"[timeline] Saved per-trial table to {args.out_csv}")
    print(f"[timeline] Saved figure to {args.out_plot}")
    print(f"[stage] Saved per-stage summary to {args.stage_csv}")
    print(f"[stage] Saved per-stage plot to {args.stage_plot}")
    print(f"[stage] Saved first-positive table to {args.first_positive_csv}")
    print(f"[stage] Saved first-positive plot to {args.first_positive_plot}")


if __name__ == "__main__":
    main()
