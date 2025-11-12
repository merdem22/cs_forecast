"""
Utility script to generate multiple sliding-window .npz datasets.

Examples:
    python scripts/build_windowed_datasets.py --only first4s
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.Hacettepe.cls import dataset_builder

DEFAULT_DATA_DIR = "data/Recordings/SplittedDataWithAssumption-EliminatedPerSession"

DATASETS = {
    # Non-overlapping slices covering the full 8 s trial
    "1s_x8": dict(window_seconds=1.0, stride_seconds=1.0, max_windows_per_trial=8, out="data/npz/cdms_1s_x8.npz"),
    "2s_x4": dict(window_seconds=2.0, stride_seconds=2.0, max_windows_per_trial=4, out="data/npz/cdms_2s_x4.npz"),
    "3s_stride1": dict(window_seconds=3.0, stride_seconds=1.0, max_windows_per_trial=0, out="data/npz/cdms_3s_stride1.npz"),
    "4s_x2": dict(window_seconds=4.0, stride_seconds=4.0, max_windows_per_trial=2, out="data/npz/cdms_4s_x2.npz"),
    "8s_x1": dict(window_seconds=8.0, stride_seconds=8.0, max_windows_per_trial=1, out="data/npz/cdms_8s_x1.npz"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Build multiple sliding-window datasets.")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR, help="Path to participant folders.")
    parser.add_argument(
        "--only",
        nargs="+",
        help=f"Subset of dataset keys to build. Available: {', '.join(DATASETS.keys())}",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    targets = args.only if args.only else DATASETS.keys()
    for key in targets:
        if key not in DATASETS:
            print(f"[builder] Skipping unknown key '{key}'.")
            continue
        spec = DATASETS[key]
        window_seconds = spec["window_seconds"]
        stride_seconds = spec["stride_seconds"]
        max_windows = spec["max_windows_per_trial"]
        # max_windows==0 means unlimited; builder expects None.
        max_windows_arg = None if max_windows in (0, None) else max_windows

        print(f"\n[build_windowed] Building '{key}' -> {spec['out']}")
        eeg, labels, pids, sids, trials, widx = dataset_builder.build_dataset(
            data_dir=args.data_dir,
            window_seconds=window_seconds,
            stride_seconds=stride_seconds,
            max_windows_per_trial=max_windows_arg,
        )

        Path(spec["out"]).parent.mkdir(parents=True, exist_ok=True)
        npz_kwargs = dict(
            eeg_windows=eeg,
            cs_labels=labels,
            participant_ids=pids,
            session_ids=sids,
            trial_ids=trials,
            window_ids=widx,
            window_seconds=float(window_seconds),
            stride_seconds=float(stride_seconds),
            max_windows=(max_windows_arg if max_windows_arg is not None else -1),
        )
        np.savez_compressed(spec["out"], **npz_kwargs)
        print(
            f"[build_windowed] Saved {len(labels)} windows to {spec['out']} "
            f"(window={window_seconds}s stride={stride_seconds}s)"
        )


if __name__ == "__main__":
    main()
