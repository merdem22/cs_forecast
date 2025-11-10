"""
Generate cross-validation plots for every outputs/job_*/cross_val_summary.json.

Usage:
    python scripts/plot_all_jobs.py
"""

import subprocess
import sys
from pathlib import Path


def main():
    project_root = Path(__file__).resolve().parents[1]
    outputs_dir = project_root / "outputs"
    plot_script = project_root / "scripts" / "plot_cv_summary.py"

    summaries = sorted(outputs_dir.glob("job_*/cross_val_summary.json"))
    if not summaries:
        print("[plot_all] No job_*/cross_val_summary.json files found under outputs/.")
        return

    for summary in summaries:
        job_name = summary.parent.name
        out_path = project_root / "reports" / "plots" / f"cv_summary_{job_name}.png"
        cmd = [sys.executable, str(plot_script), "--summary", str(summary), "--out", str(out_path)]
        print(f"[plot_all] Generating plot for {job_name}...")
        subprocess.run(cmd, check=True)

    print("[plot_all] Done.")


if __name__ == "__main__":
    main()
