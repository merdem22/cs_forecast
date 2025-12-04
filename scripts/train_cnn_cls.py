"""
train_cnn_cls.py

Main training script for the CDMS prediction task using the supervisor's
repository structure and CNN model.

Reads configuration from a YAML file, loads the specific CDMS prediction dataset,
performs Leave-One-Group-Out (LOGO) cross-validation using PyTorch Lightning,
and logs results using Weights & Biases.

**Adapted from train_cnn_cls.py for the CDMS dataset and prediction task.**
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
import numpy as np
import pytorch_lightning as pl
import torch
import yaml

from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# --- Use YOUR specific dataset loader ---
# Make sure the path 'data.Hacettepe.cls.loader' is correct relative to your project root
from data.Hacettepe.cls.loader import EEGPredictionDataset

# --- Assume these utils exist in the supervisor's repo ---
# Make sure these paths ('utils. ...') are correct relative to your project root
from utils.seed import set_seed
from utils.metrics import binary_classification_report
from utils.logging import log_config #, log_config_file # log_config_file might need adjustment

# --- Use the supervisor's training module and CNN model ---
# Make sure these paths ('tasks. ...', 'models. ...') are correct
from tasks.cs_module import CSDownstreamModule
from models.downstream.cnn import CNN
from models.downstream.shallow import ShallowStage1


DEFAULT_CONFIG = "configs/config.yaml"


# --- Helper functions (adapted from supervisor's script) ---
def _resolve_config_path(default_path: str) -> str:
    """Finds the config file, prioritizing env vars."""
    env_path = os.getenv("CONFIG_PATH")
    if env_path and os.path.exists(env_path):
        print(f"[config] Using CONFIG_PATH: {env_path}")
        return env_path
    job_root = os.getenv("JOB_ROOT")
    if job_root:
        job_cfg = os.path.join(job_root, "config.yaml")
        if os.path.exists(job_cfg):
            print(f"[config] Using JOB_ROOT config: {job_cfg}")
            return job_cfg
    # If called directly without JOB_ROOT (e.g., local debug), use default
    if os.path.exists(default_path):
        print(f"[config] Using default path: {default_path}")
        return default_path
    # Try finding it relative to the script location if run directly
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Assumes script is in scripts/ and config is in ../configs/
    rel_path = os.path.join(script_dir, '..', default_path)
    if os.path.exists(rel_path):
        print(f"[config] Using relative path: {rel_path}")
        return rel_path
    raise FileNotFoundError(f"Config file not found via CONFIG_PATH, JOB_ROOT, default ({default_path}), or relative path ({rel_path}).")


def _snapshot_config(cfg_path: str) -> str:
    """Copies the config file to the job output directory."""
    job_root = os.getenv("JOB_ROOT")
    if not job_root:
        # Create a local output dir if not running under SLURM/JOB_ROOT
        job_root = os.path.join("outputs", "local_run", os.path.basename(cfg_path).replace('.yaml',''))
        print(f"[snapshot] JOB_ROOT not set, using local path: {job_root}")
    os.makedirs(job_root, exist_ok=True)
    dst_path = os.path.join(job_root, "config.yaml")
    if os.path.abspath(cfg_path) != os.path.abspath(dst_path):
        try:
            shutil.copy2(cfg_path, dst_path)
            print(f"[snapshot] Copied config to {dst_path}")
        except Exception as e:
            print(f"[snapshot] Warning: Failed to copy config: {e}")
    else:
        print(f"[snapshot] Config already in JOB_ROOT: {dst_path}")
    return dst_path

# --- Fold helpers ---
def _logo_split(groups: np.ndarray):
    """Simple Leave-One-Group-Out split; returns test participant ids as a list."""
    unique_groups = np.unique(groups)
    for gid in unique_groups:
        mask = groups == gid
        yield [gid], np.where(~mask)[0], np.where(mask)[0]


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


def _subject_stats(groups: np.ndarray, labels: np.ndarray, pos_threshold: float) -> list[dict]:
    """Per-participant sample counts and positives."""
    uniq = np.unique(groups)
    labels_bin = (labels >= pos_threshold).astype(int)
    stats = []
    for pid in uniq:
        mask = groups == pid
        total = int(mask.sum())
        pos = int(labels_bin[mask].sum())
        stats.append({"pid": pid, "total": total, "pos": pos})
    return stats


def _balanced_kfold(groups: np.ndarray, labels: np.ndarray, k: int, seed: int, pos_threshold: float):
    """
    Greedy subject assignment to approximate balanced folds by class ratio and size.
    Returns generator of (subjects, train_idx, test_idx).
    """
    stats = _subject_stats(groups, labels, pos_threshold)
    if k <= 0 or k > len(stats):
        raise ValueError(f"Invalid num_folds={k} for {len(stats)} participants.")

    rng = np.random.default_rng(seed)
    rng.shuffle(stats)  # shuffle first to break ties deterministically per seed
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
        subj_list = fold["subjects"]
        mask = np.isin(groups, subj_list)
        yield subj_list, np.where(~mask)[0], np.where(mask)[0]


# --- Main Training Function ---
def main(overrides=None):
    overrides = overrides or {}
    cfg_path = _resolve_config_path(DEFAULT_CONFIG)
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg_path = _snapshot_config(cfg_path) # Copy config to output dir

    seed = int(cfg.get("seed", 42))
    print(f"[main] Setting random seed: {seed}")
    set_seed(seed)

    # Load configurations
    dcfg = cfg["dataset"]
    if overrides.get("local_debug"):
        dcfg["num_workers"] = 0
        dcfg["pin_memory"] = False
    tcfg = cfg.get("trainer", {})
    if overrides.get("max_epochs") is not None:
        tcfg["max_epochs"] = overrides["max_epochs"]
    if overrides.get("max_folds") is not None:
        tcfg["max_folds"] = overrides["max_folds"]
    model_name = str(cfg["model"].get("name", "cnn")).lower()
    mcfg = cfg["model"]["params"].copy()
    wcfg = cfg.get("wandb", {})
    opt_cfg = cfg.get("optimizer", {})
    report_cfg = cfg.get("report", {})
    scheduler_cfg = cfg.get("scheduler")

    # --- Instantiate YOUR Dataset ---
    print("[main] Initializing Dataset...")
    # Pass only relevant args from config to your dataset loader
    ds = EEGPredictionDataset(
        dcfg["npz"],
        input_windows=int(dcfg.get("input_windows", 1)),
        task=dcfg.get("task", "detection"),
        channel_remap=dcfg.get("channel_remap"),
        # Other args like forecast_windows, level_tau, stride are ignored
        # by EEGPredictionDataset but passed for potential signature compatibility
        forecast_windows=int(dcfg.get("forecast_windows", 0)),
        level_tau=float(dcfg.get("level_tau", 0.5)),
        stride=int(dcfg.get("stride", 1)),
    )

    N = len(ds)
    groups = ds.groups # Get participant IDs from the dataset's property
    uniq_pids = np.unique(groups)
    print(f"[main] N={N} total samples | Subjects={len(uniq_pids)}")
    if N == 0:
        raise ValueError("Dataset is empty. Check the .npz file path and contents.")

    # Dataloader params
    batch_size = int(dcfg.get("batch_size", 64))
    num_workers = int(dcfg.get("num_workers", 4))
    pin_memory = bool(dcfg.get("pin_memory", True))
    # Report threshold for metrics
    thr = float(report_cfg.get("threshold", 0.5))

    fold_acc_values = []  # Accuracies for compatibility with older summaries
    fold_metric_details: list[dict] = []  # Detailed metrics per fold

    fold_cfg = cfg.get("folds", {})
    if overrides.get("fold_mode") is not None:
        fold_cfg["mode"] = overrides["fold_mode"]
    if overrides.get("num_folds") is not None:
        fold_cfg["num_folds"] = overrides["num_folds"]

    fold_mode = fold_cfg.get("mode", "logo").lower()
    fold_num = fold_cfg.get("num_folds")
    fold_seed = int(fold_cfg.get("seed", seed))
    labels_np = ds.lbl.detach().cpu().numpy().reshape(-1)

    if fold_mode == "logo":
        fold_iter = _logo_split(groups)
        total_folds = len(uniq_pids)
    elif fold_mode == "kfold":
        if not fold_num:
            raise ValueError("folds.num_folds must be provided for kfold mode.")
        fold_iter = _participant_kfold(groups, int(fold_num), fold_seed)
        total_folds = int(fold_num)
    elif fold_mode in ("balanced", "balanced_kfold", "stratified"):
        if not fold_num:
            raise ValueError("folds.num_folds must be provided for balanced mode.")
        fold_iter = _balanced_kfold(groups, labels_np, int(fold_num), fold_seed, thr)
        total_folds = int(fold_num)
    else:
        raise ValueError(f"Unknown fold mode '{fold_mode}'. Use 'logo' or 'kfold'.")

    max_folds = tcfg.get("max_folds")
    for fold, (test_pids, tr_idx, te_idx) in enumerate(fold_iter, start=1):
        print(f"\n=== Fold {fold}/{total_folds} | Testing on participants {test_pids} ===")
        print(f"Train size = {len(tr_idx)} | Test size = {len(te_idx)}")
        test_label = "-".join(map(str, test_pids))

        # --- Per-Fold Logging Setup (W&B or CSV fallback) ---
        logger = None
        run_name = None
        job_root = os.getenv("JOB_ROOT", "outputs")
        if wcfg.get("enabled", False):
            try:
                from pytorch_lightning.loggers import WandbLogger
                base_name = wcfg.get("name", "cdms_cnn_pred")
                jobid_part = f"_job{os.getenv('SLURM_JOB_ID', 'local')}"
                group_name = f"{base_name}{jobid_part}"
                run_name = f"{base_name}_fold{fold}_test{test_label}_seed{seed}"

                # Ensure WANDB_DIR exists or is handleable by WandbLogger
                wandb_dir = os.getenv("WANDB_DIR", os.path.join(job_root, "wandb"))
                os.makedirs(wandb_dir, exist_ok=True)

                logger = WandbLogger(
                    project=wcfg.get("project", "cdms-predict"),
                    save_dir=wandb_dir,
                    group=group_name,
                    name=run_name,
                    log_model=False, # Usually don't log model checkpoints for CV
                )
                # Log the configuration to W&B for this fold
                log_config(
                    logger, cfg,
                    extras={"fold.index": fold, "fold.test_pid": test_label, "seed": seed}
                )
                # log_config_file(logger, cfg_path, artifact_name=f"{run_name}-cfg") # Optional
            except ImportError:
                print("[main] Warning: wandb not installed. Skipping W&B logging.")
            except Exception as e:
                print(f"[main] Warning: Error initializing WandbLogger: {e}")

        if logger is None:
            try:
                from pytorch_lightning.loggers import CSVLogger
                csv_base = Path(job_root) / "lightning"
                csv_base.mkdir(parents=True, exist_ok=True)
                logger = CSVLogger(save_dir=str(csv_base), name=f"{model_name}_fold{fold}_test{test_label}")
            except Exception as e:
                print(f"[main] Warning: CSVLogger could not be created: {e}")

        # --- Create DataLoaders for the current fold ---
        train_dl = DataLoader(
            Subset(ds, tr_idx), batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory, drop_last=True # Drop last incomplete batch
        )
        test_dl = DataLoader(
            Subset(ds, te_idx), batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory
        )

        # --- Initialize Model for the current fold ---
        # Ensure n_outputs=1 for binary classification with sigmoid output
        print("[main] Initializing model...")
        if model_name == "cnn":
            backbone = CNN(n_outputs=1, **mcfg)
        elif model_name in ("shallow", "shallow_stage1", "shallowconvnet"):
            # Allow either n_channels or in_channels_eeg naming
            chan = mcfg.pop("n_channels", mcfg.pop("in_channels_eeg", 14))
            # Drop params that are only relevant to the CNN configs
            mcfg.pop("input_windows", None)
            mcfg.pop("time_len", None)
            backbone = ShallowStage1(n_channels=chan, n_outputs=1, **mcfg)
        else:
            raise ValueError(f"Unknown model.name '{model_name}'. Use 'cnn' or 'shallow'.")

        # --- Initialize the PyTorch Lightning Module ---
        # CSDownstreamModule likely handles optimizer setup, loss, etc.
        print("[main] Initializing Lightning Module...")

        # Extract finetune config safely
        sched_cfg = None
        if scheduler_cfg:
            sched_cfg = dict(scheduler_cfg)
            if sched_cfg.get("type", "").lower() == "cosine" and "t_max" not in sched_cfg:
                sched_cfg["t_max"] = int(tcfg.get("max_epochs", 50))

        module = CSDownstreamModule(
            backbone,
            lr=float(opt_cfg["lr"]),
            weight_decay=float(opt_cfg["weight_decay"]),
            task="cls", # Tell the module it's a classification task
            cls_threshold=thr, # Threshold for calculating metrics
            scheduler_cfg=sched_cfg,
        )

        # --- Initialize PyTorch Lightning Trainer ---
        print("[main] Initializing Trainer...")
        trainer_kwargs = dict(
            max_epochs=int(tcfg.get("max_epochs", 100)),
            accelerator=tcfg.get("accelerator", "auto"),
            devices=tcfg.get("devices", 1),
            enable_checkpointing=bool(tcfg.get("enable_checkpointing", False)),
            log_every_n_steps=int(tcfg.get("log_every_n_steps", 25)),
            logger=logger if logger is not None else False,
            deterministic="warn",
        )
        if "min_epochs" in tcfg:
            trainer_kwargs["min_epochs"] = int(tcfg["min_epochs"])
        grad_clip = tcfg.get("gradient_clip_val")
        if grad_clip is not None:
            trainer_kwargs["gradient_clip_val"] = float(grad_clip)
        trainer = pl.Trainer(**trainer_kwargs)

        # --- Train the model ---
        print(f"[main] Starting training for fold {fold}...")
        trainer.fit(module, train_dataloaders=train_dl)

        # --- Test the model on the left-out participant ---
        print(f"[main] Starting testing for fold {fold}...")
        # Use ckpt_path='best' if checkpointing is enabled and you want the best model
        ckpt_path = "best" if bool(tcfg.get("enable_checkpointing")) else None
        test_results = trainer.test(module, dataloaders=test_dl, ckpt_path=ckpt_path)

        # Store the primary test metric (e.g., accuracy)
        # Check standard PL keys first, then custom ones if needed
        acc_key_options = ['test_acc', 'test/acc', 'test_accuracy'] # Common keys
        fold_acc = None
        if test_results and isinstance(test_results, list):
            for key in acc_key_options:
                if key in test_results[0]:
                    fold_acc = test_results[0][key]
                    break # Found the key
        metric_summary = None
        if hasattr(module, "test_Y_true") and hasattr(module, "test_Y_pred"):
            y_true = module.test_Y_true.detach().cpu().numpy().reshape(-1)
            y_prob = module.test_Y_pred.detach().cpu().numpy().reshape(-1)
            metric_summary = binary_classification_report(y_prob, y_true, threshold=thr)
        if metric_summary is None and fold_acc is not None:
            metric_summary = {"accuracy": float(fold_acc)}
        if metric_summary is not None:
            fold_metric_details.append(
                {
                    "fold": fold,
                    "test_participants": [str(pid) for pid in test_pids],
                    **metric_summary,
                }
            )
            if "accuracy" in metric_summary:
                fold_acc_values.append(metric_summary["accuracy"])
        else:
            fold_metric_details.append(
                {
                    "fold": fold,
                    "test_participants": [str(pid) for pid in test_pids],
                }
            )
            fold_acc_values.append(np.nan)
        if metric_summary:
            metric_str = " ".join(f"{k}={v:.4f}" for k, v in metric_summary.items())
            print(f"[main] Fold {fold} metrics: {metric_str}")
        elif fold_acc is not None:
            print(f"[main] Fold {fold} Test Accuracy: {fold_acc:.4f}")
        else:
            print(f"[main] Warning: Could not compute metrics for fold {fold}.")

        # --- Save per-fold predictions for later analysis ---
        try:
            pred_dir = Path(os.getenv("JOB_ROOT", "outputs")) / "predictions"
            pred_dir.mkdir(parents=True, exist_ok=True)
            pred_path = pred_dir / f"fold{fold}_test{test_label}.npz"
            np.savez_compressed(
                pred_path,
                fold=int(fold),
                test_participants=np.asarray(test_pids, dtype=object),
                y_true=y_true,
                y_prob=y_prob,
                threshold=thr,
            )
            print(f"[main] Saved fold predictions to {pred_path}")
        except Exception as e:
            print(f"[main] Warning: Could not save predictions for fold {fold}: {e}")

        # --- Clean up W&B run for the fold ---
        if logger is not None:
            try:
                # W&B: ensure all logs are sent before finishing
                exp = getattr(logger, "experiment", None)
                if exp is not None and hasattr(exp, "finish"):
                    exp.log({}, commit=True)
                    exp.finish()
                    print("[main] W&B run finished.")
            except Exception as e:
                print(f"[main] Warning: Error finishing logger: {e}")
        if max_folds and fold >= int(max_folds):
            print(f"[main] Reached max_folds={max_folds}, stopping cross-validation early.")
            break

    # --- Aggregate and Report Final Results ---
    valid_metrics = [m for m in fold_acc_values if m is not None and not np.isnan(m)]
    if valid_metrics:
        mean_acc = np.mean(valid_metrics)
        std_acc = np.std(valid_metrics)
        print("\n=== Cross-Validation Summary ===")
        print(f"Individual Fold Accuracies: {[f'{m:.4f}' for m in valid_metrics]}")
        print(f"Average Test Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")

        # Save final aggregated results
        aggregate_metrics = {}
        metric_keys = sorted(
            {
                key
                for rec in fold_metric_details
                for key in rec.keys()
                if key not in ("fold", "test_participants")
            }
        )
        for key in metric_keys:
            values = [
                rec[key]
                for rec in fold_metric_details
                if key in rec and rec[key] is not None and not np.isnan(rec[key])
            ]
            if values:
                aggregate_metrics[key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                }
        results_summary = {
            "individual_fold_accuracies": fold_acc_values, # Include potential NaNs
            "mean_accuracy": mean_acc if valid_metrics else None,
            "std_accuracy": std_acc if valid_metrics else None,
            "config": cfg, # Save the config used for this run
            "individual_fold_metrics": fold_metric_details,
            "aggregate_metrics": aggregate_metrics,
        }
        summary_path = os.path.join(os.getenv("JOB_ROOT", "outputs"), "cross_val_summary.json")
        try:
            with open(summary_path, 'w') as f:
                json.dump(results_summary, f, indent=4)
            print(f"[main] Cross-validation summary saved to: {summary_path}")
        except Exception as e:
            print(f"[main] Warning: Failed to save summary results: {e}")
    else:
        print("\n[main] No valid metrics collected across folds.")


def parse_cli():
    parser = argparse.ArgumentParser(description="Train CNN on CDMS dataset.")
    parser.add_argument("--config", type=str, help="Override CONFIG_PATH with a specific YAML.")
    parser.add_argument("--max-epochs", type=int, help="Override trainer.max_epochs (useful for quick tests).")
    parser.add_argument("--max-folds", type=int, help="Override trainer.max_folds.")
    parser.add_argument("--fold-mode", type=str, choices=["logo", "kfold", "balanced"], help="Override folds.mode.")
    parser.add_argument("--num-folds", type=int, help="Override folds.num_folds.")
    parser.add_argument("--local-debug", action="store_true", help="Force single-process dataloading for quick tests.")
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_cli()
    if cli_args.config:
        os.environ["CONFIG_PATH"] = cli_args.config
    overrides = {}
    if cli_args.max_epochs is not None:
        overrides["max_epochs"] = cli_args.max_epochs
    if cli_args.max_folds is not None:
        overrides["max_folds"] = cli_args.max_folds
    if cli_args.fold_mode is not None:
        overrides["fold_mode"] = cli_args.fold_mode
    if cli_args.num_folds is not None:
        overrides["num_folds"] = cli_args.num_folds
    if cli_args.local_debug:
        overrides["local_debug"] = True
    main(overrides=overrides)
