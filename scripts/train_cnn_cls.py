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
from utils.metrics import binary_cls_metrics # Assuming this calculates accuracy correctly
from utils.logging import log_config #, log_config_file # log_config_file might need adjustment

# --- Use the supervisor's training module and CNN model ---
# Make sure these paths ('tasks. ...', 'models. ...') are correct
from tasks.cs_module import CSDownstreamModule
from models.downstream.cnn import CNN


DEFAULT_CONFIG = "configs/config.yaml"


# --- Helper functions (adapted from supervisor's script) ---
def _logo_split(groups: np.ndarray):
    """Simple Leave-One-Group-Out split without sklearn dependency."""
    unique_groups = np.unique(groups)
    for gid in unique_groups:
        test_mask = groups == gid
        test_idx = np.where(test_mask)[0]
        train_idx = np.where(~test_mask)[0]
        yield gid, train_idx, test_idx

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
    mcfg = cfg["model"]["params"].copy()
    # Check if pretrained is specified and handle potential None value
    pretrained_cfg_raw = mcfg.pop("pretrained", None)
    pretrained_cfg = pretrained_cfg_raw if pretrained_cfg_raw is not None else {}

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

    fold_metrics = [] # Store metrics (e.g., accuracy) for each fold

    fold_cfg = cfg.get("folds", {})
    fold_mode = fold_cfg.get("mode", "logo").lower()
    fold_num = fold_cfg.get("num_folds")
    fold_seed = int(fold_cfg.get("seed", seed))

    if fold_mode == "logo":
        fold_iter = _logo_split(groups)
        total_folds = len(uniq_pids)
    elif fold_mode == "kfold":
        if not fold_num:
            raise ValueError("folds.num_folds must be provided for kfold mode.")
        fold_iter = _participant_kfold(groups, int(fold_num), fold_seed)
        total_folds = int(fold_num)
    else:
        raise ValueError(f"Unknown fold mode '{fold_mode}'. Use 'logo' or 'kfold'.")

    max_folds = tcfg.get("max_folds")
    for fold, (test_pids, tr_idx, te_idx) in enumerate(fold_iter, start=1):
        print(f"\n=== Fold {fold}/{total_folds} | Testing on participants {test_pids} ===")
        print(f"Train size = {len(tr_idx)} | Test size = {len(te_idx)}")
        test_label = "-".join(map(str, test_pids))

        # --- Per-Fold Logging Setup (W&B) ---
        logger = None
        run_name = None
        if wcfg.get("enabled", False):
            try:
                from pytorch_lightning.loggers import WandbLogger
                base_name = wcfg.get("name", "cdms_cnn_pred")
                jobid_part = f"_job{os.getenv('SLURM_JOB_ID', 'local')}"
                group_name = f"{base_name}{jobid_part}"
                run_name = f"{base_name}_fold{fold}_test{test_label}_seed{seed}"

                # Ensure WANDB_DIR exists or is handleable by WandbLogger
                wandb_dir = os.getenv("WANDB_DIR", os.path.join(os.getenv("JOB_ROOT", "outputs"), "wandb"))
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
        # Ensure n_outputs=1 for binary classification with sigmoid output in CNN model
        print("[main] Initializing model...")
        backbone = CNN(n_outputs=1, **mcfg) # Pass model params from config

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

        if fold_acc is not None:
            fold_metrics.append(fold_acc)
            print(f"[main] Fold {fold} Test Accuracy: {fold_acc:.4f}")
        else:
             print(f"[main] Warning: Could not find accuracy metric ({acc_key_options}) in test results for fold {fold}. Results: {test_results}")
             fold_metrics.append(np.nan)

        # --- Clean up W&B run for the fold ---
        if logger is not None:
            try:
                # Ensure all logs are sent before finishing
                logger.experiment.log({}, commit=True)
                logger.experiment.finish()
                print("[main] W&B run finished.")
            except Exception as e:
                print(f"[main] Warning: Error finishing W&B run: {e}")
        if max_folds and fold >= int(max_folds):
            print(f"[main] Reached max_folds={max_folds}, stopping cross-validation early.")
            break

    # --- Aggregate and Report Final Results ---
    valid_metrics = [m for m in fold_metrics if m is not None and not np.isnan(m)]
    if valid_metrics:
        mean_acc = np.mean(valid_metrics)
        std_acc = np.std(valid_metrics)
        print("\n=== Cross-Validation Summary ===")
        print(f"Individual Fold Accuracies: {[f'{m:.4f}' for m in valid_metrics]}")
        print(f"Average Test Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")

        # Save final aggregated results
        results_summary = {
            "individual_fold_accuracies": fold_metrics, # Include potential NaNs
            "mean_accuracy": mean_acc if valid_metrics else None,
            "std_accuracy": std_acc if valid_metrics else None,
            "config": cfg # Save the config used for this run
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
    if cli_args.local_debug:
        overrides["local_debug"] = True
    main(overrides=overrides)
