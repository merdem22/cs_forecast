# Cybersickness Forecasting

Code for building EEG window datasets from the Hacettepe recordings and training simple CNNs for cybersickness detection/forecasting with participant-aware cross-validation.

## Repo map
- `configs/`: YAML configs for different window lengths/stride setups (8s detection, 1s×8, 2s×4, 3s stride1, etc. plus early-window forecasting variants). Includes a `config_shallow_stage1.yaml` that approximates the paper’s Stage‑1 Shallow ConvNet.
- `data/Hacettepe/cls/`: Raw-to-window preprocessing (`preprocessing.py`), participant loop and NPZ writer (`dataset_builder.py`), and the PyTorch `EEGPredictionDataset` loader (`loader.py`).
- `models/downstream/`: 1D CNN encoder/head used for downstream training.
- `tasks/`: Lightning module wrapping the model, loss, optimizer, and test metrics.
- `scripts/`: Entry points for dataset building, training (`train_cnn_cls.py`), random baselines, and plotting helpers.
- `jobs/`: Example SLURM script showing how to launch training with `CONFIG_PATH`/`JOB_ROOT`.
- `outputs/`, `reports/`, `notebooks/`: Generated artifacts and scratch analysis.
- `utils/`: Seed helper, metrics, lightweight logging helper, plotting utilities, and classical feature extraction.

## Typical workflow
1) **Build NPZ datasets** from the raw `.mat` files (14 channels, 256 Hz, 8 s trials) using sliding-window specs in `scripts/build_windowed_datasets.py`:
   ```bash
   python scripts/build_windowed_datasets.py --only 8s_x1 2s_x4
   ```
   This reads from `data/Recordings/SplittedDataWithAssumption-EliminatedPerSession` by default and writes compressed NPZ files under `data/npz/`.

2) **Train and evaluate** with participant-aware CV via PyTorch Lightning:
   ```bash
   python scripts/train_cnn_cls.py --config configs/config.yaml
   # or pick another config_* for different windowing setups
   # or the shallow Stage-1 reproduction:
   python scripts/train_cnn_cls.py --config configs/config_shallow_stage1.yaml
   ```
   Cross-val modes: `folds.mode` can be `logo` (leave-one-participant), `kfold`, or `balanced` (greedy subject assignment to balance class ratios and fold sizes).
   - Uses LOGO or participant k-fold splits based on `folds` in the config.
   - Optional W&B logging is controlled by the `wandb` block and env vars (`WANDB_MODE`, `WANDB_DIR`).

3) **Baselines and plots**
   - `scripts/eval_random_baseline.py --npz data/npz/cdms_1s_x8.npz` writes a participant-aware random-guess benchmark and an optional bar plot.
   - Other plotting scripts summarize training jobs, CV results, label timelines, and detection curves.

4) **Cluster runs**
   - `jobs/job.sh` shows a SLURM launch recipe; set `CONFIG_PATH` to pick a config and `JOB_ROOT` for per-run outputs/checkpoints.

## Notes
- Default preprocessing: 3rd-order Butterworth band-pass (4–40 Hz) + per-channel z-score normalization before windowing.
- CNN inputs are shaped `(B, W, C, T)` where `W` is the number of windows per sample (usually 1 for detection/forecasting configs).
