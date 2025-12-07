#!/bin/bash

# ==============================================================================
# SLURM SETTINGS (edit to match your cluster)
# ==============================================================================
#SBATCH --job-name=cs-matrix
#SBATCH --partition=ai
#SBATCH --account=ai
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-12:00:00
#SBATCH --output=/home/merdem22/hacettepe_forecast/outputs/logs/matrix_%A_%a.out
#SBATCH --error=/home/merdem22/hacettepe_forecast/outputs/logs/matrix_%A_%a.err
#SBATCH --array=0-41   # (num_configs * num_modes) - 1; update if you change lists below

# ==============================================================================
# ENVIRONMENT
# ==============================================================================
echo "SLURM Job ID: $SLURM_JOB_ID / Array Task: $SLURM_ARRAY_TASK_ID"
echo "Running on node: $(hostname)"
echo "Started at: $(date)"

PROJECT_ROOT="/home/merdem22/hacettepe_forecast"
cd "$PROJECT_ROOT" || exit 1
echo "Project root: $(pwd)"

# Optional: load modules / activate conda env
module load anaconda3/2025.06
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate my-ai-env

PYTHON_BIN="${CONDA_PREFIX}/bin/python"

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Default to offline W&B unless overridden
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${WANDB_DIR:-$PROJECT_ROOT/outputs/wandb}"

# ==============================================================================
# RUN MATRIX
# ==============================================================================
# Detection + forecasting configs (CNN + Shallow)
CONFIGS=(
  # CNN detection
  "configs/config.yaml"                # 8s detection (CNN)
  "configs/config_1s8f.yaml"
  "configs/config_2s4f.yaml"
  "configs/config_4s2f.yaml"
  # CNN forecasting
  "configs/config_fc_2s_first4.yaml"   # first two 2s windows
  "configs/config_fc_3s_stride2x2.yaml" # first two 3s windows, stride 2
  "configs/config_fc_3s_stride1x3.yaml" # first three 3s windows, stride 1
  # Shallow detection
  "configs/config_shallow_1s8f.yaml"
  "configs/config_shallow_2s4f.yaml"
  "configs/config_shallow_4s2f.yaml"
  "configs/config_shallow_8s1f.yaml"
  # Shallow forecasting
  "configs/config_shallow_fc_2s_first4.yaml"
  "configs/config_shallow_fc_3s_stride2x2.yaml"
  "configs/config_shallow_fc_3s_stride1x3.yaml"
)

FOLD_MODES=(logo kfold balanced)
NUM_FOLDS_DEFAULT=5

NUM_CONFIGS=${#CONFIGS[@]}
NUM_MODES=${#FOLD_MODES[@]}
IDX_CONFIG=$(( SLURM_ARRAY_TASK_ID / NUM_MODES ))
IDX_MODE=$(( SLURM_ARRAY_TASK_ID % NUM_MODES ))

CONFIG_REL="${CONFIGS[$IDX_CONFIG]}"
MODE="${FOLD_MODES[$IDX_MODE]}"
CONFIG_PATH="$PROJECT_ROOT/$CONFIG_REL"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config not found: $CONFIG_PATH"
  exit 1
fi

CONFIG_BASENAME=$(basename "$CONFIG_PATH" .yaml)
RUN_TAG="${CONFIG_BASENAME}_${MODE}"

export CONFIG_PATH
export JOB_ROOT="$PROJECT_ROOT/outputs/job_${RUN_TAG}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$JOB_ROOT"

NUM_FOLDS_ARG=""
if [[ "$MODE" != "logo" ]]; then
  NUM_FOLDS_ARG="--num-folds ${NUM_FOLDS_DEFAULT}"
fi

echo "Running config=$CONFIG_PATH | mode=$MODE | num_folds=${NUM_FOLDS_DEFAULT} | JOB_ROOT=$JOB_ROOT"

${PYTHON_BIN:-python} -u scripts/train_cnn_cls.py \
  --config "$CONFIG_PATH" \
  --fold-mode "$MODE" \
  $NUM_FOLDS_ARG

echo "Job finished at: $(date)"
