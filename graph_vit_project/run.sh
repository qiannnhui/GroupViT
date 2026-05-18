#!/bin/bash

# Default parameters if not provided
DATASET=${1:-MUTAG}
EPOCHS=${2:-100}
EVAL_MODE=${3:-concat}
ACTIVE_THRESHOLD=${4:-0.3}
SIM_THRESHOLD=${5:-0.85}
LOG_INTERVAL=${6:-1}
FN_MODE=${7:-sim}
WARMUP_EPOCHS=${8:-20}

echo "=========================================================="
echo "Starting GraphViT (GroupViT + GraphCL) Training Pipeline"
echo "Dataset:          $DATASET"
echo "Epochs:           $EPOCHS"
echo "Eval Mode:        $EVAL_MODE (Options: concat, simclr, groupvit)"
echo "Active Threshold: $ACTIVE_THRESHOLD"
echo "Sim Threshold:    $SIM_THRESHOLD"
echo "Log Interval:     $LOG_INTERVAL"
echo "FN Mode:          $FN_MODE (Options: sim, rbo)"
echo "Warmup Epochs:    $WARMUP_EPOCHS"
echo "=========================================================="

# Run the training script
# We run from the root of the project so python path is correct
PYTHONPATH=. python -m graph_vit_project.main \
    --dataset "$DATASET" \
    --epochs "$EPOCHS" \
    --eval_mode "$EVAL_MODE" \
    --rw_length 16 \
    --hidden_dim 128 \
    --lr 0.001 \
    --active_threshold "$ACTIVE_THRESHOLD" \
    --sim_threshold "$SIM_THRESHOLD" \
    --log_interval "$LOG_INTERVAL" \
    --fn_mode "$FN_MODE" \
    --warmup_epochs "$WARMUP_EPOCHS"


