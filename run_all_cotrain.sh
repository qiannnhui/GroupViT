#!/bin/bash

# 1. 強制設定環境變數，確保 Python 導入正常
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 2. 定義資料集清單
DATASETS=("MUTAG" "PROTEINS" "NCI1" "DD" "IMDB-BINARY" "REDDIT-BINARY" "REDDIT-MULTI-5K" "COLLAB")

# 3. 設定訓練參數
EPOCHS=100
WARMUP_EPOCHS=40
EVAL_MODE="concat"
ACTIVE_THRESHOLD=0.1
SIM_THRESHOLD=0.85
LOG_INTERVAL=1
FN_MODE="causal_sim"
CAUSAL_K=2
RBO_P=0.98
RBO_THRESHOLD=0.6

# 4. 建立儲存 Log 的資料夾
LOG_DIR="benchmarks_cotrain_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

# 5. 偵測可用的 GPU 數量
NUM_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
if [ -z "$NUM_GPUS" ] || [ "$NUM_GPUS" -lt 1 ]; then
    NUM_GPUS=1
fi

echo "=========================================================="
echo "Co-Training Batch Process Started. Logs: $LOG_DIR"
echo "Total Epochs: $EPOCHS, Warmup Epochs (Transition): $WARMUP_EPOCHS"
echo "FN Mode: $FN_MODE, Sim Thresh: $SIM_THRESHOLD"
echo "Detected $NUM_GPUS GPU(s) available."
echo "=========================================================="

if [ "$NUM_GPUS" -ge 2 ]; then
    echo "Using Parallel GPU mode (GPU 0 & GPU 1) to speed up training!"
    echo "=========================================================="
    
    # 兩張卡雙工併行跑
    for ((i=0; i<${#DATASETS[@]}; i+=2))
    do
        DS1="${DATASETS[i]}"
        DS2="${DATASETS[i+1]}"
        
        # 啟動第一個資料集於 GPU 0 (背景執行)
        if [ -n "$DS1" ]; then
            LOG_FILE1="$LOG_DIR/${DS1}.log"
            echo "[$(date +%T)] Co-Training $DS1 on GPU 0Pe (background)..."
            CUDA_VISIBLE_DEVICES=0 Pe=0 uv run python -u -m graph_vit_project.main_cotrain \
                --dataset "$DS1" \
                --epochs "$EPOCHS" \
                --warmup_epochs "$WARMUP_EPOCHS" \
                --eval_mode "$EVAL_MODE" \
                --active_threshold "$ACTIVE_THRESHOLD" \
                --sim_threshold "$SIM_THRESHOLD" \
                --log_interval "$LOG_INTERVAL" \
                --fn_mode "$FN_MODE" \
                --causal_k "$CAUSAL_K" \
                --rbo_p "$RBO_P" \
                --rbo_threshold "$RBO_THRESHOLD" \
                --distill \
                > "$LOG_FILE1" 2>&1 &
            PID1=$!
        else
            PID1=""
        fi

        # 啟動第二個資料集於 GPU 1 (背景執行)
        if [ -n "$DS2" ]; then
            LOG_FILE2="$LOG_DIR/${DS2}.log"
            echo "[$(date +%T)] Co-Training $DS2 on GPUPe 1 (background)..."
            CUDA_VISIBLE_DEVICES=1 Pe=1 uv run python -u -m graph_vit_project.main_cotrain \
                --dataset "$DS2" \
                --epochs "$EPOCHS" \
                --warmup_epochs "$WARMUP_EPOCHS" \
                --eval_mode "$EVAL_MODE" \
                --active_threshold "$ACTIVE_THRESHOLD" \
                --sim_threshold "$SIM_THRESHOLD" \
                --log_interval "$LOG_INTERVAL" \
                --fn_mode "$FN_MODE" \
                --causal_k "$CAUSAL_K" \
                --rbo_p "$RBO_P" \
                --rbo_threshold "$RBO_THRESHOLD" \
                --distill \
                > "$LOG_FILE2" 2>&1 &
            PID2=$!
        else
            PID2=""
        fi

        # 等待兩者執行完畢
        if [ -n "$PID1" ]; then
            wait $PID1
            if [ $? -eq 0 ]; then
                echo "Done $DS1."
            else
                echo "ERROR: $DS1 failed."
            fi
        fi

        if [ -n "$PID2" ]; then
            wait $PID2
            if [ $? -eq 0 ]; then
                echo "Done $DS2."
            else
                echo "ERROR: $DS2 failed."
            fi
        fi
        echo "----------------------------------------------------------"
    done
else
    echo "Using Single GPU mode (sequential)..."
    echo "=========================================================="
    
    # 單卡序列跑
    for DS in "${DATASETS[@]}"
    do
        LOG_FILE="$LOG_DIR/${DS}.log"
        echo "[$(date +%T)] Co-Training $DS on GPU 0..."
        
        CUDA_VISIBLE_DEVICES=0 uv run python -u -m graph_vit_project.main_cotrain \
            --dataset "$DS" \
            --epochs "$EPOCHS" \
            --warmup_epochs "$WARMUP_EPOCHS" \
            --eval_mode "$EVAL_MODE" \
            --active_threshold "$ACTIVE_THRESHOLD" \
            --sim_threshold "$SIM_THRESHOLD" \
            --log_interval "$LOG_INTERVAL" \
            --fn_mode "$FN_MODE" \
            --causal_k "$CAUSAL_K" \
            --rbo_p "$RBO_P" \
            --rbo_threshold "$RBO_THRESHOLD" \
            --distill \
            > "$LOG_FILE" 2>&1

        if [ $? -eq 0 ]; then
            echo "Done $DS."
        else
            echo "ERROR: $DS failed."
        fi
        echo "----------------------------------------------------------"
    done
fi

echo "=========================================================="
echo "Final Summary (Best Val Acc)"
echo "----------------------------------------------------------"
printf "%-20s | %-12s | %-12s\n" "Dataset" "Best Val Acc" "Test Acc"
echo "----------------------------------------------------------"

SUMMARY_FILE="$LOG_DIR/summary.txt"
echo "==========================================================" > "$SUMMARY_FILE"
echo "Final Summary (Best Val Acc)" >> "$SUMMARY_FILE"
echo "----------------------------------------------------------" >> "$SUMMARY_FILE"
printf "%-20s | %-12s | %-12s\n" "Dataset" "Best Val Acc" "Test Acc" >> "$SUMMARY_FILE"
echo "----------------------------------------------------------" >> "$SUMMARY_FILE"

for DS in "${DATASETS[@]}"
do
    LOG_FILE="$LOG_DIR/${DS}.log"
    if [ -f "$LOG_FILE" ]; then
        # 搜尋最高 Val Acc 所在的列
        RESULT=$(grep "Val Acc:" "$LOG_FILE" | sed 's/,Pe//g' | awk '
        {
            val_acc = 0
            test_acc = 0
            for(i=1; i<=NF; i++) {
                if($i == "Val") { val_acc = $(i+2) }
                if($i == "Test") { test_acc = $(i+2) }
            }
            print val_acc, test_acc
        }' | sort -rn -k1 | head -n 1)
        
        MAX_VAL=$(echo "$RESULT" | awk '{print $1}')
        CORR_TEST=$(echo "$RESULT" | awk '{print $2}')
        
        if [ -z "$MAX_VAL" ] || [ "$MAX_VAL" = "0" ]; then
            MAX_VAL="N/A"
            CORR_TEST="N/A"
        fi
        
        printf "%-20s | %-12s | %-12s\n" "$DS" "$MAX_VAL" "$CORR_TEST"
        printf "%-20s | %-12s | %-12s\n" "$DS" "$MAX_VAL" "$CORR_TEST" >> "$SUMMARY_FILE"
    fi
done

echo "=========================================================="
echo "All tasks completed. Results in $LOG_DIR"
echo "Summary saved to $SUMMARY_FILE"
