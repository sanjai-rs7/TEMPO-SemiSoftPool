#!/bin/bash
# TEMPO Training Script for Mac (MPS GPU)
# Usage: ./run_train.sh

set -e

# Activate environment
source /Users/sanjairs/TEMPO/tempo_env/bin/activate
cd /Users/sanjairs/TEMPO/TEMPO

echo "============================================"
echo "  TEMPO Training - ETTh1 Dataset"
echo "  Device: Apple MPS GPU"
echo "============================================"

python3 train_TEMPO.py \
  --config_path ./configs/etth1_local.yml \
  --model TEMPO \
  --datasets ETTh1 \
  --target_data ETTh1 \
  --eval_data ETTh1 \
  --seq_len 336 \
  --pred_len 96 \
  --batch_size 16 \
  --train_epochs 2 \
  --gpt_layers 6 \
  --d_model 768 \
  --patch_size 16 \
  --stride 8 \
  --prompt 1 \
  --pretrain 0 \
  --freeze 1 \
  --is_gpt 1 \
  --num_nodes 1 \
  --loss_func mse \
  --stl_weight 0.01 \
  --learning_rate 0.001 \
  --checkpoints ./checkpoints_local/ \
  --model_id etth1_test \
  --itr 1 \
  --equal 0 \
  --use_token 0

echo "============================================"
echo "  Training Complete!"
echo "============================================"
