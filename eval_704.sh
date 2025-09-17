#!/bin/bash

# ---------------------------
# GPU 설정
# ---------------------------
export CUDA_VISIBLE_DEVICES=2

# ---------------------------
# Evaluation Settings
# ---------------------------
DATA_DIR="/data1/home/yhk04150/hynix/image_denoising/resize_images2"
CHECKPOINT="./results/UNetImproved/2025-09-02-12-28/epoch_model_009.pth"
SAVE_DIR="./eval_improvedunet_test_704_re_noise"
N_CHANNEL=1
N_FEATURE=48

# ---------------------------
# 실행
# ---------------------------
python3 evaluation_704.py \
    --data_dir $DATA_DIR \
    --checkpoint $CHECKPOINT \
    --save_dir $SAVE_DIR \
    --n_channel $N_CHANNEL \
    --n_feature $N_FEATURE
