#!/bin/bash
set -e

###############################
# GPU / 경로 설정
###############################
GPU_ID=5

# 704 데이터 root (noise/, clean/ 포함)
B_DOMAIN_DATA="./data/syn"

# A-domain base 모델 ckpt
BASE_CKPT="results/UNetImproved/2025-11-18-17-39/epoch_model_001.pth"

# memory adapter finetune에서 저장한 adapter-only ckpt
ADAPTER_CKPT="./results_ft_syn_memory/UNetImproved_memory_adapter_IQSL/epoch_adapter_only_020.pth"

SAVE_DIR="./results_704_eval_adapter_memory"

###############################
# 모델 / 메모리 설정 (finetune과 동일하게 맞추기)
###############################
ARCH="UNetImproved"
N_FEATURE=48
N_CHANNEL=1
ADAPTER_HIDDEN=16

PATCH_SIZE=128
PATCH_OVERLAP=64
NUM_MEMORY_IMAGES=5
MEMORY_STRIDE=64

###############################
# 실행
###############################
CUDA_VISIBLE_DEVICES=${GPU_ID} python3 evaluation_704_iqsl_memory.py \
  --data_dir "${B_DOMAIN_DATA}" \
  --base_ckpt "${BASE_CKPT}" \
  --adapter_ckpt "${ADAPTER_CKPT}" \
  --arch "${ARCH}" \
  --save_dir "${SAVE_DIR}" \
  --gpu_devices "${GPU_ID}" \
  --n_feature ${N_FEATURE} \
  --n_channel ${N_CHANNEL} \
  --adapter_hidden ${ADAPTER_HIDDEN} \
  --patch_size ${PATCH_SIZE} \
  --overlap ${PATCH_OVERLAP} \
  --num_memory_images ${NUM_MEMORY_IMAGES} \
  --memory_stride ${MEMORY_STRIDE}
  # --compute_iq_iou \
  # --iq_low_q 0.25 \
  # --iq_high_q 0.75 \
  # --parallel
