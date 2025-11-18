#!/bin/bash
set -e

###############################
# GPU / 경로 설정
###############################
GPU_ID=0

B_DOMAIN_DATA="./data"   # noise/, clean/ 포함 root

# A-domain base 모델 ckpt (pretrain 때 쓴 것과 동일)
BASE_CKPT="results/UNetImproved/2025-11-18-17-39/epoch_model_001.pth"

# finetune에서 저장한 adapter-only ckpt
# 예: results_ft/UNetImproved_adapter_IQSL/epoch_adapter_only_020.pth
ADAPTER_CKPT="./results_ft/UNetImproved_adapter_IQSL/epoch_adapter_only_020.pth"

SAVE_DIR="./results_eval_adapter"

###############################
# 모델 설정 (finetune과 동일)
###############################
ARCH="UNetImproved"
N_FEATURE=48
N_CHANNEL=1
ADAPTER_HIDDEN=16

###############################
# 실행
###############################
CUDA_VISIBLE_DEVICES=${GPU_ID} python3 evaluation_adapter_iqsl.py \
  --data_dir "${B_DOMAIN_DATA}" \
  --base_ckpt "${BASE_CKPT}" \
  --adapter_ckpt "${ADAPTER_CKPT}" \
  --arch "${ARCH}" \
  --save_dir "${SAVE_DIR}" \
  --gpu_devices "${GPU_ID}" \
  --n_feature ${N_FEATURE} \
  --n_channel ${N_CHANNEL} \
  --adapter_hidden ${ADAPTER_HIDDEN}
  # --parallel
