$ErrorActionPreference = "Stop"

# 사용할 GPU 번호
$GPU_ID = 6

# A-도메인에서 학습한 베이스 모델 체크포인트 경로
$BASE_CKPT = "results/UNetImproved/2025-07-23-13-31/epoch_model_100.pth"

# B-도메인 데이터 경로 (clean/, noise/ 포함)
$DATA_B = "data"

python finetune.py `
  --gpu_devices $GPU_ID `
  --data_dir $DATA_B `
  --pretrained_ckpt $BASE_CKPT `
  --arch UNetImproved `
  --log_name UNetImproved_adapter_ft `
  --save_model_path results_ft `
  --n_channel 1 `
  --n_feature 48 `
  --batchsize 4 `
  --n_epoch 20 `
  --lr 1e-4 `
  --adapter_hidden 16 `
  --lambda_grad 0.1 `
  --patch_size 128 `
  --patches_per_image 16 `
  --save_every 5
