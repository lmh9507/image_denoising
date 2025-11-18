$ErrorActionPreference = "Stop"

###############################
# GPU / 경로 설정
###############################
$GPU_ID = 6                              # ← 사용할 GPU 번호
$B_DOMAIN_DATA = "./data"                # ← B-domain root (clean/, noise/ 포함)
$PRETRAINED_CKPT = "results/UNetImproved/2025-11-18-17-39/epoch_model_001.pth"  # ← A-domain base 모델 ckpt 경로

$SAVE_ROOT = "./results_ft"              # ← finetune 결과(ckpt + val 이미지) 저장 루트
$LOG_NAME = "UNetImproved_adapter_IQSL"  # ← 실험 이름(서브폴더명)

###############################
# 하이퍼파라미터 설정
###############################
$ARCH = "UNetImproved"   # ← base 모델 아키텍처 (A-domain 학습 때 사용한 것과 동일해야 함)
$N_FEATURE = 48          # ← base UNet/ResNet feature 수 (기존 학습과 맞춰야 함)
$N_CHANNEL = 1           # ← 흑백 이미지면 1

$LR = 1e-4               # ← adapter 학습 learning rate
$N_EPOCH = 20            # ← finetune epoch 수
$BATCH_SIZE = 4
$NUM_WORKERS = 4

$PATCH_SIZE = 128        # ← patch 크기 (patch_size x patch_size)
$PATCHES_PER_IMAGE = 16  # ← 한 이미지당 epoch마다 뽑을 patch 개수

$ADAPTER_HIDDEN = 16     # ← adapter CNN hidden channel 수
$LAMBDA_GRAD = 0.1       # ← gradient consistency loss 가중치

# === IQSL(Intensity-Quantized Structural Loss) 관련 ===
$LAMBDA_IQSL = 0.1       # ← IQSL 전체 가중치 (0 으로 두면 구조 loss 비활성화)
$IQSL_Q1 = 0.2           # ← t1으로 사용할 intensity quantile (0~1)
$IQSL_Q2 = 0.8           # ← t2으로 사용할 intensity quantile (0~1)
$IQSL_TAU = 0.1          # ← softmax temperature (작을수록 hard assignment)
$IQSL_MARGIN = 0.0       # ← threshold 주변 don’t-care margin (0이면 사용 안 함)
$IQSL_MAX_IMAGES = 50    # ← threshold 추정에 사용할 clean 이미지 최대 개수
$IQSL_CE_FACTOR = 0.5    # ← IQSL 내부 CE 가중치 (Dice + ce_factor * CE)

###############################
# 실행
###############################
$env:CUDA_VISIBLE_DEVICES = "$GPU_ID"

python3 finetune_iqsl.py `
  --data_dir "$B_DOMAIN_DATA" `
  --pretrained_ckpt "$PRETRAINED_CKPT" `
  --arch "$ARCH" `
  --save_model_path "$SAVE_ROOT" `
  --log_name "$LOG_NAME" `
  --gpu_devices "$GPU_ID" `
  --n_feature $N_FEATURE `
  --n_channel $N_CHANNEL `
  --lr $LR `
  --n_epoch $N_EPOCH `
  --batchsize $BATCH_SIZE `
  --num_workers $NUM_WORKERS `
  --adapter_hidden $ADAPTER_HIDDEN `
  --lambda_grad $LAMBDA_GRAD `
  --save_every 1 `
  --patch_size $PATCH_SIZE `
  --patches_per_image $PATCHES_PER_IMAGE `
  --lambda_iqsl $LAMBDA_IQSL `
  --iqsl_q1 $IQSL_Q1 `
  --iqsl_q2 $IQSL_Q2 `
  --iqsl_tau $IQSL_TAU `
  --iqsl_margin $IQSL_MARGIN `
  --iqsl_max_images $IQSL_MAX_IMAGES `
  --iqsl_ce_factor $IQSL_CE_FACTOR
  # --parallel            # ← 다중 GPU(DataParallel) 쓸 경우 이 줄을 커맨드에 추가
