# finetune_memory.ps1

python finetune_memory.py `
  --data_dir "./data/syn" `
  --pretrained_ckpt "results/UNetImproved/2025-11-18-17-39/epoch_model_001.pth" `
  --arch "UNetImproved" `
  --save_model_path "./results_ft_syn_memory" `
  --log_name "UNetImproved_memory_adapter_IQSL" `
  --gpu_devices "5" `
  --n_feature 48 `
  --n_channel 1 `
  --lr 1e-4 `
  --n_epoch 20 `
  --batchsize 4 `
  --num_workers 4 `
  --adapter_hidden 16 `
  --lambda_grad 0.1 `
  --save_every 1 `
  --patch_size 128 `
  --patches_per_image 16 `
  --num_memory_images 5 `
  --memory_stride 64 `
  --lambda_iqsl 0.1 `
  --iqsl_q1 0.2 `
  --iqsl_q2 0.8 `
  --iqsl_tau 0.1 `
  --iqsl_margin 0.0 `
  --iqsl_max_images 50 `
  --iqsl_ce_factor 0.5
  # --parallel    # 다중 GPU(DataParallel)를 쓰려면 finetune_memory.py에 이 옵션을 추가해서 사용
