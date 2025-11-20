# eval_704_iqsl_memory.ps1
# Windows PowerShell에서 memory adapter 평가용

python evaluation_704_iqsl_memory.py `
  --data_dir "./data/704" `
  --base_ckpt "results/UNetImproved/2025-11-18-17-39/epoch_model_001.pth" `
  --adapter_ckpt "./results_ft_syn_memory/UNetImproved_memory_adapter_IQSL/epoch_adapter_only_020.pth" `
  --arch "UNetImproved" `
  --save_dir "./results_704_eval_adapter_memory" `
  --gpu_devices "5" `
  --n_feature 48 `
  --n_channel 1 `
  --adapter_hidden 16 `
  --patch_size 128 `
  --overlap 64 `
  --num_memory_images 5 `
  --memory_stride 4
  # --compute_iq_iou `
  # --iq_low_q 0.25 `
  # --iq_high_q 0.75 `
  # --parallel
