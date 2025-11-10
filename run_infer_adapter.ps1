$ErrorActionPreference = "Stop"

$GPU_ID = 6

$ADAPTER_CKPT = "results_ft/UNetImproved_adapter_ft/epoch_adapter_020.pth"

$DATA_B_TEST = "data"

$SAVE_DIR = "results_ft/UNetImproved_adapter_ft/infer_ep020"

python evaluation_adapter.py `
  --gpu_devices $GPU_ID `
  --data_dir $DATA_B_TEST `
  --ckpt $ADAPTER_CKPT `
  --arch UNetImproved `
  --n_channel 1 `
  --n_feature 48 `
  --adapter_hidden 16 `
  --save_dir $SAVE_DIR
