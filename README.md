# Neighbor2Neighbor: Self-Supervised Denoising from Single Noisy Images


## Python Requirements

This code was tested on:

- Python 3.7
- Pytorch 1.3

## Training

To train a network, run:

```bash
bash train.sh
```

## Data directory tree
```
data
├── clean
│   ├── sem0000_00.tif
│   ├── sem0000_01.tif
│   ├── sem0000_02.tif
│   └── ⋯
└── noise
    ├── sem0000_00.tif
    ├── sem0000_01.tif
    ├── sem0000_02.tif
    └── ⋯
```

## Finetune

```
powershell -ExecutionPolicy Bypass -File .\finetune.ps1
powershell -ExecutionPolicy Bypass -File .\run_infer_adapter.ps1
```

### 사용 시 유의사항 (clean / noise, 설정 등)

- **폴더 구조**
  - `data_dir/clean`, `data_dir/noise` 폴더 이름은 코드에서 그대로 사용하므로 바꾸지 않는 것을 권장.
  - 두 폴더의 파일 개수는 동일해야 하고, `sorted()` 기준으로 1:1 매칭된다고 가정.

- **이미지 형식**
  - 현재 구현은 1채널(grayscale) 이미지를 가정 (`--n_channel 1`).
  - 이미지 배열은 uint8 (0–255) → `/255.0` 으로 0–1 정규화되어 사용.
  - RGB(3채널)을 사용할 경우, 네트워크 정의(`in_nc`, `out_nc`)와 모든 스크립트의 `--n_channel` 값을 3으로 맞춰야 함.

- **네트워크/체크포인트 설정**
  - `--arch`, `--n_channel`, `--n_feature`는
    - A 도메인 학습(`train.py`),
    - B 도메인 finetune(`finetune.py`),
    - inference(`evaluation_adapter.py`)
    에서 모두 동일해야 함.
  - `finetune.py`의 `--pretrained_ckpt`에는 A 도메인 base 모델(`epoch_model_XXX.pth`),
    `evaluation_adapter.py`의 `--ckpt`에는 adapter까지 포함된 모델(`epoch_adapter_XXX.pth`)을 넣어야 함.

- **patch 샘플링**
  - `patch_size`는 모든 이미지의 높이/너비보다 작거나 같아야 함.
  - 한 epoch의 총 patch 수 = `이미지 개수 × patches_per_image`.
