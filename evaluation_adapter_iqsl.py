from __future__ import annotations
import os
import glob
import argparse

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms

from arch_unet import UNet, RESNET, ImprovedUNet
from adapter import DenoiserWithAdapter


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data_dir', type=str, required=True,
        help='Root dir with noise/ (and optionally clean/) for inference.'
    )
    parser.add_argument(
    '--base_ckpt', type=str, required=True,
    help='Checkpoint of the base model trained on A-domain.'
    )  
    parser.add_argument(
        '--adapter_ckpt', type=str, required=True,
        help='Checkpoint that contains ONLY adapter weights (epoch_adapter_only_xxx.pth).'
    )
    parser.add_argument(
        '--arch', type=str, default='UNetImproved',
        choices=['UNet', 'RESNET', 'UNetImproved'],
        help='Backbone architecture used in base model.'
    )
    parser.add_argument(
        '--save_dir', type=str, default='./results_infer_adapter',
        help='Directory to save denoised images.'
    )
    parser.add_argument('--gpu_devices', default='0', type=str)
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--n_feature', type=int, default=48)
    parser.add_argument('--n_channel', type=int, default=1)
    parser.add_argument('--adapter_hidden', type=int, default=16)

    # === IQSL-style 3-class IoU 옵션 (clean 폴더 있을 때만 동작) ===
    parser.add_argument(
        '--compute_iq_iou', action='store_true',
        help='If set and clean/ exists, compute 3-class intensity-quantized IoU.'
    )
    parser.add_argument(
        '--iq_low_q', type=float, default=0.25,
        help='Lower quantile for IQ thresholds.'
    )
    parser.add_argument(
        '--iq_high_q', type=float, default=0.75,
        help='Upper quantile for IQ thresholds.'
    )

    args, _ = parser.parse_known_args()
    return args


def build_base_model(arch: str, n_channel: int, n_feature: int) -> nn.Module:
    if arch == 'UNet':
        net = UNet(in_nc=n_channel, out_nc=n_channel, n_feature=n_feature)
    elif arch == 'RESNET':
        net = RESNET(in_nc=n_channel, out_nc=n_channel, n_feature=n_feature)
    elif arch == 'UNetImproved':
        net = ImprovedUNet(in_nc=n_channel, out_nc=n_channel, n_feature=n_feature)
    else:
        raise ValueError(f'Unknown arch: {arch}')
    return net

def load_base_weights(model: nn.Module, ckpt_path: str):
    state = torch.load(ckpt_path, map_location='cpu')
    # DataParallel 호환
    if any(k.startswith('module.') for k in state.keys()):
        state = {k.replace('module.', '', 1): v for k, v in state.items()}

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f'[Warning] Missing keys when loading base model: {missing}')
    if unexpected:
        print(f'[Warning] Unexpected keys when loading base model: {unexpected}')
    print(f'Loaded base weights from {ckpt_path}')


def load_adapter_weights(model: nn.Module, ckpt_path: str):
    """
    adapter_only ckpt(epoch_adapter_only_xxx.pth)을
    DenoiserWithAdapter의 adapter 서브모듈에만 로드.
    """
    state = torch.load(ckpt_path, map_location='cpu')

    # DataParallel 지원
    if isinstance(model, nn.DataParallel):
        adapter = model.module.adapter
    else:
        adapter = model.adapter

    missing, unexpected = adapter.load_state_dict(state, strict=False)
    if missing:
        print(f'[Warning] Missing keys when loading adapter: {missing}')
    if unexpected:
        print(f'[Warning] Unexpected keys when loading adapter: {unexpected}')
    print(f'Loaded adapter-only weights from {ckpt_path}')


def calculate_psnr(target: np.ndarray, ref: np.ndarray) -> float:
    img1 = target.astype(np.float32)
    img2 = ref.astype(np.float32)
    diff = img1 - img2
    mse = np.mean(np.square(diff))
    if mse == 0:
        return 99.0
    psnr = 10.0 * np.log10(255.0 * 255.0 / mse)
    return float(psnr)


def _to_gray_float01(img: np.ndarray) -> np.ndarray:
    arr = img.astype(np.float32)
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    return arr / 255.0


def _quantize_3class(gray: np.ndarray, t1: float, t2: float) -> np.ndarray:
    labels = np.zeros_like(gray, dtype=np.int32)
    labels[gray <= t1] = 0
    labels[(gray > t1) & (gray < t2)] = 1
    labels[gray >= t2] = 2
    return labels


def compute_iq_iou(pred255: np.ndarray,
                   clean255: np.ndarray,
                   low_q: float,
                   high_q: float):
    gt_gray = _to_gray_float01(clean255)
    pred_gray = _to_gray_float01(pred255)

    t1, t2 = np.quantile(gt_gray, [low_q, high_q])

    gt_lbl = _quantize_3class(gt_gray, t1, t2)
    pred_lbl = _quantize_3class(pred_gray, t1, t2)

    ious = []
    for k in range(3):
        gt_k = (gt_lbl == k)
        pr_k = (pred_lbl == k)
        inter = np.logical_and(gt_k, pr_k).sum()
        union = np.logical_or(gt_k, pr_k).sum()
        if union == 0:
            ious.append(np.nan)
        else:
            ious.append(float(inter) / float(union))
    return ious


def main():
    opt = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_devices
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    noise_dir = os.path.join(opt.data_dir, 'noise')
    clean_dir = os.path.join(opt.data_dir, 'clean')
    os.makedirs(opt.save_dir, exist_ok=True)

    noise_paths = sorted(glob.glob(os.path.join(noise_dir, '*')))
    if len(noise_paths) == 0:
        raise RuntimeError(f'No files found in {noise_dir}')

    has_clean = os.path.isdir(clean_dir) and len(glob.glob(os.path.join(clean_dir, '*'))) > 0
    if has_clean:
        clean_paths = sorted(glob.glob(os.path.join(clean_dir, '*')))
        if len(clean_paths) != len(noise_paths):
            print('[Warning] clean/ and noise/ have different counts; PSNR may be misaligned.')

    print(f'Found {len(noise_paths)} noisy images for inference.')

    # 1) base + adapter 모델 구성
    base_model = build_base_model(opt.arch, opt.n_channel, opt.n_feature)

    # 1) base A-domain weight 로드
    load_base_weights(base_model, opt.base_ckpt)

    # 2) base + adapter 래퍼 구성
    model = DenoiserWithAdapter(
        base_model=base_model,
        in_channels=opt.n_channel,
        hidden_channels=opt.adapter_hidden,
        freeze_base=True,
        use_no_grad_for_base=True,
    )

    if opt.parallel:
        model = nn.DataParallel(model)

    model.to(device)
    model.eval()

    # 3) adapter-only ckpt 로드
    load_adapter_weights(model, opt.adapter_ckpt)


    to_tensor = transforms.ToTensor()

    iou_dark_list, iou_mid_list, iou_bright_list = [], [], []

    # 3) 각 noisy 이미지에 대해 denoise
    with torch.no_grad():
        for idx, n_path in enumerate(noise_paths):
            name = os.path.basename(n_path)
            base_name = os.path.splitext(name)[0]

            noisy_img = np.array(Image.open(n_path), dtype=np.float32)
            noisy_norm = noisy_img / 255.0

            noisy_tensor = to_tensor(noisy_norm)     # [C,H,W], 0~1
            noisy_tensor = noisy_tensor.unsqueeze(0).to(device)  # [1,C,H,W]

            pred = model(noisy_tensor)               # [1,C,H,W]
            pred = pred.squeeze(0).permute(1, 2, 0).cpu().numpy()

            if pred.ndim == 2:  # [H,W]
                pred255 = np.clip(pred * 255.0 + 0.5, 0, 255).astype(np.uint8)
            else:               # [H,W,C]
                pred255 = np.clip(pred * 255.0 + 0.5, 0, 255).astype(np.uint8)

            # 저장
            if pred255.ndim == 2:
                out_img = Image.fromarray(pred255).convert('L')
            else:
                # C=1 or C=3 가정
                if pred255.shape[2] == 1:
                    out_img = Image.fromarray(pred255.squeeze(-1)).convert('L')
                else:
                    out_img = Image.fromarray(pred255).convert('RGB')

            save_path = os.path.join(opt.save_dir, f'{base_name}_denoised.png')
            out_img.save(save_path)

            # optional PSNR / IoU
            if has_clean and idx < len(clean_paths):
                clean_img = np.array(Image.open(clean_paths[idx]), dtype=np.float32)
                psnr = calculate_psnr(pred255, clean_img)

                msg = (f'[{idx+1:03d}/{len(noise_paths):03d}] '
                       f'{name} → PSNR={psnr:.2f} dB, saved to {save_path}')

                if opt.compute_iq_iou:
                    ious = compute_iq_iou(
                        pred255, clean_img,
                        low_q=opt.iq_low_q,
                        high_q=opt.iq_high_q
                    )
                    iou_d, iou_m, iou_b = ious
                    iou_dark_list.append(iou_d)
                    iou_mid_list.append(iou_m)
                    iou_bright_list.append(iou_b)
                    msg += f', IoU(d/m/b)=({iou_d:.3f},{iou_m:.3f},{iou_b:.3f})'

                print(msg)
            else:
                print(f'[{idx+1:03d}/{len(noise_paths):03d}] {name} → saved to {save_path}')

    if opt.compute_iq_iou and has_clean and len(iou_dark_list) > 0:
        print('Average IQ-3class IoU - '
              f'dark: {np.nanmean(iou_dark_list):.4f}, '
              f'mid: {np.nanmean(iou_mid_list):.4f}, '
              f'bright: {np.nanmean(iou_bright_list):.4f}')

    print('Inference with adapter model finished.')


if __name__ == '__main__':
    main()
