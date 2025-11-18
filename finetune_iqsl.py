from __future__ import annotations
import os
import time
import glob
import argparse
import datetime
from typing import Tuple, List

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from arch_unet import UNet, RESNET, ImprovedUNet
from adapter import DenoiserWithAdapter
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    # B-domain 데이터 (왼쪽/오른쪽 이미지 페어)
    parser.add_argument(
        '--data_dir', type=str, required=True,
        help='B-domain root. Must contain clean/ and noise/ subfolders.'
    )
    # A-domain에서 학습한 base 모델 checkpoint
    parser.add_argument(
        '--pretrained_ckpt', type=str, required=True,
        help='Checkpoint of the base model trained on A-domain.'
    )
    parser.add_argument(
        '--arch', type=str, default='UNetImproved',
        choices=['UNet', 'RESNET', 'UNetImproved'],
        help='Backbone architecture used for the base model.'
    )
    parser.add_argument(
        '--save_model_path', type=str, default='./results_ft',
        help='Root to save finetuned checkpoints and validation images.'
    )
    parser.add_argument(
        '--log_name', type=str, default='UNetImproved_adapter_ft',
        help='Subfolder name for logging.'
    )
    parser.add_argument('--gpu_devices', default='0', type=str)
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--n_feature', type=int, default=48)
    parser.add_argument('--n_channel', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--n_epoch', type=int, default=20)
    parser.add_argument('--batchsize', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument(
        '--adapter_hidden', type=int, default=16,
        help='Hidden channels of the small adapter CNN.'
    )
    parser.add_argument(
        '--lambda_grad', type=float, default=0.1,
        help='Weight for gradient-consistency loss.'
    )
    parser.add_argument(
        '--save_every', type=int, default=1,
        help='Save checkpoint & run validation every N epochs.'
    )

    # ★ patch 단위 샘플링 설정
    parser.add_argument(
        '--patch_size', type=int, default=128,
        help='Patch size (patch_size x patch_size).'
    )
    parser.add_argument(
        '--patches_per_image', type=int, default=16,
        help='How many patches per image per epoch.'
    )

    # === IQSL (Intensity-Quantized Structural Loss) 관련 하이퍼파라미터 ===
    parser.add_argument(
        '--lambda_iqsl', type=float, default=0.1,
        help='Weight for IQSL structural loss (0 → disable).'
    )
    parser.add_argument(
        '--iqsl_q1', type=float, default=0.2,
        help='Lower quantile for intensity threshold t1 (in [0,1]).'
    )
    parser.add_argument(
        '--iqsl_q2', type=float, default=0.8,
        help='Upper quantile for intensity threshold t2 (in [0,1]).'
    )
    parser.add_argument(
        '--iqsl_tau', type=float, default=0.1,
        help='Softmax temperature for class probabilities in IQSL.'
    )
    parser.add_argument(
        '--iqsl_margin', type=float, default=0.0,
        help='Margin around thresholds treated as ambiguous (0 = no ignore).'
    )
    parser.add_argument(
        '--iqsl_max_images', type=int, default=50,
        help='Max number of clean images used to estimate thresholds.'
    )
    parser.add_argument(
        '--iqsl_ce_factor', type=float, default=0.5,
        help='Relative weight of CE term inside IQSL (Dice + ce_factor * CE).'
    )

    args, _ = parser.parse_known_args()
    return args


def checkpoint(net: nn.Module, epoch: int, opt) -> str:
    """
    DenoiserWithAdapter에서 adapter 부분만 저장하는 checkpoint 함수.
    - DataParallel(model) 인 경우 model.module.adapter.state_dict() 사용
    - 단일 GPU/CPU 인 경우 model.adapter.state_dict() 사용
    """
    save_root = os.path.join(opt.save_model_path, opt.log_name)
    os.makedirs(save_root, exist_ok=True)
    model_name = f'epoch_adapter_only_{epoch:03d}.pth'
    save_path = os.path.join(save_root, model_name)

    if isinstance(net, nn.DataParallel):
        adapter_state = net.module.adapter.state_dict()
    else:
        adapter_state = net.adapter.state_dict()

    torch.save(adapter_state, save_path)
    print(f'Adapter checkpoint saved to {save_path}')
    return save_path


class DenoisePatchDataset(Dataset):
    """
    Supervised B-domain dataset with random patches.

    디렉토리 구조:
        data_dir/clean/*.png (또는 tif, ...)
        data_dir/noise/*.png

    clean/ 와 noise/ 는 파일명 sort 순서대로 1:1 매칭된다고 가정.
    __len__ = num_images * patches_per_image
    → 이미지 5장, patches_per_image=16이면, 한 epoch에 80개 patch 사용.
    """
    def __init__(self, data_dir: str, patch_size: int, patches_per_image: int):
        super().__init__()
        self.data_dir = data_dir
        self.clean = sorted(glob.glob(os.path.join(self.data_dir, 'clean', '*')))[:5]
        self.noise = sorted(glob.glob(os.path.join(self.data_dir, 'noise', '*')))[:5]
        assert len(self.clean) == len(self.noise) and len(self.clean) > 0, \
            'clean and noise must have the same number of images and be non-empty.'
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.to_tensor = transforms.ToTensor()  # float32 np.array 그대로 tensor로
        print(f'B-domain: {len(self.clean)} images, '
              f'{self.patches_per_image} patches/image/epoch → '
              f'{len(self)*1} samples/epoch.')

    def __len__(self) -> int:
        return len(self.clean) * self.patches_per_image

    def _load_pair(self, img_idx: int):
        clean_path, noise_path = self.clean[img_idx], self.noise[img_idx]
        clean_img = np.array(Image.open(clean_path), dtype=np.float32)
        noise_img = np.array(Image.open(noise_path), dtype=np.float32)
        return clean_img, noise_img

    def __getitem__(self, index: int):
        # index로부터 어떤 원본 이미지를 사용할지 결정
        img_idx = index // self.patches_per_image
        clean_img, noise_img = self._load_pair(img_idx)

        h, w = clean_img.shape[:2]
        ps = self.patch_size
        assert h >= ps and w >= ps, \
            f'Image size ({h},{w}) smaller than patch_size {ps}. ' \
            f'이미지가 더 크거나 patch_size를 줄여주세요.'

        # 동일 좌표에서 clean/noise patch crop
        top = np.random.randint(0, h - ps + 1)
        left = np.random.randint(0, w - ps + 1)
        clean_patch = clean_img[top:top + ps, left:left + ps]
        noise_patch = noise_img[top:top + ps, left:left + ps]

        # [0,1] 스케일로 변환
        clean_tensor = self.to_tensor(clean_patch) / 255.0
        noise_tensor = self.to_tensor(noise_patch) / 255.0

        return clean_tensor, noise_tensor


def gradient(x: torch.Tensor):
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    return dx, dy


def gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_dx, pred_dy = gradient(pred)
    tgt_dx, tgt_dy = gradient(target)
    return F.l1_loss(pred_dx, tgt_dx) + F.l1_loss(pred_dy, tgt_dy)


def calculate_psnr(target: np.ndarray, ref: np.ndarray) -> float:
    img1 = target.astype(np.float32)
    img2 = ref.astype(np.float32)
    diff = img1 - img2
    mse = np.mean(np.square(diff))
    if mse == 0:
        return 99.0
    psnr = 10.0 * np.log10(255.0 * 255.0 / mse)
    return float(psnr)


def validation_denoise(dataset_dir: str):
    clean_paths = sorted(glob.glob(os.path.join(dataset_dir, 'clean', '*')))[:10]
    noise_paths = sorted(glob.glob(os.path.join(dataset_dir, 'noise', '*')))[:10]
    images_clean, images_noise = [], []
    for fn1, fn2 in zip(clean_paths, noise_paths):
        im1, im2 = Image.open(fn1), Image.open(fn2)
        im1 = np.array(im1, dtype=np.float32)
        im2 = np.array(im2, dtype=np.float32)
        images_clean.append(im1)
        images_noise.append(im2)
    return images_clean, images_noise, clean_paths, noise_paths


def build_base_model(opt) -> nn.Module:
    if opt.arch == 'UNet':
        net = UNet(in_nc=opt.n_channel,
                   out_nc=opt.n_channel,
                   n_feature=opt.n_feature)
    elif opt.arch == 'RESNET':
        net = RESNET(in_nc=opt.n_channel,
                     out_nc=opt.n_channel,
                     n_feature=opt.n_feature)
    elif opt.arch == 'UNetImproved':
        net = ImprovedUNet(in_nc=opt.n_channel,
                           out_nc=opt.n_channel,
                           n_feature=opt.n_feature)
    else:
        raise ValueError(f'Unknown arch: {opt.arch}')
    return net


def load_base_weights(model: nn.Module, ckpt_path: str):
    state = torch.load(ckpt_path, map_location='cpu')
    # DataParallel checkpoint 지원
    if any(k.startswith('module.') for k in state.keys()):
        new_state = {k.replace('module.', '', 1): v for k, v in state.items()}
        state = new_state
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f'[Warning] Missing keys when loading base model: {missing}')
    if unexpected:
        print(f'[Warning] Unexpected keys when loading base model: {unexpected}')
    print(f'Loaded base weights from {ckpt_path}')


def estimate_intensity_thresholds(
    data_dir: str,
    q1: float = 0.2,
    q2: float = 0.8,
    max_images: int = 50
) -> Tuple[float, float]:
    """
    clean/ 폴더의 grayscale 분포를 기반으로
    [0,1] 스케일에서 intensity quantile t1, t2를 추정.
    """
    clean_paths = sorted(glob.glob(os.path.join(data_dir, 'clean', '*')))[:max_images]
    if len(clean_paths) == 0:
        raise RuntimeError(f'No clean images found in {os.path.join(data_dir, "clean")}')

    all_pixels = []
    for p in clean_paths:
        arr = np.array(Image.open(p), dtype=np.float32) / 255.0  # [0,1]
        all_pixels.append(arr.reshape(-1))
    all_pixels = np.concatenate(all_pixels, axis=0)

    q1 = float(q1)
    q2 = float(q2)
    assert 0.0 < q1 < q2 < 1.0, 'iqsl_q1, iqsl_q2 must satisfy 0 < q1 < q2 < 1.'

    t1 = float(np.quantile(all_pixels, q1))
    t2 = float(np.quantile(all_pixels, q2))
    return t1, t2


def iqsl_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    t1: float,
    t2: float,
    tau: float = 0.1,
    margin: float = 0.0,
    ce_factor: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Intensity-Quantized Structural Loss (IQSL)
    - 3-class (dark/mid/bright) surrogate segmentation 기반
    - multi-class Dice + (optional) CE

    pred, target: [B, C, H, W] or [B, H, W], 0~1, grayscale (C=1 가정)
    """
    # 3D이면 채널 차원 추가
    if pred.dim() == 3:
        pred = pred.unsqueeze(1)   # [B,1,H,W]
    if target.dim() == 3:
        target = target.unsqueeze(1)  # [B,1,H,W]

    assert pred.shape == target.shape, 'pred and target must have the same shape.'
    B, C, H, W = target.shape
    assert C == 1, 'IQSL currently assumes single-channel grayscale input.'

    y_hat = pred        # [B,1,H,W]
    y = target          # [B,1,H,W]

    # 스칼라 intensity 맵 [B,H,W]
    y_s = y[:, 0, :, :]         # [B,H,W]
    y_hat_s = y_hat[:, 0, :, :] # [B,H,W]

    num_classes = 3

    # === valid mask: threshold 주변 margin을 don’t-care로 둘 수 있음 ===
    if margin > 0.0:
        valid = (
            (y_s <= (t1 - margin))
            | ((y_s >= (t1 + margin)) & (y_s <= (t2 - margin)))
            | (y_s >= (t2 + margin))
        ).float()  # [B,H,W]
    else:
        valid = torch.ones_like(y_s)   # [B,H,W]

    # === target one-hot (3-class) ===
    # 0: dark, 1: mid, 2: bright
    dark   = (y_s <= t1).float()              # [B,H,W]
    mid    = ((y_s > t1) & (y_s < t2)).float()
    bright = (y_s >= t2).float()

    # [B,3,H,W]
    target_oh = torch.stack([dark, mid, bright], dim=1)

    # === predicted class probabilities via soft distance to centers ===
    # class centers를 threshold 기반으로 정의
    c0 = t1 / 2.0
    c1 = (t1 + t2) / 2.0
    c2 = (t2 + 1.0) / 2.0
    centers = torch.tensor([c0, c1, c2], device=y.device, dtype=y.dtype).view(
        1, num_classes, 1, 1
    )  # [1,3,1,1]

    # y_hat_s: [B,H,W] → [B,1,H,W]로 맞춰서 broadcast
    y_hat_s_4d = y_hat_s.unsqueeze(1)   # [B,1,H,W]
    dist = torch.abs(y_hat_s_4d - centers)  # [B,3,H,W]

    tau = max(float(tau), 1e-6)
    logits = -dist / tau
    prob = torch.softmax(logits, dim=1)  # [B,3,H,W]

    # valid mask broadcast: [B,1,H,W]
    valid_b = valid.unsqueeze(1)

    prob = prob * valid_b
    target_oh = target_oh * valid_b

    # === Dice loss (multi-class) ===
    inter = (prob * target_oh).sum(dim=(0, 2, 3))   # [3]
    pred_sum = prob.sum(dim=(0, 2, 3))              # [3]
    tgt_sum = target_oh.sum(dim=(0, 2, 3))          # [3]
    dice = (2.0 * inter + eps) / (pred_sum + tgt_sum + eps)
    loss_dice = 1.0 - dice.mean()

    # === Cross-Entropy (soft) ===
    # CE = - sum_i sum_k m_k log p_k / (valid 픽셀 수 * num_classes)
    ce = -(target_oh * torch.log(prob + eps)).sum()
    valid_count = valid_b.sum() * num_classes
    ce = ce / (valid_count + eps)

    total = loss_dice + ce_factor * ce
    return total


def main():
    opt = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_devices
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    systime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')

    # ★ B-domain patch dataset
    train_dataset = DenoisePatchDataset(
        opt.data_dir,
        patch_size=opt.patch_size,
        patches_per_image=opt.patches_per_image
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=opt.batchsize,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    valid = validation_denoise(opt.data_dir)

    # === IQSL용 intensity threshold 사전 추정 ===
    if opt.lambda_iqsl > 0.0:
        t1, t2 = estimate_intensity_thresholds(
            opt.data_dir,
            q1=opt.iqsl_q1,
            q2=opt.iqsl_q2,
            max_images=opt.iqsl_max_images,
        )
        print(f'[IQSL] Estimated thresholds from clean/: t1={t1:.4f}, t2={t2:.4f}')
    else:
        t1, t2 = None, None
        print('[IQSL] lambda_iqsl=0 → IQSL disabled.')

    # Base model + adapter
    base_model = build_base_model(opt)
    load_base_weights(base_model, opt.pretrained_ckpt)
    base_model.to(device)

    model = DenoiserWithAdapter(
        base_model=base_model,
        in_channels=opt.n_channel,
        hidden_channels=opt.adapter_hidden,
        freeze_base=True,
        use_no_grad_for_base=True,
    )
    if opt.parallel:
        model = nn.DataParallel(model)
    model = model.to(device)

    if isinstance(model, nn.DataParallel):
        base = model.module.base
    else:
        base = model.base
    for p in base.parameters():
        p.requires_grad = False
    base.eval()

    # Adapter만 학습
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=opt.lr,
    )
    l1_criterion = nn.L1Loss()

    print(f'==> Start finetuning with adapter + patches + IQSL. '
          f'Num epochs={opt.n_epoch}, batchsize={opt.batchsize}, '
          f'lr={opt.lr}, lambda_grad={opt.lambda_grad}, '
          f'lambda_iqsl={opt.lambda_iqsl}, '
          f'patch_size={opt.patch_size}, '
          f'patches_per_image={opt.patches_per_image}')

    for epoch in range(1, opt.n_epoch + 1):
        model.train()
        if isinstance(model, nn.DataParallel):
            model.module.base.eval()
        else:
            model.base.eval()
        epoch_st = time.time()
        losses_l1 = []
        losses_iqsl = []

        for i, (clean, noisy) in enumerate(train_loader, start=1):
            clean = clean.to(device)   # [B, C, ps, ps], 0~1
            noisy = noisy.to(device)

            optimizer.zero_grad()
            pred = model(noisy)

            loss_l1 = l1_criterion(pred, clean)
            loss_grad = gradient_loss(pred, clean)

            if opt.lambda_iqsl > 0.0:
                loss_iq = iqsl_loss(
                    pred, clean,
                    t1=t1,
                    t2=t2,
                    tau=opt.iqsl_tau,
                    margin=opt.iqsl_margin,
                    ce_factor=opt.iqsl_ce_factor,
                )
            else:
                loss_iq = torch.zeros(1, device=device)

            loss = loss_l1 + opt.lambda_grad * loss_grad + opt.lambda_iqsl * loss_iq

            loss.backward()
            optimizer.step()

            losses_l1.append(loss_l1.item())
            if opt.lambda_iqsl > 0.0:
                losses_iqsl.append(loss_iq.item())

            if i % 10 == 0 or i == len(train_loader):
                if opt.lambda_iqsl > 0.0:
                    print(
                        f'Epoch [{epoch}/{opt.n_epoch}] '
                        f'Iter [{i}/{len(train_loader)}] '
                        f'L1={loss_l1.item():.6f} '
                        f'Grad={loss_grad.item():.6f} '
                        f'IQSL={loss_iq.item():.6f} '
                        f'Total={loss.item():.6f}'
                    )
                else:
                    print(
                        f'Epoch [{epoch}/{opt.n_epoch}] '
                        f'Iter [{i}/{len(train_loader)}] '
                        f'L1={loss_l1.item():.6f} '
                        f'Grad={loss_grad.item():.6f} '
                        f'Total={loss.item():.6f}'
                    )

        mean_l1 = float(np.mean(losses_l1))
        if opt.lambda_iqsl > 0.0 and len(losses_iqsl) > 0:
            mean_iqsl = float(np.mean(losses_iqsl))
        else:
            mean_iqsl = 0.0

        print(f'End of epoch {epoch}, '
              f'mean L1={mean_l1:.6f}, '
              f'mean IQSL={mean_iqsl:.6f}, '
              f'time={time.time() - epoch_st:.2f}s')

        # checkpoint + 간단한 validation (전체 이미지 기준)
        if epoch % opt.save_every == 0 or epoch == opt.n_epoch:
            ckpt_path = checkpoint(model, epoch, opt)

            model.eval()
            save_dir = os.path.join(
                opt.save_model_path, opt.log_name,
                f'val_{systime}_ep{epoch:03d}'
            )
            os.makedirs(save_dir, exist_ok=True)

            with torch.no_grad():
                pbar = tqdm(zip(valid[0], valid[1]),
                            total=len(valid[0]),
                            desc=f'Val ep{epoch}')
                for i, (clean_np, noisy_np) in enumerate(pbar):
                    clean_name = os.path.basename(valid[2][i]).split('.')[0]
                    noisy_name = os.path.basename(valid[3][i]).split('.')[0]

                    noisy_im = noisy_np.astype(np.float32) / 255.0
                    noisy_tensor = transforms.ToTensor()(noisy_im)
                    noisy_tensor = noisy_tensor.unsqueeze(0).to(device)

                    pred = model(noisy_tensor)
                    pred = pred.squeeze(0).permute(1, 2, 0).cpu().numpy()
                    pred255 = np.clip(pred * 255.0 + 0.5, 0, 255).astype(np.uint8)

                    psnr = calculate_psnr(pred255, clean_np)
                    pbar.set_postfix(psnr=f'{psnr:.2f} dB')

                    if i == 0:
                        Image.fromarray(clean_np.astype(np.uint8)).convert("L").save(
                            os.path.join(save_dir, f'{clean_name}_clean.png'))
                        Image.fromarray(noisy_np.astype(np.uint8)).convert("L").save(
                            os.path.join(save_dir, f'{noisy_name}_noisy.png'))
                        if pred255.ndim == 3 and pred255.shape[2] == 1:
                            vis = pred255.squeeze(-1)  # (H,W)
                        else:
                            vis = pred255              # (H,W) or (H,W,3)

                        Image.fromarray(vis).convert("L").save(
                            os.path.join(save_dir, f'{noisy_name}_denoised_ep{epoch:03d}.png'))

    print('Finetuning complete.')


if __name__ == '__main__':
    main()
