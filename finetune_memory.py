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
        '--log_name', type=str, default='UNetImproved_memory_adapter_ft',
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
        help='Hidden channels of the memory-conditioned adapter CNN.'
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

    # === 메모리 뱅크(Noise/Clean) 관련 설정 ===
    parser.add_argument(
        '--num_memory_images', type=int, default=5,
        help='How many (clean,noise) image pairs to use as memory bank.'
    )
    parser.add_argument(
        '--memory_stride', type=int, default=64,
        help='Stride for extracting memory patches (<= patch_size).'
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
    DenoiserWithMemoryAdapter에서 adapter 부분만 저장하는 checkpoint 함수.
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

        print(f'B-domain: {len(self.clean)} images, '
              f'{self.patches_per_image} patches/image/epoch → '
              f'{len(self)*1} samples/epoch.')

    def __len__(self) -> int:
        return len(self.clean) * self.patches_per_image

    def _load_pair(self, img_idx: int):
        clean_path, noise_path = self.clean[img_idx], self.noise[img_idx]
        # float32 0~255
        clean_img = np.array(Image.open(clean_path), dtype=np.float32)
        noise_img = np.array(Image.open(noise_path), dtype=np.float32)
        return clean_img, noise_img

    def __getitem__(self, index: int):
        img_idx = index // self.patches_per_image
        clean_img, noise_img = self._load_pair(img_idx)

        h, w = clean_img.shape[:2]
        ps = self.patch_size
        assert h >= ps and w >= ps, \
            f'Image size ({h},{w}) smaller than patch_size {ps}. ' \
            f'이미지가 더 크거나 patch_size를 줄여주세요.'

        top = np.random.randint(0, h - ps + 1)
        left = np.random.randint(0, w - ps + 1)
        clean_patch = clean_img[top:top + ps, left:left + ps]
        noise_patch = noise_img[top:top + ps, left:left + ps]

        # ★ 명시적으로 0~1 스케일로 맞춤 (H,W)
        clean_patch = (clean_patch / 255.0).astype(np.float32)
        noise_patch = (noise_patch / 255.0).astype(np.float32)

        # (H,W) → (1,H,W)
        clean_tensor = torch.from_numpy(clean_patch).unsqueeze(0)
        noise_tensor = torch.from_numpy(noise_patch).unsqueeze(0)

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
        arr = np.array(Image.open(p), dtype=np.float32)  # 0~255
        arr = (arr / 255.0).astype(np.float32)           # 0~1
        all_pixels.append(arr.reshape(-1))
    all_pixels = np.concatenate(all_pixels, axis=0)

    q1 = float(q1)
    q2 = float(q2)
    assert 0.0 < q1 < q2 < 1.0, 'iqsl_q1, iqsl_q2 must satisfy 0 < q1 < q2 < 1.'

    t1 = float(np.quantile(all_pixels, q1))
    t2 = float(np.quantile(all_pixels, q2))
    return t1, t2


def denoise_full_image_patchwise(
    model: nn.Module,
    noisy_np: np.ndarray,
    device: torch.device,
    patch_size: int,
    overlap: int = 64,
) -> np.ndarray:
    """
    전체 noisy 이미지를 patch 단위로 inference 후 다시 합치는 함수.

    noisy_np : (H, W) 또는 (H, W, 1), 값 범위는 0~255
    return: (H, W, 1) float32, 0~1 스케일
    """
    if noisy_np.ndim == 3 and noisy_np.shape[2] == 1:
        noisy_np = noisy_np[..., 0]

    noisy_arr = noisy_np.astype(np.float32) / 255.0   # 0~1
    # (H,W) → [1,1,H,W]
    noisy_tensor = torch.from_numpy(noisy_arr).unsqueeze(0).unsqueeze(0).to(device)

    _, _, H, W = noisy_tensor.shape
    ps = patch_size
    assert H >= ps and W >= ps, f"Image ({H},{W}) smaller than patch_size {ps}"
    assert overlap < ps, "overlap must be smaller than patch_size"

    step = ps - overlap

    ys = list(range(0, max(H - ps, 0) + 1, step))
    xs = list(range(0, max(W - ps, 0) + 1, step))
    if ys[-1] != H - ps:
        ys.append(H - ps)
    if xs[-1] != W - ps:
        xs.append(W - ps)
    ys = sorted(set(int(y) for y in ys))
    xs = sorted(set(int(x) for x in xs))

    # Hann window 기반 부드러운 블렌딩
    win_1d = torch.hann_window(ps, periodic=False, device=device).view(1, 1, ps, 1)
    win_2d = win_1d * win_1d.transpose(2, 3)  # [1,1,ps,ps]
    win_2d = win_2d.clamp_min(1e-3)

    output = torch.zeros_like(noisy_tensor)
    weight = torch.zeros_like(noisy_tensor)

    for y in ys:
        for x in xs:
            patch = noisy_tensor[:, :, y:y+ps, x:x+ps]
            pred_patch = model(patch)               # [1,1,ps,ps]
            w = win_2d
            output[:, :, y:y+ps, x:x+ps] += pred_patch * w
            weight[:, :, y:y+ps, x:x+ps] += w

    output = output / (weight + 1e-8)
    pred = output.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H,W,1) 0~1
    return pred


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

    pred, target: [B, C, H, W] or [B, H, W]
    """
    if pred.dim() == 3:
        pred = pred.unsqueeze(1)
    if target.dim() == 3:
        target = target.unsqueeze(1)

    assert pred.shape == target.shape, 'pred and target must have the same shape.'
    B, C, H, W = target.shape
    assert C == 1, 'IQSL currently assumes single-channel grayscale input.'

    y_hat = pred
    y = target

    y_s = y[:, 0, :, :]
    y_hat_s = y_hat[:, 0, :, :]

    num_classes = 3

    if margin > 0.0:
        valid = (
            (y_s <= (t1 - margin))
            | ((y_s >= (t1 + margin)) & (y_s <= (t2 - margin)))
            | (y_s >= (t2 + margin))
        ).float()
    else:
        valid = torch.ones_like(y_s)

    dark   = (y_s <= t1).float()
    mid    = ((y_s > t1) & (y_s < t2)).float()
    bright = (y_s >= t2).float()

    target_oh = torch.stack([dark, mid, bright], dim=1)  # [B,3,H,W]

    c0 = t1 / 2.0
    c1 = (t1 + t2) / 2.0
    c2 = (t2 + 1.0) / 2.0
    centers = torch.tensor([c0, c1, c2], device=y.device, dtype=y.dtype).view(
        1, num_classes, 1, 1
    )

    y_hat_s_4d = y_hat_s.unsqueeze(1)
    dist = torch.abs(y_hat_s_4d - centers)

    tau = max(float(tau), 1e-6)
    logits = -dist / tau
    prob = torch.softmax(logits, dim=1)

    valid_b = valid.unsqueeze(1)
    prob = prob * valid_b
    target_oh = target_oh * valid_b

    inter = (prob * target_oh).sum(dim=(0, 2, 3))
    pred_sum = prob.sum(dim=(0, 2, 3))
    tgt_sum = target_oh.sum(dim=(0, 2, 3))
    dice = (2.0 * inter + eps) / (pred_sum + tgt_sum + eps)
    loss_dice = 1.0 - dice.mean()

    ce = -(target_oh * torch.log(prob + eps)).sum()
    valid_count = valid_b.sum() * num_classes
    ce = ce / (valid_count + eps)

    total = loss_dice + ce_factor * ce
    return total


# ============================================================
# 메모리 뱅크 + Memory-conditioned Adapter
# ============================================================

def extract_patches(img_tensor: torch.Tensor, patch_size: int, stride: int) -> torch.Tensor:
    """
    img_tensor: [C,H,W]  (여기서는 C=1)
    return: [N_patches, C, patch_size, patch_size]
    """
    C, H, W = img_tensor.shape
    img_b = img_tensor.unsqueeze(0)  # [1,C,H,W]
    patches = F.unfold(img_b, kernel_size=patch_size, stride=stride)  # [1, C*P*P, L]
    patches = patches.squeeze(0).transpose(0, 1)  # [L, C*P*P]
    patches = patches.view(-1, C, patch_size, patch_size)
    return patches


def build_memory_bank(
    clean_paths: List[str],
    noise_paths: List[str],
    patch_size: int,
    stride: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    clean/noise image 경로 리스트에서 patch를 추출하여 메모리 뱅크 구성.
    return:
        memory_noise: [N_mem, C, P, P]
        memory_clean: [N_mem, C, P, P]
    """
    assert len(clean_paths) == len(noise_paths) and len(clean_paths) > 0
    all_clean = []
    all_noise = []

    for cp, npth in zip(clean_paths, noise_paths):
        clean_arr = np.array(Image.open(cp), dtype=np.float32)  # 0~255
        noise_arr = np.array(Image.open(npth), dtype=np.float32)

        clean_arr = (clean_arr / 255.0).astype(np.float32)      # 0~1
        noise_arr = (noise_arr / 255.0).astype(np.float32)

        clean_t = torch.from_numpy(clean_arr).unsqueeze(0)      # [1,H,W]
        noise_t = torch.from_numpy(noise_arr).unsqueeze(0)

        clean_p = extract_patches(clean_t, patch_size, stride)  # [N,C,P,P]
        noise_p = extract_patches(noise_t, patch_size, stride)
        assert clean_p.shape == noise_p.shape

        all_clean.append(clean_p)
        all_noise.append(noise_p)

    memory_clean = torch.cat(all_clean, dim=0).to(device)
    memory_noise = torch.cat(all_noise, dim=0).to(device)

    print(f'[MemoryBank] #clean patches={memory_clean.shape[0]}, '
          f'patch_size={patch_size}, stride={stride}')
    return memory_noise, memory_clean



class MemoryConditionedAdapter(nn.Module):
    """
    base denoiser output과 memory clean patch를 함께 받아 residual을 예측하는 작은 CNN.

    입력 채널: [noisy, base_out, mem_clean] → 3 * in_channels
    """
    def __init__(self, in_channels: int = 1, hidden_channels: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3 * in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=3, padding=1),
        )

    def forward(
        self,
        noisy: torch.Tensor,
        base_out: torch.Tensor,
        mem_clean: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([noisy, base_out, mem_clean], dim=1)
        delta = self.net(x)
        return base_out + delta


class MemoryResidualAdapter(nn.Module):
    """
    Memory-conditioned residual adapter.

    입력: noisy, base_out, mem_clean  (모두 [B,C,H,W])
    출력: base_out + small residual(noisy, base_out, mem_clean)

    마지막 conv를 0으로 초기화해서 초기에는 정확히 base_out을 그대로 내보냄.
    """
    def __init__(self, in_channels: int = 1, hidden_channels: int = 16):
        super().__init__()
        C = in_channels
        H = hidden_channels

        self.body = nn.Sequential(
            nn.Conv2d(3 * C, H, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(H, H, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(H, C, kernel_size=3, padding=1, bias=True),
        )

        # ★ 마지막 conv를 0으로 초기화 → 초기엔 delta=0 → out=base_out
        last_conv = self.body[-1]
        nn.init.zeros_(last_conv.weight)
        nn.init.zeros_(last_conv.bias)

    def forward(
        self,
        noisy: torch.Tensor,
        base_out: torch.Tensor,
        mem_clean: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([noisy, base_out, mem_clean], dim=1)  # [B,3C,H,W]
        delta = self.body(x)
        return base_out + delta




class DenoiserWithMemoryAdapter(nn.Module):
    """
    - base_model: 미리 학습된 denoiser (freeze)
    - adapter: MemoryConditionedAdapter
    - memory_noise_bank / memory_clean_bank: [N_mem,C,P,P] (no grad)
    """
    def __init__(
        self,
        base_model: nn.Module,
        in_channels: int,
        hidden_channels: int,
        memory_noise_bank: torch.Tensor,
        memory_clean_bank: torch.Tensor,
        freeze_base: bool = True,
        use_no_grad_for_base: bool = True,
    ):
        super().__init__()
        self.base = base_model
        # self.adapter = MemoryConditionedAdapter(in_channels, hidden_channels)
        self.adapter = MemoryResidualAdapter(in_channels, hidden_channels)

        self.use_no_grad_for_base = use_no_grad_for_base

        if freeze_base:
            for p in self.base.parameters():
                p.requires_grad = False

        self.register_buffer("memory_noise_bank", memory_noise_bank)
        self.register_buffer("memory_clean_bank", memory_clean_bank)

    @torch.no_grad()
    def _select_memory_patch(self, noisy: torch.Tensor) -> torch.Tensor:
        """
        noisy: [B,C,P,P]
        memory_noise_bank: [N,C,P,P]
        return: mem_clean_selected: [B,C,P,P]
        """
        B = noisy.shape[0]
        N = self.memory_noise_bank.shape[0]

        noisy_flat = noisy.detach().reshape(B, -1)                 # [B,D]
        mem_flat = self.memory_noise_bank.view(N, -1)           # [N,D]

        # (a-b)^2 = a^2 + b^2 - 2ab
        a2 = (noisy_flat.pow(2).sum(dim=1, keepdim=True))       # [B,1]
        b2 = (mem_flat.pow(2).sum(dim=1, keepdim=True)).t()     # [1,N]
        ab = noisy_flat @ mem_flat.t()                          # [B,N]
        dists = a2 + b2 - 2.0 * ab                              # [B,N]

        idx = dists.argmin(dim=1)                               # [B]
        mem_clean_selected = self.memory_clean_bank[idx]        # [B,C,P,P]
        return mem_clean_selected

    def forward(self, noisy: torch.Tensor) -> torch.Tensor:
        """
        noisy: [B,C,P,P]
        """
        if self.use_no_grad_for_base:
            with torch.no_grad():
                base_out = self.base(noisy)
        else:
            base_out = self.base(noisy)

        mem_clean = self._select_memory_patch(noisy)
        out = self.adapter(noisy, base_out, mem_clean)
        return out


# ============================================================
# 메인 학습 루프
# ============================================================

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
        print(f'[IQSL] Estimated thresholds from clean/: t1={t1:.6f}, t2={t2:.6f}')
    else:
        t1, t2 = None, None
        print('[IQSL] lambda_iqsl=0 → IQSL disabled.')

    # === 메모리 뱅크에 사용할 이미지 경로 ===
    clean_all = sorted(glob.glob(os.path.join(opt.data_dir, 'clean', '*')))
    noise_all = sorted(glob.glob(os.path.join(opt.data_dir, 'noise', '*')))
    assert len(clean_all) == len(noise_all) and len(clean_all) > 0, \
        'clean/ and noise/ must be non-empty and match in length.'

    num_mem = min(opt.num_memory_images, len(clean_all))
    clean_mem = clean_all[:num_mem]
    noise_mem = noise_all[:num_mem]

    # === 메모리 뱅크 구축 ===
    memory_noise_bank, memory_clean_bank = build_memory_bank(
        clean_mem,
        noise_mem,
        patch_size=opt.patch_size,
        stride=opt.memory_stride,
        device=device,
    )

    # === Base model + memory adapter wrapper ===
    base_model = build_base_model(opt)
    load_base_weights(base_model, opt.pretrained_ckpt)
    base_model.to(device)

    model = DenoiserWithMemoryAdapter(
        base_model=base_model,
        in_channels=opt.n_channel,
        hidden_channels=opt.adapter_hidden,
        memory_noise_bank=memory_noise_bank,
        memory_clean_bank=memory_clean_bank,
        freeze_base=True,
        use_no_grad_for_base=True,
    )

    if opt.parallel:
        model = nn.DataParallel(model)
    model = model.to(device)

    # base freeze 확인
    if isinstance(model, nn.DataParallel):
        base = model.module.base
    else:
        base = model.base
    for p in base.parameters():
        p.requires_grad = False
    base.eval()

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=opt.lr,
    )
    l1_criterion = nn.L1Loss()

    print(f'==> Start finetuning with MEMORY adapter + patches + IQSL. '
          f'Num epochs={opt.n_epoch}, batchsize={opt.batchsize}, '
          f'lr={opt.lr}, lambda_grad={opt.lambda_grad}, '
          f'lambda_iqsl={opt.lambda_iqsl}, '
          f'patch_size={opt.patch_size}, '
          f'patches_per_image={opt.patches_per_image}, '
          f'num_memory_images={num_mem}, '
          f'memory_stride={opt.memory_stride}')

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
            clean = clean.to(device)
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

        # checkpoint + patch-wise inference 기반 validation
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

                    # ---- patchify + 재조합 inference ----
                    pred = denoise_full_image_patchwise(
                        model,
                        noisy_np,               # (H,W)
                        device,
                        patch_size=opt.patch_size,
                        overlap=opt.patch_size // 2,   # 예: 128 → overlap=64
                    )  # (H,W,1)

                    # 스케일을 원래 0~255 uint8로 환산
                    pred255 = np.clip(pred * 255.0 + 0.5, 0, 255).astype(np.uint8)

                    # PSNR은 full 352×352 기준으로 계산
                    psnr = calculate_psnr(pred255.squeeze(-1), clean_np)
                    pbar.set_postfix(psnr=f'{psnr:.2f} dB')

                    # 첫 번째 샘플은 clean/noise/denoised 이미지 저장
                    if i == 0:
                        # clean / noise 저장 (352x352)
                        Image.fromarray(clean_np.astype(np.uint8)).convert("L").save(
                            os.path.join(save_dir, f'{clean_name}_clean.png'))
                        Image.fromarray(noisy_np.astype(np.uint8)).convert("L").save(
                            os.path.join(save_dir, f'{noisy_name}_noisy.png'))

                        if pred255.ndim == 3 and pred255.shape[2] == 1:
                            vis = pred255.squeeze(-1)  # (H,W)
                        else:
                            vis = pred255              # (H,W) or (H,W,3)

                        Image.fromarray(vis).convert("L").save(
                            os.path.join(
                                save_dir,
                                f'{noisy_name}_denoised_full_ep{epoch:03d}.png'
                            )
                        )

    print('Finetuning (memory adapter) complete.')


if __name__ == '__main__':
    main()
