from __future__ import annotations
import os
import glob
import argparse

import numpy as np
from PIL import Image

import torch
import torch.nn as nn

from arch_unet import UNet, RESNET, ImprovedUNet


# ============================================================
# 옵션
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data_dir', type=str, required=True,
        help='Root dir with noise/ (and clean/) for inference.'
    )
    parser.add_argument(
        '--base_ckpt', type=str, required=True,
        help='Checkpoint of the base model trained on A-domain.'
    )
    parser.add_argument(
        '--adapter_ckpt', type=str, required=True,
        help='Adapter-only checkpoint (epoch_adapter_only_xxx.pth) for memory adapter.'
    )
    parser.add_argument(
        '--arch', type=str, default='UNetImproved',
        choices=['UNet', 'RESNET', 'UNetImproved'],
        help='Backbone architecture used in base model.'
    )
    parser.add_argument(
        '--save_dir', type=str, default='./results_infer_adapter_memory',
        help='Directory to save denoised images.'
    )
    parser.add_argument('--gpu_devices', default='0', type=str)
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--n_feature', type=int, default=48)
    parser.add_argument('--n_channel', type=int, default=1)
    parser.add_argument('--adapter_hidden', type=int, default=16)

    # === Memory bank / patch inference 관련 ===
    parser.add_argument(
        '--patch_size', type=int, default=128,
        help='Patch size used for training (P x P).'
    )
    parser.add_argument(
        '--overlap', type=int, default=64,
        help='Overlap between neighboring patches during inference.'
    )
    parser.add_argument(
        '--num_memory_images', type=int, default=5,
        help='Number of (clean,noise) image pairs used to build memory bank.'
    )
    parser.add_argument(
        '--memory_stride', type=int, default=64,
        help='Stride when extracting memory patches (<= patch_size).'
    )

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


# ============================================================
# Base model / checkpoint 로딩
# ============================================================

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
    DenoiserWithMemoryAdapter의 adapter 서브모듈에만 로드.
    """
    state = torch.load(ckpt_path, map_location='cpu')

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


# ============================================================
# IQSL-style IoU / PSNR
# ============================================================

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


# ============================================================
# Memory adapter & patch inference
# ============================================================

def extract_patches(img_tensor: torch.Tensor, patch_size: int, stride: int) -> torch.Tensor:
    """
    img_tensor: [C,H,W]  (C=1)
    return: [N, C, P, P]
    """
    C, H, W = img_tensor.shape
    img_b = img_tensor.unsqueeze(0)  # [1,C,H,W]
    patches = torch.nn.functional.unfold(
        img_b, kernel_size=patch_size, stride=stride
    )  # [1, C*P*P, L]
    patches = patches.squeeze(0).transpose(0, 1)  # [L, C*P*P]
    patches = patches.view(-1, C, patch_size, patch_size)
    return patches


def build_memory_bank(
    clean_paths,
    noise_paths,
    patch_size: int,
    stride: int,
    device: torch.device,
):
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
        clean_arr = np.array(Image.open(cp), dtype=np.float32)   # 0~255
        noise_arr = np.array(Image.open(npth), dtype=np.float32)

        clean_arr = (clean_arr / 255.0).astype(np.float32)       # 0~1
        noise_arr = (noise_arr / 255.0).astype(np.float32)

        clean_t = torch.from_numpy(clean_arr).unsqueeze(0)       # [1,H,W]
        noise_t = torch.from_numpy(noise_arr).unsqueeze(0)

        clean_p = extract_patches(clean_t, patch_size, stride)   # [N,C,P,P]
        noise_p = extract_patches(noise_t, patch_size, stride)
        assert clean_p.shape == noise_p.shape

        all_clean.append(clean_p)
        all_noise.append(noise_p)

    memory_clean = torch.cat(all_clean, dim=0).to(device)
    memory_noise = torch.cat(all_noise, dim=0).to(device)

    print(f'[MemoryBank] #patches={memory_clean.shape[0]}, '
          f'patch_size={patch_size}, stride={stride}')
    return memory_noise, memory_clean


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
    - base: frozen base denoiser
    - adapter: MemoryResidualAdapter
    - memory_noise_bank / memory_clean_bank: [N_mem,C,P,P]
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

        noisy_flat = noisy.detach().reshape(B, -1)           # [B,D]
        mem_flat = self.memory_noise_bank.view(N, -1)     # [N,D]

        a2 = (noisy_flat.pow(2).sum(dim=1, keepdim=True))     # [B,1]
        b2 = (mem_flat.pow(2).sum(dim=1, keepdim=True)).t()   # [1,N]
        ab = noisy_flat @ mem_flat.t()                        # [B,N]
        dists = a2 + b2 - 2.0 * ab                            # [B,N]

        idx = dists.argmin(dim=1)                             # [B]
        mem_clean_selected = self.memory_clean_bank[idx]      # [B,C,P,P]
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

    win_1d = torch.hann_window(ps, periodic=False, device=device).view(1, 1, ps, 1)
    win_2d = win_1d * win_1d.transpose(2, 3)  # [1,1,ps,ps]
    win_2d = win_2d.clamp_min(1e-3)

    output = torch.zeros_like(noisy_tensor)
    weight = torch.zeros_like(noisy_tensor)

    for y in ys:
        for x in xs:
            patch = noisy_tensor[:, :, y:y+ps, x:x+ps]      # [1,1,ps,ps]
            pred_patch = model(patch)                       # [1,1,ps,ps]
            w = win_2d
            output[:, :, y:y+ps, x:x+ps] += pred_patch * w
            weight[:, :, y:y+ps, x:x+ps] += w

    output = output / (weight + 1e-8)
    pred = output.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H,W,1), 0~1
    return pred


# ============================================================
# main
# ============================================================

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
    else:
        clean_paths = []
        print('[Warning] clean/ not found. PSNR / IQ IoU will not be computed.')

    print(f'Found {len(noise_paths)} noisy images for inference.')

    # === base model ===
    base_model = build_base_model(opt.arch, opt.n_channel, opt.n_feature)
    load_base_weights(base_model, opt.base_ckpt)
    base_model.to(device)

    # === memory bank ===
    if not has_clean:
        raise RuntimeError('Memory bank needs clean/ and noise/ pairs; clean/ not found.')
    num_mem = min(opt.num_memory_images, len(clean_paths))
    mem_clean_paths = clean_paths[:num_mem]
    mem_noise_paths = noise_paths[:num_mem]

    memory_noise, memory_clean = build_memory_bank(
        mem_clean_paths,
        mem_noise_paths,
        patch_size=opt.patch_size,
        stride=opt.memory_stride,
        device=device,
    )

    # === memory adapter wrapper ===
    model = DenoiserWithMemoryAdapter(
        base_model=base_model,
        in_channels=opt.n_channel,
        hidden_channels=opt.adapter_hidden,
        memory_noise_bank=memory_noise,
        memory_clean_bank=memory_clean,
        freeze_base=True,
        use_no_grad_for_base=True,
    )

    if opt.parallel:
        model = nn.DataParallel(model)

    model.to(device)
    model.eval()

    # === adapter-only ckpt 로드 ===
    load_adapter_weights(model, opt.adapter_ckpt)

    iou_dark_list, iou_mid_list, iou_bright_list = [], [], []

    with torch.no_grad():
        for idx, n_path in enumerate(noise_paths):
            name = os.path.basename(n_path)
            base_name = os.path.splitext(name)[0]

            noisy_img = np.array(Image.open(n_path), dtype=np.float32)   # 0~255
            pred = denoise_full_image_patchwise(
                model,
                noisy_img,
                device,
                patch_size=opt.patch_size,
                overlap=opt.overlap,
            )  # (H,W,1), 0~1

            pred255 = np.clip(pred * 255.0 + 0.5, 0, 255).astype(np.uint8)

            if pred255.ndim == 3 and pred255.shape[2] == 1:
                out_img = Image.fromarray(pred255.squeeze(-1)).convert('L')
            else:
                out_img = Image.fromarray(pred255).convert('L')

            save_path = os.path.join(opt.save_dir, f'{base_name}_denoised_mem.png')
            out_img.save(save_path)

            if has_clean and idx < len(clean_paths):
                clean_img = np.array(Image.open(clean_paths[idx]), dtype=np.float32)
                psnr = calculate_psnr(pred255.squeeze(-1), clean_img)

                msg = (f'[{idx+1:03d}/{len(noise_paths):03d}] '
                       f'{name} → PSNR={psnr:.2f} dB, saved to {save_path}')

                if opt.compute_iq_iou:
                    ious = compute_iq_iou(
                        pred255.squeeze(-1), clean_img,
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

    print('Inference with memory adapter model finished.')


if __name__ == '__main__':
    main()
