from __future__ import annotations
import os
import glob
import math
import argparse

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    patches = patches.reshape(-1, C, patch_size, patch_size)
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


class LowFrequencyBlendAdapter(nn.Module):
    """
    Memory-conditioned low-frequency blend adapter.

    - 구조(에지, 라인)는 base_out의 고주파(HP_base)에서만 오게 강제
    - memory clean patch는 저주파(LP_mem)만 사용
    - 출력: out = HP_base + LP_out
      where LP_out = (1-g)*LP_base + g*LP_mem, g ∈ [0,1]

    입력: noisy, base_out, mem_clean  (모두 [B,C,H,W], 0~1 스케일 가정)
    출력: out (same shape, 0~1 근처, 마지막에서 clamp 가능)
    """
    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 16,
        kernel_size: int = 9,
        sigma: float = 3.0,
        clamp_output: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.clamp_output = clamp_output

        # --- 고정 Gaussian low-pass kernel 생성 (depthwise conv용) ---
        k = kernel_size
        assert k % 2 == 1, "kernel_size must be odd."

        ax = torch.arange(k, dtype=torch.float32) - (k - 1) / 2.0
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        kernel = torch.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
        kernel = kernel / kernel.sum()  # normalize to sum=1

        # [1,1,k,k] → [C,1,k,k], depthwise conv용
        kernel = kernel.reshape(1, 1, k, k)
        kernel = kernel.repeat(in_channels, 1, 1, 1)  # [C,1,k,k]

        # register_buffer로 학습/업데이트 안 되는 고정 kernel 등록
        self.register_buffer("gauss_kernel", kernel)

        # --- 게이트 네트워크: 저주파만 보고 g(x,y) ∈ [0,1] 추정 ---
        # 입력 채널: LP_noisy, LP_base, LP_mem → 총 3*C
        C = in_channels
        Hc = hidden_channels

        self.gate_net = nn.Sequential(
            nn.Conv2d(3 * C, Hc, kernel_size=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(Hc, C, kernel_size=1, padding=0, bias=True),
            nn.Sigmoid(),  # g in [0,1]
        )

        # 초기에는 g ≈ 0 (LP_base를 거의 그대로 따르도록) 만들기 위한 초기화
        last = self.gate_net[-2]  # 마지막 Conv2d (Sigmoid 앞)
        nn.init.zeros_(last.weight)
        nn.init.constant_(last.bias, -2.0)  # sigmoid(-2) ≈ 0.12

    def _lowpass(self, x: torch.Tensor) -> torch.Tensor:
        """
        고정 Gaussian kernel을 이용한 depthwise low-pass filtering.
        x: [B,C,H,W]
        return: [B,C,H,W]
        """
        C = x.shape[1]
        k = self.kernel_size
        pad = k // 2
        # self.gauss_kernel: [C,1,k,k], groups=C depthwise conv
        return F.conv2d(x, self.gauss_kernel, padding=pad, groups=C)

    def forward(
        self,
        noisy: torch.Tensor,
        base_out: torch.Tensor,
        mem_clean: torch.Tensor,
    ) -> torch.Tensor:
        """
        noisy, base_out, mem_clean: [B,C,H,W], 0~1 스케일 가정
        """
        # --- 1) low-pass / high-pass 분해 ---
        LP_noisy = self._lowpass(noisy)
        LP_base  = self._lowpass(base_out)
        LP_mem   = self._lowpass(mem_clean)

        HP_base = base_out - LP_base  # 구조(고주파)는 base_out에서만

        # --- 2) 저주파들만 concat해서 gate g(x,y) 추정 ---
        # [B,3C,H,W]
        gate_input = torch.cat([LP_noisy, LP_base, LP_mem], dim=1)
        g = self.gate_net(gate_input)  # [B,C,H,W], ∈[0,1]

        # --- 3) 저주파 convex combination & 재조합 ---
        LP_out = (1.0 - g) * LP_base + g * LP_mem  # [B,C,H,W]
        out = HP_base + LP_out                     # [B,C,H,W]

        if self.clamp_output:
            out = torch.clamp(out, 0.0, 1.0)

        return out


class GlobalMonotoneToneCurveAdapter(nn.Module):
    """
    Global monotone tone-curve adapter.

    - 입력: noisy, base_out, mem_clean  (모두 [B,C,H,W], 0~1 스케일 가정)
    - 출력: out = f_theta(base_out),  f_theta : [0,1] -> [0,1] 단조 증가 전역 톤 커브
      * 모든 픽셀에 대해 동일한 스칼라 함수 적용 (공간 구조 불변)
      * f_theta는 noisy / mem_clean의 전역 통계(mean/std)에 의해 결정됨

    구현:
    - [0,1] 구간을 균일한 K개의 control point x_0..x_{K-1} (x_0=0, x_{K-1}=1)로 나누고,
    - 각 구간의 slope m_j >= 0 를 softplus로 파라미터화
    - y_0 = 0,   y_{j+1} = y_j + m_j * Δx 를 누적한 뒤
      전체를 y_{K-1}로 나눠서 y_{K-1} = 1 이 되도록 정규화 → 0 = y_0 < ... < y_{K-1} = 1
    - base_out 값을 [0,1]로 clamp 후, piecewise linear interpolation 으로 f_theta(base_out)를 계산
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_dim: int = 16,
        num_ctrl_points: int = 8,
        clamp_output: bool = True,
    ):
        super().__init__()
        assert num_ctrl_points >= 2, "num_ctrl_points must be >= 2"
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_ctrl_points = num_ctrl_points
        self.clamp_output = clamp_output

        # === control point grid (x_k) : [0, 1] 상 균일 분할 ===
        # x_k = k / (K-1), k=0..K-1
        K = num_ctrl_points
        x_vals = torch.linspace(0.0, 1.0, K, dtype=torch.float32).reshape(1, K)  # [1,K]
        self.register_buffer("x_vals", x_vals)  # gradient 없음, 고정 그리드

        # === 전역 통계 → slopes 로 가는 작은 MLP ===
        # 특징: noisy/base_out/mem_clean 의 전역 mean/std (각각 2개씩, 총 6차원)
        in_feat_dim = 6  # [mean_n, std_n, mean_b, std_b, mean_m, std_m]

        self.mlp = nn.Sequential(
            nn.Linear(in_feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, K - 1),  # 각 구간 slope parameter (raw) K-1개
        )

        # 초기화를 identity tone-curve 근처로 맞추기
        # - 첫 Linear는 입력에 거의 의존 안 하도록 0 초기화
        # - 두 번째 Linear의 bias를 "softplus^{-1}(1)" 로 초기화 → slope ≈ 1
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)

        # softplus^{-1}(1) ≈ 0.5413
        init_slope = 1.0
        raw_bias = math.log(math.exp(init_slope) - 1.0)
        last_linear = self.mlp[-1]
        nn.init.constant_(last_linear.bias, raw_bias)

    @staticmethod
    def _global_stats(x: torch.Tensor):
        """
        x: [B,C,H,W]
        return: mean, std  (둘 다 [B])
        - 채널까지 전역으로 평균/표준편차를 계산 (C>1이어도 전체를 하나로 봄)
        """
        B = x.shape[0]
        # flatten over C,H,W
        x_flat = x.reshape(B, -1)
        mean = x_flat.mean(dim=1)          # [B]
        std = x_flat.std(dim=1)            # [B]
        return mean, std

    def _build_tone_curve(self, noisy, base_out, mem_clean):
        """
        noisy, base_out, mem_clean: [B,C,H,W]
        return:
            y_vals: [B, K]  (각 배치별 tone curve의 y_k 값)
        """
        B = noisy.shape[0]
        K = self.num_ctrl_points
        device = noisy.device
        dtype = noisy.dtype

        # --- 전역 통계 추출 ---
        mean_n, std_n = self._global_stats(noisy)
        mean_b, std_b = self._global_stats(base_out)
        mean_m, std_m = self._global_stats(mem_clean)

        # [B,6] 피처 벡터
        feats = torch.stack(
            [mean_n, std_n, mean_b, std_b, mean_m, std_m],
            dim=1
        )  # [B,6]

        # --- MLP로 raw slopes 예측 ---
        raw_slopes = self.mlp(feats)               # [B, K-1]
        # 양수 slope로 만들기 (단조 증가의 sufficient condition)
        slopes = F.softplus(raw_slopes) + 1e-4     # [B, K-1], >0

        # --- 누적해서 monotonically increasing control point y_k 구성 ---
        # Δx = 1 / (K-1)
        delta_x = 1.0 / (K - 1)
        # 각 구간에서의 Δy = m_j * Δx
        delta_y = slopes * delta_x                 # [B, K-1]

        # y_0 = 0, y_j = sum_{i<j} Δy_i
        y0 = torch.zeros(B, 1, device=device, dtype=dtype)           # [B,1]
        y_rest = torch.cumsum(delta_y, dim=1)                         # [B, K-1]
        y_unscaled = torch.cat([y0, y_rest], dim=1)                  # [B, K]

        # 마지막 점 y_{K-1}를 1로 맞추기 위해 정규화
        y_end = y_unscaled[:, -1:].clamp_min(1e-6)   # [B,1]
        y_vals = y_unscaled / y_end                  # [B,K], 0 = y_0 < ... < y_{K-1} = 1

        return y_vals  # [B,K]

    def _apply_tone_curve(self, base_out, y_vals):
        """
        base_out: [B,C,H,W], 0~1 스케일
        y_vals:   [B,K],     control point에서의 f(x_k) 값
        return:   out: [B,C,H,W]
        """
        B, C, H, W = base_out.shape
        K = self.num_ctrl_points
        device = base_out.device
        dtype = base_out.dtype

        # x grid: [1,K] → [B,1,1,1,K]
        x_grid = self.x_vals.to(device=device, dtype=dtype)  # [1,K]
        x_grid = x_grid.reshape(1, 1, 1, 1, K)                  # [1,1,1,1,K]

        # y_vals: [B,K] → [B,1,1,1,K] for broadcasting
        y_grid = y_vals.reshape(B, 1, 1, 1, K)                  # [B,1,1,1,K]

        # base_out를 [0,1] 범위로 clamp 후 control point index로 변환
        x = base_out.clamp(0.0, 1.0)                         # [B,C,H,W]
        # pos in [0, K-1]
        pos = x * (K - 1)                                    # [B,C,H,W]
        pos = pos.clamp(0.0, K - 1 - 1e-6)

        idx0 = pos.floor().long()                            # [B,C,H,W]
        idx1 = (idx0 + 1).clamp(max=K - 1)                   # [B,C,H,W]
        t = (pos - idx0.float())                             # [B,C,H,W]  fractional

        # gather y0, y1: control point 값
        idx0_exp = idx0.unsqueeze(-1)                        # [B,C,H,W,1]
        idx1_exp = idx1.unsqueeze(-1)                        # [B,C,H,W,1]

        # y_grid: [B,1,1,1,K] → [B,C,H,W,K] (broadcast)
        y_grid_exp = y_grid.expand(B, C, H, W, K)            # [B,C,H,W,K]

        y0 = torch.gather(y_grid_exp, dim=-1, index=idx0_exp).squeeze(-1)  # [B,C,H,W]
        y1 = torch.gather(y_grid_exp, dim=-1, index=idx1_exp).squeeze(-1)  # [B,C,H,W]

        # 최종 선형 보간
        out = y0 + (y1 - y0) * t                             # [B,C,H,W]
        return out

    def forward(
        self,
        noisy: torch.Tensor,
        base_out: torch.Tensor,
        mem_clean: torch.Tensor,
    ) -> torch.Tensor:
        """
        noisy, base_out, mem_clean: [B,C,H,W], 0~1 스케일 가정
        출력: base_out에 전역 단조 톤 커브 f_theta를 적용한 결과
        """
        # 1) 톤 커브 control point y_k 계산
        y_vals = self._build_tone_curve(noisy, base_out, mem_clean)  # [B,K]

        # 2) base_out에 piecewise-linear tone curve 적용
        out = self._apply_tone_curve(base_out, y_vals)

        if self.clamp_output:
            out = torch.clamp(out, 0.0, 1.0)

        return out


class HyperGatedResidualAdapter(nn.Module):
    """
    Hyper-gated residual adapter.

    - 공간 CNN은 noisy + base_out 만 본다.
    - mem_clean 은 전역 mean/std 로 축약된 뒤, 작은 MLP가
      per-channel gate gamma, bias beta 를 생성하는 hyper-network 역할만 한다.
    - 최종 출력: out = base_out + gamma * r(noisy, base_out) + beta

    이 구조에서는 mem_clean 의 spatial 패턴이
    output 으로 직접 복사될 수 있는 경로가 없다.
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 16,
        clamp_output: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.clamp_output = clamp_output

        C = in_channels
        Hc = hidden_channels

        # --- 로컬 residual CNN: noisy + base_out 만 사용 ---
        # 입력 채널: noisy, base_out → 2C
        self.local_net = nn.Sequential(
            nn.Conv2d(2 * C, Hc, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(Hc, Hc, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(Hc, C, kernel_size=3, padding=1, bias=True),
        )

        # 마지막 conv 를 0으로 초기화 → 초기에는 r ≈ 0
        last = self.local_net[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

        # --- 전역 hyper-network: (mean/std) → gamma, beta ---
        # feature: [mean_n, std_n, mean_b, std_b, mean_m, std_m] → dim=6
        in_feat_dim = 6
        self.hyper_mlp = nn.Sequential(
            nn.Linear(in_feat_dim, Hc),
            nn.ReLU(inplace=True),
            nn.Linear(Hc, 2 * C),   # gamma_raw, beta_raw
        )

        # hyper-network 를 "almost identity" 로 초기화
        for m in self.hyper_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)

        # gamma_raw bias 를 sigmoid^{-1}(0.0) 근처로 두면 gamma ≈ 0,
        # beta_raw 는 tanh 로 작은 offset → 둘 다 0 근처에서 시작.
        # (여기선 전부 0으로 두고, activation 에서 처리)
        # 필요하면 여기서 더 세밀하게 조정 가능.

        # scale for beta (조금만 움직이도록)
        self.beta_scale = 0.1

    @staticmethod
    def _global_mean_std(x: torch.Tensor):
        """
        x: [B,C,H,W]
        return: (mean, std)  둘 다 [B]
        C/H/W 전체에 대해 scalar mean/std 를 구한다.
        """
        B = x.shape[0]
        x_flat = x.reshape(B, -1)
        mean = x_flat.mean(dim=1)
        std = x_flat.std(dim=1)
        return mean, std

    def _compute_gamma_beta(
        self,
        noisy: torch.Tensor,
        base_out: torch.Tensor,
        mem_clean: torch.Tensor,
    ):
        """
        noisy, base_out, mem_clean: [B,C,H,W]
        return:
            gamma: [B,C,1,1], in [0,1]
            beta:  [B,C,1,1], small offset
        """
        mean_n, std_n = self._global_mean_std(noisy)
        mean_b, std_b = self._global_mean_std(base_out)
        mean_m, std_m = self._global_mean_std(mem_clean)

        feats = torch.stack(
            [mean_n, std_n, mean_b, std_b, mean_m, std_m],
            dim=1
        )  # [B,6]

        hyper = self.hyper_mlp(feats)  # [B, 2C]
        B = hyper.shape[0]
        C = self.in_channels

        gamma_raw, beta_raw = hyper[:, :C], hyper[:, C:]  # [B,C], [B,C]

        # gamma ∈ [0,1] (gate), beta ∈ [-beta_scale, beta_scale]
        gamma = torch.sigmoid(gamma_raw)          # [B,C]
        beta = self.beta_scale * torch.tanh(beta_raw)

        gamma = gamma.reshape(B, C, 1, 1)
        beta = beta.reshape(B, C, 1, 1)
        return gamma, beta

    def forward(
        self,
        noisy: torch.Tensor,
        base_out: torch.Tensor,
        mem_clean: torch.Tensor,
    ) -> torch.Tensor:
        """
        noisy, base_out, mem_clean: [B,C,H,W], 0~1 스케일 가정
        """
        # 1) 로컬 residual (noisy + base_out 만 사용)
        local_in = torch.cat([noisy, base_out], dim=1)  # [B,2C,H,W]
        r = self.local_net(local_in)                    # [B,C,H,W]

        # 2) mem_clean + 전역 통계 → gamma, beta
        gamma, beta = self._compute_gamma_beta(noisy, base_out, mem_clean)

        # 3) hyper-gated residual
        r_hg = gamma * r + beta                         # [B,C,H,W]

        out = base_out + r_hg

        if self.clamp_output:
            out = torch.clamp(out, 0.0, 1.0)

        return out


class HyperGatedResidualAdapter_FFT(nn.Module):
    """
    Hyper-gated residual adapter (with row-FFT features).

    - 공간 CNN(local_net)은 noisy + base_out 만 본다.
    - mem_clean 은 전역 통계(mean/std) + row-wise FFT power 통계로
      작은 MLP(hyper_mlp)에 들어가고,
      hyper_mlp 가 per-channel gate gamma, bias beta 를 생성한다.
    - 최종 출력: out = base_out + gamma * r(noisy, base_out) + beta

    이 구조에서는 mem_clean 의 spatial 패턴이
    output 으로 직접 복사될 수 있는 경로가 없다.
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 16,
        num_fft_bins: int = 3,   # row-FFT를 몇 개 band로 요약할지
        clamp_output: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_fft_bins = num_fft_bins
        self.clamp_output = clamp_output

        C = in_channels
        Hc = hidden_channels

        # --- 로컬 residual CNN: noisy + base_out 만 사용 ---
        # 입력 채널: noisy, base_out → 2C
        self.local_net = nn.Sequential(
            nn.Conv2d(2 * C, Hc, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(Hc, Hc, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(Hc, C, kernel_size=3, padding=1, bias=True),
        )

        # 마지막 conv 를 0으로 초기화 → 초기에는 r ≈ 0
        last = self.local_net[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

        # --- 전역 hyper-network: (mean/std + FFT) → gamma, beta ---
        #
        # 기본 전역 통계: [mean_n, std_n, mean_b, std_b, mean_m, std_m] → 6차원
        # row-FFT 통계:
        #   noisy / base_out / mem_clean 각각에 대해 num_fft_bins 개씩 → 3 * num_fft_bins
        in_feat_dim = 6 + 3 * num_fft_bins

        self.hyper_mlp = nn.Sequential(
            nn.Linear(in_feat_dim, Hc),
            nn.ReLU(inplace=True),
            nn.Linear(Hc, 2 * C),   # gamma_raw, beta_raw
        )

        # hyper-network 를 identity 근처로 초기화
        for m in self.hyper_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)

        # beta 크기를 제한하기 위한 스케일
        self.beta_scale = 0.1

    # ------------------------------
    # helpers: 전역 통계 + row-FFT feature
    # ------------------------------
    @staticmethod
    def _global_mean_std(x: torch.Tensor):
        """
        x: [B,C,H,W]
        return: (mean, std)  둘 다 [B]
        C/H/W 전체에 대해 scalar mean/std 를 구한다.
        """
        B = x.shape[0]
        x_flat = x.reshape(B, -1)
        mean = x_flat.mean(dim=1)
        std = x_flat.std(dim=1)
        return mean, std

    def _row_fft_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,C,H,W], 0~1 스케일 가정
        return: [B, num_fft_bins]

        각 샘플에 대해:
          - C, H 차원을 합쳐 row-wise 1D rFFT 수행
          - power spectrum을 구해 width 방향 주파수축을
            num_fft_bins 개 band로 나눈 뒤 band별 평균 power를 사용
          - log1p 로 스케일 줄이고, per-sample 정규화(평균으로 나눔)
        """
        B, C, H, W = x.shape
        # [B, C*H, W]
        x_ch = x.reshape(B, C * H, W)
        # rFFT along width: [B, C*H, F]
        spec = torch.fft.rfft(x_ch, dim=-1)
        power = spec.real ** 2 + spec.imag ** 2  # [B, C*H, F]
        # 채널/row 평균: [B, F]
        power_mean = power.mean(dim=1)

        Freq = power_mean.shape[-1]
        nb = self.num_fft_bins
        bin_size = Freq // nb

        feats = []
        for k in range(nb):
            start = k * bin_size
            end = (k + 1) * bin_size if k < nb - 1 else Freq
            band = power_mean[:, start:end]  # [B,_band_len]
            # band power 평균
            band_mean = band.mean(dim=-1)    # [B]
            feats.append(band_mean)

        feats = torch.stack(feats, dim=1)    # [B, nb]
        # 스케일 줄이기 + 간단한 정규화
        feats = torch.log1p(feats)
        # per-sample 평균으로 나눠서 스케일 맞추기
        eps = 1e-6
        feats = feats / (feats.mean(dim=1, keepdim=True) + eps)
        return feats  # [B, nb]

    def _compute_gamma_beta(
        self,
        noisy: torch.Tensor,
        base_out: torch.Tensor,
        mem_clean: torch.Tensor,
    ):
        """
        noisy, base_out, mem_clean: [B,C,H,W]
        return:
            gamma: [B,C,1,1], in [0,1]
            beta:  [B,C,1,1], small offset
        """
        mean_n, std_n = self._global_mean_std(noisy)
        mean_b, std_b = self._global_mean_std(base_out)
        mean_m, std_m = self._global_mean_std(mem_clean)

        # row-FFT 기반 feature
        fft_n = self._row_fft_features(noisy)     # [B, nb]
        fft_b = self._row_fft_features(base_out)  # [B, nb]
        fft_m = self._row_fft_features(mem_clean) # [B, nb]

        # [B, 6 + 3*nb]
        feats = torch.cat(
            [
                torch.stack(
                    [mean_n, std_n, mean_b, std_b, mean_m, std_m],
                    dim=1
                ),  # [B,6]
                fft_n, fft_b, fft_m,                # [B, 3*nb]
            ],
            dim=1,
        )

        hyper = self.hyper_mlp(feats)             # [B, 2C]
        B = hyper.shape[0]
        C = self.in_channels

        gamma_raw, beta_raw = hyper[:, :C], hyper[:, C:]  # [B,C], [B,C]

        # gamma ∈ [0,1] (gate), beta ∈ [-beta_scale, beta_scale]
        gamma = torch.sigmoid(gamma_raw)          # [B,C]
        beta = self.beta_scale * torch.tanh(beta_raw)

        gamma = gamma.reshape(B, C, 1, 1)
        beta = beta.reshape(B, C, 1, 1)
        return gamma, beta

    # ------------------------------
    # forward
    # ------------------------------
    def forward(
        self,
        noisy: torch.Tensor,
        base_out: torch.Tensor,
        mem_clean: torch.Tensor,
    ) -> torch.Tensor:
        """
        noisy, base_out, mem_clean: [B,C,H,W], 0~1 스케일 가정
        """
        # 1) 로컬 residual (noisy + base_out 만 사용)
        local_in = torch.cat([noisy, base_out], dim=1)  # [B,2C,H,W]
        r = self.local_net(local_in)                    # [B,C,H,W]

        # 2) mem_clean + 전역 통계/FFT → gamma, beta
        gamma, beta = self._compute_gamma_beta(noisy, base_out, mem_clean)

        # 3) hyper-gated residual
        r_hg = gamma * r + beta                         # [B,C,H,W]

        out = base_out + r_hg

        if self.clamp_output:
            out = torch.clamp(out, 0.0, 1.0)

        return out


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
        # self.adapter = MemoryResidualAdapter(in_channels, hidden_channels)

        # v2 adapter
        # self.adapter = LowFrequencyBlendAdapter(
        #     in_channels=in_channels,
        #     hidden_channels=hidden_channels,
        #     kernel_size=9,   # 필요하면 7, 11 등으로 조정
        #     sigma=3.0,
        #     clamp_output=True,
        # )

        # v3 adapter
        # self.adapter = GlobalMonotoneToneCurveAdapter(
        #     in_channels=in_channels,
        #     hidden_dim=hidden_channels,  # 혹은 16, 32 등
        #     num_ctrl_points=8,           # control point 개수 (7~16 정도 권장)
        #     clamp_output=True,
        # )

        # v4 adpater
        # self.adapter = HyperGatedResidualAdapter(
        #     in_channels=in_channels,
        #     hidden_channels=hidden_channels,
        #     clamp_output=True,
        # )

        
        # v5 adapter
        self.adapter = HyperGatedResidualAdapter_FFT(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_fft_bins=3,
            clamp_output=True,
        )

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
        mem_flat = self.memory_noise_bank.reshape(N, -1)     # [N,D]

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

    win_1d = torch.hann_window(ps, periodic=False, device=device).reshape(1, 1, ps, 1)
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
