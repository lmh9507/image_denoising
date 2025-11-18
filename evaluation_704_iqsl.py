import os
import argparse
import numpy as np
from PIL import Image

import torch
from torchvision import transforms
import torch.nn as nn
from arch_unet import UNet, RESNET, ImprovedUNet
from utils_eval import validation_denoise, calculate_psnr, calculate_ssim

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./dataset/m1', help='dataset dir')
parser.add_argument('--checkpoint', type=str, required=True, help='path to checkpoint .pth file')
parser.add_argument('--save_dir', type=str, default='./eval_results', help='directory to save denoised images')
parser.add_argument('--n_feature', type=int, default=48)
parser.add_argument('--n_channel', type=int, default=1)
parser.add_argument('--log_name', type=str, default='UNetImproved')
parser.add_argument('--gpu_devices', default='0', type=str)

# === IQSL-style 3-class 구조 IoU 계산 옵션 ===
parser.add_argument(
    '--compute_iq_iou', action='store_true',
    help='If set, compute 3-class intensity-quantized IoU between prediction and GT.'
)
parser.add_argument(
    '--iq_low_q', type=float, default=0.25,
    help='Lower quantile for intensity thresholds (e.g., 0.25).'
)
parser.add_argument(
    '--iq_high_q', type=float, default=0.75,
    help='Upper quantile for intensity thresholds (e.g., 0.75).'
)

opt = parser.parse_args()


def _to_gray_float01(img: np.ndarray) -> np.ndarray:
    """Convert 2D or 3D image (0~255) to grayscale float [0,1]."""
    arr = img.astype(np.float32)
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    return arr / 255.0


def _quantize_3class(gray: np.ndarray, t1: float, t2: float) -> np.ndarray:
    """gray: [H,W] float in [0,1]. Return labels in {0,1,2}."""
    labels = np.zeros_like(gray, dtype=np.int32)
    labels[gray <= t1] = 0
    labels[(gray > t1) & (gray < t2)] = 1
    labels[gray >= t2] = 2
    return labels


def compute_iq_iou(pred255: np.ndarray,
                   clean255: np.ndarray,
                   low_q: float,
                   high_q: float):
    """
    pred255, clean255: 0~255 uint/float.
    1) clean에서 quantile로 t1,t2 추정
    2) 두 이미지를 각각 3-class로 quantize
    3) class별 IoU 반환 (길이 3)
    """
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
    return ious  # [IoU_dark, IoU_mid, IoU_bright]


def evaluate():
    os.makedirs(opt.save_dir, exist_ok=True)

    clean_imgs, noisy_imgs, clean_paths, noisy_paths = validation_denoise(opt.data_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if ('UNET' in opt.log_name) and ('blindspot' in opt.log_name):
        network = UNet(in_nc=opt.n_channel,
                       out_nc=opt.n_channel,
                       n_feature=opt.n_feature,
                       blindspot=True)
    elif 'UNET' in opt.log_name:
        network = UNet(in_nc=opt.n_channel,
                       out_nc=opt.n_channel,
                       n_feature=opt.n_feature)
    elif 'RESNET' in opt.log_name:
        network = RESNET(in_nc=opt.n_channel,
                         out_nc=opt.n_channel,
                         n_feature=opt.n_feature)
    elif 'UNetImproved' in opt.log_name:
        network = ImprovedUNet(in_nc=opt.n_channel,
                               out_nc=opt.n_channel,
                               n_feature=opt.n_feature)

    state_dict = torch.load(opt.checkpoint, map_location=device)
    network.load_state_dict(state_dict)
    network = network.to(device)
    network.eval()
    print(f"Loaded checkpoint from {opt.checkpoint}")

    transformer = transforms.Compose([transforms.ToTensor()])
    criterion = nn.L1Loss()
    
    psnr_list, ssim_list, l1_list = [], [], []
    iou_dark_list, iou_mid_list, iou_bright_list = [], [], []

    model_patch_size = 352
    overlap_size = 64
    stride = model_patch_size - overlap_size

    # === 가중 마스크 생성 (패치 중앙이 가장 크고 가장자리는 작은 값) ===
    yy, xx = np.meshgrid(
        np.linspace(0, 1, model_patch_size),
        np.linspace(0, 1, model_patch_size),
        indexing="ij"
    )
    weight_mask = (1 - np.abs(yy - 0.5) * 2) * (1 - np.abs(xx - 0.5) * 2)
    weight_mask = weight_mask.astype(np.float32)

    for i, (clean, noisy) in enumerate(zip(clean_imgs, noisy_imgs)):
        clean_name = os.path.basename(clean_paths[i]).split('.')[0]
        noisy_name = os.path.basename(noisy_paths[i]).split('.')[0]

        clean_uint8 = clean.astype(np.uint8)
        noisy_uint8 = noisy.astype(np.uint8)
        h, w = noisy_uint8.shape

        denoised_image = np.zeros((h, w), dtype=np.float32)
        contribution_map = np.zeros((h, w), dtype=np.float32)
        l1_vals = []
        
        for r_start in range(0, h, stride):
            for c_start in range(0, w, stride):
                r_end = min(r_start + model_patch_size, h)
                c_end = min(c_start + model_patch_size, w)
                
                patch = noisy_uint8[r_start:r_end, c_start:c_end]

                patch_norm = patch.astype(np.float32) / 255.0
                
                padded_patch = np.pad(
                    patch_norm,
                    ((0, model_patch_size - patch.shape[0]), 
                     (0, model_patch_size - patch.shape[1])),
                    mode='reflect'
                )

                with torch.no_grad():
                    noisy_input = transformer(padded_patch).unsqueeze(0).to(device)
                    prediction_patch = network(noisy_input)

                    l1_val = criterion(prediction_patch, noisy_input).item()
                    l1_vals.append(l1_val)

                    prediction_patch = prediction_patch.permute(0, 2, 3, 1).cpu().clamp(0, 1).numpy().squeeze()

                prediction_patch = prediction_patch[:patch.shape[0], :patch.shape[1]]

                # === 가중 마스크 적용 ===
                wm = weight_mask[:patch.shape[0], :patch.shape[1]]
                denoised_image[r_start:r_end, c_start:c_end] += prediction_patch * wm
                contribution_map[r_start:r_end, c_start:c_end] += wm

        contribution_map[contribution_map == 0] = 1
        denoised_image = denoised_image / contribution_map  # [0,1]

        avg_l1_val = np.mean(l1_vals)
        l1_list.append(avg_l1_val)

        pred255 = np.clip(denoised_image * 255.0, 0, 255).astype(np.uint8)

        Image.fromarray(noisy_uint8).save(
            os.path.join(opt.save_dir, f"{noisy_name}_{i:03d}_noisy.png"))
        Image.fromarray(clean_uint8).save(
            os.path.join(opt.save_dir, f"{clean_name}_{i:03d}_clean.png"))
        Image.fromarray(pred255).save(
            os.path.join(opt.save_dir, f"{noisy_name}_{i:03d}_denoised.png"))
        
        psnr_val = calculate_psnr(pred255, clean_uint8)
        ssim_val = calculate_ssim(pred255, clean_uint8)
        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)

        msg = f"[{i+1}/{len(clean_imgs)}] {noisy_name} -> PSNR: {psnr_val:.2f}, SSIM: {ssim_val:.4f}, L1: {avg_l1_val:.6f}"

        # === 옵션: 3-class IQ IoU 계산 ===
        if opt.compute_iq_iou:
            ious = compute_iq_iou(
                pred255, clean_uint8,
                low_q=opt.iq_low_q,
                high_q=opt.iq_high_q
            )
            iou_dark, iou_mid, iou_bright = ious
            iou_dark_list.append(iou_dark)
            iou_mid_list.append(iou_mid)
            iou_bright_list.append(iou_bright)
            msg += f", IoU(d/m/b)=({iou_dark:.3f},{iou_mid:.3f},{iou_bright:.3f})"

        print(msg)

    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)
    avg_l1 = np.mean(l1_list)

    log_path = os.path.join(opt.save_dir, "metrics.txt")
    with open(log_path, "w") as f:
        f.write(f"Average PSNR: {avg_psnr:.2f}\n")
        f.write(f"Average SSIM: {avg_ssim:.4f}\n")
        f.write(f"Average L1 Loss: {avg_l1:.6f}\n")
        if opt.compute_iq_iou and len(iou_dark_list) > 0:
            avg_iou_dark = float(np.nanmean(iou_dark_list))
            avg_iou_mid = float(np.nanmean(iou_mid_list))
            avg_iou_bright = float(np.nanmean(iou_bright_list))
            f.write(
                f"Average 3-class IoU (dark/mid/bright): "
                f"{avg_iou_dark:.4f}, {avg_iou_mid:.4f}, {avg_iou_bright:.4f}\n"
            )
    print(f"Saved metrics to {log_path}")
    print(f"Average PSNR: {avg_psnr:.2f}, Average SSIM: {avg_ssim:.4f}, Average L1 Loss: {avg_l1:.6f}")
    if opt.compute_iq_iou and len(iou_dark_list) > 0:
        print("Average IQ-3class IoU - "
              f"dark: {np.nanmean(iou_dark_list):.4f}, "
              f"mid: {np.nanmean(iou_mid_list):.4f}, "
              f"bright: {np.nanmean(iou_bright_list):.4f}")


if __name__ == "__main__":
    evaluate()
