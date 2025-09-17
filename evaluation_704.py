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
opt = parser.parse_args()

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

        clean = clean.astype(np.uint8)
        noisy = noisy.astype(np.uint8)
        h, w = noisy.shape

        denoised_image = np.zeros((h, w), dtype=np.float32)
        contribution_map = np.zeros((h, w), dtype=np.float32)
        l1_vals = []
        
        for r_start in range(0, h, stride):
            for c_start in range(0, w, stride):
                r_end = min(r_start + model_patch_size, h)
                c_end = min(c_start + model_patch_size, w)
                
                patch = noisy[r_start:r_end, c_start:c_end]

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
        denoised_image = denoised_image / contribution_map

        avg_l1_val = np.mean(l1_vals)
        l1_list.append(avg_l1_val)

        pred255 = np.clip(denoised_image * 255.0, 0, 255).astype(np.uint8)

        Image.fromarray(noisy).save(
            os.path.join(opt.save_dir, f"{noisy_name}_{i:03d}_noisy.png"))
        Image.fromarray(clean).save(
            os.path.join(opt.save_dir, f"{clean_name}_{i:03d}_clean.png"))
        Image.fromarray(pred255).save(
            os.path.join(opt.save_dir, f"{noisy_name}_{i:03d}_denoised.png"))
        
        psnr_val = calculate_psnr(pred255, clean)
        ssim_val = calculate_ssim(pred255, clean)
        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)

        print(f"[{i+1}/{len(clean_imgs)}] {noisy_name} -> PSNR: {psnr_val:.2f}, SSIM: {ssim_val:.4f}, L1: {avg_l1_val:.6f}")

    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)
    avg_l1 = np.mean(l1_list)
    log_path = os.path.join(opt.save_dir, "metrics.txt")
    with open(log_path, "w") as f:
        f.write(f"Average PSNR: {avg_psnr:.2f}\n")
        f.write(f"Average SSIM: {avg_ssim:.4f}\n")
        f.write(f"Average L1 Loss: {avg_l1:.6f}\n")
    print(f"Saved metrics to {log_path}")
    print(f"Average PSNR: {avg_psnr:.2f}, Average SSIM: {avg_ssim:.4f}, Average L1 Loss: {avg_l1:.6f}")

if __name__ == "__main__":
    evaluate()
