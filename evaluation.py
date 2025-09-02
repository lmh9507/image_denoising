import os
import glob
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
    # 저장 폴더
    os.makedirs(opt.save_dir, exist_ok=True)

    # 데이터셋 로드
    clean_imgs, noisy_imgs, clean_paths, noisy_paths = validation_denoise(opt.data_dir)

    # 모델 초기화
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


    # 체크포인트 로드
    state_dict = torch.load(opt.checkpoint, map_location=device)
    network.load_state_dict(state_dict)
    network = network.to(device)
    network.eval()
    print(f"Loaded checkpoint from {opt.checkpoint}")

    transformer = transforms.Compose([transforms.ToTensor()])
    criterion = nn.L1Loss()   # L1 loss 정의
    # 평가
    psnr_list, ssim_list, l1_list = [], [], []
    for i, (clean, noisy) in enumerate(zip(clean_imgs, noisy_imgs)):
        clean_name = os.path.basename(clean_paths[i]).split('.')[0]
        noisy_name = os.path.basename(noisy_paths[i]).split('.')[0]

        clean = clean.astype(np.float32)
        noisy = noisy.astype(np.float32)

        # normalize
        noisy_input = noisy / 255.0
        noisy_input = transformer(noisy_input).unsqueeze(0).to(device)

        with torch.no_grad():
            prediction = network(noisy_input)

        # L1 Loss 계산
        l1_val = criterion(prediction, noisy_input).item()
        l1_list.append(l1_val)
    

        # 후처리
        prediction = prediction.permute(0, 2, 3, 1).cpu().clamp(0, 1).numpy().squeeze()
        pred255 = np.clip(prediction * 255.0 + 0.5, 0, 255).astype(np.uint8)
        clean255 = clean.astype(np.uint8)
        noisy255 = noisy.astype(np.uint8)

        # 결과 저장
        Image.fromarray(noisy255).convert('RGB').save(
            os.path.join(opt.save_dir, f"{noisy_name}_{i:03d}_noisy.png"))
        Image.fromarray(clean255).convert('RGB').save(
            os.path.join(opt.save_dir, f"{clean_name}_{i:03d}_clean.png"))
        Image.fromarray(pred255).convert('RGB').save(
            os.path.join(opt.save_dir, f"{noisy_name}_{i:03d}_denoised.png"))

        # 지표 계산
        psnr_val = calculate_psnr(pred255, clean255)
        ssim_val = calculate_ssim(pred255, clean255)
        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)

        print(f"[{i+1}/{len(clean_imgs)}] {noisy_name} -> PSNR: {psnr_val:.2f}, SSIM: {ssim_val:.4f}, L1: {l1_val:.6f}")

    # 평균 결과 로그 저장
    
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
