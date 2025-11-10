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

    args, _ = parser.parse_known_args()
    return args


def checkpoint(net: nn.Module, epoch: int, opt) -> str:
    save_model_path = os.path.join(opt.save_model_path, opt.log_name)
    os.makedirs(save_model_path, exist_ok=True)
    model_name = f'epoch_adapter_{epoch:03d}.pth'
    save_model_path = os.path.join(save_model_path, model_name)
    torch.save(net.state_dict(), save_model_path)
    print(f'Checkpoint saved to {save_model_path}')
    return save_model_path


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
    clean_paths = sorted(glob.glob(os.path.join(dataset_dir, 'clean', '*')))
    noise_paths = sorted(glob.glob(os.path.join(dataset_dir, 'noise', '*')))
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

    # Adapter만 학습
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=opt.lr,
    )
    l1_criterion = nn.L1Loss()

    print(f'==> Start finetuning with adapter + patches. '
          f'Num epochs={opt.n_epoch}, batchsize={opt.batchsize}, '
          f'lr={opt.lr}, lambda_grad={opt.lambda_grad}, '
          f'patch_size={opt.patch_size}, '
          f'patches_per_image={opt.patches_per_image}')

    for epoch in range(1, opt.n_epoch + 1):
        model.train()
        epoch_st = time.time()
        losses = []

        for i, (clean, noisy) in enumerate(train_loader, start=1):
            clean = clean.to(device)   # [B, C, ps, ps], 0~1
            noisy = noisy.to(device)

            optimizer.zero_grad()
            pred = model(noisy)
            loss_l1 = l1_criterion(pred, clean)
            loss_grad = gradient_loss(pred, clean)
            loss = loss_l1 + opt.lambda_grad * loss_grad

            loss.backward()
            optimizer.step()

            losses.append(loss_l1.item())
            if i % 10 == 0 or i == len(train_loader):
                print(
                    f'Epoch [{epoch}/{opt.n_epoch}] '
                    f'Iter [{i}/{len(train_loader)}] '
                    f'L1={loss_l1.item():.6f} '
                    f'Grad={loss_grad.item():.6f} '
                    f'Total={loss.item():.6f}'
                )

        mean_loss = float(np.mean(losses))
        print(f'End of epoch {epoch}, mean L1 loss={mean_loss:.6f}, '
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
                pbar = tqdm(zip(valid[0], valid[1]), total=len(valid[0]), desc=f'Val ep{epoch}')
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
