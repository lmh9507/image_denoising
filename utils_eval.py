import os, glob
import numpy as np
from PIL import Image
import cv2

def validation_denoise(dataset_dir):
    clean = glob.glob(os.path.join(dataset_dir, 'clean', "*"))
    noise = glob.glob(os.path.join(dataset_dir, 'noise', "*"))
    clean.sort(); noise.sort()
    images1, images2 = [], []
    for fn1, fn2 in zip(clean, noise):
        im1, im2 = Image.open(fn1), Image.open(fn2)
        im1 = np.array(im1, dtype=np.float32)
        im2 = np.array(im2, dtype=np.float32)
        images1.append(im1)
        images2.append(im2)
    return images1, images2, clean, noise

def ssim(prediction, target):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    img1 = prediction.astype(np.float64)
    img2 = target.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1*mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12  = cv2.filter2D(img1*img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2*mu1_mu2+C1)*(2*sigma12+C2)) / ((mu1_sq+mu2_sq+C1)*(sigma1_sq+sigma2_sq+C2))
    return ssim_map.mean()

def calculate_ssim(target, ref):
    img1, img2 = np.array(target, dtype=np.float64), np.array(ref, dtype=np.float64)
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            return np.mean([ssim(img1[:,:,i], img2[:,:,i]) for i in range(3)])
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def calculate_psnr(target, ref):
    img1, img2 = np.array(target, dtype=np.float32), np.array(ref, dtype=np.float32)
    diff = img1 - img2
    psnr = 10.0 * np.log10(255.0*255.0 / np.mean(np.square(diff)))
    return psnr
