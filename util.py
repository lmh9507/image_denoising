import torch
import torch.nn as nn


class L1FFT(nn.Module):
    def __init__(self, alpha: float = 1.0, beta: float = 1.0, reduction: str = 'mean'):
        """
        Args:
            alpha: pixel domain weight
            beta: frequency domain weight
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction
        self.l1 = nn.L1Loss(reduction=reduction)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:  (B, C, H, W)
            target: (B, C, H, W)
        Returns:
            total_loss: alpha·L1_pixel + beta·L1_freq
        """
        pixel_loss = self.l1(pred, target)
        fft_pred   = torch.fft.fft2(pred)
        fft_target = torch.fft.fft2(target)
        freq_diff = torch.abs(fft_pred - fft_target)

        if self.reduction == 'mean':
            freq_loss = freq_diff.mean()
        elif self.reduction == 'sum':
            freq_loss = freq_diff.sum()
        else:
            freq_loss = freq_diff
        return self.alpha * pixel_loss + self.beta * freq_loss