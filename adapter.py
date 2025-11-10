import torch
import torch.nn as nn


class OutputAdapter(nn.Module):
    """
    Small residual CNN that refines the output of a frozen base denoiser.

    Given noisy input y and base output x_base, it predicts a residual delta
    and returns x_base + delta.
    """
    def __init__(self, in_channels: int = 1, hidden_channels: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2 * in_channels, hidden_channels,
                      kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, in_channels,
                      kernel_size=3, padding=1, bias=True),
        )

    def forward(self, noisy: torch.Tensor, base_out: torch.Tensor) -> torch.Tensor:
        # concat [noisy, base_out] along channel
        x = torch.cat([noisy, base_out], dim=1)
        delta = self.net(x)
        return base_out + delta


class DenoiserWithAdapter(nn.Module):
    """
    Wrapper that combines a frozen base denoiser and a small trainable adapter.
    base_model: A-domain에서 학습된 UNet / RESNET / ImprovedUNet 등
    """
    def __init__(
        self,
        base_model: nn.Module,
        in_channels: int = 1,
        hidden_channels: int = 16,
        freeze_base: bool = True,
        use_no_grad_for_base: bool = True,
    ):
        super().__init__()
        self.base = base_model
        self.in_channels = in_channels
        self.freeze_base = freeze_base
        self.use_no_grad_for_base = use_no_grad_for_base

        # base model freeze
        if freeze_base:
            for p in self.base.parameters():
                p.requires_grad = False

        # very small CNN as adapter
        self.adapter = OutputAdapter(
            in_channels=in_channels,
            hidden_channels=hidden_channels
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # base 출력은 gradient 필요 없으므로 no_grad로 래핑
        if self.use_no_grad_for_base:
            with torch.no_grad():
                base_out = self.base(x)
        else:
            base_out = self.base(x)
        out = self.adapter(x, base_out)
        return out
