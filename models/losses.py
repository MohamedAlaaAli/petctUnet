import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth=0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)

        # Assume shape: [B, C, D, H, W] or [B, C, H, W]
        dims = tuple(range(2, pred.dim()))  # dims = (2, 3) for 2D, (2, 3, 4) for 3D

        intersection = (pred * target).sum(dim=dims)
        union = pred.sum(dim=dims) + target.sum(dim=dims)
        dice = (2. * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice.mean()
