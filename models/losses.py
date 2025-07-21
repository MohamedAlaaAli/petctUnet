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


class gradientLoss3d(nn.Module):
    def __init__(self, penalty='l1'):
        super(gradientLoss3d, self).__init__()
        self.penalty = penalty

    def forward(self, input):
        # Compute gradients in 3 directions
        dD = torch.abs(input[:, :, 1:, :, :] - input[:, :, :-1, :, :])  # depth
        dH = torch.abs(input[:, :, :, 1:, :] - input[:, :, :, :-1, :])  # height
        dW = torch.abs(input[:, :, :, :, 1:] - input[:, :, :, :, :-1])  # width

        if self.penalty == "l2":
            dD = dD ** 2
            dH = dH ** 2
            dW = dW ** 2

        loss = torch.sum(dD) + torch.sum(dH) + torch.sum(dW)
        return loss

		
class levelsetLoss3d(nn.Module):
    def __init__(self):
        super(levelsetLoss3d, self).__init__()

    def forward(self, output, target):
        # output: (B, C, D, H, W)
        # target: (B, C_t, D, H, W)
        B, C, D, H, W = output.shape
        B_t, C_t, _, _, _ = target.shape

        loss = 0.0
        for ich in range(C_t):
            target_ = target[:, ich:ich+1]  # shape: (B, 1, D, H, W)
            target_ = target_.expand(B, C, D, H, W)
            
            # Compute centroid
            pcentroid = (target_ * output).sum(dim=(2,3,4)) / (output.sum(dim=(2,3,4)) + 1e-8)
            pcentroid = pcentroid.view(B, C, 1, 1, 1)
            
            plevel = target_ - pcentroid.expand_as(target_)
            pLoss = plevel * plevel * output
            loss += torch.sum(pLoss)

        return loss

