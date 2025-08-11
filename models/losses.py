import torch
import torch.nn as nn

import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, apply_sigmoid=True, skip_empty=False):
        super().__init__()
        self.smooth = smooth
        self.apply_sigmoid = apply_sigmoid
        self.skip_empty = skip_empty

    def forward(self, pred, target):
        """
        Args:
            pred: [B, 1, D, H, W] raw logits
            target: [B, 1, D, H, W] binary masks
        """
        if self.apply_sigmoid:
            pred = torch.sigmoid(pred)

        dims = tuple(range(2, pred.dim()))  # (2, 3, 4) for 3D

        intersection = (pred * target).sum(dim=dims)
        union = pred.sum(dim=dims) + target.sum(dim=dims)
        dice = (2. * intersection + self.smooth) / (union + self.smooth)

        if self.skip_empty:
            # Only keep samples with non-zero target
            mask_has_fg = (target.sum(dim=dims) > 0)
            if mask_has_fg.any():
                dice = dice[mask_has_fg]
            else:
                # all empty targets; return 0 loss
                return torch.tensor(0.0, device=pred.device, requires_grad=True)

        return 1 - dice.mean()

class DiceBCELoss(nn.Module):
    def __init__(self, dice_weight=0.8, bce_weight=0.2):
        super().__init__()
        self.dice = DiceLoss(apply_sigmoid=True, skip_empty=False)
        self.bce = nn.BCEWithLogitsLoss()
        self.dw = dice_weight
        self.bw = bce_weight

    def forward(self, pred, target):
        return self.dw * self.dice(pred, target) + self.bw * self.bce(pred, target)
    

class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, gamma=1.5, smooth=1.0, apply_sigmoid=True, skip_empty=False):
        """
        alpha: weight for false positives
        beta:  weight for false negatives (set higher than alpha to boost recall)
        gamma: focal exponent (>1 focuses more on hard examples)
        smooth: smoothing constant to avoid div by zero
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        self.apply_sigmoid = apply_sigmoid
        self.skip_empty = skip_empty

    def forward(self, pred, target):
        if self.apply_sigmoid:
            pred = torch.sigmoid(pred)

        dims = tuple(range(2, pred.dim()))  # for 3D: (2, 3, 4)

        # True positives, false negatives, false positives
        TP = (pred * target).sum(dim=dims)
        FN = ((1 - pred) * target).sum(dim=dims)
        FP = (pred * (1 - target)).sum(dim=dims)

        # Tversky index
        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)

        if self.skip_empty:
            mask_has_fg = (target.sum(dim=dims) > 0)
            if mask_has_fg.any():
                tversky = tversky[mask_has_fg]
            else:
                return torch.tensor(0.0, device=pred.device, requires_grad=True)

        focal_tversky = (1 - tversky) ** self.gamma
        return focal_tversky.mean()
    

class FocalTverskyBCELoss(nn.Module):
    def __init__(self, ft_weight=0.8, bce_weight=0.2, alpha=0.3, beta=0.7, gamma=0.75):
        super().__init__()
        self.ft = FocalTverskyLoss(alpha=alpha, beta=beta, gamma=gamma, apply_sigmoid=True, skip_empty=False)
        self.bce = nn.BCEWithLogitsLoss(reduction="mean")
        self.fw = ft_weight
        self.bw = bce_weight

    def forward(self, pred, target):
        # Focal Tversky part
        focal = self.fw * self.ft(pred, target)

        # BCE part (patch-wise mean)
        bce_loss_all = self.bce(pred, target)
        bce = self.bw * bce_loss_all.mean()

        return focal + bce   


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

