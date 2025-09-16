import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch
import numpy as np
from torch import Tensor, einsum

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
        # self.w_lesion = 1.5
        # self.w_rim = 1.2

    def forward(self, pred, target, w_map):
        if self.apply_sigmoid:
            pred = torch.sigmoid(pred)
        
        # weight_map = torch.ones_like(target)
        # weight_map[target == 1] = self.w_lesion
        # if self.w_rim > 1.0:
        #     rim = torch.nn.functional.max_pool3d(target.float(), kernel_size=3, stride=1, padding=1) - target.float() 
        #     rim = rim.clamp(min=0) 
        #     weight_map[rim == 1] = self.w_rim

        dims = tuple(range(2, pred.dim()))  # for 3D: (2, 3, 4)

        #True positives, false negatives, false positives
        # TP = (weight_map * pred * target).sum(dim=dims)
        # FN = (weight_map * (1 - pred) * target).sum(dim=dims)
        # FP = (weight_map * pred * (1 - target)).sum(dim=dims)
        if w_map is None:
            w_map = torch.ones_like(pred)
            
        TP = (w_map * pred * target).sum(dim=dims)
        FN = (w_map * (1 - pred) * target).sum(dim=dims)
        FP = (w_map * pred * (1 - target)).sum(dim=dims)

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
    def __init__(self, ft_weight=1, bce_weight=0, alpha=0.3, beta=0.7, gamma=4/3):
        super().__init__()
        self.ft = FocalTverskyLoss(alpha=alpha, beta=beta, gamma=gamma, apply_sigmoid=True, skip_empty=False)
        self.bce = nn.BCEWithLogitsLoss(reduction="mean")
        self.fw = ft_weight
        self.bw = bce_weight

    def forward(self, pred, target, w_map=None):
        # Focal Tversky part
        focal = self.fw * self.ft(pred, target, w_map=w_map)

        # BCE part (patch-wise mean)
        bce_loss_all = self.bce(pred, target)
        bce = self.bw * bce_loss_all.mean()

        return focal + bce   


class gradientLoss3d(nn.Module):
    def __init__(self, penalty='l1'):
        super(gradientLoss3d, self).__init__()
        self.penalty = penalty

    def forward(self, input):
        _, _ , h,w,d = input.shape
        # Compute gradients in 3 directions
        dD = torch.abs(input[:, :, 1:, :, :] - input[:, :, :-1, :, :])  # depth
        dH = torch.abs(input[:, :, :, 1:, :] - input[:, :, :, :-1, :])  # height
        dW = torch.abs(input[:, :, :, :, 1:] - input[:, :, :, :, :-1])  # width

        if self.penalty == "l2":
            dD = dD ** 2
            dH = dH ** 2
            dW = dW ** 2

        loss = torch.sum(dD) + torch.sum(dH) + torch.sum(dW)
        return loss/(h*w*d)

		
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
    

class BoundaryDoULoss3D(nn.Module):
    def __init__(self):
        super(BoundaryDoULoss3D, self).__init__()

        # 3D cross kernel (6-neighbourhood)
        kernel = torch.zeros((3, 3, 3))
        kernel[1, 1, :] = 1
        kernel[1, :, 1] = 1
        kernel[:, 1, 1] = 1
        self.register_buffer("kernel3d", kernel.unsqueeze(0).unsqueeze(0), persistent=False)  # (1,1,3,3,3)

    def _adaptive_size(self, score, target):
        smooth = 1e-5

        with torch.autocast(device_type="cuda", enabled=False):
            Y = F.conv3d(target, self.kernel3d.to("cuda"), padding=1)

        Y = Y * target
        Y[Y == 7] = 0  # remove full-neighbour voxels

        C = torch.count_nonzero(Y)
        S = torch.count_nonzero(target)

        alpha = 1 - (C + smooth) / (S + smooth)
        alpha = 2 * alpha - 1
        alpha = torch.clamp(alpha, max=0.8)

        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)

        loss = (z_sum + y_sum - 2 * intersect + smooth) / \
               (z_sum + y_sum - (1 + alpha) * intersect + smooth)

        return loss

    def forward(self, inputs, target):
        """
        inputs: (B,1,W,H,D) logits
        target: (B,1,W,H,D) binary ground truth {0,1}
        """
        # apply sigmoid inside AMP context
        inputs = torch.sigmoid(inputs)

        return self._adaptive_size(inputs, target)
