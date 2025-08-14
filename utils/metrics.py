import torch
import numpy as np
from scipy.ndimage import distance_transform_edt


def dice_coefficient(pred, target, epsilon=1e-6):
    """
        Compute Dice coefficient for 3D binary segmentation.
        - pred_logits: torch.Tensor of shape [1, 1, D, W, H] 
        - target: torch.Tensor of shape [1, 1, D, W, H] (binary labels: 0 or 1)
        - threshold: float for binarizing predictions
        - epsilon: small value to avoid division by zero
    """
    
    # Flatten to compute over all voxels
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    pred_sum = pred_flat.sum()
    target_sum = target_flat.sum()
    dice = (2. * intersection + epsilon) / (pred_sum + target_sum + epsilon)
    return dice.item()


def compute_metrics(pred, target, smooth=1e-6):
    """
    Compute Dice, Precision, Recall, and IoU for binary segmentation.
    Args:
        pred (torch.Tensor): Binary prediction mask (0/1), any shape.
        target (torch.Tensor): Binary ground truth mask (0/1), same shape as pred.
        smooth (float): Small constant to avoid division by zero.
    Returns:
        dict: {'dice': ..., 'precision': ..., 'recall': ..., 'iou': ...}
    """
    pred = pred.float().view(-1)
    target = target.float().view(-1)

    intersection = (pred * target).sum()

    dice = (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)

    tp = intersection
    fp = pred.sum() - tp
    fn = target.sum() - tp

    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    iou = (tp + smooth) / (pred.sum() + target.sum() - tp + smooth)

    return {
        "dice": dice.item(),
        "precision": precision.item(),
        "recall": recall.item(),
        "iou": iou.item()
    }


