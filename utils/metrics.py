import torch
import numpy as np
from scipy.ndimage import distance_transform_edt

def binarize_predictions(logits, threshold=0.5):
    """
    Binarize raw logits to 0/1 predictions using sigmoid and thresholding.
    Assumes binary segmentation (single channel).
    """
    probs = torch.sigmoid(logits)
    return (probs > threshold).float()

def dice_coefficient(pred_logits, target, threshold=0.5, epsilon=1e-6):
    """
        Compute Dice coefficient for 3D binary segmentation.
        - pred_logits: torch.Tensor of shape [1, 1, D, W, H] (raw logits)
        - target: torch.Tensor of shape [1, 1, D, W, H] (binary labels: 0 or 1)
        - threshold: float for binarizing predictions
        - epsilon: small value to avoid division by zero
    """
    pred = binarize_predictions(pred_logits, threshold)
    target = (target > 0).float()  # Ensure binary
    
    # Flatten to compute over all voxels
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    pred_sum = pred_flat.sum()
    target_sum = target_flat.sum()
    dice = (2. * intersection + epsilon) / (pred_sum + target_sum + epsilon)
    return dice.item()

