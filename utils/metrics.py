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
    
    Handles no foreground case: If both empty, returns 1.0; if target empty but pred not, returns 0.0.
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


# def hausdorff_distance(pred_logits, target, threshold=0.5, voxel_spacing=(1.0, 1.0, 1.0)):
#     """
#     Compute 95th percentile Hausdorff distance for 3D binary segmentation (common variant for robustness).
#     - pred_logits: torch.Tensor of shape [1, 1, D, W, H]
#     - target: torch.Tensor of shape [1, 1, D, W, H]
#     - threshold: for binarizing
#     - voxel_spacing: tuple (z, y, x) for anisotropic spacing; default isotropic
    
#     Uses scipy's distance_transform_edt. Converts to numpy for computation.
#     Returns float distance in voxel units (scaled by spacing).
    
#     Handles no foreground: Returns 0.0 if both empty; raises ValueError if one is empty but not the other
#     (as Hausdorff is undefined; handle upstream if needed).
#     """
#     pred = binarize_predictions(pred_logits, threshold).cpu().numpy()[0, 0]  # Shape [D, W, H]
#     target = (target > 0).cpu().numpy()[0, 0]  # Shape [D, W, H]
    
#     if np.sum(target) == 0 and np.sum(pred) == 0:
#         return 0.0
#     elif np.sum(target) == 0 or np.sum(pred) == 0:
#         raise ValueError("Hausdorff distance undefined when one mask is empty.")
    
#     # Compute distance transforms
#     dist_target = distance_transform_edt(1 - target)  # Distance to target boundary
#     dist_pred = distance_transform_edt(1 - pred)      # Distance to pred boundary
    
#     # Distances from pred surface to target
#     pred_boundary = (pred == 1) & (dist_pred <= 1)  # Approximate surface
#     hausdorff_pred_to_target = dist_target[pred > 0]
    
#     # Distances from target surface to pred
#     target_boundary = (target == 1) & (dist_target <= 1)
#     hausdorff_target_to_pred = dist_pred[target > 0]
    
#     # Combine and take 95th percentile (HD95, more robust to outliers)
#     all_distances = np.concatenate([hausdorff_pred_to_target, hausdorff_target_to_pred])
#     hd95 = np.percentile(all_distances, 95)
    
#     # Apply voxel spacing (assuming uniform in each direction)
#     # For simplicity, take average or scale appropriately; here we scale the distance
#     effective_hd95 = hd95 * np.mean(voxel_spacing)  # If anisotropic, adjust logic as needed
    
#     return effective_hd95

# # Example usage (assuming tensors on GPU or CPU)
# # pred_logits = torch.randn(1, 1, 64, 128, 128)  # Example logits
# # target = torch.randint(0, 2, (1, 1, 64, 128, 128)).float()  # Example binary target
# # print(dice_coefficient(pred_logits, target))
# # print(jaccard_index(pred_logits, target))
# # print(hausdorff_distance(pred_logits, target))