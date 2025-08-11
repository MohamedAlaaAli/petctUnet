import torch
import time
import torch.nn.functional as F
from torchmetrics.classification import (
    BinaryJaccardIndex,
    BinaryAccuracy,
    BinaryPrecision,
    BinaryRecall
)

@torch.no_grad
def get_binary_preds(preds: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    Convert logits to binary predictions using sigmoid + thresholding.
    Input shape: (B, 1, D, H, W)
    Output shape: (B, D, H, W)
    """
    probs = torch.sigmoid(preds)
    return (probs > threshold).long()  # Remove channel dim

@torch.no_grad
def binary_iou(preds: torch.Tensor, targets: torch.Tensor) -> float:
    metric = BinaryJaccardIndex().to(preds.device)
    return metric(preds, targets).item()

@torch.no_grad
def binary_dice(preds: torch.Tensor, targets: torch.Tensor, epsilon=1e-6) -> float:
    preds = get_binary_preds(preds).float()
    targets = targets.float()

    intersection = torch.sum(preds * targets)
    union = torch.sum(preds) + torch.sum(targets)

    dice = (2. * intersection + epsilon) / (union + epsilon)
    return dice.item()

@torch.no_grad
def binary_accuracy(preds: torch.Tensor, targets: torch.Tensor) -> float:
    metric = BinaryAccuracy().to(preds.device)
    return metric(preds, targets).item()

@torch.no_grad
def binary_precision(preds: torch.Tensor, targets: torch.Tensor) -> float:
    metric = BinaryPrecision().to(preds.device)
    return metric(preds, targets).item()

@torch.no_grad
def binary_recall(preds: torch.Tensor, targets: torch.Tensor) -> float:
    metric = BinaryRecall().to(preds.device)
    return metric(preds, targets).item()

@torch.no_grad
def measure_inference_time(model, sample: torch.Tensor,
                           device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> float:
    """
    Measure inference time per volume sample in seconds.

    Args:
        model (torch.nn.Module): Segmentation model.
        sample (torch.Tensor): Input of shape (1, C=2, D, H, W)
        device (str): 'cuda' or 'cpu'

    Returns:
        float: Inference time per volume in seconds.
    """
    model.eval()
    model.to(device)
    sample = sample.to(device)

    with torch.no_grad():
        if device == 'cuda':
            torch.cuda.synchronize()
        start_time = time.time()
        _ = model(sample)
        if device == 'cuda':
            torch.cuda.synchronize()
        end_time = time.time()

    return end_time - start_time
