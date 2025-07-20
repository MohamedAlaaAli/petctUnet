import torch
import time
import torch.nn.functional as F
from torchmetrics.classification import (
    MulticlassJaccardIndex,
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall
)

@torch.no_grad
def get_pred_labels(preds: torch.Tensor) -> torch.Tensor:
    """
    Convert raw logits or probabilities to predicted labels using argmax over channel axis.

    Args:
        preds (torch.Tensor): Tensor of shape (B, C, D, H, W)

    Returns:
        torch.Tensor: Tensor of shape (B, D, H, W) with integer labels.
    """
    return torch.argmax(preds, dim=1)  # (B, D, H, W)

@torch.no_grad
def iou_score(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    """
    Compute mean IoU for multi-class 3D segmentation.

    Args:
        preds (torch.Tensor): Raw predictions (B, C, D, H, W)
        targets (torch.Tensor): Ground truth labels (B, D, H, W)
        num_classes (int): Number of classes (including background)

    Returns:
        float: Mean IoU score
    """
    preds = get_pred_labels(preds)
    metric = MulticlassJaccardIndex(num_classes=num_classes, average='macro')
    return metric(preds, targets).item()

@torch.no_grad
def dice_score(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    """
    Compute mean Dice score for multi-class 3D segmentation.

    Returns:
        float: Mean Dice coefficient
    """
    preds = get_pred_labels(preds)
    return multiclass_dice_score(preds, targets, num_classes)

@torch.no_grad
def accuracy_score(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    """
    Compute accuracy for multi-class 3D segmentation.

    Returns:
        float: Accuracy score
    """
    preds = get_pred_labels(preds)
    metric = MulticlassAccuracy(num_classes=num_classes, average='micro')
    return metric(preds, targets).item()

@torch.no_grad
def precision_score(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    """
    Compute macro precision for multi-class 3D segmentation.

    Returns:
        float: Precision score
    """
    preds = get_pred_labels(preds)
    metric = MulticlassPrecision(num_classes=num_classes, average='macro')
    return metric(preds, targets).item()

@torch.no_grad
def recall_score(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    """
    Compute macro recall for multi-class 3D segmentation.

    Returns:
        float: Recall score
    """
    preds = get_pred_labels(preds)
    metric = MulticlassRecall(num_classes=num_classes, average='macro')
    return metric(preds, targets).item()

@torch.no_grad
def multiclass_dice_score(preds: torch.Tensor, targets: torch.Tensor, num_classes: int, epsilon=1e-6) -> float:
    """
    Compute macro-average Dice score for multi-class 3D segmentation.

    Args:
        preds (torch.Tensor): Logits or softmax of shape (B, C, D, H, W)
        targets (torch.Tensor): Ground truth of shape (B, D, H, W) with integer labels
        num_classes (int): Number of classes
        epsilon (float): Small value to avoid division by zero

    Returns:
        float: Macro-average Dice score
    """
    preds = torch.argmax(preds, dim=1)  # shape: (B, D, H, W)
    dice_per_class = []

    for cls in range(num_classes):
        pred_cls = (preds == cls).float()
        target_cls = (targets == cls).float()

        intersection = torch.sum(pred_cls * target_cls)
        union = torch.sum(pred_cls) + torch.sum(target_cls)

        dice = (2 * intersection + epsilon) / (union + epsilon)
        dice_per_class.append(dice)

    return torch.mean(torch.stack(dice_per_class)).item()

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
        torch.cuda.synchronize() if device == 'cuda' else None
        start_time = time.time()
        _ = model(sample)
        torch.cuda.synchronize() if device == 'cuda' else None
        end_time = time.time()

    return end_time - start_time
