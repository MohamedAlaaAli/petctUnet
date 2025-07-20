from models.unet import Unet
from utils.metrics import *
import torch


@torch.no_grad
def test_out_shape(model: Unet):
    """
        test the sahpe of output segmentation masks.
    """
    try:
        model.eval()
        x = torch.randn(1, 2, 128, 128, 128)  # (B, C, D, H, W)
        out = model(x)
        print(f"Output shape: {out.shape}")

    except Exception as e:
        raise e


@torch.no_grad
def test_segmentation_metrics():
    """
    Comprehensive test:
     - Validates get_pred_labels() using controlled logits
     - Computes all segmentation metrics using random data
       (Dice, IoU, accuracy, precision, recall)
     - Ensures metric values are valid (between 0 and 1)
    """
    # ---- Part 1: test get_pred_labels() ----
    B, C, D, H, W = 1, 3, 2, 2, 2
    preds = torch.zeros(B, C, D, H, W)
    preds[:, 2] = 5.0  # class 2 highest logits
    preds[:, 1] = 3.0
    preds[:, 0] = 1.0

    labels = get_pred_labels(preds)
    expected = torch.full((B, D, H, W), 2)
    assert torch.equal(labels, expected), f"get_pred_labels() failed: expected all 2s, got {labels}"
    print("get_pred_labels test passed.")

    # ---- Part 2: test metrics end-to-end ----
    batch_size = 2
    num_classes = 4
    D, H, W = 128, 128, 128

    preds = torch.randn(batch_size, num_classes, D, H, W)
    targets = torch.randint(0, num_classes, (batch_size, D, H, W))

    dice = multiclass_dice_score(preds, targets, num_classes)
    iou = iou_score(preds, targets, num_classes)
    acc = accuracy_score(preds, targets, num_classes)
    prec = precision_score(preds, targets, num_classes)
    rec = recall_score(preds, targets, num_classes)

    print("\nMetrics Test Results")
    print(f"Dice      : {dice:.4f}")
    print(f"IoU       : {iou:.4f}")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")

    for name, value in [("Dice", dice), ("IoU", iou), ("Accuracy", acc),
                        ("Precision", prec), ("Recall", rec)]:
        assert 0.0 <= value <= 1.0, f"{name} out of range: {value}"

    print("All metrics are within [0, 1] range.")
    print("Segmentation metrics test suite passed successfully.")






if __name__ == "__main__":
    model = Unet(2, 2, 32, 4, 0.2, True, True, leaky_negative_slope=0.1)
    test_out_shape(model)
    test_segmentation_metrics()

