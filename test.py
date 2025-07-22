from models.unet import Unet
from utils.metrics import *
from models.losses import DiceLoss
import torch
from models.text_model import TextEmbedder


@torch.no_grad
def test_out_wth_txt(model:Unet):
    """
        test the model utilizing text grounding feature. 
    """
    try:
        model.eval()
        x = torch.randn(1, 2, 64, 64, 64)  # (B, C, D, H, W)
        text = torch.randn(1, 2, 768)
        out = model(x, text)
        print(f"Output shape: {out.shape}")

    except Exception as e:
        raise e
    
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


def test_dice_loss():
    loss_fn = DiceLoss()

    # Case 1: Perfect match
    pred = torch.tensor([[[[1., 1.], [1., 1.]]]])  # Shape: (1, 1, 2, 2)
    target = torch.tensor([[[[1., 1.], [1., 1.]]]])
    loss = loss_fn(pred, target)
    print(f"Perfect match loss: {loss.item():.4f}")
    assert abs(loss.item() - 0.0) < 1e-4, "Dice loss should be near 0 for perfect match"

    # Case 2: No overlap
    pred = torch.tensor([[[[0., 0.], [0., 0.]]]])
    target = torch.tensor([[[[1., 1.], [1., 1.]]]])
    loss = loss_fn(pred, target)
    print(f"No overlap loss: {loss.item():.4f}")
    assert loss.item() > 0.9, "Dice loss should be near 1 for no overlap"

    # Case 3: Half overlap
    pred = torch.tensor([[[[1., 0.], [0., 0.]]]])
    target = torch.tensor([[[[1., 1.], [0., 0.]]]])
    loss = loss_fn(pred, target)
    print(f"Half overlap loss: {loss.item():.4f}")
    assert 0.3 < loss.item() < 0.8, "Dice loss should reflect partial overlap"

@torch.no_grad
def test_embedder():
    try:
        embedder = TextEmbedder()
        res = embedder("Define what a tumor is ")
        print(res.shape)

    except Exception as e:
        raise e


@torch.no_grad
def test_img_text_fusion(model):
    model.eval()
    embedder = TextEmbedder()
    res = embedder("Define what a tumor is ")
    x = torch.randn(1, 2, 64, 64, 64)  # (B, C, D, H, W)
    out = model(x, res)
    print(f"Output shape: {out.shape}")


     


if __name__ == "__main__":
    model = Unet(2, 2, 32, 4, 0.2, True, True, leaky_negative_slope=0.1)
    #test_out_wth_txt(model)
    # test_out_shape(model)
    #test_segmentation_metrics()
    # test_dice_loss()
    # test_embedder()
    test_img_text_fusion(model)
    


