from models.unet import Unet
import torch


@torch.no_grad
def test_out_shape(model: Unet):
    """
        test the sahpe of output segmentation masks.
    """
    model.eval()
    x = torch.randn(1, 2, 128, 128, 128)  # (B, C, D, H, W)
    out = model(x)
    print(f"Output shape: {out.shape}")




if __name__ == "__main__":
    model = Unet(2, 2, 32, 4, 0.2, True, True, leaky_negative_slope=0.1)
    test_out_shape(model)
