import argparse
import torch
from torch.utils.data import DataLoader

from models.unet import Unet
from models.text_model import TextEmbedder
from models.losses import DiceLoss
from utils.metrics import *
from utils.data import PETCTDataset

def main(args):
    # Dataset and loader
    train_set = PETCTDataset(root_dir=args.train_dir)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    # Model setup
    model = Unet(
        in_chans=2,
        out_chans=1,
        chans=16,
        num_pool_layers=4,
        drop_prob=args.dropout,
        use_att=False,
        use_res=True,
        leaky_negative_slope=0.1
    )
    model.eval()

    # Text embedder
    embedder = TextEmbedder()

    # Get one batch
    ct, pet, mask, text = next(iter(train_loader))
    print(f"CT: {ct.shape}, PET: {pet.shape}, MASK: {mask.shape}")

    # Forward pass
    out = model(torch.cat([ct, pet], dim=1), embedder(text))
    print(f"Output shape: {out.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PET/CT Tumor Segmentation")

    parser.add_argument("--train_dir", type=str, required=True, help="Path to training dataset")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")

    args = parser.parse_args()
    main(args)
