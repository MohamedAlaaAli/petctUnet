from models.unet import Unet
from utils.metrics import *
from models.losses import DiceLoss
import torch
from models.text_model import TextEmbedder
from utils.data import PETCTDataset
from torch.utils.data import DataLoader

train_set_dir = '/home/muhamed/mntdrive/zips/FDG-PET-CT-Lesions'
#val_set_dir = '/home/muhamed/mntdrive/zips/test'

train_set = PETCTDataset(root_dir=train_set_dir)
#val_set = PETCTDataset(root_dir=val_set_dir)
train_loader = DataLoader(train_set, batch_size=1, shuffle=True)


model = Unet(2, 1, 16, 4, 0.2, False, True, leaky_negative_slope=0.1)
model.eval()
#embedder = TextEmbedder()
#res = embedder(["Define what a tumor is "]*1)
ct, pet, mask, text = next(iter(train_loader))
print(ct.shape, pet.shape, mask.shape)
out = model(torch.cat([ct, pet], dim=1), TextEmbedder()(text))
print(f"Output shape: {out.shape}")

