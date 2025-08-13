import torch
from pathlib import Path
import os
import torch.nn as nn
import torch.optim as optim
from models.unet import Unet
from models.losses import FocalTverskyBCELoss
from utils.data import create_petct_datasets
import wandb
from tqdm import tqdm
from utils.metrics import (
    dice_coefficient,
)
from torch.optim.lr_scheduler import CosineAnnealingLR
import nibabel as nib
import numpy as np
from monai.inferers import sliding_window_inference
from torch.cuda.amp import autocast, GradScaler


class Trainer(nn.Module):
    def __init__(self, model, 
                 datadir, 
                 device='cuda' if torch.cuda.is_available() else 'cpu', 
                 patch_size=(96, 96, 96), epochs=300):
        
        super(Trainer, self).__init__()
        self.model = model.to(device)
        for param in self.model.parameters():
            param.requires_grad=True
        self.device = device
        #self.ema_model = copy.deepcopy(self.model)
        self.ema_decay = 0.999  
        self.patch_size = patch_size
        self.epochs = epochs
        # train_set = torch.utils.data.Subset(PETCTDataset(os.path.join(Path(datadir), "train"), patch_size=patch_size),
        #                                     [0,10,23])
        # val_set = PETCTDataset(os.path.join(Path(datadir), "val"))

        # self.train_loader = torch.utils.data.DataLoader(
        #     train_set, batch_size=1, shuffle=True, num_workers=4, prefetch_factor=2, persistent_workers=True,pin_memory=True
        # )
        # self.val_loader = torch.utils.data.DataLoader(
        #     val_set, batch_size=1, shuffle=False, num_workers=4, prefetch_factor=2, persistent_workers=True, pin_memory=True
        # )
        self.train_loader, self.val_loader = create_petct_datasets(
                                                            train_dir=os.path.join(Path(datadir), "train"),
                                                            val_dir=os.path.join(Path(datadir), "val"),
                                                            patch_size=(96, 96, 96),
                                                            num_samples=1
                                                        )
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-5, betas=(0.9, 0.999))
        self.criterion = FocalTverskyBCELoss()
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=len(self.train_loader)*self.epochs)  # T_max = total steps or epochs

        # AMP GradScaler
        self.scaler = torch.amp.GradScaler("cuda")

        wandb.init(project="unet_petct", name="unet-training")
        wandb.log({
            "patch_size": patch_size,
            "ema_decay": self.ema_decay
        })

    def load_lastckpt(self, pth):
        state_dict = torch.load(pth, map_location="cuda")  
        self.model.load_state_dict(state_dict)
        print(f"loaded model from checkpoint: {pth}")

    @torch.no_grad()
    def update_ema(self):
        model_params = dict(self.model.named_parameters())
        ema_params = dict(self.ema_model.named_parameters())
        for name in model_params.keys():
            ema_params[name].data.mul_(self.ema_decay).add_(model_params[name].data, alpha=1 - self.ema_decay)


    def train(self):
        best=0
        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0

            progress_bar = tqdm(self.train_loader, desc=f"[Epoch {epoch+1}] Training", leave=False)
            for batch in progress_bar:
                self.optimizer.zero_grad()
                ct, pet, targets, _ = batch['ct'], batch['pet'], batch['seg'], batch['pth']
                print(ct.shape)
                inputs = torch.cat((ct,pet), dim=1).to(self.device)
                targets = targets.to(self.device)


                with torch.amp.autocast(device_type="cuda"):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)

                print("Pred min:", outputs.min().item(), "max:", outputs.max().item())
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                #self.update_ema()
                running_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())
                wandb.log({"train/batchLoss":loss.item()})

            n= len(self.train_loader)
            avg_loss = running_loss /n
            wandb.log(
                {
                "epoch": epoch + 1,
                "train/train_loss":avg_loss

                }
            )
            print(f"[Epoch {epoch+1}] Train Loss: {avg_loss:.4f}")
            val_dice = self.validate(epoch)
            torch.save(self.model.state_dict(), "ckpts/model_ckpt.pth")
            if val_dice > best:
                best=val_dice
                torch.save(self.model.state_dict(), "ckpts/best_modeltrn.pth")


    @torch.no_grad
    def validate(self, epoch, save_dir="nifti_predictions", max_nifti_to_save=5):
        self.model.eval()

        total_dice = 0.0

        progress_bar = tqdm(self.val_loader, desc=f"[Epoch {epoch+1}] Validation", leave=False)

        with torch.no_grad():
            for batch in progress_bar:
                ct, pet, targets, pth = batch['ct'], batch['pet'], batch['seg'], batch['pth']
                print(ct.shape)
                im = nib.load(pth[0]+"/CTres.nii.gz")
                affine = im.affine
                inputs = torch.cat((ct,pet), dim=1).to(self.device)
                targets = targets.to(self.device)
                outputs = sliding_window_inference(
                    inputs=inputs, 
                    roi_size=self.patch_size,  
                    sw_batch_size=1, 
                    predictor=self.model, 
                    overlap=0.25
                    )
                print(outputs.shape)
                # Compute metrics
                dice = dice_coefficient(outputs, targets)

                total_dice += dice

                progress_bar.set_postfix(dice=dice)
                outputs = (torch.nn.functional.sigmoid(outputs) > 0.5).float()
                # Save NIfTI volumes (input, label, prediction) for a few samples
                if max_nifti_to_save:
                    max_nifti_to_save-=1
                    nib.save(nib.Nifti1Image(ct.cpu().squeeze(0).squeeze(0).numpy(), affine), "val_outs/CTres"+str(epoch)+str(max_nifti_to_save)+".nii.gz")
                    nib.save(nib.Nifti1Image(pet.cpu().squeeze(0).squeeze(0).numpy(), affine), "val_outs/PET"+str(epoch)+str(max_nifti_to_save)+".nii.gz")
                    nib.save(nib.Nifti1Image(targets.cpu().squeeze(0).squeeze(0).numpy(), affine), "val_outs/GT"+str(epoch)+str(max_nifti_to_save)+".nii.gz")
                    nib.save(nib.Nifti1Image(outputs.cpu().detach().squeeze(0).squeeze(0).numpy(), affine), "val_outs/PRED"+str(epoch)+str(max_nifti_to_save)+".nii.gz")


        n = len(self.val_loader)
        avg_metrics = {
            "val/dice": total_dice / n,
        }

        wandb.log(avg_metrics)
        print(f"[Epoch {epoch+1}] Validation Dice: {avg_metrics['val/dice']:.4f}")
        return total_dice


def main():
    datadir = "../../../Storage/fdg_pet_ct/FDG-PET-CT-Lesions"
    model = Unet(in_chans=2, 
                 out_chans=1, 
                 chans=32, 
                 num_pool_layers=4, 
                 use_att=True, 
                 use_res=True, 
                 leaky_negative_slope=0)
    
    trainer = Trainer(model, datadir, device="cuda")
    trainer.load_lastckpt("ckpts/best_modeltrn.pth")
    trainer.train()

    #trainer.validate(1)



if __name__ == "__main__":
    main()