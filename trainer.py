import torch
from pathlib import Path
import os
import torch.nn as nn
import torch.optim as optim
from models.unet import Unet
from models.losses import FocalTverskyBCELoss, levelsetLoss3d, gradientLoss3d, BoundaryDoULoss3D
from utils.data import create_petct_datasets
import wandb
from tqdm import tqdm
from utils.metrics import (
    compute_metrics
)
from torch.optim.lr_scheduler import CosineAnnealingLR
import nibabel as nib
import numpy as np
from monai.inferers import sliding_window_inference
from torch.cuda.amp import autocast, GradScaler
import random
import pandas as pd
from utils.data import get_small_tumor_weights


class Trainer(nn.Module):
    def __init__(self, model, 
                 datadir, 
                 device='cuda' if torch.cuda.is_available() else 'cpu', 
                 patch_size=(96, 96, 96), epochs=1000):
        
        super(Trainer, self).__init__()
        self.model = model.to(device)
        for param in self.model.parameters():
            param.requires_grad=True
        self.device = device
        #self.ema_model = copy.deepcopy(self.model)
        self.ema_decay = 0.999  
        self.patch_size = patch_size
        self.epochs = epochs
        # train_set = torch.utils.data.Subset(PETCTDataset(os.path.join(Pa/th(datadir), "train"), patch_size=patch_size),
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
                                                            patch_size=self.patch_size,
                                                            num_samples=1
                                                        )
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-5, weight_decay=1e-6, betas=(0.9, 0.999))
        self.criterion = FocalTverskyBCELoss()
        self.dou = BoundaryDoULoss3D()
        #self.mumfordsah = levelsetLoss3d()
        #self.tv = gradientLoss3d()
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=len(self.train_loader)*self.epochs)  # T_max = total steps or epochs

        # AMP GradScaler
        self.scaler = torch.amp.GradScaler("cuda")

        wandb.init(project="unet_petct", name="unet-tvresky-rim-petInj-diceovun")
        wandb.log({
            "patch_size": patch_size,
            "ema_decay": self.ema_decay
        })

    def load_lastckpt(self, pth):
        try:
            state_dict = torch.load(pth, map_location="cuda")  
            self.model.load_state_dict(state_dict, strict=False)
            print(f"loaded model from checkpoint: {pth}")
        except Exception as e:
            print(f"failed to load ckpt error msg {e}")
            pass

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
                ct, pet, targets, pth = batch['ct'], batch['pet'], batch['seg'], batch['pth']
                #w_map = get_small_tumor_weights(targets) outputs.max().item())
                print(ct.shape)
                inputs = torch.cat((ct,pet), dim=1).to(self.device)
                targets = targets.to(self.device)
                with torch.amp.autocast(device_type="cuda"):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets) + self.dou(outputs, targets)

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
            if epoch % 20 == 0 :
                val_dice = self.validate(epoch)
                torch.save(self.model.state_dict(), "ckpts/diceovunNW.pth")
                if val_dice > best:
                    best=val_dice
                    artifact = wandb.Artifact("model_ckpt", type="model")
                    torch.save(self.model.state_dict(), "ckpts/injectPet/diceovunNW.pth") # note one is saved with rimijn but it is inj only
                    artifact.add_file("ckpts/best_modeltrn.pth")
                    wandb.log_artifact(artifact)


    @torch.no_grad
    def validate(self, epoch, save_dir="val_outs", max_nifti_to_save=200, per_category=True):

        if per_category:
            df = pd.read_csv("fdg_metadata.csv")
            melanoma=[]
            lymph=[]
            lung=[]

        self.model.eval()

        total_dice = []
        total_precision=[]
        total_recall = []
        total_iou=[]

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
                    overlap=0.5
                    )
                print(outputs.shape)
                outputs = (torch.nn.functional.sigmoid(outputs) > 0.2).float()
                # Compute metrics
                if targets.sum() > 0:
                    results = compute_metrics(outputs, targets)
                    wandb.log({
                        "val/batch_results": results
                    })
                    total_dice.append(results["dice"])
                    total_precision.append(results["precision"])
                    total_recall.append(results["recall"])
                    total_iou.append(results["iou"])
                    print(results["dice"])
                    print(pth)
                    if per_category:
                        after_val = pth[0].split("val/")[1]
                        print(after_val)
                        result = df.loc[df["File Location"].str.contains(after_val), "diagnosis"]
                        if result.empty:
                            raise ValueError("extraction error")
                        else:
                            diagnosis = result.iloc[0]
                            if diagnosis == "MELANOMA":
                                melanoma.append(results["dice"])
                            elif diagnosis == "LUNG_CANCER":
                                lung.append(results["dice"])
                            elif diagnosis == "LYMPHOMA":
                                lymph.append(results["dice"])
                            else:
                                raise ValueError("extraction error")
                    
                # Save NIfTI volumes (input, label, prediction) for a few samples
                if max_nifti_to_save and random.choice([True, False]):
                    max_nifti_to_save-=1
                    print("nifti_predictions/CTres"+str(epoch)+str(max_nifti_to_save))
                    nib.save(nib.Nifti1Image(ct.cpu().squeeze(0).squeeze(0).numpy(), affine), save_dir+"/CTres"+str(epoch)+str(max_nifti_to_save)+".nii.gz")
                    nib.save(nib.Nifti1Image(pet.cpu().squeeze(0).squeeze(0).numpy(), affine), save_dir+"/PET"+str(epoch)+str(max_nifti_to_save)+".nii.gz")
                    nib.save(nib.Nifti1Image(targets.cpu().squeeze(0).squeeze(0).numpy(), affine), save_dir+"/GT"+str(epoch)+str(max_nifti_to_save)+".nii.gz")
                    nib.save(nib.Nifti1Image((outputs>0.5).float().cpu().detach().squeeze(0).squeeze(0).numpy(), affine), save_dir+"/PRED"+str(epoch)+str(max_nifti_to_save)+".nii.gz")


        avg_metrics = {
            "val/dice": sum(total_dice)/len(total_dice) if total_dice.__len__() != 0 else None,
            "val/precision": sum(total_precision)/len(total_precision) if total_precision.__len__() != 0 else None,
            "val/recall": sum(total_recall)/len(total_recall) if total_recall.__len__() != 0 else None,
            "val/iou": sum(total_iou)/len(total_iou) if total_iou.__len__() != 0 else None,

        }

        if per_category:
            avg_metrics["val/dice_mela"] = sum(melanoma)/len(melanoma) if melanoma else None
            avg_metrics["val/dice_lym"] = sum(lymph)/len(lymph) if lymph else None
            avg_metrics["val/dice_lung"] = sum(lung)/len(lung) if lung else None


        wandb.log(avg_metrics)
        print(f"[Epoch {epoch+1}] Validation Dice: {avg_metrics['val/dice']:.4f}")
        return sum(total_dice)/len(total_dice)


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
    #trainer.load_lastckpt("ckpts/injectPet/diceovun.pth")
    #trainer.train()
    #trainer.validate(222, per_category=True)



if __name__ == "__main__":
    main()