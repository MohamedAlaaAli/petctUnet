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
    binary_dice,
    binary_iou,
    binary_accuracy,
    binary_precision,
    binary_recall,
    measure_inference_time,
    get_binary_preds,
)
import nibabel as nib
import numpy as np
from monai.inferers import sliding_window_inference
import copy


class Trainer(nn.Module):
    def __init__(self, model, 
                 datadir, 
                 device='cuda' if torch.cuda.is_available() else 'cpu', 
                 patch_size=(96, 96, 96)):
        
        super(Trainer, self).__init__()
        self.model = model.to(device)
        for param in self.model.parameters():
            param.requires_grad=True
        self.device = device
        #self.ema_model = copy.deepcopy(self.model)
        self.ema_decay = 0.999  
        self.patch_size = patch_size

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
                                                            patch_size=(2, 96, 96),
                                                            num_samples=1
                                                        )
        self.optimizer = optim.AdamW(self.model.parameters())
        self.criterion = FocalTverskyBCELoss()

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


    def train(self, epochs=100):
        best=0
        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            total_dice = 0.0
            total_iou = 0.0
            total_acc = 0.0
            total_precision = 0.0
            total_recall = 0.0

            progress_bar = tqdm(self.train_loader, desc=f"[Epoch {epoch+1}] Training", leave=False)
            i=0
            for batch in progress_bar:
                self.optimizer.zero_grad()
                i+=1
                ct, pet, targets, _ = batch['ct'], batch['pet'], batch['seg'], batch['pth']
                print(ct.shape)

                inputs = torch.cat((ct,pet), dim=1).to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(inputs)

                print("Pred min:", outputs.min().item(), "max:", outputs.max().item())

                if targets.sum() != 0:
                    dice = binary_dice(outputs, targets)
                    iou = binary_iou(outputs, targets)
                    acc = binary_accuracy(outputs, targets)
                    precision = binary_precision(outputs, targets)
                    recall = binary_recall(outputs, targets)

                    total_dice += dice
                    total_iou += iou
                    total_acc += acc
                    total_precision += precision
                    total_recall += recall
                    i+=1


                loss = self.criterion(outputs, targets)
                loss.backward()

                self.optimizer.step()
                #self.update_ema()

                running_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())

            n= len(self.train_loader)
            avg_loss = running_loss /n


            wandb.log(
                {
                "epoch": epoch + 1,
                "train/loss": avg_loss,
                "train/dice": total_dice / (i+1),
                "train/iou": total_iou / (i+1),
                "train/accuracy": total_acc / (i+1),
                "train/precision": total_precision / (i+1),
                "train/recall": total_recall / (i+1),
                }
            )

            print(f"[Epoch {epoch+1}] Train Loss: {avg_loss:.4f}")
            val_dice = self.validate(epoch)
            torch.save(self.model.state_dict(), "ckpts/model_ckpt.pth")
            if epoch%4==0:
                self.save_train_sample(index=0, save_dir="train_sample")

            if val_dice > best:
                best=val_dice
                torch.save(self.model.state_dict(), "ckpts/best_modeltrn.pth")



    def validate(self, epoch, save_dir="nifti_predictions", max_nifti_to_save=2):
        self.model.eval()

        total_dice = 0.0
        total_iou = 0.0
        total_acc = 0.0
        total_precision = 0.0
        total_recall = 0.0
        total_inference_time = 0.0
        saved_nifti = 0
        progress_bar = tqdm(self.val_loader, desc=f"[Epoch {epoch+1}] Validation", leave=False)
        i=0
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
                if targets.sum() != 0:
                    i+=1
                    dice = binary_dice(outputs, targets)
                    iou = binary_iou(outputs, targets)
                    acc = binary_accuracy(outputs, targets)
                    precision = binary_precision(outputs, targets)
                    recall = binary_recall(outputs, targets)

                    total_dice += dice
                    total_iou += iou
                    total_acc += acc
                    total_precision += precision
                    total_recall += recall

                    progress_bar.set_postfix(dice=dice, iou=iou)
                    outputs = (torch.nn.functional.sigmoid(outputs) > 0.5).float()
                # Save NIfTI volumes (input, label, prediction) for a few samples
                    if saved_nifti < max_nifti_to_save:
                        nib.save(nib.Nifti1Image(ct.cpu().squeeze(0).squeeze(0).permute(1,2,0).numpy(), affine), "val_outs/CTres"+str(epoch)+".nii.gz")
                        nib.save(nib.Nifti1Image(pet.cpu().squeeze(0).squeeze(0).permute(1,2,0).numpy(), affine), "val_outs/PET"+str(epoch)+".nii.gz")
                        nib.save(nib.Nifti1Image(targets.cpu().squeeze(0).squeeze(0).permute(1,2,0).numpy(), affine), "val_outs/GT"+str(epoch)+".nii.gz")
                        nib.save(nib.Nifti1Image(outputs.cpu().detach().squeeze(0).squeeze(0).permute(1,2,0).numpy(), affine), "val_outs/PRED"+str(epoch)+".nii.gz")


        n = len(self.val_loader)
        avg_metrics = {
            "val/dice": total_dice / i,
            "val/iou": total_iou / i,
            "val/accuracy": total_acc / i,
            "val/precision": total_precision / i,
            "val/recall": total_recall / i,
            "val/inference_time_per_volume": total_inference_time / i,
        }

        wandb.log(avg_metrics)
        print(f"[Epoch {epoch+1}] Validation Dice: {avg_metrics['val/dice']:.4f}, IoU: {avg_metrics['val/iou']:.4f}")
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
    #trainer.train()
    trainer.validate(1)



if __name__ == "__main__":
    main()