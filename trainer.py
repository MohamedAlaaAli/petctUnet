import torch
from pathlib import Path
import os
import torch.nn as nn
import torch.optim as optim
from models.unet import Unet
from models.losses import DiceBCELoss
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
                 patch_size=(32, 128, 128)):
        
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
                                                            val_dir=os.path.join(Path(datadir), "tt"),
                                                            patch_size=(32, 128, 128),
                                                            num_samples=2
                                                        )
        self.optimizer = optim.AdamW(self.model.parameters())
        self.criterion = DiceBCELoss()

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
        best=100
        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0

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


                loss = self.criterion(outputs, targets)
                loss.backward()

                self.optimizer.step()
                #self.update_ema()

                running_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())

            avg_loss = running_loss / len(self.train_loader)

            wandb.log({
                "epoch": epoch + 1,
                "train/loss": avg_loss,
            })

            print(f"[Epoch {epoch+1}] Train Loss: {avg_loss:.4f}")
            # if epoch>=10:
            #     self.validate(epoch)
            torch.save(self.model.state_dict(), "ckpts/model_ckpt.pth")
            if avg_loss < best:
                best=avg_loss
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

        with torch.no_grad():
            for i, batch in enumerate(progress_bar):
                ct, pet, targets, pth = batch['ct'], batch['pet'], batch['seg'], batch['pth']
                pth = pth +"/CTres.nii.gz"

                inputs = torch.cat((ct,pet), dim=1).to(self.device)
                targets = targets.to(self.device)
                outputs = sliding_window_inference(
                    inputs=inputs, 
                    roi_size=self.patch_size,  
                    sw_batch_size=1, 
                    predictor=self.model, 
                    overlap=0.25
                    )
                # Compute metrics
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

                # Log one sample per epoch
                if i == 0:
                    image_np = inputs[0, 0].cpu().numpy()
                    label_np = targets[0, 0].cpu().numpy()
                    pred_np = get_binary_preds(outputs)[0].cpu().numpy()

                    middle = image_np.shape[0] // 2
                    slice_label = label_np[middle]
                    slice_pred = pred_np[middle]

                    wandb.log({
                        f"val/label_epoch_{epoch+1}": wandb.Image(slice_label * 255, caption="Ground Truth"),
                        f"val/pred_epoch_{epoch+1}": wandb.Image(slice_pred * 255, caption="Prediction"),
                    })

                # Save NIfTI volumes (input, label, prediction) for a few samples
                if saved_nifti < max_nifti_to_save:
                    affine = nib.load(pth).affine
                    label_np = targets[0, 0].cpu().numpy()
                    pred_np = get_binary_preds(outputs)[0].cpu().numpy()

                    base_name = f"epoch{epoch+1}_sample{i}"
                    nib.save(nib.Nifti1Image(label_np.astype(np.uint8), affine=affine), os.path.join(save_dir, f"{base_name}_label.nii.gz"))
                    nib.save(nib.Nifti1Image(pred_np.astype(np.uint8), affine=affine), os.path.join(save_dir, f"{base_name}_pred.nii.gz"))
                    saved_nifti += 1

        n = len(self.val_loader)
        avg_metrics = {
            "val/dice": total_dice / n,
            "val/iou": total_iou / n,
            "val/accuracy": total_acc / n,
            "val/precision": total_precision / n,
            "val/recall": total_recall / n,
            "val/inference_time_per_volume": total_inference_time / n,
        }

        wandb.log(avg_metrics)
        print(f"[Epoch {epoch+1}] Validation Dice: {avg_metrics['val/dice']:.4f}, IoU: {avg_metrics['val/iou']:.4f}")


    def save_train_sample(self, index=0, save_dir="sample_outputs"):
        """
        Saves a single training sample's CT, PET, and segmentation mask as NIfTI files.

        Args:
            index (int): Index of the sample to save from the training dataset.
            save_dir (str): Directory where the files will be saved.
        """
        os.makedirs(save_dir, exist_ok=True)

        # Get the dataset directly from the DataLoader's dataset
        dataset = self.val_loader.dataset
        sample = dataset[index]  # Returns a dict: {"ct", "pet", "seg", "pth"}

        ct = sample["ct"].numpy()       # Shape: (1, D, H, W)
        pet = sample["pet"].numpy()     # Shape: (1, D, H, W)
        seg = sample["seg"].numpy()     # Shape: (1, D, H, W)
        pth = sample["pth"]+"/CTres.nii.gz"
        affine = nib.load(pth).affine

        # Remove channel dim for saving
        ct = np.squeeze(ct)
        pet = np.squeeze(pet)
        seg = np.squeeze(seg)
        
        print(affine)
        # Save to NIfTI
        nib.save(nib.Nifti1Image(ct.astype(np.float32), affine=affine), 
                os.path.join(save_dir, f"sample_{index}_ct.nii.gz"))
        nib.save(nib.Nifti1Image(pet.astype(np.float32), affine=affine), 
                os.path.join(save_dir, f"sample_{index}_pet.nii.gz"))
        nib.save(nib.Nifti1Image(seg.astype(np.uint8), affine=affine), 
                os.path.join(save_dir, f"sample_{index}_seg.nii.gz"))

        print(f"Saved CT, PET, and SEG for sample {index} in {save_dir}")


def main():
    datadir = "../../../Storage/fdg_pet_ct/FDG-PET-CT-Lesions"
    model = Unet(in_chans=2, 
                 out_chans=1, 
                 chans=32, 
                 num_pool_layers=4, 
                 use_att=True, 
                 use_res=True, 
                 leaky_negative_slope=0.2)
    
    trainer = Trainer(model, datadir, device="cuda")
    trainer.save_train_sample(index=0, save_dir="train_sample")
    #trainer.train()
    #trainer.validate(1)



if __name__ == "__main__":
    main()