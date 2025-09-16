import os, random
import torch
import nibabel as nib
import pandas as pd
from tqdm import tqdm
from models.unet import Unet
from monai.inferers import sliding_window_inference
from utils.data import create_petct_datasets
from utils.metrics import compute_metrics
import wandb
from pathlib import Path
from scipy.ndimage import label
import numpy as np
from scipy.ndimage import find_objects


class Evaluator:
    def __init__(self, model, model_dir, data_dir, device="cuda", patch_size=(128,128,128)):
        self.model = model.to(device)
        self.device = device
        self.patch_size = patch_size
        self.data_dir = data_dir
        self._load_model(model_dir)
        self.train_loader, self.val_loader = create_petct_datasets(
            train_dir=os.path.join(Path(data_dir), "train"),
            val_dir=os.path.join(Path(data_dir), "val"),
            patch_size=self.patch_size,
            num_samples=1
        )
        self.tnr = []
        self.accumulator = []


    def calculate_suvpeak(self, suv, mask, voxel_volume):
        """
        Calculate SUVpeak for a single lesion.
        
        Args:
            suv: 3D NumPy array of SUV values.
            mask: 3D binary mask (same shape as suv) of the lesion.
            voxel_volume: volume of a single voxel in mL.
        
        Returns:
            suvpeak: float, the SUVpeak value.
        """
        # Extract SUV values inside the lesion
        suv_values = suv[mask > 0]
        
        if len(suv_values) == 0:
            return 0.0  # No voxels in lesion
        
        # Number of voxels in 1 cm³
        voxels_in_1cm3 = int(np.round(1 / voxel_volume))
        
        # Sort voxel values descending
        suv_sorted = np.sort(suv_values)[::-1]
        
        # Take the mean of top N voxels corresponding to 1 cm³
        top_voxels = suv_sorted[:voxels_in_1cm3]
        suvpeak = top_voxels.mean()
        
        return suvpeak


    def _load_model(self, model_dir):
        try:
            state_dict = torch.load(model_dir, map_location=self.device)
            self.model.load_state_dict(state_dict, strict=False)
            print(f"Loaded model from checkpoint: {model_dir}")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            pass

    @torch.no_grad()
    def infer_batch(self, inputs):
        inputs = inputs.to(self.device)
        outputs = sliding_window_inference(
            inputs=inputs,
            roi_size=self.patch_size,
            sw_batch_size=1,
            predictor=self.model,
            overlap=0.75,
            progress=True
        )
        return outputs

    def postprocess_outputs(self, outputs, threshold=0.2):
        return (torch.sigmoid(outputs) > threshold).float()

    def get_components(self, mask):
        """Return list of connected lesion components."""
        structure = np.ones((3,3,3), dtype=np.int32)
        labeled, n = label(mask, structure=structure)
        lesions = [(labeled == i).astype(np.uint8) for i in range(1, n+1)]
        return lesions

    def iou_score(self, a, b):
        inter = np.logical_and(a, b).sum()
        union = np.logical_or(a, b).sum()
        return inter / union if union > 0 else 0

    def match_lesions(self, gt_lesions, pred_lesions, iou_thresh=0.1):
        matches = []
        used_preds = set()
        for gi, g in enumerate(gt_lesions):
            best_iou, best_j = 0, None
            for pj, p in enumerate(pred_lesions):
                if pj in used_preds: continue
                iou = self.iou_score(g, p)
                if iou > best_iou:
                    best_iou, best_j = iou, pj
            if best_iou >= iou_thresh:
                matches.append((gi, best_j, best_iou))
                used_preds.add(best_j)
        return matches

    def compute_detection_rate_from_matches(self, gt_lesions, pred_lesions, matched_lesions):
        total_gt = len(gt_lesions)
        detected_gt = len(set([gi for gi, _, _ in matched_lesions]))
        return detected_gt / total_gt if total_gt > 0 else None

    def compute_batch_metrics(self, preds, targets, compute_metrics_fn, suv,spacing=None, affine=None):
        preds_np = preds.cpu().detach().numpy().squeeze().astype(np.uint8)
        targets_np = targets.cpu().detach().numpy().squeeze().astype(np.uint8)

        # --- Handle non-tumor patients ---
        if targets_np.sum() == 0:
            self.tnr.append(1 if preds_np.sum() == 0 else 0)
            return None

        # --- Lesion metrics ---
        gt_lesions = self.get_components(targets_np)
        pred_lesions = self.get_components(preds_np)
        matched_lesions = self.match_lesions(gt_lesions, pred_lesions, iou_thresh=0.1)
        detection_rate = self.compute_detection_rate_from_matches(gt_lesions, pred_lesions, matched_lesions)

        voxel_volume = np.prod(spacing) / 1000  # mm^3 -> mL
        suv_threshold = 2.5  # SUV threshold for MTV
        mtv_pred, mtv_gt = 0.0, 0.0
        vsis = []

        per_finding_metrics_list = []

        voxels_in_1cm3 = int(np.round(1 / voxel_volume))  # For SUVpeak

        for gi, pj, _ in matched_lesions:
            suv_np = suv.cpu().numpy().squeeze()

            # Extract voxel values inside lesion masks
            suv_values_gt = suv_np[gt_lesions[gi] > 0]
            suv_values_pred = suv_np[pred_lesions[pj] > 0]

            # MTV calculation (thresholded)
            suv_values_gt_thr = suv_values_gt[suv_values_gt > suv_threshold]
            suv_values_pred_thr = suv_values_pred[suv_values_pred > suv_threshold]

            vol_gt = suv_values_gt_thr.size * voxel_volume
            vol_pred = suv_values_pred_thr.size * voxel_volume
            mtv_gt += vol_gt
            mtv_pred += vol_pred

            # VSI
            if vol_gt > 0:
                vsi = 1 - abs(vol_pred - vol_gt) / (vol_pred + vol_gt)
                vsis.append(vsi)

            # SUV statistics (all lesion voxels)
            suvmax_gt = suv_values_gt.max() if suv_values_gt.size > 0 else 0
            suvmax_pred = suv_values_pred.max() if suv_values_pred.size > 0 else 0
            suvmean_gt = suv_values_gt.mean() if suv_values_gt.size > 0 else 0
            suvmean_pred = suv_values_pred.mean() if suv_values_pred.size > 0 else 0

            # SUVpeak (mean of top voxels_in_1cm3 voxels)
            suvpeak_gt = suv_values_gt.flatten()
            suvpeak_gt = np.sort(suvpeak_gt)[-voxels_in_1cm3:].mean() if suvpeak_gt.size >= voxels_in_1cm3 else suvpeak_gt.mean()

            suvpeak_pred = suv_values_pred.flatten()
            suvpeak_pred = np.sort(suvpeak_pred)[-voxels_in_1cm3:].mean() if suvpeak_pred.size >= voxels_in_1cm3 else suvpeak_pred.mean()

            # --- Lesion diameter (longest axis) ---
            coords_gt = np.argwhere(gt_lesions[gi] > 0)
            coords_pred = np.argwhere(pred_lesions[pj] > 0)
            diameter_gt = max((coords_gt[:, i].max() - coords_gt[:, i].min() + 1) * spacing[i] for i in range(3)) if coords_gt.size > 0 else 0
            diameter_pred = max((coords_pred[:, i].max() - coords_pred[:, i].min() + 1) * spacing[i] for i in range(3)) if coords_pred.size > 0 else 0

            suv_std_gt = suv_values_gt.std() if suv_values_gt.size > 0 else 0
            suv_std_pred = suv_values_pred.std() if suv_values_pred.size > 0 else 0

            per_finding_metrics_list.append(
                {
                "val/perfinding/mtv_gt": vol_gt,
                "val/perfinding/mtv_pred": vol_pred,
                "val/perfinding/suvmax_gt": suvmax_gt,
                "val/perfinding/suvmax_pred": suvmax_pred,
                "val/perfinding/suvmean_gt": suvmean_gt,
                "val/perfinding/suvmean_pred": suvmean_pred,
                "val/perfinding/suvpeak_gt": suvpeak_gt,
                "val/perfinding/suvpeak_pred": suvpeak_pred,
                "val/perfinding/tlg_gt": suvmean_gt*vol_gt,
                "val/perfinding/tlg_pred": suvmean_pred*vol_pred,
                "val/perfinding/diameter_gt": diameter_gt,
                "val/perfinding/diameter_pred": diameter_pred,
                "val/perfinding/suv_std_gt": suv_std_gt,
                "val/perfinding/suv_std_pred": suv_std_pred,

                }
            )

        csv_path = "per_finding_metrics.csv"

        # write header if file doesn't exist
        if not os.path.exists(csv_path) and per_finding_metrics_list:
            with open(csv_path, "w") as f:
                f.write(",".join(per_finding_metrics_list[0].keys()) + "\n")

        # append per-finding metrics
        with open(csv_path, "a") as f:
            for m in per_finding_metrics_list:
                line = ",".join(str(v) for v in m.values())
                f.write(line + "\n")
            # add a blank line to separate batches
            f.write("\n")

        metrics1 = {
                "val/volume_vsi": np.mean(vsis) if vsis else 0.0,
                "val/total_mtv_predicted": mtv_pred,
                "val/total_mtv_gt": mtv_gt,
                "val/detection_rate": detection_rate if detection_rate is not None else 0.0
            }

        self.accumulator.append([mtv_pred, mtv_gt])
        metrics2 = compute_metrics_fn(preds, targets)
        return {**metrics1, **metrics2}

    def categorize_metrics(self, results, pth, df, melanoma, lung, lymph):
        after_val = pth[0].split("val/")[1]
        result = df.loc[df["File Location"].str.contains(after_val), "diagnosis"]
        if result.empty: raise ValueError("Extraction error")
        diagnosis = result.iloc[0]
        if diagnosis == "MELANOMA":
            melanoma.append(results["dice"])
        elif diagnosis == "LUNG_CANCER":
            lung.append(results["dice"])
        elif diagnosis == "LYMPHOMA":
            lymph.append(results["dice"])
        else:
            raise ValueError("Unknown diagnosis")

    def save_sample_lesions(self, lesions, affine, out_dir='cca', prefix="gt"):
        os.makedirs(out_dir, exist_ok=True)
        for i, lesion in enumerate(lesions):
            out_path = os.path.join(out_dir, f"{prefix}_lesion_{i+1}.nii.gz")
            nib.save(nib.Nifti1Image(lesion.astype(np.uint8), affine), out_path)

    def save_nifti_samples(self, ct, pet, targets, preds, affine, save_dir, epoch, idx):
        os.makedirs(save_dir, exist_ok=True)
        nib.save(nib.Nifti1Image(ct.cpu().squeeze().numpy(), affine), f"{save_dir}/CTres{epoch}_{idx}.nii.gz")
        nib.save(nib.Nifti1Image(pet.cpu().squeeze().numpy(), affine), f"{save_dir}/PET{epoch}_{idx}.nii.gz")
        nib.save(nib.Nifti1Image(targets.cpu().squeeze().numpy(), affine), f"{save_dir}/GT{epoch}_{idx}.nii.gz")
        nib.save(nib.Nifti1Image(preds.cpu().squeeze().numpy(), affine), f"{save_dir}/PRED{epoch}_{idx}.nii.gz")

        # save components
        self.save_sample_lesions(self.get_components(targets.cpu().detach().numpy().squeeze()), affine, f"{save_dir}_{epoch}_{idx}", 'gt')
        self.save_sample_lesions(self.get_components(preds.cpu().detach().numpy().squeeze()), affine, f"{save_dir}_{epoch}_{idx}", 'pred')

    def validate(self, epoch, compute_metrics_fn, save_dir="cca",
                 max_nifti_to_save=200, per_category=True, log_wandb=True):
        self.model.eval()
        if per_category:
            df = pd.read_csv("fdg_metadata.csv")
            melanoma, lymph, lung = [], [], []

        total_dice, total_precision, total_recall, total_iou = [], [], [], []

        step = 0
        for batch in tqdm(self.val_loader, desc=f"[Epoch {epoch+1}] Validation", leave=False):
            ct, pet, targets, pth, suv = batch['ct'], batch['pet'], batch['seg'], batch['pth'], batch['suv']
            ni = nib.load(pth[0]+"/CTres.nii.gz")
            affine = ni.affine
            spacing = ni.header.get_zooms()
            inputs = torch.cat((ct, pet), dim=1).to(self.device)
            targets = targets.to(self.device)

            outputs = self.infer_batch(inputs)
            preds = self.postprocess_outputs(outputs)

            results = self.compute_batch_metrics(preds, targets, compute_metrics_fn, suv=suv, spacing=spacing, affine=affine)
            if results:
                total_dice.append(results["dice"])
                total_precision.append(results["precision"])
                total_recall.append(results["recall"])
                total_iou.append(results["iou"])

                if per_category:
                    self.categorize_metrics(results, pth, df, melanoma, lung, lymph)

                if log_wandb:
                    wandb.log({"val/batch_results": results})
                    print("logged")

            # --- Save NIfTI samples ---
            if max_nifti_to_save > 0 and random.choice([True, False]) and targets.sum() > 0:
                max_nifti_to_save -= 1
                self.save_nifti_samples(ct, pet, targets, preds, affine, save_dir, epoch, max_nifti_to_save)

            step += 1

        # --- Aggregate metrics ---
        avg_metrics = {
            "val/dice": np.mean(total_dice) if total_dice else None,
            "val/precision": np.mean(total_precision) if total_precision else None,
            "val/recall": np.mean(total_recall) if total_recall else None,
            "val/iou": np.mean(total_iou) if total_iou else None,
        }
        if per_category:
            avg_metrics.update({
                "val/dice_mela": np.mean(melanoma) if melanoma else None,
                "val/dice_lym": np.mean(lymph) if lymph else None,
                "val/dice_lung": np.mean(lung) if lung else None,
            })

        if log_wandb:
            wandb.log(avg_metrics, step=step)

        print(f"[Epoch {epoch+1}] Validation Dice: {avg_metrics['val/dice']:.4f}")
        return avg_metrics


if __name__ == "__main__":
    model = Unet(in_chans=2, out_chans=1, chans=32, num_pool_layers=4, use_att=True, use_res=True, leaky_negative_slope=0)
    evaluator = Evaluator(
        model=model,
        model_dir="ckpts/injectPet/focaltvskt.pth",
        data_dir="../../../Storage/fdg_pet_ct/FDG-PET-CT-Lesions",
        device="cuda" if torch.cuda.is_available() else "cpu",
        patch_size=(128,128,128)
    )

    wandb.init(project="petct-eval", name="eval_run_001", config={"patch_size": (128,128,128), "threshold":0.2})

    metrics = evaluator.validate(epoch=0, compute_metrics_fn=compute_metrics, save_dir="nifti_predictions", max_nifti_to_save=2, per_category=True, log_wandb=True)

    print("Validation metrics:", metrics)
    print("TNR:", np.mean(evaluator.tnr))

    mtvs = np.array(evaluator.accumulator)
    gt, pred = mtvs[:,1], mtvs[:,0]
    from scipy.stats import spearmanr, pearsonr
    spearman_corr, spearman_p = spearmanr(gt, pred)
    pearson_corr, pearson_p = pearsonr(gt, pred)
    gt_mean, gt_std = gt.mean(), gt.std()
    pred_mean, pred_std = pred.mean(), pred.std()

    print(f"Spearman r = {spearman_corr:.4f}, p = {spearman_p:.4e}")
    print(f"Pearson r = {pearson_corr:.4f}, p = {pearson_p:.4e}")
    print(f"GT MTV Mean = {gt_mean:.2f}, Std = {gt_std:.2f}")
    print(f"Pred MTV Mean = {pred_mean:.2f}, Std = {pred_std:.2f}")

    table = wandb.Table(data=[[g, p] for g, p in zip(gt, pred)], columns=["GT_MTV","Pred_MTV"])
    wandb.log({
        "val/mtv_spearman_r": spearman_corr,
        "val/mtv_spearman_p": spearman_p,
        "val/mtv_pearson_r": pearson_corr,
        "val/mtv_pearson_p": pearson_p,
        "val/mtv_gt_mean": gt_mean,
        "val/mtv_gt_std": gt_std,
        "val/mtv_pred_mean": pred_mean,
        "val/mtv_pred_std": pred_std,
        "val/mtv_scatter": wandb.plot.scatter(table, "GT_MTV","Pred_MTV", title="GT vs Pred MTV")
    })
