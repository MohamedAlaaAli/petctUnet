# import os
# import torch
# import nibabel as nib
# from torch.utils.data import Dataset
# import numpy as np
# from pathlib import Path


# class PETCTDataset(Dataset):
#     def __init__(self, root_dir, patch_size=None,transform=None):
#         """
#         Args:
#             root_dir (str): Root directory containing patients.
#             transform (callable, optional): Function to apply to all volumes (pet, ct, mask).
#         """
#         self.root_dir = root_dir
#         self.transform = transform
#         self.patch_size = patch_size
#         self.samples = []

#         for patient in os.listdir(root_dir):
#             patient_path = os.path.join(root_dir, patient)
#             if not os.path.isdir(patient_path):
#                 continue
#             for study in os.listdir(patient_path):
#                 study_path = os.path.join(patient_path, study)
#                 if not os.path.isdir(study_path):
#                     continue

#                 nii_files = [
#                     os.path.join(study_path, f, inner_f)
#                     if os.path.isdir(os.path.join(study_path, f)) else os.path.join(study_path, f)
#                     for f in os.listdir(study_path)
#                     for inner_f in ([os.listdir(os.path.join(study_path, f))[0]] if os.path.isdir(os.path.join(study_path, f)) else [f])
#                     if inner_f.endswith(('.nii', '.nii.gz'))
#                     ]
                                
#                 pet_file = next(
#                     (f for f in nii_files if Path(f).name.replace('.nii.gz', '').split('_')[-1].lower() == 'pet' or 
#                      Path(f).name.replace('.nii', '').split('_')[-1].lower() == 'pet'),
#                     None
#                 )
#                 ct_file = next((f for f in nii_files if 'ctres' in f.lower()), None)
#                 mask_file = next((f for f in nii_files if 'seg' in f.lower()), None)
                

#                 if pet_file and ct_file:
#                     self.samples.append({
#                     'pet_path': str(Path(pet_file).resolve()),
#                     'ct_path': str(Path(ct_file).resolve()),
#                     'mask_path': str(Path(mask_file).resolve()) if mask_file else None,
#                     })

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         sample = self.samples[idx]

#         pet_vol = nib.load(sample['pet_path']).get_fdata().astype(np.float32)
#         ct_vol = nib.load(sample['ct_path']).get_fdata().astype(np.float32)
        
#         pet_vol = (pet_vol - pet_vol.min()) / (np.ptp(pet_vol) + 1e-6)
#         ct_vol = (ct_vol - ct_vol.min()) / (np.ptp(ct_vol) + 1e-6)

#         if sample['mask_path'] is not None:
#             mask_vol = nib.load(sample['mask_path']).get_fdata().astype(np.float32)
#         else:
#             mask_vol = np.zeros_like(ct_vol, dtype=np.float32)

#         if self.patch_size:
#             D, H, W = pet_vol.shape
#             pd, ph, pw = self.patch_size

#             foreground_indices = np.argwhere(mask_vol > 0)

#             if foreground_indices.size > 0:
#                 # Pick a foreground voxel
#                 center = foreground_indices[np.random.choice(len(foreground_indices))]
#                 cd, ch, cw = center

#                 # Compute start and end positions (clipped to image boundaries)
#                 d_start = np.clip(cd - pd // 2, 0, D - pd)
#                 h_start = np.clip(ch - ph // 2, 0, H - ph)
#                 w_start = np.clip(cw - pw // 2, 0, W - pw)
#             else:
#                 # Fallback: random patch
#                 d_start = np.random.randint(0, max(1, D - pd + 1))
#                 h_start = np.random.randint(0, max(1, H - ph + 1))
#                 w_start = np.random.randint(0, max(1, W - pw + 1))

#             d_end, h_end, w_end = d_start + pd, h_start + ph, w_start + pw

#             pet_patch = pet_vol[d_start:d_end, h_start:h_end, w_start:w_end]
#             ct_patch = ct_vol[d_start:d_end, h_start:h_end, w_start:w_end]
#             mask_patch = mask_vol[d_start:d_end, h_start:h_end, w_start:w_end]

#             pet_tensor = torch.from_numpy(pet_patch).unsqueeze(0)
#             ct_tensor = torch.from_numpy(ct_patch).unsqueeze(0)
#             mask_tensor = torch.from_numpy(mask_patch).unsqueeze(0)

#         else:
#             pet_tensor = torch.from_numpy(pet_vol).unsqueeze(0)
#             ct_tensor = torch.from_numpy(ct_vol).unsqueeze(0)
#             mask_tensor = torch.from_numpy(mask_vol).unsqueeze(0)

#         if self.transform:
#             pet_tensor, ct_tensor, mask_tensor = self.transform(pet_tensor, ct_tensor, mask_tensor)

#         return ct_tensor, pet_tensor, mask_tensor, sample['mask_path']


import os
import numpy as np
from pathlib import Path
from glob import glob
from monai.data import Dataset, CacheDataset, DataLoader
from monai.transforms import RandCropByPosNegLabeld
from monai.transforms import (
    LoadImaged,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    Compose,
    ToTensord,
    LambdaD,
    ScaleIntensityRangeD
)
from scipy import ndimage
import torch


def get_small_tumor_weights(mask, small_threshold=100, weight_factor=1.5):
    """
    Generate weight map that gives higher weights to small tumors.
    
    Args:
        mask: numpy array or torch tensor of shape (H, W, D) with binary mask
        small_threshold: lesions smaller than this (in voxels) get extra weight
        weight_factor: multiplier for small lesions (e.g., 3.0 = 3x weight)
    
    Returns:
        weight_map: same shape as mask, with higher weights for small lesions
    """
    
    # Convert to numpy if torch tensor
    if isinstance(mask, torch.Tensor):
        mask_np = mask.cpu().numpy()
        return_torch = True
    else:
        mask_np = mask
        return_torch = False
    
    # Initialize weight map with ones (normal weight)
    weight_map = np.ones_like(mask_np, dtype=np.float32)
    
    if mask_np.sum() == 0:  # no lesions in this patch
        if return_torch:
            return torch.from_numpy(weight_map)
        return weight_map
    
    # Label connected components (individual lesions)
    labeled_lesions, num_lesions = ndimage.label(mask_np)
    
    # For each lesion, check size and apply weight
    for lesion_id in range(1, num_lesions + 1):
        lesion_mask = (labeled_lesions == lesion_id)
        lesion_size = lesion_mask.sum()
        
        if lesion_size < small_threshold:
            # Apply higher weight to small lesions
            weight_map[lesion_mask] = weight_factor
    
    if return_torch:
        return torch.from_numpy(weight_map)
    
    return weight_map


def collect_study_paths(root_dir):
    samples = []
    for patient in os.listdir(root_dir):
        patient_path = os.path.join(root_dir, patient)
        if not os.path.isdir(patient_path):
            continue
        for study in os.listdir(patient_path):
            study_path = os.path.join(patient_path, study)
            if not os.path.isdir(study_path):
                continue

            nii_files = glob(os.path.join(study_path, "**/*.nii*"), recursive=True)
            pet  = next(
                    (f for f in nii_files if Path(f).name.replace('.nii.gz', '').split('_')[-1].lower() == 'pet' or 
                     Path(f).name.replace('.nii', '').split('_')[-1].lower() == 'pet'),
                    None
                )
            ct = next((f for f in nii_files if 'ctres' in f.lower()), None)
            seg = next((f for f in nii_files if 'seg' in f.lower()), None)
            suv = next((f for f in nii_files if 'suv' in f.lower()), None)
            if pet and ct:
                samples.append({
                    "pet": pet,
                    "ct": ct,
                    "seg": seg if seg else None,  # fallback if missing mask
                    "pth": study_path,
                    "suv": suv
                })
    return samples


def create_petct_datasets(
    train_dir,
    val_dir,
    patch_size=(128, 128, 32),
    num_samples=1,
    cache_train=False,
    cache_val=False,
    batch_size=1,
):
    # -- collect samples
    train_samples = collect_study_paths(train_dir)
    val_samples = collect_study_paths(val_dir)

    # -- train transforms
    train_transforms = Compose([
        LoadImaged(keys=["pet", "ct", "seg"]),
        EnsureChannelFirstd(keys=["pet", "ct", "seg"]),
        NormalizeIntensityd(keys=["pet"], nonzero=True, channel_wise=True),
        ScaleIntensityRangeD(keys=["ct"], a_min=-1024, a_max=1024,
                         b_min=0.0, b_max=1.0, clip=True),

        RandCropByPosNegLabeld(
            keys=["pet", "ct", "seg"],
            label_key="seg",
            spatial_size=patch_size,
            pos=10,
            neg=1,
            num_samples=num_samples,
            image_key="pet",
            allow_smaller=True,
        ),
        ToTensord(keys=["pet", "ct", "seg"]),
    ])

    # -- validation transforms: full volume
    val_transforms = Compose([
        LoadImaged(keys=["pet", "ct", "seg", "suv"]),
        EnsureChannelFirstd(keys=["pet", "ct", "seg", "suv"]),
        NormalizeIntensityd(keys=["pet"], nonzero=True, channel_wise=True),
        ScaleIntensityRangeD(keys=["ct"], a_min=-1024, a_max=1024,
                         b_min=0.0, b_max=1.0, clip=True),
        ToTensord(keys=["pet", "ct", "seg", "suv"]),
    ])

    # -- choose dataset class
    TrainDataset = CacheDataset if cache_train else Dataset
    ValDataset = CacheDataset if cache_val else Dataset

    train_ds = TrainDataset(data=train_samples, transform=train_transforms)
    val_ds = ValDataset(data=val_samples, transform=val_transforms)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=8)

    return train_loader, val_loader

# datadir = "../../../Storage/fdg_pet_ct/FDG-PET-CT-Lesions"
# train_loader, val_loader = create_petct_datasets(datadir+"/train", datadir+"/val")

# for batch in train_loader:
#     print("Train batch shape:", batch["ct"].shape) 
#     break

# for batch in val_loader:
#     print("Val batch shape:", batch["ct"].shape)    
#     break
