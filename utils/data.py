import os
import torch
import nibabel as nib
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path


class PETCTDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Root directory containing patients.
            transform (callable, optional): Function to apply to all volumes (pet, ct, mask).
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        for patient in os.listdir(root_dir):
            patient_path = os.path.join(root_dir, patient)
            if not os.path.isdir(patient_path):
                continue
            for study in os.listdir(patient_path):
                study_path = os.path.join(patient_path, study)
                if not os.path.isdir(study_path):
                    continue

                nii_files = [
                    os.path.join(study_path, f, inner_f)
                    if os.path.isdir(os.path.join(study_path, f)) else os.path.join(study_path, f)
                    for f in os.listdir(study_path)
                    for inner_f in ([os.listdir(os.path.join(study_path, f))[0]] if os.path.isdir(os.path.join(study_path, f)) else [f])
                    if inner_f.endswith(('.nii', '.nii.gz'))
                    ]
                                
                pet_file = next(
                    (f for f in nii_files if Path(f).name.replace('.nii.gz', '').split('_')[-1].lower() == 'pet' or 
                     Path(f).name.replace('.nii', '').split('_')[-1].lower() == 'pet'),
                    None
                )
                ct_file = next((f for f in nii_files if 'ctres' in f.lower()), None)
                mask_file = next((f for f in nii_files if 'seg' in f.lower()), None)
                

                if pet_file and ct_file:
                    self.samples.append({
                        'pet_path': os.path.join(study_path, pet_file),
                        'ct_path': os.path.join(study_path, ct_file),
                        'mask_path': os.path.join(study_path, mask_file) if mask_file else None,
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        pet_vol = nib.load(sample['pet_path']).get_fdata().astype(np.float32)
        ct_vol = nib.load(sample['ct_path']).get_fdata().astype(np.float32)
        print(sample["pet_path"])

        # Normalize PET and CT
        pet_vol = (pet_vol - pet_vol.min()) / (np.ptp(pet_vol) + 1e-6)
        ct_vol = (ct_vol - ct_vol.min()) / (np.ptp(ct_vol) + 1e-6)

        # Load mask (if exists)
        if sample['mask_path'] is not None:
            mask_vol = nib.load(sample['mask_path']).get_fdata().astype(np.float32)
        else:
            mask_vol = np.zeros_like(ct_vol, dtype=np.float32)

        # Convert to tensors with shape [1, D, H, W]
        pet_tensor = torch.from_numpy(pet_vol).unsqueeze(0).permute(0, 3, 1, 2)
        ct_tensor = torch.from_numpy(ct_vol).unsqueeze(0).permute(0, 3, 1, 2)
        mask_tensor = torch.from_numpy(mask_vol).unsqueeze(0).permute(0, 3, 1, 2)


        if self.transform:
            pet_tensor, ct_tensor, mask_tensor = self.transform(pet_tensor, ct_tensor, mask_tensor)

        return ct_tensor, pet_tensor, mask_tensor