import os
import torch
import nibabel as nib
from torch.utils.data import Dataset
import numpy as np
from models.text_model import TextEmbedder


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

                nii_files = [f for f in os.listdir(study_path) if f.endswith('.nii.gz')]

                pet_file = next((f for f in nii_files if 'pet' in f.lower()), None)
                ct_file = next((f for f in nii_files if 'ctres' in f.lower()), None)
                mask_file = next((f for f in nii_files if 'seg' in f.lower()), None)
                txt = os.path.join(study_path, "lesion_report.txt")

                if pet_file and ct_file:
                    self.samples.append({
                        'pet_path': os.path.join(study_path, pet_file),
                        'ct_path': os.path.join(study_path, ct_file),
                        'mask_path': os.path.join(study_path, mask_file) if mask_file else None,
                        "txt":txt
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        pet_vol = nib.load(sample['pet_path']).get_fdata().astype(np.float32)
        ct_vol = nib.load(sample['ct_path']).get_fdata().astype(np.float32)

        # Normalize PET and CT
        pet_vol = (pet_vol - pet_vol.min()) / (np.ptp(pet_vol) + 1e-6)
        ct_vol = (ct_vol - ct_vol.min()) / (np.ptp(ct_vol) + 1e-6)

        # Optional: Clip extreme values if needed

        # Load mask (if exists)
        if sample['mask_path'] is not None:
            mask_vol = nib.load(sample['mask_path']).get_fdata().astype(np.float32)
        else:
            mask_vol = np.zeros_like(ct_vol, dtype=np.float32)

        # Convert to tensors with shape [1, D, H, W]
        pet_tensor = torch.from_numpy(pet_vol).unsqueeze(0)
        ct_tensor = torch.from_numpy(ct_vol).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask_vol).unsqueeze(0)

        with open(sample["txt"], "r") as s:
            text = s.read()

        
        if self.transform:
            pet_tensor, ct_tensor, mask_tensor = self.transform(pet_tensor, ct_tensor, mask_tensor)

        return ct_tensor, pet_tensor, mask_tensor, text
