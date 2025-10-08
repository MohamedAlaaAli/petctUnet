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
    ScaleIntensityRangeD,
    ResampleToMatchD
)
from scipy import ndimage
import torch




import SimpleITK as sitk



def collect_study_paths(root_dir):
    samples = []
    for patient in os.listdir(root_dir):
        patient_path = os.path.join(root_dir, patient)
        if not os.path.isdir(patient_path):
            continue

        nii_files = glob(os.path.join(patient_path, "*.nii*"))
        pet = next((f for f in nii_files if "pt" in f.lower()), None)
        ct  = next((f for f in nii_files if "ct" in f.lower()), None)
        seg = next((f for f in nii_files if f.lower().endswith(".nii.gz") 
                    and "ct" not in f.lower() and "pt" not in f.lower()), None)

        if pet and ct:
            samples.append({
                "patient": patient,
                "pet": pet,
                "ct": ct,
                "seg": seg if seg else None,
                "pth": patient_path
            })
    return samples

def create_dataset(rootdir):

    samples = collect_study_paths(rootdir)
    val_transforms = Compose([
        LoadImaged(keys=["pet", "ct", "seg", ]),
        EnsureChannelFirstd(keys=["pet", "ct", "seg"]),
        ResampleToMatchD(keys=["ct", "seg"], key_dst="pet", mode="trilinear"),
        NormalizeIntensityd(keys=["pet"], nonzero=True, channel_wise=True),
        ScaleIntensityRangeD(keys=["ct"], a_min=-1024, a_max=1024,
                         b_min=0.0, b_max=1.0, clip=True),
        ToTensord(keys=["pet", "ct", "seg"]),
    ])

    ds = Dataset(data=samples, transform=val_transforms)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=8)
    return loader






# datadir = "/home/Storage/hecktordata/HECKTOR 2025 Training Data/Task 1"
# loader = create_dataset(datadir)
# l = next(iter(loader))
# print(l["ct"].shape, l["pet"].shape, l["seg"].shape)
