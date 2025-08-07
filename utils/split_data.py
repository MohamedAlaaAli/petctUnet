import os
import shutil
import random
from tqdm import tqdm
import json


def split_dataset_in_place(
    dataset_dir,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42,
):

    random.seed(seed)

    train_dir = os.path.join(dataset_dir, 'train')
    val_dir = os.path.join(dataset_dir, 'val')
    test_dir = os.path.join(dataset_dir, 'test')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    all_patients = [
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d)) and d not in ('train', 'val', 'test')
    ]

    random.shuffle(all_patients)

    total = len(all_patients)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_patients = all_patients[:train_end]
    val_patients = all_patients[train_end:val_end]
    test_patients = all_patients[val_end:]

    def move_patients(patients, target_dir, label):
        for patient in tqdm(patients, desc=f"Moving to {label}", unit="patient"):
            src = os.path.join(dataset_dir, patient)
            dst = os.path.join(target_dir, patient)
            shutil.move(src, dst)

    move_patients(train_patients, train_dir, "train")
    move_patients(val_patients, val_dir, "val")
    move_patients(test_patients, test_dir, "test")

    print(f"✅ Split complete (in: {dataset_dir})")
    print(f"  → Train: {len(train_patients)}")
    print(f"  → Val:   {len(val_patients)}")
    print(f"  → Test:  {len(test_patients)}")

    manifest = {
    "train": train_patients,
    "val": val_patients,
    "test": test_patients
    }

    with open(os.path.join(dataset_dir, "split_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":

    dataset_dir = "../../../Storage/fdg_pet_ct/FDG-PET-CT-Lesions"
    split_dataset_in_place(
        dataset_dir=dataset_dir,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )
