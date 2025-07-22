import os
import nibabel as nib
import numpy as np
import json
from tqdm import tqdm

def compute_statistics(image_data: np.ndarray):
    image_data = image_data.astype(np.float32)
    flat = image_data.flatten()
    flat = flat[~np.isnan(flat)]
    stats = {
        'mean': float(np.mean(flat)),
        'std': float(np.std(flat)),
        'percentile_00_5': float(np.percentile(flat, 0.5)),
        'percentile_99_5': float(np.percentile(flat, 99.5)),
    }
    return stats

def find_files(root_dir, filename):
    for dirpath, _, files in os.walk(root_dir):
        if filename in files:
            yield os.path.join(dirpath, filename)

def compute_modality_stats(root_dir):
    ct_stats = []
    pet_stats = []
    for ct_path in tqdm(find_files(root_dir, "CT.nii.gz"), desc="Processing CT"):
        ct_img = nib.load(ct_path).get_fdata()
        ct_stats.append(compute_statistics(ct_img))

    for pet_path in tqdm(find_files(root_dir, "PET.nii.gz"), desc="Processing PET"):
        pet_img = nib.load(pet_path).get_fdata()
        pet_stats.append(compute_statistics(pet_img))

    def aggregate(stats_list):
        means = np.array([s['mean'] for s in stats_list])
        stds = np.array([s['std'] for s in stats_list])
        p005 = np.array([s['percentile_00_5'] for s in stats_list])
        p995 = np.array([s['percentile_99_5'] for s in stats_list])

        return {
            'mean': float(np.mean(means)),
            'std': float(np.mean(stds)),
            'percentile_00_5': float(np.mean(p005)),
            'percentile_99_5': float(np.mean(p995)),
        }

    return {
        'CT': aggregate(ct_stats),
        'PET': aggregate(pet_stats)
    }

if __name__ == "__main__":
    dataset_root = "../dataset"  
    out_json = "intensity_properties.json"

    stats = compute_modality_stats(dataset_root)

    with open(out_json, "w") as f:
        json.dump(stats, f, indent=4)

    print(f"\nSaved intensity stats to {out_json}")
