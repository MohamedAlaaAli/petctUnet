import os
import numpy as np
import nibabel as nib
from scipy.ndimage import center_of_mass

def get_approximate_location(z, total_slices):
    if z < total_slices * 0.25:
        return "pelvic region"
    elif z < total_slices * 0.5:
        return "abdominal region"
    elif z < total_slices * 0.75:
        return "thoracic (chest) region"
    else:
        return "neck/head region"

def generate_petct_report(pet_path, suv_path, mask_path):
    # Load the NIfTI files
    pet_img = nib.load(pet_path)
    suv_img = nib.load(suv_path)
    mask_img = nib.load(mask_path)

    # Extract image data
    pet_data = pet_img.get_fdata()
    suv_data = suv_img.get_fdata()
    mask_data = mask_img.get_fdata()

    assert pet_data.shape == suv_data.shape == mask_data.shape, "Input images must have the same shape."

    tumor_mask = mask_data > 0
    if not np.any(tumor_mask):
        report = "No suspicious lesion detected."
    else:
        suv_values = suv_data[tumor_mask]
        pet_values = pet_data[tumor_mask]

        voxel_volume_mm3 = np.prod(mask_img.header.get_zooms()[:3])
        volume_cm3 = np.sum(tumor_mask) * voxel_volume_mm3 / 1000

        center = center_of_mass(tumor_mask)
        z_location = get_approximate_location(center[2], pet_data.shape[2])

        report = (
            f"Lesion detected in the {z_location} with an estimated volume of {round(volume_cm3, 2)} cm³. "
            f"The maximum SUV is {round(float(np.max(suv_values)), 2)}, "
            f"with a mean of {round(float(np.mean(suv_values)), 2)} and a standard deviation of {round(float(np.std(suv_values)), 2)}. "
        )

    # Save report to text file in same directory as mask
    output_path = os.path.join(os.path.dirname(mask_path), "lesion_report.txt")
    with open(output_path, "w") as f:
        f.write(report)

    print(f"Report saved to {output_path}")

base = "/home/muhamed/mntdrive/zips/FDG-PET-CT-Lesions/PETCT_a61d8768d0/03-26-2004-NA-PET-CT Ganzkoerper  primaer mit KM-57711"

generate_petct_report(base+"/PET.nii.gz", base+"/SUV.nii.gz", base+"/SEG.nii.gz")
