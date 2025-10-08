import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom, label
import pydicom
from pydicom.dataset import FileDataset
from pydicom.uid import generate_uid
import datetime

def create_sc_from_segmentation(ct_path, mask_path, output_dir="SC_Fusion", orientation="axial"):
    """
    Generate Secondary Capture DICOM from CT and segmentation mask.

    Args:
        ct_path (str): Path to CT NIfTI.
        mask_path (str): Path to segmentation NIfTI.
        output_dir (str): Folder to save SC images.
        orientation (str): 'axial', 'coronal', or 'sagittal'.
    """
    os.makedirs(output_dir, exist_ok=True)

    # --- Load ---
    ct = nib.load(ct_path).get_fdata().copy()
    mask = nib.load(mask_path).get_fdata().astype(np.uint8)

    # --- Resample mask to CT shape if needed ---
    if mask.shape != ct.shape:
        zoom_factors = np.array(ct.shape) / np.array(mask.shape)
        mask = zoom(mask, zoom_factors, order=0)
        print(f"Resampled mask to CT shape: {ct.shape}")

    # --- Normalize CT to grayscale ---
    ct_clip = np.clip(ct, -100, 400)  # soft-tissue window
    ct_norm = (ct_clip - np.min(ct_clip)) / np.ptp(ct_clip)
    ct_gray = np.stack([ct_norm] * 3, axis=-1)

    # --- Overlay segmentation mask (semi-transparent red) ---
    alpha = 0.5
    overlay = ct_gray.copy()
    red = np.array([1, 0, 0])
    overlay[mask > 0] = (1 - alpha) * ct_gray[mask > 0] + alpha * red

    # --- Find connected components (individual tumors) ---
    labeled, n_components = label(mask)
    if n_components == 0:
        raise ValueError("No tumor found in segmentation mask.")
    print(f"Detected {n_components} tumor components")

    # --- Loop through each tumor ---
    for i in range(1, n_components + 1):
        tumor_coords = np.where(labeled == i)
        if len(tumor_coords[0]) == 0:
            continue

        # --- Select slice depending on orientation ---
        if orientation == "axial":
            zmin, zmax = np.min(tumor_coords[2]), np.max(tumor_coords[2])
            mid = (zmin + zmax) // 2
            slice_rgb = (overlay[:, :, mid] * 255).astype(np.uint8)
            slice_rgb = np.rot90(slice_rgb, 1)

        elif orientation == "coronal":
            ymin, ymax = np.min(tumor_coords[1]), np.max(tumor_coords[1])
            mid = (ymin + ymax) // 2
            slice_rgb = (overlay[:, mid, :] * 255).astype(np.uint8)
            slice_rgb = np.rot90(slice_rgb, 1)

        elif orientation == "sagittal":
            xmin, xmax = np.min(tumor_coords[0]), np.max(tumor_coords[0])
            mid = (xmin + xmax) // 2
            slice_rgb = (overlay[mid, :, :] * 255).astype(np.uint8)
            slice_rgb = np.rot90(slice_rgb, 1)

        else:
            raise ValueError("Orientation must be 'axial', 'coronal', or 'sagittal'")

        # --- Show preview ---
        plt.imshow(slice_rgb)
        plt.title(f"Tumor {i} ({orientation} view)")
        plt.axis("off")
        plt.show()
        # --- Create DICOM SC ---
        file_meta = pydicom.Dataset()
        ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)

        ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.7"  # Secondary Capture Image Storage
        ds.SOPInstanceUID = generate_uid()
        ds.StudyInstanceUID = generate_uid()
        ds.SeriesInstanceUID = generate_uid()

        ds.Modality = "SC"
        ds.PatientName = "Anonymous"
        ds.PatientID = "0001"
        ds.StudyDate = datetime.datetime.now().strftime("%Y%m%d")
        ds.ContentDate = ds.StudyDate
        ds.StudyTime = datetime.datetime.now().strftime("%H%M%S")

        ds.Rows, ds.Columns = slice_rgb.shape[:2]
        ds.SamplesPerPixel = 3
        ds.Photometcreate_sc_from_segmentationricInterpretation = "RGB"
        ds.PlanarConfiguration = 0
        ds.BitsAllocated = 8
        ds.BitsStored = 8
        ds.HighBit = 7
        ds.PixelRepresentation = 0
        ds.PixelData = slice_rgb.tobytes()

        output_path = os.path.join(output_dir, f"SC_tumor{i}_{orientation}.dcm")
        ds.save_as(output_path)
        print(f"✅ Saved: {output_path}")




create_sc_from_segmentation(
    ct_path="val_v/CTres069.nii.gz",
    mask_path="val_v/PRED069.nii.gz",
    orientation="coronal"
)
