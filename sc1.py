import os
import datetime
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import zoom, label, binary_erosion, center_of_mass
from scipy import ndimage
import pydicom
from pydicom.dataset import FileDataset
from pydicom.uid import generate_uid
import matplotlib.cm as cm
from PIL import ImageDraw


try:
    logo = Image.open("logo.png")
    logo_size = (300, 64)  
    logo_x = 1010 - logo_size[0]
    logo_y = 600-5

except Exception as e:
    print(f"Tumor {lab}: Error loading or placing logo: {e}")


# -------------------------- Utilities -------------------------- #
def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def load_nifti(path: str):
    nii = nib.load(path)
    arr = nii.get_fdata()
    zooms = nii.header.get_zooms()[:3]
    return arr, zooms, nii


def resample_to_target(src: np.ndarray, target_shape, order=0) -> np.ndarray:
    if src.shape == target_shape:
        return src
    factors = np.array(target_shape) / np.array(src.shape)
    return zoom(src, factors, order=order)


def draw_left_edge_arrow(img: Image.Image, y: int,
                         color=(0, 102, 255),  # solid blue (RGB)
                         size=14, width=2):
    """
    Draw a small, right-pointing blue arrowhead on the left edge of the image.

    Args:
        img: PIL Image.
        y (int): vertical coordinate for arrow center.
        color (tuple): RGB color of the arrow.
        size (int): arrow size in pixels.
        width (int): arrow line thickness.
    """
    if img.mode != "RGB":
        img = img.convert("RGB")

    draw = ImageDraw.Draw(img)
    h = img.height
    y = max(size, min(h - size, y))

    # Coordinates for right-pointing arrowhead (▶)
    arrow = [
        (0, y - size // 2),   # top-left
        (0, y + size // 2),   # bottom-left
        (size, y)             # tip pointing right
    ]

    draw.polygon(arrow, fill=color, outline=color)
    return img

# -------------------- Connected Components -------------------- #
def compute_connected_components(mask: np.ndarray, min_voxels: int = 0):
    """
    Label connected components in a binary mask using SciPy.
    
    Args:
        mask (np.ndarray): Binary or integer mask (3D or 2D).
        min_voxels (int): Minimum voxel count to keep a component.
    
    Returns:
        labeled (np.ndarray): Labeled mask where each connected region 
                              has a unique integer label starting from 1.
        n (int): Number of valid connected components.
    """
    # Ensure binary mask
    binary = (mask > 0).astype(np.uint8)
    
    # Label connected components (connectivity=1 for 6-neighbors in 3D)
    labeled, num = ndimage.label(binary)
    
    if min_voxels > 0 and num > 0:
        # Compute voxel count per label
        sizes = ndimage.sum(binary, labeled, range(1, num + 1))
        # Create mask of labels that meet the voxel size criterion
        valid_labels = [i + 1 for i, s in enumerate(sizes) if s >= min_voxels]
        # Keep only valid components
        mask_valid = np.isin(labeled, valid_labels)
        labeled, num = ndimage.label(mask_valid)
    
    return labeled.astype(np.int32), int(num)


# ---------------------- Per-tumor metrics ---------------------- #
def compute_tumor_metrics(ct: np.ndarray, pet: np.ndarray, labeled_mask: np.ndarray, voxel_sizes) -> tuple:
    """
    Compute per-tumor SUV metrics and total metabolic metrics.

    Args:
        pet (np.ndarray): PET volume in SUV units.
        labeled_mask (np.ndarray): Labeled mask (each tumor has unique ID > 0).
        voxel_sizes (tuple or list): (dx, dy, dz) in mm.
    
    Returns:
        results (list of dict): per-tumor metrics
        totals (dict): total TmTV, Total_TLG, tumor_count
    """
    voxel_sizes = np.array(voxel_sizes, dtype=np.float32)
    voxel_vol_ml = float(np.prod(voxel_sizes)) / 1000.0  # mm³ → mL
    results = []
    
    labels = np.unique(labeled_mask)
    labels = labels[labels != 0]

    total_volume = 0.0
    total_tlg = 0.0

    r_mm = ((3 * 1000) / (4 * np.pi)) ** (1/3)  # 1 mL sphere ≈ 6.2 mm radius

    for lab in labels:
        mask_vox = (labeled_mask == lab)
        count = int(mask_vox.sum())
        vol_ml = count * voxel_vol_ml
        total_volume += vol_ml

        tumor_data = pet[mask_vox]
        if tumor_data.size == 0:
            continue

        suv_mean = float(np.nanmean(tumor_data))
        suv_max = float(np.nanmax(tumor_data))

        # SUVpeak within 1 mL sphere
        max_idx = np.unravel_index(np.nanargmax(pet * mask_vox), pet.shape)
        zz, yy, xx = np.indices(pet.shape)
        dist = np.sqrt(((zz - max_idx[0]) * voxel_sizes[2])**2 +
                       ((yy - max_idx[1]) * voxel_sizes[1])**2 +
                       ((xx - max_idx[2]) * voxel_sizes[0])**2)
        local_region = dist <= r_mm
        suvpeak_vals = pet[local_region & mask_vox]
        suv_peak = float(np.nanmean(suvpeak_vals)) if suvpeak_vals.size > 0 else suv_max

        # MTV and TLG
        mtv_mask = mask_vox & (pet >= 2.5)
        mtv_count = int(mtv_mask.sum())
        mtv_ml = mtv_count * voxel_vol_ml

        if mtv_ml > 0:
            suv_mean_mtv = float(np.nanmean(pet[mtv_mask]))
            tlg = suv_mean_mtv * mtv_ml
        else:
            tlg = 0.0

        total_tlg += tlg

        hu_mean = np.mean(ct[mask_vox])

        results.append({
            "tumor_id": int(lab),
            "voxel_count": count,
            "volume_ml": round(vol_ml, 2),
            "SUVmean": round(suv_mean, 2),
            "SUVmax": round(suv_max, 2),
            "SUVpeak": round(suv_peak, 2),
            "MTV_ml": round(mtv_ml, 2),
            "TLG": round(tlg, 2),
            "Mean_HU": round(hu_mean, 2)
        })

    totals = {
        "TmTV_ml": round(total_volume, 2),
        "Total_TLG": round(total_tlg, 2),
        "tumor_count": len(labels)
    }

    return results, totals


# ---------------------- PET/CT normalization utilities ---------------------- #
def normalize_pet_suv(vol: np.ndarray, mode: str = "percentile", fixed_max: float = 12.0, pct_low: float = 1.0, pct_high: float = 99.0):
    vol = np.asarray(vol, dtype=np.float32)
    vol = np.nan_to_num(vol, nan=0.0)
    vol[vol < 0] = 0.0

    if mode == "percentile":
        p_low, p_high = np.percentile(vol, [pct_low, pct_high])
        if p_high <= p_low:
            p_low, p_high = float(vol.min()), float(vol.max())
        v = np.clip((vol - p_low) / (p_high - p_low + 1e-8), 0.0, 1.0)
        return v

    if mode == "fixed":
        v = np.clip(vol / float(max(fixed_max, 1e-6)), 0.0, 1.0)
        return v

    if mode == "log":
        v = np.log1p(vol)
        v = (v - v.min()) / (v.max() - v.min() + 1e-8)
        return v

    if mode == "zscore":
        mu = vol.mean()
        sigma = vol.std() + 1e-8
        z = (vol - mu) / sigma
        z = np.clip(z, -1.0, 4.0)
        z = (z - z.min()) / (z.max() - z.min() + 1e-8)
        return z

    v = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)
    return v

def normalize_ct(ct: np.ndarray):
    """Normalize CT data to [0, 1] using Hounsfield Units range."""
    ct = np.asarray(ct, dtype=np.float32)
    ct = np.nan_to_num(ct, nan=0.0)
    ct = np.clip(ct, -1000, 1000)  # Typical HU range for CT
    return (ct + 1000) / 2000  # Map [-1000, 1000] to [0, 1]

# ---------------------- MIP and Edges ---------------------- #
def compute_mips(volume: np.ndarray, normalize_mode: str = "percentile", fixed_max: float = 12.0, pct_low: float = 1.0, pct_high: float = 99.0):
    vol = normalize_pet_suv(volume, mode=normalize_mode, fixed_max=fixed_max, pct_low=pct_low, pct_high=pct_high)
    sagittal_mip = np.max(vol, axis=0)
    coronal_mip = np.max(vol, axis=1)
    return sagittal_mip, coronal_mip

# ---------------------- Save SC DICOM ---------------------- #
def save_sc_dicom(img_rgb: np.ndarray, output_path: str, patient_name="Anonymous", patient_id="0001", modality="SC"):
    if img_rgb.dtype != np.uint8:
        img_rgb = (np.clip(img_rgb, 0, 255)).astype(np.uint8)

    sc_uid = "1.2.840.10008.5.1.4.1.1.7"
    file_meta = pydicom.Dataset()
    file_meta.MediaStorageSOPClassUID = sc_uid
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.ImplementationClassUID = generate_uid()

    ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.SOPClassUID = sc_uid
    ds.SOPInstanceUID = generate_uid()
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()

    ds.Modality = modality
    ds.PatientName = patient_name
    ds.PatientID = patient_id
    ds.StudyDate = datetime.datetime.now().strftime("%Y%m%d")
    ds.ContentDate = ds.StudyDate
    ds.StudyTime = datetime.datetime.now().strftime("%H%M%S")
    ds.ContentTime = ds.StudyTime

    ds.Rows, ds.Columns = img_rgb.shape[0], img_rgb.shape[1]
    ds.SamplesPerPixel = 3
    ds.PhotometricInterpretation = "RGB"
    ds.PlanarConfiguration = 0
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PixelData = img_rgb.tobytes()

    ds.save_as(output_path)
    return output_path

# ---------------------- SC Pipelines ---------------------- #
def crop_around_center(volume, center, crop_size):
    x, y, z = [int(round(c)) for c in center]
    sx, sy, sz = crop_size
    x0, y0, z0 = x - sx // 2, y - sy // 2, z - sz // 2
    x1, y1, z1 = x0 + sx, y0 + sy, z0 + sz

    pad_x0, pad_y0, pad_z0 = max(0, -x0), max(0, -y0), max(0, -z0)
    pad_x1 = max(0, x1 - volume.shape[0])
    pad_y1 = max(0, y1 - volume.shape[1])
    pad_z1 = max(0, z1 - volume.shape[2])

    x0c, x1c = max(0, x0), min(volume.shape[0], x1)
    y0c, y1c = max(0, y0), min(volume.shape[1], y1)
    z0c, z1c = max(0, z0), min(volume.shape[2], z1)

    crop = volume[x0c:x1c, y0c:y1c, z0c:z1c]

    if any([pad_x0, pad_x1, pad_y0, pad_y1, pad_z0, pad_z1]):
        crop = np.pad(crop, ((pad_x0, pad_x1),
                             (pad_y0, pad_y1),
                             (pad_z0, pad_z1)),
                      mode='constant', constant_values=0)
    return crop

def overlay_ct_pet(ct_img: np.ndarray, pet_img: np.ndarray, pet_cmap='hot', alpha=0.5):
    """Overlay CT (grayscale) and PET (color-mapped) images."""
    if ct_img.ndim != 2 or pet_img.ndim != 2:
        raise ValueError(f"Expected 2D images, got CT shape={ct_img.shape}, PET shape={pet_img.shape}")
    if ct_img.shape != pet_img.shape:
        raise ValueError(f"Mismatch: CT shape={ct_img.shape}, PET shape={pet_img.shape}")
    
    ct_img = np.clip(ct_img, 0, 1)
    pet_img = np.clip(pet_img, 0, 1)
    
    # Convert CT to grayscale RGB
    ct_rgb = np.stack([ct_img]*3, axis=-1)
    
    # Apply colormap to PET
    cmap = cm.get_cmap(pet_cmap)
    pet_rgb = cmap(pet_img)[..., :3]  # Get RGB, discard alpha channel if any
    
    # Blend images
    overlay = ct_rgb * (1 - alpha) + pet_rgb * alpha
    return np.clip(overlay, 0, 1)

def overlay_mask_on_rgb(rgb_img: np.ndarray, mask: np.ndarray, color=(0, 0, 1),
                        alpha=0.4, color_split=False):
    """
    Overlay a binary mask on an RGB image.

    Args:
        rgb_img (np.ndarray): RGB image in [0, 1].
        mask (np.ndarray): 2D binary mask.
        color (tuple): Default overlay color (R, G, B) in [0–1].
        alpha (float): Transparency of overlay.
        color_split (bool): If True, choose shade of blue by vertical location.
    """
    # --- Sanity checks ---
    if rgb_img.ndim != 3 or rgb_img.shape[-1] != 3:
        raise ValueError(f"Expected RGB image, got shape {rgb_img.shape}")
    if mask.ndim != 2 or mask.shape != rgb_img.shape[:2]:
        raise ValueError(f"Mismatch: RGB shape={rgb_img.shape}, mask shape={mask.shape}")

    rgb_img = np.clip(rgb_img, 0, 1)
    overlay = rgb_img.copy()
    mask = (mask > 0).astype(np.uint8)
    H = mask.shape[1]

    # --- Option 1: simple overlay (no color split) ---
    if not color_split:
        overlay[mask > 0] = overlay[mask > 0] * (1 - alpha) + np.array(color) * alpha
        return np.clip(overlay, 0, 1)

    # --- Option 2: color split by height (blue gradient) ---
    y_indices, x_indices = np.where(mask > 0)
    if len(y_indices) == 0:
        return rgb_img  # empty mask

    y_mean = np.mean(y_indices)
    rel_pos = y_mean / H

    if rel_pos < 1/3:
        region = "upper"
        color = (0.7, 0.8, 1.0)   # light blue
    elif rel_pos < 2/3:
        region = "middle"
        color = (0.3, 0.5, 0.9)   # medium blue
    else:
        region = "lower"
        color = (0.0, 0.2, 0.6)   # dark blue

    overlay[mask > 0] = overlay[mask > 0] * (1 - alpha) + np.array(color) * alpha
    return np.clip(overlay, 0, 1)


def make_square_image(img: np.ndarray, target_size: int) -> np.ndarray:
    """Ensure 2D image is square by padding with zeros."""
    h, w = img.shape[:2]  # Handle both 2D and 3D (RGB) inputs
    if h == w:
        return img
    size = max(h, w)
    if img.ndim == 2:
        new_img = np.zeros((size, size), dtype=img.dtype)
        y0 = (size - h) // 2
        x0 = (size - w) // 2
        new_img[y0:y0+h, x0:x0+w] = img
    else:
        new_img = np.zeros((size, size, img.shape[2]), dtype=img.dtype)
        y0 = (size - h) // 2
        x0 = (size - w) // 2
        new_img[y0:y0+h, x0:x0+w, :] = img
    return new_img

# ---------------------- Global SC Generation ---------------------- #
def generate_global_sc(mask, pet, ct, totals):

    # --- CORONAL VIEW ---
    cor_full = np.max(pet, axis=1)
    cor_full_mask = np.max(mask, axis=1)
    ct_cor_full = np.max(ct, axis=1)  # CT coronal projection for body

    non_zero = np.where(ct_cor_full > 0.1)  # Threshold CT to detect body
    if len(non_zero[0]) > 0:
        y_min, y_max = np.min(non_zero[0]), np.max(non_zero[0])
        z_min, z_max = np.min(non_zero[1]), np.max(non_zero[1])
        margin = 20  # Margin for body context
        y_min, y_max = max(0, y_min - margin), min(cor_full.shape[0], y_max + margin + 1)
        z_min, z_max = max(0, z_min - margin), min(cor_full.shape[1], z_max + margin + 1)
        cor_full = cor_full[y_min:y_max, z_min:z_max]
        cor_full_mask = cor_full_mask[y_min:y_max, z_min:z_max]

    cor_full = make_square_image(cor_full, max(cor_full.shape))
    cor_full_mask = make_square_image(cor_full_mask, max(cor_full_mask.shape))
    cor_full = np.rot90(cor_full)
    cor_full_mask = np.rot90(cor_full_mask)
    cmap = cm.get_cmap('gray')
    cor_full_rgb = cmap(cor_full)[..., :3]  # Apply gray colormap
    cor_full_overlay = overlay_mask_on_rgb(1 - cor_full_rgb, cor_full_mask, color_split=False)


    # --- SAGITTAL VIEW ---
    sag_full = np.max(pet, axis=0)
    sag_full_mask = np.max(mask, axis=0)
    ct_sag_full = np.max(ct, axis=0)  # CT sagittal projection for body

    non_zero = np.where(ct_sag_full > 0.1)
    if len(non_zero[0]) > 0:
        y_min, y_max = np.min(non_zero[0]), np.max(non_zero[0])
        x_min, x_max = np.min(non_zero[1]), np.max(non_zero[1])
        margin = 20
        y_min, y_max = max(0, y_min - margin), min(sag_full.shape[0], y_max + margin + 1)
        x_min, x_max = max(0, x_min - margin), min(sag_full.shape[1], x_max + margin + 1)
        sag_full = sag_full[y_min:y_max, x_min:x_max]
        sag_full_mask = sag_full_mask[y_min:y_max, x_min:x_max]

    sag_full = make_square_image(sag_full, max(sag_full.shape))
    sag_full_mask = make_square_image(sag_full_mask, max(sag_full_mask.shape))
    sag_full = np.rot90(sag_full)
    sag_full_mask = np.rot90(sag_full_mask)
    sag_full_rgb = cmap(sag_full)[..., :3]
    sag_full_overlay = overlay_mask_on_rgb(1 - sag_full_rgb, sag_full_mask, color_split=False)

    collage_width = 1010
    collage_height = 600+64
    collage = Image.new('RGB', (collage_width, collage_height), color='white')

    # Convert NumPy arrays to Pillow images
    def numpy_to_pil(img_array):
        try:
            if img_array.max() <= 1.0:
                img_array = (img_array * 255).astype(np.uint8)
            else:
                img_array = img_array.astype(np.uint8)
            return Image.fromarray(img_array, 'RGB')
        except Exception as e:
            raise ValueError(f"Failed to convert array to Pillow image: {e}")

    try:
        img_cor_full = numpy_to_pil(cor_full_overlay)
        img_sag_full = numpy_to_pil(sag_full_overlay)

    except Exception as e:
        print(e)
    
    left_width = 350 
    img_cor_full_resized = img_cor_full.resize((left_width, 600))
    collage.paste(img_cor_full_resized, (5, 5))

    img_sag_full_resized = img_sag_full.resize((left_width, 600))
    collage.paste(img_sag_full_resized, (5+left_width, 5))

    global logo, logo_x, logo_y
    collage.paste(logo, (logo_x, logo_y), logo if logo.mode == 'RGBA' else None)

    # Add text box on the right side (fixed size)
    text_box_x = left_width*2 + 10
    text_box_y = 30+5
    text_box_width = 280
    text_box_height = 200
    draw = ImageDraw.Draw(collage)
    # Draw rounded rectangle with baby blue outline, no fill
    draw.rounded_rectangle(
        [(text_box_x, text_box_y), (text_box_x + text_box_width, text_box_y + text_box_height)],
        radius=20,  # Rounded corners
        outline=(135, 206, 250),  # Baby blue
        width=2,  # Thin outline
        fill=None  # Transparent background
    )

    # Load fonts
    title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 32)
    metrics_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)


    # Title: "Lesion x/n" (bold, centered)
    title_text = "MIP"
    title_bbox = draw.textbbox((0, 0), title_text, font=title_font)
    title_width = title_bbox[2] - title_bbox[0]
    title_x = text_box_x + (text_box_width - title_width) // 2
    title_y = text_box_y + 10  # Top of text box with padding
    draw.text((title_x, title_y), title_text, fill='black', font=title_font)

    # Metrics: Left-aligned below title
    metrics_text = (
        f"Lesions Count: {int(totals['tumor_count'])}\n\n"
        f"TMTV: {totals['TmTV_ml']:.2f}mL\n\n"
        f"TLG: {totals['Total_TLG']:.2f}Suv.mL"
    )
    metrics_x = text_box_x + 20  # Left-aligned with padding
    metrics_y = title_y + 50  # Below title with spacing
    draw.text((metrics_x, metrics_y), metrics_text, fill='black', font=metrics_font, align='left')

    # Warning text
    warning_text = "⚠️ Attention: This image contains AI-generated content"
    warning_font = metrics_font
    # Draw warning text (red for visibility)
    draw.text((10, 664-36), warning_text, fill='red', font=warning_font)

    try:
        rgb = np.array(collage)
        if rgb.dtype != np.uint8:
            rgb = rgb.astype(np.uint8)

    except Exception as e:
        print(e)
    

    fname = f"SC_GLOBAL.dcm"
    os.makedirs("SC_Fusion_Output/Global", exist_ok=True)
    outpath = os.path.join("SC_Fusion_Output/Global", fname)

    try:
        save_sc_dicom(rgb, outpath)
    except Exception as e:
        print(f"Error saving DICOM: {e}")

    collage.save(os.path.join("SC_Fusion_Output/Global", "SC_GLOBAL.png"))


# ---------------------- Master function ---------------------- #
def create_all_scs(ct_path: str, mask_path: str, pet_path: str,
                   out_dir: str = "SC_Output", min_voxels: int = 0,
                   edge_method: str = "full", normalize_mode: str = "percentile", fixed_max: float = 12.0,
                   crop_size=(64,64,64)):
    ensure_dir(out_dir)
    per_tumor_dir = os.path.join(out_dir, "per_tumor")
    ensure_dir(per_tumor_dir)

    ct, ct_vox, ct_nii = load_nifti(ct_path)
    mask, mask_vox, _ = load_nifti(mask_path)
    if mask.shape != ct.shape:
        mask = resample_to_target(mask, ct.shape, order=0).astype(np.uint8)

    if pet_path is None:
        raise ValueError("pet_path is required (we operate on PET/SUV for MIPs and metrics).")
    pet, pet_vox, _ = load_nifti(pet_path)
    if pet.shape != ct.shape:
        pet = resample_to_target(pet, ct.shape, order=1).astype(np.float32)

    labeled_mask, num = compute_connected_components(mask, min_voxels=min_voxels)
    results, totals = compute_tumor_metrics(ct, pet, labeled_mask, ct_nii.header.get_zooms()[:3])
    saved_paths = []

    pet_norm = normalize_pet_suv(pet, normalize_mode, fixed_max)
    ct_norm = normalize_ct(ct)

    for r in results:
        lab = r["tumor_id"]
        tumor_mask = (labeled_mask == lab).astype(np.uint8)
        if tumor_mask.sum() < max(1, min_voxels):
            print(f"Skipping tumor {lab}: too few voxels ({tumor_mask.sum()} < {max(1, min_voxels)})")
            continue

        cx, cy, cz = center_of_mass(tumor_mask)
        if np.isnan(cx):
            print(f"Skipping tumor {lab}: invalid center of mass")
            continue

        # Full coronal MIP 
        cor_full = np.max(pet_norm, axis=1)
        cor_full_mask = np.max(tumor_mask, axis=1)
        ct_cor_full = np.max(ct_norm, axis=1)  # CT coronal projection for body
        non_zero = np.where(ct_cor_full > 0.1)  # Threshold CT to detect body
        if len(non_zero[0]) > 0:
            y_min, y_max = np.min(non_zero[0]), np.max(non_zero[0])
            z_min, z_max = np.min(non_zero[1]), np.max(non_zero[1])
            margin = 20  # Margin for body context
            y_min, y_max = max(0, y_min - margin), min(cor_full.shape[0], y_max + margin + 1)
            z_min, z_max = max(0, z_min - margin), min(cor_full.shape[1], z_max + margin + 1)
            cor_full = cor_full[y_min:y_max, z_min:z_max]
            cor_full_mask = cor_full_mask[y_min:y_max, z_min:z_max]
            
        cor_full = make_square_image(cor_full, max(cor_full.shape))
        cor_full_mask = make_square_image(cor_full_mask, max(cor_full_mask.shape))
        cor_full = np.rot90(cor_full)
        cor_full_mask = np.rot90(cor_full_mask)
        cmap = cm.get_cmap('gray')
        cor_full_rgb = cmap(cor_full)[..., :3]  # Apply gray colormap
        cor_full_overlay = overlay_mask_on_rgb(1-cor_full_rgb, cor_full_mask)
        print(f"Tumor {lab}: full_cor shape={cor_full.shape}, full_cor_rgb shape={cor_full_rgb.shape}, full_cor_overlay shape={cor_full_overlay.shape}")

        # Cropped PET, CT, and mask with consistent size
        pet_crop = crop_around_center(pet_norm, (cx, cy, cz), crop_size)
        ct_crop = crop_around_center(ct_norm, (cx, cy, cz), crop_size)
        mask_crop = crop_around_center(tumor_mask, (cx, cy, cz), crop_size)
        print(f"Tumor {lab}: pet_crop shape={pet_crop.shape}, ct_crop shape={ct_crop.shape}, mask_crop shape={mask_crop.shape}")

        # Sagittal, coronal, axial cropped slices (ensure square)
        sag_crop_pet = np.max(pet_crop, axis=0)
        sag_crop_pet = make_square_image(sag_crop_pet, max(sag_crop_pet.shape))
        sag_crop_pet = np.rot90(sag_crop_pet)
        sag_crop_ct = np.max(ct_crop, axis=0)
        sag_crop_ct = make_square_image(sag_crop_ct, max(sag_crop_ct.shape))
        sag_crop_ct = np.rot90(sag_crop_ct)
        sag_crop_mask = np.max(mask_crop, axis=0)
        sag_crop_mask = make_square_image(sag_crop_mask, max(sag_crop_mask.shape))
        sag_crop_mask = np.rot90(sag_crop_mask)

        cor_crop_pet = np.max(pet_crop, axis=1)
        cor_crop_pet = make_square_image(cor_crop_pet, max(cor_crop_pet.shape))
        cor_crop_pet = np.rot90(cor_crop_pet)
        cor_crop_ct = np.max(ct_crop, axis=1)
        cor_crop_ct = make_square_image(cor_crop_ct, max(cor_crop_ct.shape))
        cor_crop_ct = np.rot90(cor_crop_ct)
        cor_crop_mask = np.max(mask_crop, axis=1)
        cor_crop_mask = make_square_image(cor_crop_mask, max(cor_crop_mask.shape))
        cor_crop_mask = np.rot90(cor_crop_mask)

        ax_crop_pet = pet_crop[pet_crop.shape[0]//2]
        ax_crop_pet = make_square_image(ax_crop_pet, max(ax_crop_pet.shape))
        ax_crop_pet = np.rot90(ax_crop_pet)
        ax_crop_ct = ct_crop[ct_crop.shape[0]//2]
        ax_crop_ct = make_square_image(ax_crop_ct, max(ax_crop_ct.shape))
        ax_crop_ct = np.rot90(ax_crop_ct)
        ax_crop_mask = mask_crop[mask_crop.shape[0]//2]
        ax_crop_mask = make_square_image(ax_crop_mask, max(ax_crop_mask.shape))
        ax_crop_mask = np.rot90(ax_crop_mask)

        print(f"Tumor {lab}: sag_crop_pet shape={sag_crop_pet.shape}, cor_crop_pet shape={cor_crop_pet.shape}, ax_crop_pet shape={ax_crop_pet.shape}")
        try:
            ys, xs = np.where(cor_full_mask > 0)
            if len(ys) > 0:
                centroid_y = int(np.round(ys.mean()))
                tmp = (cor_full_overlay * 255).astype(np.uint8)
                pil_img = Image.fromarray(tmp, "RGB")
                pil_img = draw_left_edge_arrow(pil_img, centroid_y,
                                            color=(0, 102, 255),
                                            size=14, width=2)
                cor_full_overlay = np.asarray(pil_img).astype(np.float32) / 255.0
        except Exception as e:
            print(f"Tumor {lab}: Warning: failed to draw left arrow: {e}")



        # Overlay CT, PET, and mask for crops
        try:
            sag_crop_ct_pet = overlay_ct_pet(sag_crop_ct, sag_crop_pet)
            sag_crop_overlay = overlay_mask_on_rgb(sag_crop_ct_pet, sag_crop_mask)
            cor_crop_ct_pet = overlay_ct_pet(cor_crop_ct, cor_crop_pet)
            cor_crop_overlay = overlay_mask_on_rgb(cor_crop_ct_pet, cor_crop_mask)
            ax_crop_ct_pet = overlay_ct_pet(ax_crop_ct, ax_crop_pet)
            ax_crop_overlay = overlay_mask_on_rgb(ax_crop_ct_pet, ax_crop_mask)
            print(f"Tumor {lab}: overlay shapes - full_cor={cor_full_overlay.shape}, sag_crop={sag_crop_overlay.shape}, cor_crop={cor_crop_overlay.shape}, ax_crop={ax_crop_overlay.shape}")
        except Exception as e:
            print(f"Tumor {lab}: Error in overlay: {e}")
            continue

        # Validate overlay arrays
        for name, arr in [("full_coronal", cor_full_overlay), ("sagittal_crop", sag_crop_overlay),
                          ("coronal_crop", cor_crop_overlay), ("axial_crop", ax_crop_overlay)]:
            if arr.ndim != 3 or arr.shape[-1] != 3:
                print(f"Tumor {lab}: Invalid {name} overlay shape: {arr.shape}")
                continue
            if arr.max() > 1.0 or arr.min() < 0.0:
                print(f"Tumor {lab}: {name} overlay values out of range [0,1]: min={arr.min()}, max={arr.max()}")

        # Create collage canvas (horizontal: wider than tall)
        collage_width = 1010
        collage_height = 600+64
        collage = Image.new('RGB', (collage_width, collage_height), color='white')

        # Convert NumPy arrays to Pillow images
        def numpy_to_pil(img_array):
            try:
                if img_array.max() <= 1.0:
                    img_array = (img_array * 255).astype(np.uint8)
                else:
                    img_array = img_array.astype(np.uint8)
                return Image.fromarray(img_array, 'RGB')
            except Exception as e:
                raise ValueError(f"Failed to convert array to Pillow image: {e}")

        try:
            img_cor_full = numpy_to_pil(cor_full_overlay)
            img_sag_crop = numpy_to_pil(sag_crop_overlay)
            img_cor_crop = numpy_to_pil(cor_crop_overlay)
            img_ax_crop = numpy_to_pil(ax_crop_overlay)
            print(f"Tumor {lab}: Successfully converted overlays to Pillow images")
        except Exception as e:
            print(f"Tumor {lab}: Error converting arrays to images: {e}")
            continue

        # Resize and place images
        try:
            left_width = collage_width // 2  # 500 pixels
            img_cor_full_resized = img_cor_full.resize((left_width, 600))
            collage.paste(img_cor_full_resized, (5, 5))
            small_size = (200, 200)
            img_sag_crop_resized = img_sag_crop.resize(small_size)
            img_cor_crop_resized = img_cor_crop.resize(small_size)
            img_ax_crop_resized = img_ax_crop.resize(small_size)
            collage.paste(img_ax_crop_resized, (left_width+5, 0+5))
            collage.paste(img_cor_crop_resized, (left_width+5, 200+5+2))
            collage.paste(img_sag_crop_resized, (left_width+5, 400+5+4))

            global logo, logo_x, logo_y
            collage.paste(logo, (logo_x, logo_y), logo if logo.mode == 'RGBA' else None)

            print(f"Tumor {lab}: Images pasted successfully")
        except Exception as e:
            print(f"Tumor {lab}: Error resizing or pasting images: {e}")
            continue

        # Add text box on the right side (fixed size)
        text_box_x = left_width + 215  +5
        text_box_y = 30+5
        text_box_width = 270
        text_box_height = 200
        draw = ImageDraw.Draw(collage)
        # Draw rounded rectangle with baby blue outline, no fill
        draw.rounded_rectangle(
            [(text_box_x, text_box_y), (text_box_x + text_box_width, text_box_y + text_box_height)],
            radius=20,  # Rounded corners
            outline=(135, 206, 250),  # Baby blue
            width=2,  # Thin outline
            fill=None  # Transparent background
        )

        # Load fonts
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 32)
        metrics_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)


        # Title: "Lesion x/n" (bold, centered)
        title_text = f"Lesion {lab}/{totals['tumor_count']}"
        title_bbox = draw.textbbox((0, 0), title_text, font=title_font)
        title_width = title_bbox[2] - title_bbox[0]
        title_x = text_box_x + (text_box_width - title_width) // 2
        title_y = text_box_y + 10  # Top of text box with padding
        draw.text((title_x, title_y), title_text, fill='black', font=title_font)

        # Metrics: Left-aligned below title
        metrics_text = (
            f"SUVmax: {r['SUVmax']:.2f}\n"
            f"SUVmean: {r['SUVmean']:.2f}\n"
            f"SUVpeak: {r['SUVpeak']:.2f}\n"
            f"MTV≥2.5: {r['MTV_ml']:.2f}mL\n"
            f"Mean HU: {r['Mean_HU']:.2f}"
            
        )
        metrics_x = text_box_x + 20  # Left-aligned with padding
        metrics_y = title_y + 40  # Below title with spacing
        draw.text((metrics_x, metrics_y), metrics_text, fill='black', font=metrics_font, align='left')

        # Warning text
        warning_text = "⚠️ Attention: This image contains AI-generated content"
        warning_font = metrics_font
        # Draw warning text (red for visibility)
        draw.text((10, 664-36), warning_text, fill='red', font=warning_font)


        print(f"Tumor {lab}: Text box added with title 'Lesion {lab}/{totals['tumor_count']}' at ({title_x}, {title_y}), metrics at ({metrics_x}, {metrics_y})")

        # Save collage as PNG for debugging
        debug_png = os.path.join(per_tumor_dir, f"debug_tumor{lab:03d}.png")
        try:
            collage.save(debug_png)
            print(f"Tumor {lab}: Saved debug PNG at {debug_png}")
        except Exception as e:
            print(f"Tumor {lab}: Error saving debug PNG: {e}")

        # Convert collage to NumPy array for DICOM
        try:
            rgb = np.array(collage)
            if rgb.dtype != np.uint8:
                rgb = rgb.astype(np.uint8)
            print(f"Tumor {lab}: RGB array shape={rgb.shape}, dtype={rgb.dtype}")
        except Exception as e:
            print(f"Tumor {lab}: Error converting collage to RGB array: {e}")
            continue

        # Save SC DICOM
        fname = f"SC_tumor{lab:03d}.dcm"
        outpath = os.path.join(per_tumor_dir, fname)
        try:
            save_sc_dicom(rgb, outpath)
            saved_paths.append(outpath)
            print(f"Tumor {lab}: Saved DICOM at {outpath}")
        except Exception as e:
            print(f"Tumor {lab}: Error saving DICOM: {e}")
            continue

    # Global SC generation
    generate_global_sc(
        mask, pet_norm, ct_norm, totals,
    )

    summary = {
        "num_tumors": totals["tumor_count"],
        "TmTV_ml": totals["TmTV_ml"],
        "Total_TLG": totals["Total_TLG"],
        "per_tumor_results": results,
        "per_tumor_files": saved_paths,
    }
    return summary

# ---------------------- Example usage ---------------------- #
if __name__ == "__main__":
    ct_path = "dta/CTres.nii.gz"
    mask_path = "dta/SEG.nii.gz"
    pet_path = "dta/SUV.nii.gz"
    out_dir = "SC_Fusion_Output"

    summary = create_all_scs(ct_path, mask_path, pet_path,
                             out_dir=out_dir,
                             min_voxels=5,
                             edge_method="centroid",
                             normalize_mode="log",
                             fixed_max=200)

    print("=== SUMMARY ===")
    print(f"Num tumors: {summary['num_tumors']}")
    print(f"TmTV (mL): {summary['TmTV_ml']:.2f}")
    print(f"Total TLG: {summary['Total_TLG']:.2f}")
    print("Per-tumor results:")
    for r in summary["per_tumor_results"]:
        print(r)
    print("Per-tumor SC files:", summary["per_tumor_files"])

