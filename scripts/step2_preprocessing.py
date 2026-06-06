# =========================================================
# NIFD MRI PREPROCESSING PIPELINE
# =========================================================
# Step 2:
#   - Load MRI volumes
#   - Verify integrity
#   - Normalize intensity
#   - Resize to 96x96x96
#   - Save preprocessed volumes
#
# Output:
#   project/data/preprocessed/
# =========================================================

from pathlib import Path
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
from tqdm import tqdm

# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

NIFTI_ROOT = Path("/teamspace/studios/this_studio/project/data/nifti")
OUTPUT_ROOT = Path("/teamspace/studios/this_studio/project/data/preprocessed")

TARGET_SHAPE = (96, 96, 96)

# ---------------------------------------------------------
# LOAD MRI
# ---------------------------------------------------------

def load_nifti(path):
    nii = nib.load(str(path))
    data = nii.get_fdata()
    affine = nii.affine
    return data, affine

# ---------------------------------------------------------
# Z-SCORE NORMALIZATION
# ---------------------------------------------------------

def normalize_volume(volume):

    mean = np.mean(volume)
    std = np.std(volume)

    if std == 0:
        return volume

    volume = (volume - mean) / std

    return volume

# ---------------------------------------------------------
# RESIZE VOLUME
# ---------------------------------------------------------

def resize_volume(volume, target_shape=(96,96,96)):

    factors = [
        target_shape[0] / volume.shape[0],
        target_shape[1] / volume.shape[1],
        target_shape[2] / volume.shape[2],
    ]

    resized = zoom(volume, factors, order=1)

    return resized

# ---------------------------------------------------------
# SAVE NIFTI
# ---------------------------------------------------------

def save_nifti(volume, affine, out_path):

    nii = nib.Nifti1Image(
        volume.astype(np.float32),
        affine
    )

    nib.save(nii, str(out_path))

# ---------------------------------------------------------
# MAIN PREPROCESSING
# ---------------------------------------------------------

def preprocess_dataset():

    nii_files = sorted(
        NIFTI_ROOT.rglob("*.nii.gz")
    )

    print("=" * 60)
    print("NIFTI FILES FOUND:", len(nii_files))
    print("=" * 60)

    for nii_path in tqdm(nii_files):

        try:

            # -----------------------------------------
            # LOAD MRI
            # -----------------------------------------
            volume, affine = load_nifti(nii_path)

            print(f"\nProcessing: {nii_path.name}")
            print("Original Shape:", volume.shape)

            # -----------------------------------------
            # VERIFY 3D
            # -----------------------------------------
            if volume.ndim != 3:
                print("Skipping non-3D scan")
                continue

            # -----------------------------------------
            # NORMALIZE
            # -----------------------------------------
            volume = normalize_volume(volume)

            # -----------------------------------------
            # RESIZE
            # -----------------------------------------
            volume = resize_volume(
                volume,
                TARGET_SHAPE
            )

            print("Resized Shape:", volume.shape)

            # -----------------------------------------
            # OUTPUT PATH
            # -----------------------------------------
            relative_path = nii_path.relative_to(
                NIFTI_ROOT
            )

            out_path = OUTPUT_ROOT / relative_path

            out_path.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            # Rename output file
            out_path = out_path.with_name(
                out_path.stem.replace(".nii", "")
                + "_prep.nii.gz"
            )

            # -----------------------------------------
            # SAVE
            # -----------------------------------------
            save_nifti(
                volume,
                affine,
                out_path
            )

            print("Saved:", out_path)

        except Exception as e:

            print(f"\nERROR processing {nii_path}")
            print(e)

    print("\nPREPROCESSING COMPLETE")

# ---------------------------------------------------------
# RUN SCRIPT
# ---------------------------------------------------------

if __name__ == "__main__":
    preprocess_dataset()