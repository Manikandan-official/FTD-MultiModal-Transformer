# =========================================================
# MRI VISUALIZATION + QUALITY CHECK
# =========================================================

from pathlib import Path
import nibabel as nib
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# SELECT ONE MRI
# ---------------------------------------------------------

MRI_PATH = Path(
    "/teamspace/studios/this_studio/project/data/preprocessed"
)

# Find first MRI automatically
nii_files = sorted(MRI_PATH.rglob("*.nii.gz"))

print(f"Found {len(nii_files)} preprocessed MRIs")

sample_path = nii_files[0]

print("\nLoading MRI:")
print(sample_path)

# ---------------------------------------------------------
# LOAD MRI
# ---------------------------------------------------------

nii = nib.load(str(sample_path))

volume = nii.get_fdata()

print("\nMRI Shape:", volume.shape)

# ---------------------------------------------------------
# EXTRACT MIDDLE SLICES
# ---------------------------------------------------------

axial_idx = volume.shape[2] // 2
coronal_idx = volume.shape[1] // 2
sagittal_idx = volume.shape[0] // 2

# ---------------------------------------------------------
# PLOT
# ---------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Sagittal
axes[0].imshow(
    volume[sagittal_idx, :, :],
    cmap="gray"
)
axes[0].set_title("Sagittal")
axes[0].axis("off")

# Coronal
axes[1].imshow(
    volume[:, coronal_idx, :],
    cmap="gray"
)
axes[1].set_title("Coronal")
axes[1].axis("off")

# Axial
axes[2].imshow(
    volume[:, :, axial_idx],
    cmap="gray"
)
axes[2].set_title("Axial")
axes[2].axis("off")

plt.tight_layout()

output_path = "mri_visualization.png"

plt.savefig(output_path)

print(f"\nSaved visualization to: {output_path}")

print("\nVISUALIZATION COMPLETE")