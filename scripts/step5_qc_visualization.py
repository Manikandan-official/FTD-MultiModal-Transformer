# =========================================================
# MRI QUALITY CHECK VISUALIZATION PIPELINE
# =========================================================

from pathlib import Path
import random

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

DATA_ROOT = Path(
    "/teamspace/studios/this_studio/project/data/skull_stripped"
)

OUTPUT_DIR = Path(
    "/teamspace/studios/this_studio/project/data/qc_outputs"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ---------------------------------------------------------
# FIND MRI FILES
# ---------------------------------------------------------

nii_files = sorted(
    DATA_ROOT.rglob("*.nii.gz")
)

print(f"\nFound {len(nii_files)} skull-stripped MRIs")

# ---------------------------------------------------------
# RANDOM SAMPLE
# ---------------------------------------------------------

NUM_SAMPLES = 25

sample_files = random.sample(
    nii_files,
    min(NUM_SAMPLES, len(nii_files))
)

# ---------------------------------------------------------
# QC STORAGE
# ---------------------------------------------------------

failed_scans = []

# ---------------------------------------------------------
# VISUALIZATION GRID
# ---------------------------------------------------------

fig, axes = plt.subplots(
    5,
    5,
    figsize=(18, 18)
)

axes = axes.flatten()

# ---------------------------------------------------------
# PROCESS SCANS
# ---------------------------------------------------------

for idx, nii_path in enumerate(tqdm(sample_files)):

    try:

        img = nib.load(str(nii_path))

        data = img.get_fdata()

        if np.sum(data) == 0:

            failed_scans.append(
                str(nii_path)
            )

            continue

        z_center = data.shape[2] // 2

        slice_img = data[:, :, z_center]

        slice_img = np.rot90(slice_img)

        axes[idx].imshow(
            slice_img,
            cmap="gray"
        )

        axes[idx].set_title(
            nii_path.stem[:20],
            fontsize=8
        )

        axes[idx].axis("off")

    except Exception as e:

        print(f"\nERROR: {nii_path.name}")
        print(e)

        failed_scans.append(
            str(nii_path)
        )

# ---------------------------------------------------------
# CLEAN EMPTY PLOTS
# ---------------------------------------------------------

for i in range(len(sample_files), len(axes)):

    axes[i].axis("off")

# ---------------------------------------------------------
# SAVE FIGURE
# ---------------------------------------------------------

plt.tight_layout()

output_image = OUTPUT_DIR / "qc_grid.png"

plt.savefig(
    output_image,
    dpi=300,
    bbox_inches="tight"
)

print(f"\nQC grid saved to: {output_image}")

# ---------------------------------------------------------
# SAVE FAILED SCANS
# ---------------------------------------------------------

failed_report = OUTPUT_DIR / "failed_scans.txt"

with open(failed_report, "w") as f:

    for item in failed_scans:

        f.write(item + "\n")

print(f"Failed scans report saved to: {failed_report}")

# ---------------------------------------------------------
# COMPLETE
# ---------------------------------------------------------

print("\nMRI QC COMPLETE")
