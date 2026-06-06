from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

import nibabel as nib
from nilearn import datasets
from nilearn.maskers import NiftiLabelsMasker

# ============================================================
# PATHS
# ============================================================

INPUT_ROOT = Path("/teamspace/studios/this_studio/project/data/mni_registered")
OUTPUT_ROOT = Path("/teamspace/studios/this_studio/project/data/roi_features")

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# ============================================================
# LOAD ATLAS
# ============================================================

print("Loading Harvard Oxford atlas...")

atlas = datasets.fetch_atlas_harvard_oxford(
    'cort-maxprob-thr25-2mm'
)

atlas_img = atlas.maps
labels = atlas.labels

print(f"Loaded {len(labels)} atlas regions")

# ============================================================
# ROI FEATURE EXTRACTOR
# ============================================================

masker = NiftiLabelsMasker(
    labels_img=atlas_img,
    standardize=True
)

# ============================================================
# FIND ALL REGISTERED SCANS
# ============================================================

nii_files = sorted(INPUT_ROOT.rglob("*_mni.nii.gz"))

print(f"Found {len(nii_files)} MNI registered scans")

# ============================================================
# EXTRACTION LOOP
# ============================================================

failed_scans = []

for nii_path in tqdm(nii_files):

    try:

        subject_id = nii_path.parts[-3]
        modality = nii_path.parts[-2]

        output_subject_dir = OUTPUT_ROOT / subject_id / modality
        output_subject_dir.mkdir(parents=True, exist_ok=True)

        output_csv = output_subject_dir / (
            nii_path.stem.replace(".nii", "") + "_roi.csv"
        )

        # LOAD IMAGE
        img = nib.load(str(nii_path))

        # EXTRACT ROI SIGNALS
        roi_features = masker.fit_transform(img)

        roi_features = roi_features.flatten()

        # SAVE CSV
        df = pd.DataFrame({
            "ROI": labels[1:],
            "Value": roi_features
        })

        df.to_csv(output_csv, index=False)

    except Exception as e:

        print(f"\nERROR: {nii_path.name}")
        print(e)

        failed_scans.append(str(nii_path))

# ============================================================
# SAVE FAILED REPORT
# ============================================================

failed_report = OUTPUT_ROOT / "failed_roi_extraction.txt"

with open(failed_report, "w") as f:
    for item in failed_scans:
        f.write(item + "\n")

print("\nROI EXTRACTION COMPLETE")
print(f"Failed scans: {len(failed_scans)}")
