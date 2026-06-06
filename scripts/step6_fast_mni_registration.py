import ants
import nibabel as nib
from pathlib import Path
from tqdm import tqdm

# =========================
# PATHS
# =========================
INPUT_ROOT = Path("/teamspace/studios/this_studio/project/data/skull_stripped")
OUTPUT_ROOT = Path("/teamspace/studios/this_studio/project/data/mni_registered")

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# =========================
# LOAD MNI TEMPLATE
# =========================
print("Loading MNI template...")

mni = ants.get_ants_data('mni')
template = ants.image_read(mni)

print("MNI template loaded")

# =========================
# FIND ALL MRI FILES
# =========================
nii_files = sorted(INPUT_ROOT.rglob("*.nii.gz"))

print(f"Found {len(nii_files)} skull-stripped scans")

failed_scans = []

# =========================
# FAST RIGID REGISTRATION
# =========================
for nii_path in tqdm(nii_files):

    try:
        subject = nii_path.parts[-3]
        modality = nii_path.parts[-2]

        output_dir = OUTPUT_ROOT / subject / modality
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / nii_path.name.replace(
            ".nii.gz",
            "_mni.nii.gz"
        )

        moving = ants.image_read(str(nii_path))

        registration = ants.registration(
            fixed=template,
            moving=moving,
            type_of_transform='Rigid'
        )

        ants.image_write(
            registration['warpedmovout'],
            str(output_file)
        )

    except Exception as e:
        print(f"\nERROR: {nii_path.name}")
        print(e)
        failed_scans.append(str(nii_path))

# =========================
# SAVE FAILED REPORT
# =========================
failed_report = OUTPUT_ROOT / "failed_mni_registration.txt"

with open(failed_report, "w") as f:
    for item in failed_scans:
        f.write(item + "\n")

print("\nMNI REGISTRATION COMPLETE")
print(f"Failed scans: {len(failed_scans)}")
