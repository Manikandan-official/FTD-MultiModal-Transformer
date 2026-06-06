from pathlib import Path
from tqdm import tqdm
import ants

# =========================================================
# PATHS
# =========================================================

INPUT_ROOT = Path(
    "/teamspace/studios/this_studio/project/data/skull_stripped"
)

OUTPUT_ROOT = Path(
    "/teamspace/studios/this_studio/project/data/mni_registered"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

# =========================================================
# LOAD MNI TEMPLATE
# =========================================================

print("\nLoading MNI template...")

mni_template = ants.image_read(
    ants.get_ants_data("mni")
)

print("MNI template loaded")

# =========================================================
# FIND ALL SCANS
# =========================================================

nii_files = sorted(
    INPUT_ROOT.rglob("*.nii.gz")
)

print(f"\nFound {len(nii_files)} skull-stripped scans")

# =========================================================
# REGISTER TO MNI
# =========================================================

failed_scans = []

for nii_path in tqdm(nii_files):

    try:

        relative_path = nii_path.relative_to(INPUT_ROOT)

        output_dir = OUTPUT_ROOT / relative_path.parent

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        output_file = output_dir / (
            nii_path.stem.replace(".nii", "")
            + "_mni.nii.gz"
        )

        # ---------------------------------------------
        # SKIP EXISTING
        # ---------------------------------------------

        if output_file.exists():

            print(f"\nSkipping existing: {output_file.name}")
            continue

        print(f"\nRegistering: {nii_path.name}")

        # ---------------------------------------------
        # LOAD MRI
        # ---------------------------------------------

        moving = ants.image_read(
            str(nii_path)
        )

        # ---------------------------------------------
        # FAST AFFINE REGISTRATION
        # ---------------------------------------------

        registration = ants.registration(
            fixed=mni_template,
            moving=moving,
            type_of_transform="Affine"
        )

        # ---------------------------------------------
        # SAVE REGISTERED MRI
        # ---------------------------------------------

        ants.image_write(
            registration["warpedmovout"],
            str(output_file)
        )

        print(f"Saved: {output_file}")

    except Exception as e:

        print(f"\nERROR: {nii_path.name}")
        print(e)

        failed_scans.append(
            str(nii_path)
        )

# =========================================================
# SAVE FAILED SCANS
# =========================================================

failed_report = OUTPUT_ROOT / "failed_mni_registration.txt"

with open(failed_report, "w") as f:

    for item in failed_scans:

        f.write(item + "\n")

# =========================================================
# COMPLETE
# =========================================================

print("\nMNI REGISTRATION COMPLETE")
print(f"Failed scans: {len(failed_scans)}")

