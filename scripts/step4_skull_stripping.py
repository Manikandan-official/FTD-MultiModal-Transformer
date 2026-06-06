# =========================================================
# HD-BET SKULL STRIPPING PIPELINE (GPU VERSION)
# =========================================================

from pathlib import Path
import subprocess
from tqdm import tqdm
import torch

# ---------------------------------------------------------
# CHECK GPU
# ---------------------------------------------------------

print("\nChecking GPU availability...\n")

if torch.cuda.is_available():

    print(f"GPU Detected: {torch.cuda.get_device_name(0)}")
    DEVICE = "cuda"

else:

    print("No GPU detected. Falling back to CPU.")
    DEVICE = "cpu"

# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

INPUT_ROOT = Path(
    "/teamspace/studios/this_studio/project/data/preprocessed"
)

OUTPUT_ROOT = Path(
    "/teamspace/studios/this_studio/project/data/skull_stripped"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

# ---------------------------------------------------------
# FIND ALL MRI FILES
# ---------------------------------------------------------

nii_files = sorted(
    INPUT_ROOT.rglob("*.nii.gz")
)

print(f"\nFound {len(nii_files)} preprocessed MRIs")

# ---------------------------------------------------------
# RUN HD-BET
# ---------------------------------------------------------

for nii_path in tqdm(nii_files):

    try:

        # -------------------------------------------------
        # CREATE OUTPUT STRUCTURE
        # -------------------------------------------------

        relative_path = nii_path.relative_to(INPUT_ROOT)

        output_dir = OUTPUT_ROOT / relative_path.parent

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # -------------------------------------------------
        # OUTPUT FILE
        # -------------------------------------------------

        output_file = output_dir / (
            nii_path.stem.replace(".nii", "")
            + "_brain.nii.gz"
        )

        # -------------------------------------------------
        # SKIP IF ALREADY PROCESSED
        # -------------------------------------------------

        if output_file.exists():

            print(f"\nSkipping existing: {output_file.name}")
            continue

        # -------------------------------------------------
        # PROCESS MRI
        # -------------------------------------------------

        print(f"\nProcessing: {nii_path.name}")

        command = [
            "hd-bet",
            "-i", str(nii_path),
            "-o", str(output_file),
            "-device", DEVICE
        ]

        subprocess.run(
            command,
            check=True
        )

        print(f"Saved: {output_file}")

    except Exception as e:

        print(f"\nERROR processing: {nii_path.name}")
        print(e)

# ---------------------------------------------------------
# COMPLETE
# ---------------------------------------------------------

print("\nSKULL STRIPPING COMPLETE")