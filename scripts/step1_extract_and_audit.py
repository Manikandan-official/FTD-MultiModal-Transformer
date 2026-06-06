"""
NIFD Dataset Extraction & Audit Pipeline
=========================================
Step 1: Scans all subjects, identifies all usable T1/T2 DICOM series,
        builds a complete manifest, then converts DICOM → NIfTI.

Run from Lightning.ai terminal:
    python step1_extract_and_audit.py

Output:
    ~/project/data/nifti/          ← converted NIfTI volumes
    ~/project/data/manifest.csv    ← full dataset manifest
    ~/project/data/audit_report.txt ← what was kept / skipped and why
"""

import os
import re
import csv
import json
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# ─────────────────────────────────────────────
# CONFIG — edit only these paths if needed
# ─────────────────────────────────────────────
RAW_ROOT    = Path("/teamspace/studios/this_studio/full_dataset/NIFD")
PROJECT_DIR = Path("/teamspace/studios/this_studio/project")
NIFTI_DIR   = PROJECT_DIR / "data" / "nifti"
REPORT_PATH = PROJECT_DIR / "data" / "audit_report.txt"
MANIFEST    = PROJECT_DIR / "data" / "manifest.csv"

# ─────────────────────────────────────────────
# SERIES SELECTION RULES
# Priority order: DIS3D variants first (best quality),
# then plain mprage as fallback.
# We collect T1, T2-FLAIR, and T2-SPC per visit.
# ─────────────────────────────────────────────

# Each entry: (modality_label, regex_pattern, priority)
# Lower priority number = preferred. Per visit we keep
# the SINGLE best match per modality.
SERIES_RULES = [
    # T1 — DIS3D variants (distortion corrected, best quality)
    ("T1", r"(?i)^t1_mprage.*dis3d$",           1),
    ("T1", r"(?i)^T1_mprage.*dis3d$",           1),
    ("T1", r"(?i)^t1_mprage_short.*dis3d$",      2),
    ("T1", r"(?i)^T1_mprage_short.*dis3d$",      2),
    ("T1", r"(?i)^T1_MPRAGE_long.*dis3d$",       2),
    ("T1", r"(?i)^Sag_3D_MP-RAGE$",              3),
    ("T1", r"(?i)^Sagittal_MP-Rage$",            3),
    # T1 plain fallback (only used if NO DIS3D found for that visit)
    ("T1_fallback", r"(?i)^t1_mprage$",          10),
    ("T1_fallback", r"(?i)^T1_mprage$",          10),
    ("T1_fallback", r"(?i)^t1_mprage_short$",    11),
    ("T1_fallback", r"(?i)^T1_mprage_short$",    11),
    ("T1_fallback", r"(?i)^T1_MPRAGE_long$",     11),

    # T2-FLAIR — DIS3D variants
    ("T2F", r"(?i)^t2_flair.*dis3d$",            1),
    ("T2F", r"(?i)^T2_flair.*dis3d$",            1),
    ("T2F", r"(?i)^T2_FLAIR$",                   2),
    ("T2F", r"(?i)^AXIAL_FLAIR$",                3),
    # T2-FLAIR plain fallback
    ("T2F_fallback", r"(?i)^t2_flair$",          10),
    ("T2F_fallback", r"(?i)^T2_flair$",          10),

    # T2-SPC (3D isotropic T2, great for hippocampus)
    ("T2S", r"(?i)^t2_spc.*dis3d$",              1),
    ("T2S", r"(?i)^T2_spc.*dis3d$",              1),
    ("T2S", r"(?i)^t2_space.*",                  2),
    ("T2S", r"(?i)^t2_flair_sag.*",              3),
    ("T2S_fallback", r"(?i)^t2_spc_ns_sag.*$",   10),
]

# Modalities we actually want in the final dataset
KEEP_MODALITIES = {"T1", "T1_fallback", "T2F", "T2F_fallback", "T2S", "T2S_fallback"}

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def match_series(series_name):
    """Return (modality, priority) for a series folder name, or None if skip."""
    best = None
    for modality, pattern, priority in SERIES_RULES:
        if re.match(pattern, series_name):
            if best is None or priority < best[1]:
                best = (modality, priority)
    return best


def parse_date(date_str):
    """Parse NIFD date folder format: 2012-02-06_09_06_43.0"""
    try:
        return datetime.strptime(date_str.split(".")[0], "%Y-%m-%d_%H_%M_%S")
    except Exception:
        return datetime.min


def find_dicom_dir(i_folder: Path):
    """Return path if I-folder contains DICOM files."""
    dcm_files = list(i_folder.glob("*.dcm"))
    if dcm_files:
        return i_folder, len(dcm_files)
    return None, 0


def convert_dicom_to_nifti(dicom_dir: Path, out_dir: Path, out_name: str):
    """
    Convert a DICOM folder to NIfTI using dcm2niix.
    Returns path to output .nii.gz or None on failure.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "dcm2niix",
        "-z", "y",           # gzip output
        "-f", out_name,      # output filename
        "-o", str(out_dir),  # output directory
        "-m", "n",           # don't merge 2D slices
        "-v", "0",           # silent
        str(dicom_dir)
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        nii_files = list(out_dir.glob(f"{out_name}*.nii.gz"))
        if nii_files:
            # Keep only the first (dcm2niix sometimes outputs extras)
            primary = nii_files[0]
            # Remove extra files if any
            for f in nii_files[1:]:
                f.unlink()
            return primary
        return None
    except subprocess.TimeoutExpired:
        return None
    except FileNotFoundError:
        return "dcm2niix_missing"


# ─────────────────────────────────────────────
# MAIN AUDIT PASS
# ─────────────────────────────────────────────

def audit_dataset():
    """
    Pass 1: Walk the full dataset, build complete picture of what exists.
    Returns: dict of subject_id → {visit_date → {modality → (series_name, i_folder, priority)}}
    """
    print("\n" + "="*60)
    print("  NIFD DATASET AUDIT")
    print("="*60)

    subject_dirs = sorted([d for d in RAW_ROOT.iterdir() if d.is_dir()])
    print(f"  Found {len(subject_dirs)} subject folders\n")

    # subject_id → visit_date → modality → (series_name, i_folder, priority)
    dataset = defaultdict(lambda: defaultdict(dict))

    skipped_series = defaultdict(int)
    kept_series    = defaultdict(int)
    total_dcm      = 0

    for subj_dir in subject_dirs:
        subject_id = subj_dir.name

        # Each subdirectory = one series type
        series_dirs = sorted([s for s in subj_dir.iterdir() if s.is_dir()])

        for series_dir in series_dirs:
            series_name = series_dir.name
            match = match_series(series_name)

            if match is None:
                skipped_series[series_name] += 1
                continue

            modality, priority = match

            # Each subdirectory of series = one visit date
            date_dirs = sorted([d for d in series_dir.iterdir() if d.is_dir()])

            for date_dir in date_dirs:
                visit_date = date_dir.name

                # Each subdirectory of date = one image collection (I######)
                i_folders = sorted([i for i in date_dir.iterdir() if i.is_dir()])

                for i_folder in i_folders:
                    dcm_dir, dcm_count = find_dicom_dir(i_folder)
                    if dcm_dir is None:
                        continue

                    total_dcm += dcm_count

                    # Keep best (lowest priority number) per modality per visit
                    existing = dataset[subject_id][visit_date].get(modality)
                    if existing is None or priority < existing[2]:
                        dataset[subject_id][visit_date][modality] = (
                            series_name, dcm_dir, priority
                        )
                        kept_series[modality] += 1

    print(f"  Total DICOM files found : {total_dcm:,}")
    print(f"\n  Series kept by modality:")
    for mod, count in sorted(kept_series.items()):
        print(f"    {mod:20s}: {count:4d} series")

    print(f"\n  Series skipped (not needed for project):")
    for name, count in sorted(skipped_series.items(), key=lambda x: -x[1])[:15]:
        print(f"    {name:45s}: {count:4d}")

    return dataset, subject_dirs


# ─────────────────────────────────────────────
# RESOLVE T1 FALLBACK
# Keep T1_fallback only if no T1 (DIS3D) found for that visit
# Same logic for T2F_fallback and T2S_fallback
# ─────────────────────────────────────────────

def resolve_fallbacks(dataset):
    """Promote fallbacks only where primary is missing."""
    resolved = defaultdict(lambda: defaultdict(dict))

    fallback_map = {
        "T1_fallback": "T1",
        "T2F_fallback": "T2F",
        "T2S_fallback": "T2S",
    }

    for subj, visits in dataset.items():
        for visit_date, modalities in visits.items():
            for mod, data in modalities.items():
                primary = fallback_map.get(mod)
                if primary and primary in modalities:
                    # Primary exists — skip fallback
                    continue
                # Rename fallback to primary label
                final_mod = fallback_map.get(mod, mod)
                resolved[subj][visit_date][final_mod] = data

    return resolved


# ─────────────────────────────────────────────
# BUILD MANIFEST & CONVERT
# ─────────────────────────────────────────────

def build_manifest_and_convert(dataset):
    """
    Pass 2: For each subject, sort visits by date, assign visit indices,
    convert DICOM → NIfTI, write manifest CSV.
    """
    print("\n" + "="*60)
    print("  DICOM → NIfTI CONVERSION")
    print("="*60)

    PROJECT_DIR.mkdir(parents=True, exist_ok=True)
    (PROJECT_DIR / "data").mkdir(exist_ok=True)
    NIFTI_DIR.mkdir(parents=True, exist_ok=True)

    # Check dcm2niix is available
    result = shutil.which("dcm2niix")
    if result is None:
        print("\n  ⚠️  dcm2niix not found. Installing...")
        os.system("pip install dcm2niix 2>/dev/null || conda install -c conda-forge dcm2niix -y 2>/dev/null")
        if shutil.which("dcm2niix") is None:
            print("  Installing via apt...")
            os.system("sudo apt-get install -y dcm2niix 2>/dev/null")

    manifest_rows = []
    conversion_log = []
    subjects_processed = 0
    total_converted = 0
    total_failed = 0

    all_subjects = sorted(dataset.keys())

    for subj_idx, subject_id in enumerate(all_subjects):
        visits = dataset[subject_id]

        # Sort visits chronologically
        sorted_visits = sorted(visits.items(), key=lambda x: parse_date(x[0]))

        print(f"\n  [{subj_idx+1:3d}/{len(all_subjects)}] Subject: {subject_id} "
              f"({len(sorted_visits)} visits)")

        subj_nifti_dir = NIFTI_DIR / subject_id
        subj_nifti_dir.mkdir(parents=True, exist_ok=True)

        for visit_idx, (visit_date, modalities) in enumerate(sorted_visits):
            visit_label = f"visit_{visit_idx:02d}"
            visit_dt    = parse_date(visit_date)
            visit_str   = visit_dt.strftime("%Y%m%d") if visit_dt != datetime.min else visit_date[:10].replace("-","")

            row = {
                "subject_id":  subject_id,
                "site":        subject_id.split("_")[0],
                "visit_idx":   visit_idx,
                "visit_label": visit_label,
                "visit_date":  visit_str,
                "T1_path":     "",
                "T2F_path":    "",
                "T2S_path":    "",
                "T1_series":   "",
                "T2F_series":  "",
                "T2S_series":  "",
                "status":      "ok",
            }

            for modality in ["T1", "T2F", "T2S"]:
                if modality not in modalities:
                    continue

                series_name, dcm_dir, priority = modalities[modality]
                out_name  = f"{subject_id}_{visit_str}_{modality}"
                out_subdir = subj_nifti_dir / modality
                out_subdir.mkdir(exist_ok=True)

                # Skip if already converted (resume-safe)
                existing = list(out_subdir.glob(f"{out_name}*.nii.gz"))
                if existing:
                    nii_path = existing[0]
                    print(f"       {modality} ✓ (cached) {nii_path.name}")
                else:
                    nii_path = convert_dicom_to_nifti(dcm_dir, out_subdir, out_name)

                if nii_path == "dcm2niix_missing":
                    print(f"       {modality} ✗ dcm2niix not available!")
                    row["status"] = "dcm2niix_missing"
                    total_failed += 1
                elif nii_path and Path(nii_path).exists():
                    rel_path = str(nii_path.relative_to(PROJECT_DIR))
                    row[f"{modality}_path"]   = rel_path
                    row[f"{modality}_series"] = series_name
                    print(f"       {modality} ✓ {nii_path.name}")
                    total_converted += 1
                else:
                    print(f"       {modality} ✗ conversion failed ({dcm_dir})")
                    row["status"] = "partial_fail"
                    total_failed += 1

            manifest_rows.append(row)

        subjects_processed += 1

    # Write manifest CSV
    fieldnames = [
        "subject_id","site","visit_idx","visit_label","visit_date",
        "T1_path","T2F_path","T2S_path",
        "T1_series","T2F_series","T2S_series","status"
    ]
    with open(MANIFEST, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\n{'='*60}")
    print(f"  CONVERSION COMPLETE")
    print(f"  Subjects processed : {subjects_processed}")
    print(f"  NIfTI files created: {total_converted}")
    print(f"  Failures           : {total_failed}")
    print(f"  Manifest saved to  : {MANIFEST}")

    return manifest_rows


# ─────────────────────────────────────────────
# AUDIT REPORT
# ─────────────────────────────────────────────

def write_audit_report(manifest_rows):
    """Write a human-readable report of what was extracted."""

    subjects = defaultdict(list)
    for row in manifest_rows:
        subjects[row["subject_id"]].append(row)

    t1_count  = sum(1 for r in manifest_rows if r["T1_path"])
    t2f_count = sum(1 for r in manifest_rows if r["T2F_path"])
    t2s_count = sum(1 for r in manifest_rows if r["T2S_path"])
    visit_counts = defaultdict(int)
    for subj, rows in subjects.items():
        visit_counts[len(rows)] += 1

    lines = [
        "=" * 60,
        "  NIFD EXTRACTION AUDIT REPORT",
        f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 60,
        "",
        f"  Total subjects      : {len(subjects)}",
        f"  Total visit records : {len(manifest_rows)}",
        "",
        "  Modality coverage (visits with data):",
        f"    T1  (structural)  : {t1_count:4d} visits",
        f"    T2F (FLAIR)       : {t2f_count:4d} visits",
        f"    T2S (SPC/3D-T2)   : {t2s_count:4d} visits",
        "",
        "  Visit distribution (subjects by number of visits):",
    ]
    for n_visits in sorted(visit_counts.keys()):
        lines.append(f"    {n_visits} visits: {visit_counts[n_visits]} subjects")

    lines += [
        "",
        "  Per-subject summary (subject | visits | T1 | T2F | T2S):",
        "  " + "-" * 55,
    ]
    for subj in sorted(subjects.keys()):
        rows = subjects[subj]
        t1  = sum(1 for r in rows if r["T1_path"])
        t2f = sum(1 for r in rows if r["T2F_path"])
        t2s = sum(1 for r in rows if r["T2S_path"])
        lines.append(
            f"  {subj:15s} | {len(rows):7d} visits | "
            f"T1:{t1:2d} | T2F:{t2f:2d} | T2S:{t2s:2d}"
        )

    lines += [
        "",
        "=" * 60,
        "  NEXT STEP: Run step2_preprocess.py",
        "=" * 60,
    ]

    report_text = "\n".join(lines)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report_text)

    print("\n" + report_text)
    print(f"\n  Full report saved to: {REPORT_PATH}")


# ─────────────────────────────────────────────
# DATASET STRUCTURE SUMMARY (before conversion)
# ─────────────────────────────────────────────

def print_what_will_be_kept(dataset):
    print("\n" + "="*60)
    print("  SERIES SELECTION SUMMARY (before conversion)")
    print("="*60)
    print("  We will KEEP these modalities per visit:")
    print("    T1  — best available DIS3D T1_mprage variant")
    print("    T2F — best available DIS3D T2_flair variant")
    print("    T2S — best available 3D-T2 SPC variant")
    print("")
    print("  We will SKIP everything else:")
    print("    localizer, scout, field_map, GRE, ASL, ep2d,")
    print("    BOLD/resting-state, DTI, clinical 2D sequences")
    print("")

    t1_total  = sum(
        1 for v in dataset.values()
        for mods in v.values()
        if "T1" in mods or "T1_fallback" in mods
    )
    t2f_total = sum(
        1 for v in dataset.values()
        for mods in v.values()
        if "T2F" in mods or "T2F_fallback" in mods
    )
    t2s_total = sum(
        1 for v in dataset.values()
        for mods in v.values()
        if "T2S" in mods or "T2S_fallback" in mods
    )

    print(f"  T1  series to convert : {t1_total}")
    print(f"  T2F series to convert : {t2f_total}")
    print(f"  T2S series to convert : {t2s_total}")
    print(f"  Total conversions     : {t1_total + t2f_total + t2s_total}")
    print("="*60)


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("\n🧠 NIFD Dataset Extraction Pipeline — Step 1")
    print("   This will audit, filter, and convert your dataset.")
    print("   Resume-safe: already-converted files will be skipped.\n")

    # Pass 1: Audit
    dataset, subject_dirs = audit_dataset()

    # Resolve fallbacks
    dataset = resolve_fallbacks(dataset)

    # Show summary of what will be kept
    print_what_will_be_kept(dataset)

    # Confirm before converting (optional — comment out to auto-proceed)
    print("\n  Proceed with DICOM → NIfTI conversion? [y/N]: ", end="")
    ans = input().strip().lower()
    if ans != "y":
        print("  Aborted. Run again and type 'y' to proceed.")
        exit(0)

    # Pass 2: Convert and build manifest
    manifest_rows = build_manifest_and_convert(dataset)

    # Write audit report
    write_audit_report(manifest_rows)

    print("\n✅ Step 1 complete.")
    print(f"   NIfTI files : {NIFTI_DIR}")
    print(f"   Manifest    : {MANIFEST}")
    print(f"   Report      : {REPORT_PATH}")
    print("\n   Next → run step2_preprocess.py")
