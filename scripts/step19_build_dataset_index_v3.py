import os
import glob
import pandas as pd

# ============================================
# PATHS
# ============================================

MRI_ROOT = "/teamspace/studios/this_studio/project/data/skull_stripped"
GRAPH_ROOT = "/teamspace/studios/this_studio/project/data/graphs"

# ============================================
# FIND FILES RECURSIVELY
# ============================================

mri_files = glob.glob(
    os.path.join(MRI_ROOT, "**", "*.nii.gz"),
    recursive=True
)

graph_files = glob.glob(
    os.path.join(GRAPH_ROOT, "**", "*.pkl"),
    recursive=True
)

print("MRI files found:", len(mri_files))
print("Graph files found:", len(graph_files))

# ============================================
# BUILD GRAPH LOOKUP
# ============================================

graph_map = {}

for g in graph_files:

    base = os.path.basename(g)

    graph_id = (
        base
        .replace("_prep_brain_mni_roi.pkl", "")
    )

    graph_map[graph_id] = g

# ============================================
# MATCH MRI + GRAPH
# ============================================

rows = []

for mri in mri_files:

    base = os.path.basename(mri)

    mri_id = (
        base
        .replace("_prep_brain.nii.gz", "")
    )

    if mri_id in graph_map:

        rows.append({
            "id": mri_id,
            "mri_path": mri,
            "graph_path": graph_map[mri_id]
        })

# ============================================
# DATAFRAME
# ============================================

df = pd.DataFrame(rows)

print("\nMatched MRI-Graph pairs:", len(df))

print("\nFirst 5 rows:")
print(df.head())

# ============================================
# SAVE
# ============================================

save_path = "/teamspace/studios/this_studio/project/data/dataset_index.csv"

df.to_csv(save_path, index=False)

print("\nSaved dataset index:")
print(save_path)

print("\nDONE")
