import os
from glob import glob
import pandas as pd

# ============================================
# PATHS
# ============================================

MRI_ROOT = "/teamspace/studios/this_studio/project/data/skull_stripped"
GRAPH_ROOT = "/teamspace/studios/this_studio/project/data/graphs"

# ============================================
# FIND FILES
# ============================================

mri_files = glob(
    os.path.join(MRI_ROOT, "**", "*.nii.gz"),
    recursive=True
)

graph_files = glob(
    os.path.join(GRAPH_ROOT, "*.pkl")
)

print("\nMRI files found:", len(mri_files))
print("Graph files found:", len(graph_files))

# ============================================
# CREATE MRI MAP
# ============================================

mri_map = {}

for f in mri_files:

    base = os.path.basename(f)

    base = base.replace(".nii.gz", "")

    mri_map[base] = f

# ============================================
# CREATE GRAPH MAP
# ============================================

graph_map = {}

for f in graph_files:

    base = os.path.basename(f)

    base = base.replace(".pkl", "")

    graph_map[base] = f

# ============================================
# MATCH FILES
# ============================================

common_ids = sorted(
    set(mri_map.keys()).intersection(
        set(graph_map.keys())
    )
)

print("\nMatched MRI-Graph pairs:", len(common_ids))

# ============================================
# BUILD DATAFRAME
# ============================================

rows = []

for item in common_ids:

    rows.append({
        "id": item,
        "mri_path": mri_map[item],
        "graph_path": graph_map[item]
    })

df = pd.DataFrame(rows)

# ============================================
# SAVE
# ============================================

save_path = "/teamspace/studios/this_studio/project/data/dataset_index.csv"

df.to_csv(save_path, index=False)

print("\nSaved dataset index:")
print(save_path)

print("\nFirst 5 rows:")
print(df.head())

print("\nDONE")

