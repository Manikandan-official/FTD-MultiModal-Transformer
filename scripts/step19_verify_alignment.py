import os
import pickle
from glob import glob

# ============================================
# PATHS
# ============================================

MRI_DIR = "/teamspace/studios/this_studio/project/data/preprocessed"
GRAPH_DIR = "/teamspace/studios/this_studio/project/data/graphs"

# ============================================
# LOAD FILES
# ============================================

mri_files = sorted(
    glob(os.path.join(MRI_DIR, "*.npy"))
)

graph_files = sorted(
    glob(os.path.join(GRAPH_DIR, "*.pkl"))
)

print("\nMRI files:", len(mri_files))
print("Graph files:", len(graph_files))

# ============================================
# CREATE BASENAMES
# ============================================

mri_ids = set([
    os.path.basename(f).replace(".npy", "")
    for f in mri_files
])

graph_ids = set([
    os.path.basename(f).replace(".pkl", "")
    for f in graph_files
])

# ============================================
# INTERSECTION
# ============================================

common_ids = mri_ids.intersection(graph_ids)

print("\nMatching MRI + Graph pairs:", len(common_ids))

# ============================================
# CHECK MISSING
# ============================================

missing_graphs = mri_ids - graph_ids
missing_mris = graph_ids - mri_ids

print("\nMissing graph files:", len(missing_graphs))
print("Missing MRI files:", len(missing_mris))

# ============================================
# SHOW EXAMPLES
# ============================================

print("\nSample aligned IDs:")

for i, item in enumerate(sorted(common_ids)[:5]):
    print(item)

print("\nDONE")

