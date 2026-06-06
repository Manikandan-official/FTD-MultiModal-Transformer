import os
import pickle
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
from torch_geometric.data import Data

# ============================================
# LOAD CSV
# ============================================

csv_path = "/teamspace/studios/this_studio/project/data/dataset_index.csv"

df = pd.read_csv(csv_path)

print("Total samples:", len(df))

# ============================================
# LABELS
# ============================================

def get_label(subject_id):

    if subject_id.startswith("1_"):
        return 1

    return 0

df["label"] = df["id"].apply(get_label)

# ============================================
# DATASET
# ============================================

class MultimodalDataset(Dataset):

    def __init__(self, dataframe):

        self.df = dataframe.reset_index(drop=True)

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        # ====================================
        # MRI
        # ====================================

        mri_path = row["mri_path"]

        mri = nib.load(mri_path).get_fdata()

        mri = torch.tensor(
            mri,
            dtype=torch.float32
        )

        # add channel dimension
        mri = mri.unsqueeze(0)

        # ====================================
        # GRAPH
        # ====================================

        graph_path = row["graph_path"]

        with open(graph_path, "rb") as f:
            graph = pickle.load(f)

        # ====================================
        # LABEL
        # ====================================

        label = torch.tensor(
            row["label"],
            dtype=torch.long
        )

        return {
            "mri": mri,
            "graph": graph,
            "label": label,
            "id": row["id"]
        }

# ============================================
# CREATE DATASET
# ============================================

dataset = MultimodalDataset(df)

print("\nDataset created.")

print("Dataset size:", len(dataset))

# ============================================
# TEST SAMPLE
# ============================================

sample = dataset[0]

print("\nSample ID:")
print(sample["id"])

print("\nMRI shape:")
print(sample["mri"].shape)

print("\nGraph type:")
print(type(sample["graph"]))

print("\nGraph x shape:")
print(sample["graph"].x.shape)

print("\nEdge index shape:")
print(sample["graph"].edge_index.shape)

print("\nLabel:")
print(sample["label"])

print("\nDONE")
