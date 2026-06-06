import os
import torch
import torch.nn as nn
import pandas as pd
import nibabel as nib
import pickle

from torch_geometric.data import Data
from torch_geometric.nn import GATConv

# ============================================
# DEVICE
# ============================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("\nUsing device:", device)

# ============================================
# LOAD DATASET INDEX
# ============================================

csv_path = "/teamspace/studios/this_studio/project/data/dataset_index.csv"

df = pd.read_csv(csv_path)

print("\nDataset loaded:")
print("Total samples:", len(df))

# ============================================
# LOAD ONE SAMPLE
# ============================================

sample = df.iloc[0]

mri_path = sample["mri_path"]
graph_path = sample["graph_path"]

# ============================================
# MRI LOAD
# ============================================

mri = nib.load(mri_path).get_fdata()

mri_tensor = torch.tensor(
    mri,
    dtype=torch.float32
).unsqueeze(0).unsqueeze(0).to(device)

print("\nMRI tensor shape:")
print(mri_tensor.shape)

# ============================================
# GRAPH LOAD
# ============================================

with open(graph_path, "rb") as f:
    graph = pickle.load(f)

x = graph.x.float().to(device)
edge_index = graph.edge_index.long().to(device)

print("\nGraph node feature shape:")
print(x.shape)

print("\nGraph edge shape:")
print(edge_index.shape)

# ============================================
# SWIN ENCODER
# ============================================

class SimpleSwinEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1)
        )

        self.fc = nn.Linear(64, 512)

    def forward(self, x):

        x = self.encoder(x)

        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x

# ============================================
# GAT ENCODER
# ============================================

class GATEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.gat1 = GATConv(
            in_channels=2,
            out_channels=64,
            heads=4
        )

        self.gat2 = GATConv(
            in_channels=256,
            out_channels=128,
            heads=1
        )

    def forward(self, x, edge_index):

        x = self.gat1(x, edge_index)

        x = torch.relu(x)

        x = self.gat2(x, edge_index)

        x = x.mean(dim=0)

        return x.unsqueeze(0)

# ============================================
# FUSION
# ============================================

class FusionModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc = nn.Linear(640, 512)

    def forward(self, swin_feat, gat_feat):

        x = torch.cat([swin_feat, gat_feat], dim=1)

        x = self.fc(x)

        return x

# ============================================
# BUILD MODELS
# ============================================

swin_model = SimpleSwinEncoder().to(device)

gat_model = GATEncoder().to(device)

fusion_model = FusionModel().to(device)

# ============================================
# FORWARD PASS
# ============================================

print("\nRunning Swin encoder...")

swin_embedding = swin_model(mri_tensor)

print("Swin embedding shape:")
print(swin_embedding.shape)

print("\nRunning GAT encoder...")

gat_embedding = gat_model(x, edge_index)

print("GAT embedding shape:")
print(gat_embedding.shape)

print("\nRunning fusion...")

fused_embedding = fusion_model(
    swin_embedding,
    gat_embedding
)

print("Fused embedding shape:")
print(fused_embedding.shape)

print("\nREAL MULTIMODAL PIPELINE SUCCESSFUL")
