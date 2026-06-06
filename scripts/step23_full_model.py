import os
import torch
import torch.nn as nn
import pandas as pd
import nibabel as nib
import pickle

from torch_geometric.nn import GATConv

# ============================================
# DEVICE
# ============================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("\nUsing device:", device)

# ============================================
# LOAD DATASET
# ============================================

csv_path = "/teamspace/studios/this_studio/project/data/dataset_index.csv"

df = pd.read_csv(csv_path)

print("\nDataset size:", len(df))

sample = df.iloc[0]

mri_path = sample["mri_path"]
graph_path = sample["graph_path"]

# ============================================
# LOAD MRI
# ============================================

mri = nib.load(mri_path).get_fdata()

mri_tensor = torch.tensor(
    mri,
    dtype=torch.float32
).unsqueeze(0).unsqueeze(0).to(device)

# ============================================
# LOAD GRAPH
# ============================================

with open(graph_path, "rb") as f:
    graph = pickle.load(f)

x = graph.x.float().to(device)
edge_index = graph.edge_index.long().to(device)

# ============================================
# SWIN ENCODER
# ============================================

class SimpleSwinEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv3d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(32, 64, 3, padding=1),
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
# FEATURE FUSION
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
# LONGITUDINAL TRANSFORMER
# ============================================

class LongitudinalTransformer(nn.Module):

    def __init__(self):

        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=512,
            nhead=4,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

    def forward(self, x):

        return self.transformer(x)

# ============================================
# CLASSIFIER
# ============================================

class RiskClassifier(nn.Module):

    def __init__(self):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, x):

        x = x[:, -1, :]

        return self.classifier(x)

# ============================================
# BUILD MODELS
# ============================================

swin_model = SimpleSwinEncoder().to(device)

gat_model = GATEncoder().to(device)

fusion_model = FusionModel().to(device)

transformer_model = LongitudinalTransformer().to(device)

classifier_model = RiskClassifier().to(device)

# ============================================
# FORWARD PASS
# ============================================

print("\nRunning Swin encoder...")

swin_embedding = swin_model(mri_tensor)

print("Swin shape:", swin_embedding.shape)

print("\nRunning GAT encoder...")

gat_embedding = gat_model(x, edge_index)

print("GAT shape:", gat_embedding.shape)

print("\nRunning fusion...")

fused_embedding = fusion_model(
    swin_embedding,
    gat_embedding
)

print("Fusion shape:", fused_embedding.shape)

# ============================================
# CREATE TEMPORAL SEQUENCE
# ============================================

sequence = fused_embedding.unsqueeze(1)

sequence = sequence.repeat(1, 4, 1)

print("\nTemporal sequence shape:")
print(sequence.shape)

# ============================================
# TRANSFORMER
# ============================================

print("\nRunning transformer...")

transformer_output = transformer_model(sequence)

print("Transformer output shape:")
print(transformer_output.shape)

# ============================================
# CLASSIFIER
# ============================================

print("\nRunning classifier...")

prediction = classifier_model(transformer_output)

print("Prediction shape:")
print(prediction.shape)

print("\nPrediction logits:")
print(prediction)

print("\nFULL MODEL SUCCESSFUL")
