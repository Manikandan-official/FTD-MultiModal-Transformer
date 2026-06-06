import torch
import torch.nn as nn

# ============================================
# DEVICE
# ============================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("\nUsing device:", device)

# ============================================
# DUMMY INPUTS
# ============================================

# Swin output
swin_embedding = torch.randn(2, 512).to(device)

# GAT output
gat_embedding = torch.randn(2, 128).to(device)

print("\nSwin embedding shape:")
print(swin_embedding.shape)

print("\nGAT embedding shape:")
print(gat_embedding.shape)

# ============================================
# FEATURE FUSION MODULE
# ============================================

class FeatureFusion(nn.Module):

    def __init__(self):

        super().__init__()

        # 512 + 128 = 640
        self.projection = nn.Sequential(

            nn.Linear(640, 512),

            nn.GELU(),

            nn.Dropout(0.2)

        )

    def forward(self, swin_feat, gat_feat):

        # concatenate
        fused = torch.cat(
            [swin_feat, gat_feat],
            dim=1
        )

        print("\nConcatenated feature shape:")
        print(fused.shape)

        # learned projection
        fused = self.projection(fused)

        return fused

# ============================================
# MODEL
# ============================================

model = FeatureFusion().to(device)

# ============================================
# TEST PIPELINE
# ============================================

print("\nTesting Fusion pipeline...\n")

output = model(
    swin_embedding,
    gat_embedding
)

print("\nFinal fused embedding shape:")
print(output.shape)

print("\nDONE")
