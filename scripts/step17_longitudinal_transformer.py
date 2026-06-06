import os
import torch
import torch.nn as nn

# ============================================
# DEVICE
# ============================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("\nUsing device:", device)

# ============================================
# LONGITUDINAL TRANSFORMER
# ============================================

class LongitudinalTransformer(nn.Module):

    def __init__(
        self,
        embedding_dim=512,
        num_heads=4,
        num_layers=2,
        dropout=0.1
    ):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

    def forward(self, x):

        # x shape:
        # [batch, timepoints, embedding_dim]

        out = self.transformer(x)

        return out

# ============================================
# DUMMY LONGITUDINAL DATA
# ============================================

batch_size = 2
timepoints = 4
embedding_dim = 512

dummy_sequence = torch.randn(
    batch_size,
    timepoints,
    embedding_dim
).to(device)

# ============================================
# MODEL
# ============================================

model = LongitudinalTransformer().to(device)

# ============================================
# TEST PIPELINE
# ============================================

print("\nTesting Longitudinal Transformer...\n")

output = model(dummy_sequence)

print("Input sequence shape:")
print(dummy_sequence.shape)

print("\nTransformer output shape:")
print(output.shape)

print("\nDONE")

