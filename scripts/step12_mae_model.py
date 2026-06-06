import torch
import torch.nn as nn

# ============================================
# PATCH EMBEDDING
# ============================================

class PatchEmbed3D(nn.Module):

    def __init__(
        self,
        in_channels=1,
        embed_dim=128,
        patch_size=16
    ):
        super().__init__()

        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):

        x = self.proj(x)

        return x


# ============================================
# SIMPLE TRANSFORMER ENCODER
# ============================================

class MAEEncoder(nn.Module):

    def __init__(
        self,
        embed_dim=128,
        num_heads=4,
        depth=4
    ):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth
        )

    def forward(self, x):

        return self.encoder(x)


# ============================================
# SIMPLE DECODER
# ============================================

class MAEDecoder(nn.Module):

    def __init__(
        self,
        embed_dim=128,
        out_dim=4096
    ):
        super().__init__()

        self.decoder = nn.Sequential(

            nn.Linear(embed_dim, 512),
            nn.ReLU(),

            nn.Linear(512, out_dim)
        )

    def forward(self, x):

        return self.decoder(x)


# ============================================
# FULL MAE MODEL
# ============================================

class MAE3D(nn.Module):

    def __init__(self):

        super().__init__()

        self.patch_embed = PatchEmbed3D()

        self.encoder = MAEEncoder()

        self.decoder = MAEDecoder()

    def forward(self, x):

        # patch embedding
        x = self.patch_embed(x)

        # flatten patches
        B, C, D, H, W = x.shape

        x = x.flatten(2).transpose(1, 2)

        # transformer
        encoded = self.encoder(x)

        # reconstruct
        reconstructed = self.decoder(encoded)

        return reconstructed


# ============================================
# TEST MODEL
# ============================================

model = MAE3D()

dummy = torch.randn(2, 1, 96, 96, 96)

output = model(dummy)

print("\nMODEL TEST")

print("Input:")
print(dummy.shape)

print("\nOutput:")
print(output.shape)

print("\nDONE")
