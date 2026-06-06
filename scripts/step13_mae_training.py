import glob
import nibabel as nib
import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader


# =====================================================
# DATASET
# =====================================================

class MRIDataset(Dataset):

    def __init__(self, files):

        self.files = files

    def __len__(self):

        return len(self.files)

    def __getitem__(self, idx):

        path = self.files[idx]

        img = nib.load(path).get_fdata()

        img = img.astype(np.float32)

        img = torch.tensor(img)

        img = img.unsqueeze(0)

        return img


# =====================================================
# LOAD MRI FILES
# =====================================================

mri_files = glob.glob(
    "/teamspace/studios/this_studio/project/data/preprocessed/**/*.nii.gz",
    recursive=True
)

print(f"\nTotal MRI scans: {len(mri_files)}")


# =====================================================
# DATALOADER
# =====================================================

dataset = MRIDataset(mri_files)

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True
)


# =====================================================
# PATCH EMBEDDING
# =====================================================

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

        return self.proj(x)


# =====================================================
# ENCODER
# =====================================================

class MAEEncoder(nn.Module):

    def __init__(
        self,
        embed_dim=128,
        num_heads=4,
        depth=4
    ):
        super().__init__()

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=depth
        )

    def forward(self, x):

        return self.encoder(x)


# =====================================================
# DECODER
# =====================================================

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


# =====================================================
# FULL MAE MODEL
# =====================================================

class MAE3D(nn.Module):

    def __init__(self):

        super().__init__()

        self.patch_embed = PatchEmbed3D()

        self.encoder = MAEEncoder()

        self.decoder = MAEDecoder()

    def forward(self, x):

        x = self.patch_embed(x)

        B, C, D, H, W = x.shape

        x = x.flatten(2).transpose(1, 2)

        encoded = self.encoder(x)

        reconstructed = self.decoder(encoded)

        return reconstructed


# =====================================================
# DEVICE
# =====================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"\nUsing device: {device}")


# =====================================================
# MODEL
# =====================================================

model = MAE3D().to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4
)

criterion = nn.MSELoss()


# =====================================================
# TRAIN LOOP
# =====================================================

print("\nStarting MAE training...\n")

epochs = 2

for epoch in range(epochs):

    total_loss = 0

    for batch_idx, batch in enumerate(loader):

        batch = batch.to(device)

        optimizer.zero_grad()

        output = model(batch)

        target = torch.zeros_like(output)

        loss = criterion(output, target)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        print(
            f"Epoch {epoch+1} | "
            f"Batch {batch_idx+1} | "
            f"Loss: {loss.item():.4f}"
        )

        if batch_idx == 5:
            break

    avg_loss = total_loss / (batch_idx + 1)

    print(
        f"\nEpoch {epoch+1} Complete | "
        f"Average Loss: {avg_loss:.4f}\n"
    )


# =====================================================
# SAVE MODEL
# =====================================================

torch.save(
    model.state_dict(),
    "/teamspace/studios/this_studio/project/mae_pretrained.pth"
)

print("\nMODEL SAVED")

print("\nDONE")
