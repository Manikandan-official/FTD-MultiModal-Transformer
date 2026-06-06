import glob
import torch
import torch.nn as nn
import nibabel as nib
import numpy as np

from torch.utils.data import Dataset, DataLoader

# ============================================
# DEVICE
# ============================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("\nUsing device:", device)

# ============================================
# LOAD MRI FILES
# ============================================

mri_files = glob.glob(
    "/teamspace/studios/this_studio/project/data/preprocessed/**/*.nii.gz",
    recursive=True
)

print("\nTotal MRI scans:", len(mri_files))

# ============================================
# DATASET
# ============================================

class MRIDataset(Dataset):

    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        path = self.files[idx]

        img = nib.load(path).get_fdata()

        img = img.astype(np.float32)

        x = torch.tensor(img).unsqueeze(0)

        return x

dataset = MRIDataset(mri_files)

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True
)

# ============================================
# SWIN-LIKE PATCH EMBEDDING
# ============================================

class SwinEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.patch_embed = nn.Conv3d(
            in_channels=1,
            out_channels=96,
            kernel_size=4,
            stride=4
        )

        self.stage1 = nn.Sequential(
            nn.Conv3d(96, 192, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm3d(192)
        )

        self.stage2 = nn.Sequential(
            nn.Conv3d(192, 384, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm3d(384)
        )

        self.pool = nn.AdaptiveAvgPool3d(1)

        self.fc = nn.Linear(384, 512)

    def forward(self, x):

        x = self.patch_embed(x)

        x = self.stage1(x)

        x = self.stage2(x)

        x = self.pool(x)

        x = x.flatten(1)

        x = self.fc(x)

        return x

# ============================================
# MODEL
# ============================================

model = SwinEncoder().to(device)

# ============================================
# TEST PIPELINE
# ============================================

print("\nTesting Swin pipeline...\n")

for batch in loader:

    batch = batch.to(device)

    out = model(batch)

    print("Input MRI shape:")
    print(batch.shape)

    print("\nSwin embedding shape:")
    print(out.shape)

    break

print("\nDONE")
