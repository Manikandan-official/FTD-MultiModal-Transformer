from pathlib import Path
import nibabel as nib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import glob
import random

# ============================================
# FIND PREPROCESSED MRI FILES
# ============================================

PREP_DIR = "/teamspace/studios/this_studio/project/data/preprocessed"

mri_files = glob.glob(
    PREP_DIR + "/**/*_prep.nii.gz",
    recursive=True
)

print(f"\nTotal preprocessed MRIs: {len(mri_files)}")

# ============================================
# MRI DATASET
# ============================================

class MRIDataset(Dataset):

    def __init__(self, file_list):
        self.files = file_list

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        mri_path = self.files[idx]

        img = nib.load(mri_path)

        volume = img.get_fdata()

        # normalize
        volume = (volume - volume.mean()) / (
            volume.std() + 1e-8
        )

        volume = torch.tensor(
            volume,
            dtype=torch.float32
        )

        volume = volume.unsqueeze(0)

        return volume

# ============================================
# PATCH EMBEDDING
# ============================================

class PatchEmbed3D(nn.Module):

    def __init__(
        self,
        patch_size=16,
        embed_dim=128
    ):

        super().__init__()

        self.proj = nn.Conv3d(
            1,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):

        x = self.proj(x)

        print("\nPatch embeddings shape:")
        print(x.shape)

        return x

# ============================================
# RANDOM MASKING
# ============================================

def random_masking(x, mask_ratio=0.75):

    B, C, D, H, W = x.shape

    total_patches = D * H * W

    num_keep = int(
        total_patches * (1 - mask_ratio)
    )

    mask = torch.zeros(
        total_patches
    )

    keep_indices = random.sample(
        range(total_patches),
        num_keep
    )

    mask[:] = 1

    mask[keep_indices] = 0

    print("\nTotal patches:", total_patches)
    print("Masked patches:", int(mask.sum()))

    return mask

# ============================================
# DATASET + LOADER
# ============================================

dataset = MRIDataset(mri_files)

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True
)

# ============================================
# PATCH MODEL
# ============================================

patch_embed = PatchEmbed3D()

# ============================================
# TEST
# ============================================

print("\nTesting MAE pipeline...")

for batch in loader:

    print("\nMRI batch shape:")
    print(batch.shape)

    patches = patch_embed(batch)

    mask = random_masking(
        patches,
        mask_ratio=0.75
    )

    break

print("\nDONE")
