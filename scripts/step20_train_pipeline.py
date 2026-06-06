import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# ============================================
# LOAD DATASET INDEX
# ============================================

csv_path = "/teamspace/studios/this_studio/project/data/dataset_index.csv"

df = pd.read_csv(csv_path)

print("Total samples:", len(df))

# ============================================
# CREATE LABELS
# ============================================

def get_label(subject_id):

    if subject_id.startswith("1_"):
        return 1   # FTD

    return 0       # Control

df["label"] = df["id"].apply(get_label)

print("\nLabel distribution:")
print(df["label"].value_counts())

# ============================================
# TRAIN / VAL / TEST SPLIT
# ============================================

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["label"],
    random_state=42
)

train_df, val_df = train_test_split(
    train_df,
    test_size=0.1,
    stratify=train_df["label"],
    random_state=42
)

print("\nTrain:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))

# ============================================
# DUMMY DATASET
# ============================================

class DummyDataset(Dataset):

    def __init__(self, dataframe):
        self.df = dataframe

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        x = torch.randn(512)

        y = torch.tensor(
            self.df.iloc[idx]["label"],
            dtype=torch.long
        )

        return x, y

# ============================================
# DATALOADERS
# ============================================

train_loader = DataLoader(
    DummyDataset(train_df),
    batch_size=8,
    shuffle=True
)

val_loader = DataLoader(
    DummyDataset(val_df),
    batch_size=8
)

# ============================================
# MODEL
# ============================================

class RiskClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.net(x)

# ============================================
# DEVICE
# ============================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\nUsing device:", device)

# ============================================
# SETUP
# ============================================

model = RiskClassifier().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4
)

# ============================================
# TRAIN LOOP
# ============================================

epochs = 3

for epoch in range(epochs):

    model.train()

    total_loss = 0

    for x, y in train_loader:

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        outputs = model(x)

        loss = criterion(outputs, y)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    print(f"\nEpoch {epoch+1}/{epochs}")
    print("Training Loss:", round(avg_loss, 4))

print("\nTRAINING COMPLETE")
