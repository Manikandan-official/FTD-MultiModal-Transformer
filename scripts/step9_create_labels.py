from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# ====================================
# LOAD MANIFEST
# ====================================

MANIFEST_PATH = Path(
    "/teamspace/studios/this_studio/project/data/manifest.csv"
)

df = pd.read_csv(MANIFEST_PATH)

# ====================================
# CREATE LABELS FROM SUBJECT PREFIX
# ====================================

def get_label(subject_id):

    if str(subject_id).startswith("1_"):
        return 1   # Patient

    elif str(subject_id).startswith("2_"):
        return 0   # Control

    elif str(subject_id).startswith("3_"):
        return 1   # Patient

    else:
        return None

df["label"] = df["subject_id"].apply(get_label)

# remove unknown labels
df = df[df["label"].notnull()]

# ====================================
# SUBJECT-LEVEL UNIQUE TABLE
# ====================================

subjects = df[["subject_id", "label"]].drop_duplicates()

# ====================================
# TRAIN SPLIT
# ====================================

train_subj, temp_subj = train_test_split(
    subjects,
    test_size=0.30,
    stratify=subjects["label"],
    random_state=42
)

# ====================================
# VAL / TEST SPLIT (STRATIFIED)
# ====================================

val_subj, test_subj = train_test_split(
    temp_subj,
    test_size=0.50,
    stratify=temp_subj["label"],
    random_state=42
)

# ====================================
# SAVE SPLITS
# ====================================

OUTPUT = Path(
    "/teamspace/studios/this_studio/project/data/splits"
)

OUTPUT.mkdir(exist_ok=True)

train_subj.to_csv(
    OUTPUT / "train_subjects.csv",
    index=False
)

val_subj.to_csv(
    OUTPUT / "val_subjects.csv",
    index=False
)

test_subj.to_csv(
    OUTPUT / "test_subjects.csv",
    index=False
)

# ====================================
# PRINT RESULTS
# ====================================

print("\nTRAIN")
print(train_subj["label"].value_counts())

print("\nVAL")
print(val_subj["label"].value_counts())

print("\nTEST")
print(test_subj["label"].value_counts())

print("\nDONE")
