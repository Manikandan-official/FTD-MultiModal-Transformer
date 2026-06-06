from pathlib import Path
import pandas as pd
import pickle

# ====================================
# PATHS
# ====================================

GRAPH_DIR = Path(
    "/teamspace/studios/this_studio/project/data/graphs"
)

SPLIT_DIR = Path(
    "/teamspace/studios/this_studio/project/data/splits"
)

# ====================================
# LOAD SPLITS
# ====================================

train_df = pd.read_csv(SPLIT_DIR / "train_subjects.csv")
val_df = pd.read_csv(SPLIT_DIR / "val_subjects.csv")
test_df = pd.read_csv(SPLIT_DIR / "test_subjects.csv")

# ====================================
# FUNCTION TO LOAD GRAPH FILES
# ====================================

def load_graphs(split_df, split_name):

    dataset = []

    graph_files = list(GRAPH_DIR.rglob("*.pkl"))

    print(f"\nLoading {split_name} graphs...")

    for _, row in split_df.iterrows():

        subject_id = row["subject_id"]
        label = row["label"]

        matched = False

        for graph_path in graph_files:

            if subject_id in graph_path.name:

                with open(graph_path, "rb") as f:
                    graph_data = pickle.load(f)

                dataset.append({
                    "subject_id": subject_id,
                    "label": label,
                    "graph": graph_data
                })

                matched = True

        if not matched:
            print(f"Missing graph for: {subject_id}")

    print(f"{split_name} loaded: {len(dataset)} graphs")

    return dataset

# ====================================
# LOAD DATASETS
# ====================================

train_dataset = load_graphs(train_df, "TRAIN")
val_dataset = load_graphs(val_df, "VAL")
test_dataset = load_graphs(test_df, "TEST")

# ====================================
# SUMMARY
# ====================================

print("\n========== DATASET SUMMARY ==========")

print(f"Train graphs : {len(train_dataset)}")
print(f"Val graphs   : {len(val_dataset)}")
print(f"Test graphs  : {len(test_dataset)}")

print("\nDONE")
