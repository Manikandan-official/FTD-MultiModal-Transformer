from pathlib import Path
import pandas as pd
import numpy as np
import networkx as nx
import pickle

# ============================================================
# PATHS
# ============================================================

ROI_ROOT = Path("/teamspace/studios/this_studio/project/data/roi_features")
GRAPH_ROOT = Path("/teamspace/studios/this_studio/project/data/graphs")

GRAPH_ROOT.mkdir(parents=True, exist_ok=True)

# ============================================================
# GET ROI FILES
# ============================================================

roi_files = list(ROI_ROOT.rglob("*.csv"))

print(f"Found {len(roi_files)} ROI feature files")

failed = []

# ============================================================
# PROCESS EACH ROI CSV
# ============================================================

for csv_file in roi_files:

    try:

        print(f"Processing: {csv_file.name}")

        df = pd.read_csv(csv_file)

        roi_values = df["Value"].values
        roi_labels = df["ROI"].values

        # ====================================================
        # CREATE GRAPH
        # ====================================================

        G = nx.Graph()

        # ADD NODES
        for idx, roi in enumerate(roi_labels):
            G.add_node(idx, label=str(roi))

        # ADD EDGES
        for i in range(len(roi_values)):
            for j in range(i + 1, len(roi_values)):

                weight = abs(float(roi_values[i]) - float(roi_values[j]))

                G.add_edge(i, j, weight=weight)

        # ====================================================
        # SAVE GRAPH
        # ====================================================

        relative = csv_file.relative_to(ROI_ROOT)

        output_dir = GRAPH_ROOT / relative.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / csv_file.name.replace(".csv", ".pkl")

        with open(output_file, "wb") as f:
            pickle.dump(G, f)

        print(f"Saved: {output_file}")

    except Exception as e:

        print(f"ERROR: {csv_file.name}")
        print(e)

        failed.append(str(csv_file))

# ============================================================
# SAVE FAILED REPORT
# ============================================================

with open(GRAPH_ROOT / "failed_graphs.txt", "w") as f:
    for item in failed:
        f.write(item + "\n")

print("\nGRAPH CONSTRUCTION COMPLETE")
print(f"Failed scans: {len(failed)}")
