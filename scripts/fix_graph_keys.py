import os
import glob
import pickle
import torch
from torch_geometric.data import Data

graph_files = glob.glob(
    "/teamspace/studios/this_studio/project/data/graphs/**/*.pkl",
    recursive=True
)

fixed = 0
failed = 0

for file in graph_files:

    try:
        with open(file, "rb") as f:
            obj = pickle.load(f)

        # already pyg Data
        if isinstance(obj, Data):
            continue

        # inspect keys
        print("\nFILE:", file)
        print("KEYS:", obj.keys())

        # possible node feature key
        if "node_features" in obj:
            x = torch.tensor(obj["node_features"], dtype=torch.float)

        elif "features" in obj:
            x = torch.tensor(obj["features"], dtype=torch.float)

        elif "x" in obj:
            x = torch.tensor(obj["x"], dtype=torch.float)

        else:
            print("No node features found")
            failed += 1
            continue

        # possible edge key
        if "edge_index" in obj:
            edge_index = torch.tensor(obj["edge_index"], dtype=torch.long)

        elif "edges" in obj:
            edge_index = torch.tensor(obj["edges"], dtype=torch.long)

        else:
            print("No edge index found")
            failed += 1
            continue

        data = Data(
            x=x,
            edge_index=edge_index
        )

        with open(file, "wb") as f:
            pickle.dump(data, f)

        fixed += 1

    except Exception as e:
        print("FAILED:", file)
        print(e)
        failed += 1

print("\nDONE")
print("Fixed:", fixed)
print("Failed:", failed)

