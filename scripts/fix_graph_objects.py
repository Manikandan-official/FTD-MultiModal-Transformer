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
            g = pickle.load(f)

        print("\nFILE:", file)
        print("TYPE:", type(g))

        # inspect attributes
        print("ATTRIBUTES:", dir(g))

        # get node features
        if hasattr(g, "x"):
            x = torch.tensor(g.x, dtype=torch.float)

        elif hasattr(g, "node_features"):
            x = torch.tensor(g.node_features, dtype=torch.float)

        elif hasattr(g, "features"):
            x = torch.tensor(g.features, dtype=torch.float)

        else:
            print("No node features")
            failed += 1
            continue

        # get edge index
        if hasattr(g, "edge_index"):
            edge_index = torch.tensor(g.edge_index, dtype=torch.long)

        elif hasattr(g, "edges"):
            edge_index = torch.tensor(g.edges, dtype=torch.long)

        else:
            print("No edge index")
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

