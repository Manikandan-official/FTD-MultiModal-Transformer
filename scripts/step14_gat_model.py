import glob
import pickle

import torch
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv


# =====================================================
# LOAD GRAPH FILES
# =====================================================

graph_files = glob.glob(
    "/teamspace/studios/this_studio/project/data/graphs/**/*.pkl",
    recursive=True
)

print(f"\nTotal graph files: {len(graph_files)}")


# =====================================================
# LOAD SINGLE GRAPH
# =====================================================

graphs = []

for path in graph_files:

    try:

        with open(path, "rb") as f:

            graph = pickle.load(f)

        x = torch.tensor(
            graph["x"],
            dtype=torch.float
        )

        edge_index = torch.tensor(
            graph["edge_index"],
            dtype=torch.long
        )

        data = Data(
            x=x,
            edge_index=edge_index
        )

        graphs.append(data)

    except Exception as e:

        print(f"Failed: {path}")
        print(e)


print(f"\nLoaded graphs: {len(graphs)}")


# =====================================================
# DATALOADER
# =====================================================

loader = DataLoader(
    graphs,
    batch_size=2,
    shuffle=True
)


# =====================================================
# GAT MODEL
# =====================================================

class GAT(torch.nn.Module):

    def __init__(self):

        super().__init__()

        self.gat1 = GATConv(
            in_channels=2,
            out_channels=64,
            heads=4
        )

        self.gat2 = GATConv(
            in_channels=256,
            out_channels=128,
            heads=1
        )

    def forward(self, x, edge_index):

        x = self.gat1(x, edge_index)

        x = F.elu(x)

        x = self.gat2(x, edge_index)

        return x


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

model = GAT().to(device)


# =====================================================
# TEST PIPELINE
# =====================================================

print("\nTesting GAT pipeline...\n")

for batch in loader:

    batch = batch.to(device)

    out = model(
        batch.x,
        batch.edge_index
    )

    print("Input node shape:")
    print(batch.x.shape)

    print("\nOutput embedding shape:")
    print(out.shape)

    break


print("\nDONE")
