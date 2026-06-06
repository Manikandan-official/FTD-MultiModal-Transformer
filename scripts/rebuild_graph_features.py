import os
import glob
import pickle
import torch
import networkx as nx
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
            G = pickle.load(f)

        if not isinstance(G, nx.Graph):
            failed += 1
            continue

        nodes = list(G.nodes())

        # node features
        features = []

        for n in nodes:

            degree = G.degree[n]
            clustering = nx.clustering(G, n)

            features.append([
                degree,
                clustering
            ])

        x = torch.tensor(features, dtype=torch.float)

        # edge index
        edges = list(G.edges())

        if len(edges) == 0:
            failed += 1
            continue

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

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

