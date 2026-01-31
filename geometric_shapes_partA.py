import matplotlib.pyplot as plt
import numpy as np
from torch_geometric.datasets import GeometricShapes
from torch_geometric.transforms import FaceToEdge


def plot_graph(data, idx):
    pos = data.pos.cpu().numpy()
    edge_index = data.edge_index.cpu().numpy()
    dim = pos.shape[1]

    fig = plt.figure(figsize=(5, 5))
    if dim == 3:
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=8, c="tab:blue")
        for u, v in edge_index.T:
            xs = [pos[u, 0], pos[v, 0]]
            ys = [pos[u, 1], pos[v, 1]]
            zs = [pos[u, 2], pos[v, 2]]
            ax.plot(xs, ys, zs, color="black", linewidth=0.5, alpha=0.6)
    else:
        ax = fig.add_subplot(111)
        ax.scatter(pos[:, 0], pos[:, 1], s=8, c="tab:blue")
        for u, v in edge_index.T:
            xs = [pos[u, 0], pos[v, 0]]
            ys = [pos[u, 1], pos[v, 1]]
            ax.plot(xs, ys, color="black", linewidth=0.5, alpha=0.6)
        ax.set_aspect("equal", adjustable="box")

    ax.set_title(f"GeometricShapes graph {idx}")
    ax.axis("off")


# Load dataset with FaceToEdge so edge_index is populated
transform = FaceToEdge()

dataset = GeometricShapes(root="data/GeometricShapes", pre_transform=transform)

# Pick any 3 graphs
indices = [0, 1, 2]
for idx in indices:
    plot_graph(dataset[idx], idx)

plt.show()
