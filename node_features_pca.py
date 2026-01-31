import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.datasets import KarateClub


def pca_2d(features):
    x = np.array(features, dtype=float)
    x_centered = x - x.mean(axis=0, keepdims=True)
    cov = np.cov(x_centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    components = eigvecs[:, order][:, :2]
    emb_2d = x_centered @ components
    return emb_2d


def main():
    dataset = KarateClub()
    data = dataset[0]

    x = data.x.cpu().numpy()
    labels = data.y.cpu().numpy()

    emb_2d = pca_2d(x)

    plt.figure(figsize=(5, 4))
    scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels, cmap=plt.cm.Set2, s=60, edgecolors="k")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA of Node Features")
    plt.tight_layout()
    plt.savefig("node_features_pca.png", dpi=300)


if __name__ == "__main__":
    main()
