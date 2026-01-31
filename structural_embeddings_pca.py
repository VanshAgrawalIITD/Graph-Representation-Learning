import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from torch_geometric.datasets import KarateClub

from structural_features import build_undirected_adj, compute_structural_features


def pca_2d(features):
    X = np.array(features, dtype=float)
    X_centered = X - X.mean(axis=0, keepdims=True)

    cov = np.cov(X_centered, rowvar=False)

    eigvals, eigvecs = np.linalg.eigh(cov)

    order = np.argsort(eigvals)[::-1]
    eigvals_sorted = eigvals[order]
    eigvecs_sorted = eigvecs[:, order]

    components = eigvecs_sorted[:, :2]

    emb_2d = X_centered @ components

    return emb_2d, components, eigvals_sorted


def main():
    dataset = KarateClub()
    data = dataset[0]

    adj = build_undirected_adj(data.num_nodes, data.edge_index)
    node_order = list(range(data.num_nodes))

    X_struct = compute_structural_features(
        adj,
        node_order=node_order,
        should_normalize=True,
    )
    emb_2d, components, eigvals_sorted = pca_2d(X_struct)

    feature_names = [
        "degree",
        "betweenness",
        "closeness",
        "pagerank",
        "eigenvector",
        "clustering",
    ]

    if components.shape[0] != len(feature_names):
        print(
            f"[WARN] components has {components.shape[0]} rows (features), but feature_names has {len(feature_names)}. "
            "Update feature_names to match compute_structural_features column order/length."
        )

    total_var = float(np.sum(eigvals_sorted)) if np.sum(eigvals_sorted) != 0 else 1.0
    evr = eigvals_sorted / total_var
    print("\nExplained variance ratio:")
    print(f"  PC1: {evr[0]:.4f}")
    print(f"  PC2: {evr[1]:.4f}")

    for pc_idx in range(2):
        pc_name = f"PC{pc_idx + 1}"
        print(f"\n{pc_name} loadings (feature weights):")

        pairs = []
        for i in range(min(len(feature_names), components.shape[0])):
            pairs.append((feature_names[i], float(components[i, pc_idx])))

        pairs.sort(key=lambda x: abs(x[1]), reverse=True)

        for fname, w in pairs:
            print(f"  {fname:12s} {w:+.6f}")

    labels = data.y.tolist()

    classes = sorted(set(int(v) for v in labels))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    labels_idx = np.array([class_to_idx[int(v)] for v in labels], dtype=int)

    cmap = plt.cm.get_cmap("Set2", len(classes))
    bounds = np.arange(-0.5, len(classes) + 0.5, 1)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(
        emb_2d[:, 0],
        emb_2d[:, 1],
        c=labels_idx,
        cmap=cmap,
        norm=norm,
        s=70,
        edgecolors="k",
        linewidths=0.4,
    )


    plt.title("PCA of Structural Features")
    plt.xlabel("PC1")
    plt.ylabel("PC2")

    cbar = plt.colorbar(scatter, ticks=np.arange(len(classes)))
    cbar.set_ticklabels([str(c) for c in classes])
    cbar.set_label("data.y class")

    plt.tight_layout()
    plt.savefig("structural_pca.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    main()
