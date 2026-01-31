import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch_geometric.datasets import GeometricShapes
from torch_geometric.transforms import FaceToEdge


def build_undirected_adj(num_nodes, edge_index):
    adj = [set() for _ in range(num_nodes)]
    src, dst = edge_index
    for u, v in zip(src, dst):
        u = int(u)
        v = int(v)
        adj[u].add(v)
        adj[v].add(u)
    return adj


def node2vec_walk(start, walk_length, adj, p, q, rng):
    walk = [start]
    while len(walk) < walk_length:
        v = walk[-1]
        neighbors = list(adj[v])
        if not neighbors:
            break
        if len(walk) == 1:
            next_node = rng.choice(neighbors)
        else:
            t = walk[-2]
            weights = []
            t_neighbors = adj[t]
            for x in neighbors:
                if x == t:
                    w = 1.0 / p
                elif x in t_neighbors:
                    w = 1.0
                else:
                    w = 1.0 / q
                weights.append(w)
            weights = np.asarray(weights, dtype=np.float64)
            probs = weights / weights.sum()
            next_node = rng.choice(neighbors, p=probs)
        walk.append(int(next_node))
    return walk


def generate_walks(adj, num_walks_per_node, walk_length, p, q, seed=42):
    rng = np.random.default_rng(seed)
    walks = []
    nodes = list(range(len(adj)))
    for _ in range(num_walks_per_node):
        rng.shuffle(nodes)
        for start in nodes:
            walks.append(node2vec_walk(start, walk_length, adj, p, q, rng))
    return walks


def build_pairs(walks, window_size):
    pairs = []
    for walk in walks:
        for i, center in enumerate(walk):
            left = max(0, i - window_size)
            right = min(len(walk), i + window_size + 1)
            for j in range(left, right):
                if i == j:
                    continue
                pairs.append((center, walk[j]))
    return pairs


def build_negative_sampling_dist(walks, num_nodes, power=0.75):
    counts = np.zeros(num_nodes, dtype=np.float64)
    for walk in walks:
        for node in walk:
            counts[node] += 1
    counts = np.power(counts, power)
    counts[counts == 0] = 1.0
    probs = counts / counts.sum()
    return torch.tensor(probs, dtype=torch.float32)


def train_skipgram(
    num_nodes,
    pairs,
    neg_dist,
    embedding_dim=2,
    num_negatives=5,
    epochs=50,
    batch_size=256,
    lr=0.01,
    seed=42,
):
    torch.manual_seed(seed)
    device = torch.device("cpu")

    emb_in = nn.Embedding(num_nodes, embedding_dim).to(device)
    emb_out = nn.Embedding(num_nodes, embedding_dim).to(device)
    optimizer = optim.Adam(list(emb_in.parameters()) + list(emb_out.parameters()), lr=lr)

    pairs = np.asarray(pairs, dtype=np.int64)
    num_pairs = len(pairs)

    for epoch in range(epochs):
        perm = np.random.permutation(num_pairs)
        for start in range(0, num_pairs, batch_size):
            idx = perm[start : start + batch_size]
            batch = pairs[idx]
            centers = torch.tensor(batch[:, 0], dtype=torch.long, device=device)
            contexts = torch.tensor(batch[:, 1], dtype=torch.long, device=device)

            center_vecs = emb_in(centers)
            context_vecs = emb_out(contexts)
            pos_scores = torch.sum(center_vecs * context_vecs, dim=1)
            pos_loss = -F.logsigmoid(pos_scores).mean()

            neg_samples = torch.multinomial(
                neg_dist, centers.size(0) * num_negatives, replacement=True
            ).to(device)
            neg_samples = neg_samples.view(centers.size(0), num_negatives)
            neg_vecs = emb_out(neg_samples)
            neg_scores = torch.sum(center_vecs.unsqueeze(1) * neg_vecs, dim=2)
            neg_loss = -F.logsigmoid(-neg_scores).mean()

            loss = pos_loss + neg_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return emb_in.weight.detach().cpu().numpy()


def plot_embedding(data, emb, title, out_path):
    edge_index = data.edge_index.cpu().numpy()
    fig, ax = plt.subplots(figsize=(5, 5))
    for u, v in edge_index.T:
        xs = [emb[u, 0], emb[v, 0]]
        ys = [emb[u, 1], emb[v, 1]]
        ax.plot(xs, ys, color="black", linewidth=0.5, alpha=0.6)
    ax.scatter(emb[:, 0], emb[:, 1], s=12, c="tab:blue")
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    # Node2Vec parameters (reported in the writeup)
    p = 10
    q_bfs = 0.5
    q_dfs = 5
    print(f"Node2Vec parameters: p={p}, q_bfs={q_bfs}, q_dfs={q_dfs}")

    dataset = GeometricShapes(root="data/GeometricShapes", pre_transform=FaceToEdge())
    indices = [0, 1, 2]

    for idx in indices:
        data = dataset[idx]
        adj = build_undirected_adj(data.num_nodes, data.edge_index)
        for label, q in [("bfs", q_bfs), ("dfs", q_dfs)]:
            walks = generate_walks(
                adj,
                num_walks_per_node=10,
                walk_length=20,
                p=p,
                q=q,
                seed=42,
            )
            pairs = build_pairs(walks, window_size=5)
            neg_dist = build_negative_sampling_dist(walks, data.num_nodes)
            emb = train_skipgram(
                data.num_nodes,
                pairs,
                neg_dist,
                embedding_dim=2,
                num_negatives=5,
                epochs=50,
                batch_size=256,
                lr=0.01,
                seed=42,
            )
            out_path = f"node2vec_graph{idx}_{label}.png"
            title = f"Node2Vec ({label}) - Graph {idx}"
            plot_embedding(data, emb, title, out_path)
            print(
                f"Graph {idx} [{label}]: walks={len(walks)}, pairs={len(pairs)}, saved {out_path}"
            )
