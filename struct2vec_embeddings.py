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


def k_hop_signatures(adj, k):
    num_nodes = len(adj)
    degrees = [len(adj[u]) for u in range(num_nodes)]
    signatures = []
    for u in range(num_nodes):
        visited = {u}
        frontier = [u]
        for _ in range(k):
            next_frontier = []
            for v in frontier:
                for w in adj[v]:
                    if w not in visited:
                        visited.add(w)
                        next_frontier.append(w)
            frontier = next_frontier
            if not frontier:
                break
        sig = sorted(degrees[w] for w in visited)
        signatures.append(sig)
    return signatures


def gk_similarity(sig_u, sig_v):
    len_u = len(sig_u)
    len_v = len(sig_v)
    L = max(len_u, len_v)
    if L == 0:
        return 0.0
    if len_u < L:
        sig_u = sig_u + [0] * (L - len_u)
    if len_v < L:
        sig_v = sig_v + [0] * (L - len_v)
    diff = np.abs(np.array(sig_u, dtype=np.float64) - np.array(sig_v, dtype=np.float64)).sum()
    return math.exp(-diff / L)


def build_struct2vec_graph(signatures, top_m=10):
    n = len(signatures)
    weighted_adj = [[] for _ in range(n)]
    for u in range(n):
        sims = []
        for v in range(n):
            if u == v:
                continue
            s = gk_similarity(signatures[u], signatures[v])
            sims.append((v, s))
        sims.sort(key=lambda x: x[1], reverse=True)
        if top_m is not None:
            sims = sims[:top_m]
        weighted_adj[u] = sims
    return weighted_adj


def weighted_walk(start, walk_length, weighted_adj, rng):
    walk = [start]
    while len(walk) < walk_length:
        v = walk[-1]
        neighbors = weighted_adj[v]
        if not neighbors:
            break
        nodes = [n for n, _ in neighbors]
        weights = np.array([w for _, w in neighbors], dtype=np.float64)
        weights_sum = weights.sum()
        if weights_sum == 0:
            break
        probs = weights / weights_sum
        next_node = rng.choice(nodes, p=probs)
        walk.append(int(next_node))
    return walk


def generate_walks(weighted_adj, num_walks_per_node, walk_length, seed=42):
    rng = np.random.default_rng(seed)
    walks = []
    nodes = list(range(len(weighted_adj)))
    for _ in range(num_walks_per_node):
        rng.shuffle(nodes)
        for start in nodes:
            walks.append(weighted_walk(start, walk_length, weighted_adj, rng))
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
    emb_in = nn.Embedding(num_nodes, embedding_dim)
    emb_out = nn.Embedding(num_nodes, embedding_dim)
    optimizer = optim.Adam(list(emb_in.parameters()) + list(emb_out.parameters()), lr=lr)

    pairs = np.asarray(pairs, dtype=np.int64)
    num_pairs = len(pairs)

    for epoch in range(epochs):
        perm = np.random.permutation(num_pairs)
        for start in range(0, num_pairs, batch_size):
            idx = perm[start : start + batch_size]
            batch = pairs[idx]
            centers = torch.tensor(batch[:, 0], dtype=torch.long)
            contexts = torch.tensor(batch[:, 1], dtype=torch.long)

            center_vecs = emb_in(centers)
            context_vecs = emb_out(contexts)
            pos_scores = torch.sum(center_vecs * context_vecs, dim=1)
            pos_loss = -F.logsigmoid(pos_scores).mean()

            neg_samples = torch.multinomial(
                neg_dist, centers.size(0) * num_negatives, replacement=True
            )
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
    k = 3
    top_m = 10
    num_walks_per_node = 10
    walk_length = 20
    window_size = 5

    dataset = GeometricShapes(root="data/GeometricShapes", pre_transform=FaceToEdge())
    indices = [0, 1, 2]

    for idx in indices:
        data = dataset[idx]
        adj = build_undirected_adj(data.num_nodes, data.edge_index)
        signatures = k_hop_signatures(adj, k=k)
        weighted_adj = build_struct2vec_graph(signatures, top_m=top_m)

        walks = generate_walks(weighted_adj, num_walks_per_node, walk_length, seed=42)
        pairs = build_pairs(walks, window_size=window_size)
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

        out_path = f"struct2vec_graph{idx}.png"
        title = f"Struct2Vec (k={k}) - Graph {idx}"
        plot_embedding(data, emb, title, out_path)
        print(
            f"Graph {idx}: walks={len(walks)}, pairs={len(pairs)}, saved {out_path}"
        )
