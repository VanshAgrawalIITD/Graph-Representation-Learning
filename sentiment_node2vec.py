import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_imdb_split(base_dir: Path, split: str, max_per_class: int | None = None):
    texts, labels = [], []
    for label, subdir in [(1, "pos"), (0, "neg")]:
        dir_path = base_dir / split / subdir
        files = sorted(dir_path.glob("*.txt"))
        if max_per_class is not None:
            files = files[:max_per_class]
        for path in files:
            texts.append(path.read_text(encoding="utf-8", errors="ignore"))
            labels.append(label)
    return texts, labels


def preprocess(text: str, stopwords: set[str]):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [t for t in text.split() if t and t not in stopwords]
    return tokens


def build_vocab(tokenized_texts, vocab_size: int):
    counter = Counter()
    for tokens in tokenized_texts:
        counter.update(tokens)
    most_common = counter.most_common(vocab_size)
    vocab = {word: idx for idx, (word, _) in enumerate(most_common)}
    freqs = np.array([count for _, count in most_common], dtype=np.float64)
    return vocab, freqs


def build_cooc_graph(tokenized_texts, vocab, window_size: int):
    edge_counts = defaultdict(int)
    for tokens in tokenized_texts:
        filtered = [t for t in tokens if t in vocab]
        for i, token in enumerate(filtered):
            u = vocab[token]
            for j in range(i + 1, min(i + window_size + 1, len(filtered))):
                v = vocab[filtered[j]]
                if u == v:
                    continue
                a, b = (u, v) if u < v else (v, u)
                edge_counts[(a, b)] += 1

    adj = [defaultdict(float) for _ in range(len(vocab))]
    for (u, v), w in edge_counts.items():
        adj[u][v] += w
        adj[v][u] += w
    return adj


def node2vec_walk(start, walk_length, adj, p, q, rng):
    walk = [start]
    while len(walk) < walk_length:
        v = walk[-1]
        neighbors = list(adj[v].keys())
        if not neighbors:
            break
        if len(walk) == 1:
            weights = np.array([adj[v][x] for x in neighbors], dtype=np.float64)
        else:
            t = walk[-2]
            t_neighbors = adj[t]
            weights = []
            for x in neighbors:
                w = adj[v][x]
                if x == t:
                    w *= 1.0 / p
                elif x in t_neighbors:
                    w *= 1.0
                else:
                    w *= 1.0 / q
                weights.append(w)
            weights = np.asarray(weights, dtype=np.float64)
        probs = weights / weights.sum()
        next_node = rng.choice(neighbors, p=probs)
        walk.append(int(next_node))
    return walk


def generate_walks(adj, num_walks, walk_length, p, q, seed=42):
    rng = np.random.default_rng(seed)
    nodes = np.arange(len(adj))
    walks = []
    for _ in range(num_walks):
        rng.shuffle(nodes)
        for node in nodes:
            if adj[node]:
                walks.append(node2vec_walk(int(node), walk_length, adj, p, q, rng))
    return walks


def build_pairs(walks, window_size):
    pairs = []
    for walk in walks:
        for i, center in enumerate(walk):
            left = max(0, i - window_size)
            right = min(len(walk), i + window_size + 1)
            for j in range(left, right):
                if j == i:
                    continue
                pairs.append((center, walk[j]))
    return pairs


def train_skipgram(pairs, num_nodes, embed_dim, neg_dist, num_neg=5, epochs=3, batch_size=256, lr=0.01):
    device = torch.device("cpu")
    emb_in = nn.Embedding(num_nodes, embed_dim).to(device)
    emb_out = nn.Embedding(num_nodes, embed_dim).to(device)
    nn.init.uniform_(emb_in.weight, -0.5 / embed_dim, 0.5 / embed_dim)
    nn.init.zeros_(emb_out.weight)

    optimizer = optim.Adam([*emb_in.parameters(), *emb_out.parameters()], lr=lr)
    pairs = np.asarray(pairs, dtype=np.int64)

    for epoch in range(epochs):
        rng = np.random.default_rng(42 + epoch)
        rng.shuffle(pairs)
        total_loss = 0.0
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]
            centers = torch.tensor(batch[:, 0], dtype=torch.long, device=device)
            contexts = torch.tensor(batch[:, 1], dtype=torch.long, device=device)

            pos_score = (emb_in(centers) * emb_out(contexts)).sum(dim=1)
            pos_loss = F.logsigmoid(pos_score).mean()

            neg_samples = rng.choice(num_nodes, size=(len(batch), num_neg), p=neg_dist)
            neg_samples = torch.tensor(neg_samples, dtype=torch.long, device=device)
            neg_emb = emb_out(neg_samples)
            center_emb = emb_in(centers).unsqueeze(2)
            neg_score = torch.bmm(neg_emb, center_emb).squeeze(2)
            neg_loss = F.logsigmoid(-neg_score).mean()

            loss = -(pos_loss + neg_loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)
        avg_loss = total_loss / len(pairs)
        print(f"Epoch {epoch + 1}/{epochs} - loss: {avg_loss:.4f}")

    return emb_in.weight.detach().cpu().numpy()


def embed_documents(tokenized_texts, vocab, embeddings):
    dim = embeddings.shape[1]
    doc_vecs = np.zeros((len(tokenized_texts), dim), dtype=np.float32)
    for i, tokens in enumerate(tokenized_texts):
        ids = [vocab[t] for t in tokens if t in vocab]
        if not ids:
            continue
        doc_vecs[i] = embeddings[ids].mean(axis=0)
    return doc_vecs


def plot_tsne(embeddings, vocab, freqs, out_path):
    idx_to_word = {idx: word for word, idx in vocab.items()}
    top_indices = np.argsort(freqs)[-200:]
    words = [idx_to_word[i] for i in top_indices]

    pos_words = {"good", "great", "excellent", "amazing", "love", "wonderful", "enjoy"}
    neg_words = {"bad", "worst", "awful", "terrible", "boring", "hate", "poor"}

    colors = []
    labels = []
    for w in words:
        if w in pos_words:
            colors.append("#2ca02c")
            labels.append(w)
        elif w in neg_words:
            colors.append("#d62728")
            labels.append(w)
        else:
            colors.append("#7f7f7f")
            labels.append("")

    coords = TSNE(n_components=2, init="pca", perplexity=30, random_state=42, learning_rate="auto").fit_transform(
        embeddings[top_indices]
    )

    plt.figure(figsize=(6, 5))
    plt.scatter(coords[:, 0], coords[:, 1], c=colors, s=12, alpha=0.8)
    for i, label in enumerate(labels):
        if label:
            plt.text(coords[i, 0], coords[i, 1], label, fontsize=8)
    plt.title("t-SNE of word embeddings (top 200 words)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    base_dir = Path("data/aclImdb")
    if not base_dir.exists():
        raise FileNotFoundError("Missing data/aclImdb. Download and extract the IMDb dataset.")

    max_train = 2000
    max_test = 1000
    vocab_size = 5000
    window_size = 2

    print("Loading dataset...")
    train_texts, train_labels = load_imdb_split(base_dir, "train", max_per_class=max_train)
    test_texts, test_labels = load_imdb_split(base_dir, "test", max_per_class=max_test)

    stopwords = set(ENGLISH_STOP_WORDS)
    print("Preprocessing...")
    train_tokens = [preprocess(text, stopwords) for text in train_texts]
    test_tokens = [preprocess(text, stopwords) for text in test_texts]

    vocab, freqs = build_vocab(train_tokens, vocab_size)
    print(f"Vocab size: {len(vocab)}")

    print("Building co-occurrence graph...")
    adj = build_cooc_graph(train_tokens, vocab, window_size=window_size)

    print("Generating Node2Vec walks...")
    walks = generate_walks(adj, num_walks=5, walk_length=10, p=1.0, q=0.5, seed=42)

    print("Building skip-gram pairs...")
    pairs = build_pairs(walks, window_size=2)
    print(f"Pairs: {len(pairs)}")

    print("Training skip-gram...")
    neg_dist = freqs ** 0.75
    neg_dist = neg_dist / neg_dist.sum()
    embeddings = train_skipgram(
        pairs,
        num_nodes=len(vocab),
        embed_dim=64,
        neg_dist=neg_dist,
        num_neg=5,
        epochs=3,
        batch_size=256,
        lr=0.01,
    )

    print("Embedding documents...")
    X_train = embed_documents(train_tokens, vocab, embeddings)
    X_test = embed_documents(test_tokens, vocab, embeddings)

    print("Training classifier...")
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, train_labels)
    acc = clf.score(X_test, test_labels)
    print(f"Test accuracy: {acc:.4f}")

    print("Plotting t-SNE...")
    plot_tsne(embeddings, vocab, freqs, "word_tsne.png")


if __name__ == "__main__":
    main()
