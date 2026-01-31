import random
import numpy as np
import torch
from torch import nn
from torch_geometric.datasets import KarateClub
from sklearn.model_selection import StratifiedKFold

from structural_features import build_undirected_adj, compute_structural_features


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_mlp(input_dim, num_classes):
    return nn.Sequential(
        nn.Linear(input_dim, 16),
        nn.ReLU(),
        nn.Linear(16, num_classes),
    )


def build_logreg(input_dim, num_classes):
    return nn.Linear(input_dim, num_classes)


def train_classifier(model, x, y, train_idx, epochs=500, lr=0.05, weight_decay=1e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        model.train()
        logits = model(x)
        loss = criterion(logits[train_idx], y[train_idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model


def accuracy(model, x, y, idx):
    model.eval()
    with torch.no_grad():
        preds = model(x).argmax(dim=1)
        correct = (preds[idx] == y[idx]).sum().item()
        total = idx.numel()
    return correct / total if total > 0 else 0.0


def main():
    set_seed(42)

    dataset = KarateClub()
    data = dataset[0]

    adj = build_undirected_adj(data.num_nodes, data.edge_index)
    node_order = list(range(data.num_nodes))
    x_struct = compute_structural_features(
        adj,
        node_order=node_order,
        should_normalize=True,
    )

    x = torch.tensor(x_struct, dtype=torch.float32)
    y = data.y.clone().long()

    num_nodes = data.num_nodes
    num_classes = dataset.num_classes

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    train_accuracies = []
    test_accuracies = []

    def eval_model(build_fn, label):
        train_accuracies.clear()
        test_accuracies.clear()

        for fold, (train_idx_np, test_idx_np) in enumerate(skf.split(x, y)):
            set_seed(42 + fold)
            train_idx = torch.tensor(train_idx_np, dtype=torch.long)
            test_idx = torch.tensor(test_idx_np, dtype=torch.long)

            model = build_fn(x.size(1), num_classes)
            model = train_classifier(model, x, y, train_idx)

            train_acc = accuracy(model, x, y, train_idx)
            test_acc = accuracy(model, x, y, test_idx)

            train_accuracies.append(train_acc)
            test_accuracies.append(test_acc)

            print(f"{label} Fold {fold + 1}: Train acc = {train_acc:.4f}, Test acc = {test_acc:.4f}")

        print(f"\n{label} overall performance:")
        print(f"Train accuracy: {np.mean(train_accuracies):.4f} ± {np.std(train_accuracies):.4f}")
        print(f"Test accuracy:  {np.mean(test_accuracies):.4f} ± {np.std(test_accuracies):.4f}\n")

        return (float(np.mean(train_accuracies)), float(np.std(train_accuracies)),
                float(np.mean(test_accuracies)), float(np.std(test_accuracies)))

    mlp_stats = eval_model(build_mlp, "MLP")
    logreg_stats = eval_model(build_logreg, "LogReg")

    print("Summary:")
    print(f"MLP    Train {mlp_stats[0]:.4f} ± {mlp_stats[1]:.4f} | Test {mlp_stats[2]:.4f} ± {mlp_stats[3]:.4f}")
    print(f"LogReg Train {logreg_stats[0]:.4f} ± {logreg_stats[1]:.4f} | Test {logreg_stats[2]:.4f} ± {logreg_stats[3]:.4f}")


if __name__ == "__main__":
    main()
