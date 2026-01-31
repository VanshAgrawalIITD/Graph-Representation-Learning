import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch_geometric.datasets import Planetoid

from structural_features import build_undirected_adj, compute_structural_features


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_mlp(input_dim: int, num_classes: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(64, num_classes),
    )


def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    correct = (preds[mask] == y[mask]).sum().item()
    total = int(mask.sum().item())
    return correct / total if total > 0 else 0.0


def train_with_early_stopping(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    epochs: int = 500,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    patience: int = 50,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val = -1.0
    best_state = None
    patience_left = patience

    for _ in range(epochs):
        model.train()
        logits = model(x)
        loss = criterion(logits[train_mask], y[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(x)
            val_acc = accuracy_from_logits(val_logits, y, val_mask)

        if val_acc > best_val + 1e-6:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left == 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        logits = model(x)
        val_acc = accuracy_from_logits(logits, y, val_mask)
        test_acc = accuracy_from_logits(logits, y, test_mask)

    return val_acc, test_acc


def load_or_compute_structural_features(data, cache_path: Path) -> torch.Tensor:
    if cache_path.exists():
        cached = np.load(cache_path)
        if cached.shape[0] == data.num_nodes:
            return torch.from_numpy(cached).float()

    adj = build_undirected_adj(data.num_nodes, data.edge_index)
    node_order = list(range(data.num_nodes))
    x_struct = compute_structural_features(
        adj,
        node_order=node_order,
        should_normalize=True,
    )
    x_struct = np.asarray(x_struct, dtype=np.float32)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, x_struct)

    return torch.from_numpy(x_struct).float()


def main():
    set_seed(42)

    dataset = Planetoid(root="data/Planetoid", name="Cora")
    data = dataset[0]

    x_attr = data.x.clone().float()
    y = data.y.clone().long()

    cache_path = Path("data/cora_structural_features.npy")
    x_struct = load_or_compute_structural_features(data, cache_path)
    x_comb = torch.cat([x_struct, x_attr], dim=1)

    train_mask = data.train_mask
    val_mask = data.val_mask
    test_mask = data.test_mask

    settings = [
        ("Structural only", x_struct),
        ("Node features only", x_attr),
        ("Combined", x_comb),
    ]

    for name, x in settings:
        model = build_mlp(x.size(1), dataset.num_classes)
        val_acc, test_acc = train_with_early_stopping(
            model,
            x,
            y,
            train_mask,
            val_mask,
            test_mask,
        )
        print(f"{name}: Val acc = {val_acc:.4f}, Test acc = {test_acc:.4f}")


if __name__ == "__main__":
    main()
