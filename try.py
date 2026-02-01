import torch
from torch_geometric.datasets import Planetoid
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx
import numpy as np

# Load Cora dataset
dataset = Planetoid(root='data/Planetoid', name='Cora')
data = dataset[0]

# Print dataset properties
print("=" * 60)
print("CORA DATASET PROPERTIES")
print("=" * 60)
print(f"Number of graphs: {len(dataset)}")
print(f"Number of classes: {dataset.num_classes}")
print(f"Number of node features: {dataset.num_node_features}")
print()

print("Graph Properties:")
print(f"  Number of nodes: {data.num_nodes}")
print(f"  Number of edges: {data.num_edges}")
print(f"  Average node degree: {data.num_edges / data.num_nodes:.2f}")
print(f"  Has isolated nodes: {data.has_isolated_nodes()}")
print(f"  Has self-loops: {data.has_self_loops()}")
print(f"  Is undirected: {data.is_undirected()}")
print()

print("Node Features (data.x):")
print(f"  Shape: {data.x.shape}")
print(f"  Feature dimension: {data.x.shape[1]}")
print(f"  Data type: {data.x.dtype}")
print()

print("Edge Index (data.edge_index):")
print(f"  Shape: {data.edge_index.shape}")
print(f"  First 10 edges:\n{data.edge_index[:, :10]}")
print()

print("Labels (data.y):")
print(f"  Shape: {data.y.shape}")
print(f"  Unique classes: {torch.unique(data.y).tolist()}")
print(f"  Class distribution:")
for class_idx in range(dataset.num_classes):
    count = (data.y == class_idx).sum().item()
    print(f"    Class {class_idx}: {count} nodes")
print()

print("Train/Val/Test Split:")
print(f"  Training nodes: {data.train_mask.sum().item()}")
print(f"  Validation nodes: {data.val_mask.sum().item()}")
print(f"  Test nodes: {data.test_mask.sum().item()}")
print("=" * 60)

# Visualize the graph
print("\nCreating visualization...")
G = to_networkx(data, to_undirected=True)

# Create figure with subplots
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Plot 1: Full graph structure
ax1 = axes[0]
pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
nx.draw_networkx(G, pos, 
                 node_size=20, 
                 node_color='lightblue',
                 with_labels=False,
                 edge_color='gray',
                 alpha=0.6,
                 width=0.3,
                 ax=ax1)
ax1.set_title('Cora Citation Network - Full Graph', fontsize=14, fontweight='bold')
ax1.axis('off')

# Plot 2: Graph colored by class labels
ax2 = axes[1]
node_colors = data.y.numpy()
cmap = plt.cm.get_cmap('tab10', dataset.num_classes)
nx.draw_networkx(G, pos,
                 node_size=20,
                 node_color=node_colors,
                 cmap=cmap,
                 with_labels=False,
                 edge_color='gray',
                 alpha=0.6,
                 width=0.3,
                 ax=ax2)
ax2.set_title('Cora Citation Network - Colored by Class', fontsize=14, fontweight='bold')
ax2.axis('off')

# Add colorbar legend
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=dataset.num_classes-1))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax2, fraction=0.046, pad=0.04)
cbar.set_label('Class Label', rotation=270, labelpad=20)

plt.tight_layout()
plt.savefig('cora_visualization.png', dpi=300, bbox_inches='tight')
print("Visualization saved as 'cora_visualization.png'")
plt.show()