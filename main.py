import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.datasets import KarateClub
from torch_geometric.utils import to_networkx

dataset = KarateClub()
data = dataset[0]

# Convert PyG graph to NetworkX
G = to_networkx(data, to_undirected=True)

# Node colors based on labels
labels = data.y.tolist()

# Draw the graph
pos = nx.spring_layout(G, seed=42)
plt.figure(figsize=(6, 5))
nx.draw(
    G,
    pos,
    node_color=labels,
    cmap=plt.cm.Set2,
    with_labels=True,
    node_size=400,
    font_size=8,
)
plt.title("Karate Club Graph")
plt.show()
