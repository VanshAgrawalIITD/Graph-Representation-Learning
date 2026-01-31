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

# Define colors to match the figure legend
color_map = {
    0: '#66C2A5',  # Green
    1: '#FC8D62',  # Orange/Salmon
    2: '#8DA0CB',  # Blue
    3: '#FFD92F'   # Yellow
}
node_colors = [color_map[label] for label in labels]

# Draw the graph
pos = nx.spring_layout(G, seed=42)
plt.figure(figsize=(6, 5))
nx.draw(
    G,
    pos,
    node_color=node_colors,
    with_labels=True,
    node_size=400,
    font_size=8,
)
plt.title("Karate Club Graph")
plt.show()
