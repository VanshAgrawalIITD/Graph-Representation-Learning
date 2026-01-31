import math
from torch_geometric.datasets import KarateClub


def build_undirected_adj(num_nodes, edge_index):
    adj = [set() for _ in range(num_nodes)]
    for u, v in zip(edge_index[0], edge_index[1]):
        u = int(u)
        v = int(v)
        if u == v:
            continue
        adj[u].add(v)
        adj[v].add(u)
    return adj


def bfs_distances(adj, source):
    n = len(adj)
    dist = [-1] * n
    dist[source] = 0
    queue = [source]
    head = 0
    while head < len(queue):
        u = queue[head]
        head += 1
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                queue.append(v)
    return dist


def betweenness_centrality(adj):
    num_nodes = len(adj)
    centrality_scores = [0.0] * num_nodes
    
    for source_node in range(num_nodes):
        path_counts = [0.0] * num_nodes
        path_counts[source_node] = 1.0
        
        node_distances = [-1] * num_nodes
        node_distances[source_node] = 0
        
        predecessors = [[] for _ in range(num_nodes)]
        visit_order = []
        
        frontier = [source_node]
        frontier_pos = 0
        
        while frontier_pos < len(frontier):
            current_node = frontier[frontier_pos]
            frontier_pos += 1
            visit_order.append(current_node)
            
            for adjacent_node in adj[current_node]:
                if node_distances[adjacent_node] < 0:
                    node_distances[adjacent_node] = node_distances[current_node] + 1
                    frontier.append(adjacent_node)
                
                if node_distances[adjacent_node] == node_distances[current_node] + 1:
                    path_counts[adjacent_node] += path_counts[current_node]
                    predecessors[adjacent_node].append(current_node)
        
        node_dependencies = [0.0] * num_nodes

        # Back-propagate dependencies in reverse BFS order (Brandes algorithm)
        for w in reversed(visit_order):
            for v in predecessors[w]:
                if path_counts[w] != 0:
                    node_dependencies[v] += (path_counts[v] / path_counts[w]) * (1.0 + node_dependencies[w])
            if w != source_node:
                centrality_scores[w] += node_dependencies[w]
    
    # For undirected graphs, each shortest path is counted twice (s->t and t->s)
    for idx in range(num_nodes):
        centrality_scores[idx] /= 2.0

    # Normalize to [0, 1] using the standard factor for undirected graphs
    # (n-1)(n-2)/2 is the maximum possible unnormalized betweenness in an undirected graph
    if num_nodes > 2:
        scale_factor = 2.0 / ((num_nodes - 1) * (num_nodes - 2))
        centrality_scores = [score * scale_factor for score in centrality_scores]
    
    return centrality_scores


def closeness_centrality(adj):
    n = len(adj)
    cc = [0.0] * n
    for u in range(n):
        dist = bfs_distances(adj, u)
        total = 0
        reachable = 0
        for d in dist:
            if d > 0:
                total += d
                reachable += 1
        if total > 0 and reachable > 0:
            cc[u] = reachable / total
    return cc


def pagerank(adj, damping=0.85, max_iter=100, tol=1e-6):
    n = len(adj)
    pr = [1.0 / n] * n
    out_deg = [len(adj[u]) for u in range(n)]
    for _ in range(max_iter):
        new_pr = [(1.0 - damping) / n] * n
        for u in range(n):
            if out_deg[u] == 0:
                share = damping * pr[u] / n
                for v in range(n):
                    new_pr[v] += share
            else:
                share = damping * pr[u] / out_deg[u]
                for v in adj[u]:
                    new_pr[v] += share
        diff = 0.0
        for i in range(n):
            diff += abs(new_pr[i] - pr[i])
        pr = new_pr
        if diff < tol:
            break
    return pr


def eigenvector_centrality(adj, max_iter=100, tol=1e-6):
    num_nodes = len(adj)
    centrality = [1.0] * num_nodes
    
    for iteration in range(max_iter):
        new_centrality = [0.0] * num_nodes
        
        for node in range(num_nodes):
            for neighbor in adj[node]:
                new_centrality[node] += centrality[neighbor]
        
        magnitude = math.sqrt(sum(score ** 2 for score in new_centrality))
        
        if magnitude == 0:
            break
        
        normalized_centrality = [score / magnitude for score in new_centrality]
        
        convergence_error = sum(abs(normalized_centrality[i] - centrality[i]) for i in range(num_nodes))
        centrality = normalized_centrality
        
        if convergence_error < tol:
            break
    
    return centrality


def clustering_coefficient(adj):
    num_nodes = len(adj)
    cluster_coeff = [0.0] * num_nodes
    
    for node_idx in range(num_nodes):
        neighbor_list = list(adj[node_idx])
        neighbor_count = len(neighbor_list)
        
        if neighbor_count < 2:
            cluster_coeff[node_idx] = 0.0
            continue
        
        triangle_edges = 0
        
        for first_idx in range(neighbor_count):
            for second_idx in range(first_idx + 1, neighbor_count):
                neighbor_a = neighbor_list[first_idx]
                neighbor_b = neighbor_list[second_idx]
                if neighbor_b in adj[neighbor_a]:
                    triangle_edges += 1
        
        max_possible_edges = neighbor_count * (neighbor_count - 1) / 2.0
        cluster_coeff[node_idx] = triangle_edges / max_possible_edges if max_possible_edges > 0 else 0.0
    
    return cluster_coeff


def zscore_columns(data_table):
    row_count = len(data_table)
    col_count = len(data_table[0])
    
    for col_idx in range(col_count):
        column_sum = 0.0
        for row_idx in range(row_count):
            column_sum += data_table[row_idx][col_idx]
        group_mean = column_sum / row_count
        
        variance_sum = 0.0
        for row_idx in range(row_count):
            deviation = data_table[row_idx][col_idx] - group_mean
            variance_sum += deviation * deviation
        
        group_std = math.sqrt(variance_sum / row_count)
        if group_std == 0:
            group_std = 1.0
            
        for row_idx in range(row_count):
            data_table[row_idx][col_idx] = (data_table[row_idx][col_idx] - group_mean) / group_std
            
    return data_table


def compute_structural_features(adjacency_list, node_order=None, should_normalize=True):
    total_nodes = len(adjacency_list)
    
    if node_order is None:
        node_order = list(range(total_nodes))
    
    node_degrees = [len(adjacency_list[node]) for node in range(total_nodes)]
    betweenness_scores = betweenness_centrality(adjacency_list)
    closeness_scores = closeness_centrality(adjacency_list)
    pagerank_scores = pagerank(adjacency_list)
    eigenvector_scores = eigenvector_centrality(adjacency_list)
    clustering_scores = clustering_coefficient(adjacency_list)
    
    feature_matrix = []
    for current_node in node_order:
        feature_matrix.append([
            node_degrees[current_node],
            betweenness_scores[current_node],
            closeness_scores[current_node],
            pagerank_scores[current_node],
            eigenvector_scores[current_node],
            clustering_scores[current_node],
        ])
    
    if should_normalize:
        feature_matrix = zscore_columns(feature_matrix)
    
    return feature_matrix


def main():
    karate_dataset = KarateClub()
    graph_data = karate_dataset[0]
    
    adjacency_list = build_undirected_adj(graph_data.num_nodes, graph_data.edge_index)
    
    node_order = list(range(graph_data.num_nodes))
    
    structural_embeddings = compute_structural_features(adjacency_list, node_order=node_order, should_normalize=True)
    
    print("Shape of the structural embedding matrix:", (len(structural_embeddings), len(structural_embeddings[0])))
    print("First 5 rows of the matrix:")
    for row in structural_embeddings[:5]:
        print(row)
    
    # Tiny sanity-check block vs NetworkX on Karate Club
    try:
        import networkx as nx
    except Exception as exc:
        print("NetworkX not available; skipping sanity check.", exc)
        return
    
    G = nx.Graph()
    G.add_nodes_from(range(graph_data.num_nodes))
    edge_list = list(zip(graph_data.edge_index[0].tolist(), graph_data.edge_index[1].tolist()))
    G.add_edges_from(edge_list)
    
    nx_deg = [d for _, d in G.degree(range(graph_data.num_nodes))]
    nx_bet = nx.betweenness_centrality(G, normalized=True)
    nx_clo = nx.closeness_centrality(G)
    nx_pr = nx.pagerank(G, alpha=0.85, tol=1e-6)
    nx_eig = nx.eigenvector_centrality(G, max_iter=100, tol=1e-6)
    nx_clu = nx.clustering(G)
    
    local_deg = [len(adjacency_list[i]) for i in range(graph_data.num_nodes)]
    local_bet = betweenness_centrality(adjacency_list)
    local_clo = closeness_centrality(adjacency_list)
    local_pr = pagerank(adjacency_list, damping=0.85, max_iter=100, tol=1e-6)
    local_eig = eigenvector_centrality(adjacency_list, max_iter=100, tol=1e-6)
    local_clu = clustering_coefficient(adjacency_list)
    
    def max_abs_diff(a, b):
        return max(abs(x - y) for x, y in zip(a, b))
    
    diffs = {
        "degree": max_abs_diff(local_deg, nx_deg),
        "betweenness": max_abs_diff(local_bet, [nx_bet[i] for i in range(graph_data.num_nodes)]),
        "closeness": max_abs_diff(local_clo, [nx_clo[i] for i in range(graph_data.num_nodes)]),
        "pagerank": max_abs_diff(local_pr, [nx_pr[i] for i in range(graph_data.num_nodes)]),
        "eigenvector": max_abs_diff(local_eig, [nx_eig[i] for i in range(graph_data.num_nodes)]),
        "clustering": max_abs_diff(local_clu, [nx_clu[i] for i in range(graph_data.num_nodes)]),
    }
    
    tol = 1e-5
    print("Sanity check max abs diffs vs NetworkX:")
    for name, diff in diffs.items():
        print(f"  {name}: {diff}")
    print("All within tolerance:", all(diff <= tol for diff in diffs.values()))


if __name__ == "__main__":
    main()
