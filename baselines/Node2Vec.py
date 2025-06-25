import networkx as nx
from node2vec import Node2Vec
from sklearn.cluster import KMeans
    # prepare embedding matrix
import numpy as np
from networkx.algorithms.community import modularity
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

# your existing functions
def load_graph_with_attributes(node_file, edge_file):
    G = nx.Graph()
    node_map = {}
    node_id_counter = 0
    with open(node_file, 'r') as node_file:
        for line in node_file:
            line = line.strip()
            if line:
                parts = line.split()
                original_node_id = parts[0]
                actual_community = int(parts[1])
                if original_node_id not in node_map:
                    node_map[original_node_id] = node_id_counter
                    node_id_counter += 1
                new_node_id = node_map[original_node_id]
                G.add_node(new_node_id, label=original_node_id, actual_community=actual_community)
    with open(edge_file, 'r') as edge_file:
        for line in edge_file:
            node_pair = line.strip().split()
            src_new_id = node_map[node_pair[0]]
            tgt_new_id = node_map[node_pair[1]]
            if src_new_id != tgt_new_id: 
                G.add_edge(src_new_id, tgt_new_id)
    isolated_nodes = list(nx.isolates(G))
    G.remove_nodes_from(isolated_nodes)
    new_node_map = {old_id: new_id for new_id, old_id in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, new_node_map)

    return G


def cal_q(G, pred_labels):
    # modularity
    coms = {}
    for idx, lab in enumerate(pred_labels):
        coms.setdefault(int(lab), []).append(idx)
    community_sets = [set(nodes) for nodes in coms.values()]
    q = modularity(G, community_sets)
    return q

def extract_hop_subgraph(G,cutoff=3):
    # 1) find node with max degree
    max_deg_node, max_deg = max(G.degree, key=lambda x: x[1])
    lengths = nx.single_source_shortest_path_length(G, max_deg_node, cutoff=cutoff)
    nodes_3hop = list(lengths.keys())
    # 2) extract subgraph
    G_sub = G.subgraph(nodes_3hop).copy()
    new_node_map = {old_id: new_id for new_id, old_id in enumerate(G_sub.nodes())}
    G_sub = nx.relabel_nodes(G_sub, new_node_map)
    return G_sub

if __name__ == '__main__':
    # parameters
    file_name = 'email-Eu-core'
    node_file = f"norm_dataset/{file_name}/{file_name}_nodes.txt"
    edge_file = f"norm_dataset/{file_name}/{file_name}_edges.txt"

    # load graph
    G = load_graph_with_attributes(node_file, edge_file)
    cutoff = None
    if cutoff is not None:
        G = extract_hop_subgraph(G, cutoff=cutoff)
    
    node2vec = Node2Vec(
        G,
        dimensions=128,
        walk_length=20,
        num_walks=3,
        p=1.0,
        q=1.0,
        workers=4
    )
    model = node2vec.fit(window=10, min_count=1)


    X = np.array([model.wv[str(node)] for node in G.nodes()])

    # determine number of communities
    true_coms = [G.nodes[n]['actual_community'] for n in G.nodes()]
    n_clusters = len(set(true_coms))

    # clustering
    kmeans = KMeans(n_clusters=n_clusters)
    pred_labels = kmeans.fit_predict(X)

    # evaluation
    q, nmi, ari = cal_q(G, pred_labels)
    print(f"Modularity: {q:.4f}")