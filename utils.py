import os
import numpy as np
import networkx as nx
import community as community_louvain
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, accuracy_score
import igraph as ig
from infomap import Infomap

from scipy.sparse import csr_matrix


def adjust_weight(G,cd_mode,pair_predictions):
    G_weighted = G.copy()
    for node1, node2, prediction in pair_predictions:
        if G.has_edge(node1, node2) and not node1==node2:
            new_weight = prediction*prediction
            if new_weight == 0:
                new_weight = 1e-6
            G_weighted.add_edge(node1, node2, weight=new_weight) 
    G_weighted = community_detection(G_weighted,cd_mode,when='weighted')
    return G_weighted


def cal_accuracy_return(X_train, y_train, clf):
    y_pred_train = clf.predict(X_train)
    train_acc = accuracy_score(y_train, y_pred_train)
    return train_acc

def compare_q_return(q_origin, q_new, nmi_score, nmi_new, ari_score, ari_new):
    result = []
    
    if q_new > q_origin:
        result.append(f"Modularity improved by {q_new - q_origin}")
    else:
        result.append(f"Modularity not improved by {q_new - q_origin}")
    
    if nmi_new > nmi_score:
        result.append(f"NMI improved by {nmi_new - nmi_score}")
    else:
        result.append(f"NMI not improved by {nmi_new - nmi_score}")
    
    if ari_new > ari_score:
        result.append(f"ARI improved by {ari_new - ari_score}")
    else:
        result.append(f"ARI not improved by {ari_new - ari_score}")
    return "\n".join(result)


def cal_q_nmi_ari(G, when):  
    true_labels = [G.nodes[node].get('actual_community') for node in G.nodes()]
    communities = {}
    
    for node in G.nodes():
        community_id = G.nodes[node].get(f'{when}_community')
        if community_id not in communities:
            communities[community_id] = []
        communities[community_id].append(node)
    q_score = nx.algorithms.community.modularity(G, communities.values())
    clusters = [G.nodes[node].get(f'{when}_community') for node in G.nodes()]
    nmi_score = normalized_mutual_info_score(true_labels, clusters)
    ari_score = adjusted_rand_score(true_labels, clusters)
    return q_score, nmi_score, ari_score

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
            if src_new_id != tgt_new_id:  # Avoid self-loops
                G.add_edge(src_new_id, tgt_new_id)
    isolated_nodes = list(nx.isolates(G))
    G.remove_nodes_from(isolated_nodes)
    new_node_map = {old_id: new_id for new_id, old_id in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, new_node_map)
    
    degree_dict = dict(G.degree())
    clustering_dict = nx.clustering(G)

    nx.set_node_attributes(G, degree_dict, 'degree')
    nx.set_node_attributes(G, clustering_dict, 'clustering')

    return G

def extract_pairwise_features_with_common_neighbors(G, all_pairs, am):
    G = G.copy()
    node1 = all_pairs[:, 0]
    node2 = all_pairs[:, 1]

    edges = list(G.edges())
    row, col = zip(*edges)
    data = np.ones(len(edges))
    adj_matrix = csr_matrix((data, (row, col)), shape=(len(G.nodes()), len(G.nodes())))
    
    degree_diff = np.abs(np.array([G.nodes[node1].get('degree') for node1 in node1]) - np.array([G.nodes[node2].get('degree') for node2 in node2]))
    clustering_diff = np.abs(np.array([G.nodes[node1].get('clustering') for node1 in node1]) - np.array([G.nodes[node2].get('clustering') for node2 in node2]))
    
    if am == 'gt':
        labels = (np.array([G.nodes[node1].get('actual_community') for node1 in node1]) == np.array([G.nodes[node2].get('actual_community') for node2 in node2])).astype(int)
    if am == 'sp':
        labels = (np.array([G.nodes[node1].get('origin_community') for node1 in node1]) == np.array([G.nodes[node2].get('origin_community') for node2 in node2])).astype(int)
    
    common_neighbors_matrix = adj_matrix @ adj_matrix
    common_neighbors = np.array(common_neighbors_matrix[node1, node2]).flatten()
    features_with_ids = np.stack((node1, node2, degree_diff, clustering_diff, common_neighbors), axis=1)
    return features_with_ids, labels


def sample_edges_balanced_with_local_search1(G):
    edges = [(u, v) for u, v in G.edges()]
    samples = edges
    return samples


def sample_edges_balanced_with_local_search2(G):
    edges = [(u, v) for u, v in G.edges()]
    row, col = zip(*edges)
    data = np.ones(len(edges))
    adj_matrix = csr_matrix((data, (row, col)), shape=(len(G.nodes()), len(G.nodes())))
    
    A_2 = adj_matrix@adj_matrix
    A_2 = A_2 + adj_matrix
    A_2.setdiag(0)
    
    samples = []
    for i in range(A_2.shape[0]):
        row = A_2.getrow(i).toarray().flatten()
        for j, value in enumerate(row):
            if value > 0:
                if i != j:
                    samples.append([i, j])
    return samples
        
def community_detection(G,cd_mode,when):
    if cd_mode == 'louvain':
        if when == 'origin':
            partition = community_louvain.best_partition(G)
        elif when == 'weighted':
            partition = community_louvain.best_partition(G, weight='weight')
        for node, community_id in partition.items():
            G.nodes[node][f'{when}_community'] = community_id
    elif cd_mode == 'infomap':
        im = Infomap(num_trials=3)
        if when == 'origin':
            for edge in G.edges():
                im.add_link(edge[0], edge[1], 1)
        elif when == 'weighted':
            for u, v, data in G.edges(data=True):
                weight = data.get("weight")
                im.add_link(u, v, weight)
        im.run()
        for node in im.tree:
            if node.is_leaf:
                G.nodes[node.node_id][f'{when}_community'] = node.module_id
    elif cd_mode == 'leiden':
        G_ig = ig.Graph.from_networkx(G)
        if when == 'origin':
            communities = G_ig.community_leiden(objective_function="modularity")
        elif when == 'weighted':
            communities = G_ig.community_leiden(objective_function="modularity", weights='weight')
        for cid, community in enumerate(communities):
            for v in community:
                G.nodes[v][f'{when}_community'] = cid
    elif cd_mode == 'fastgreedy':
        G_ig = ig.Graph.from_networkx(G)
        if when == 'origin':
            communities = G_ig.community_fastgreedy().as_clustering()
        elif when == 'weighted':
            communities = G_ig.community_fastgreedy(weights='weight').as_clustering()
        for cid, community in enumerate(communities):
            for v in community:
                G.nodes[v][f'{when}_community'] = cid
    return G
