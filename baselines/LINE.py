import networkx as nx
from node2vec import Node2Vec
from sklearn.cluster import KMeans
    # prepare embedding matrix
import numpy as np
from networkx.algorithms.community import modularity
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import math
import random

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
class LINE:
    def __init__(self, G, dim=128, neg_samples=5, lr=0.025, epochs=5):
        self.G = G
        self.dim = dim
        self.neg = neg_samples
        self.lr = lr
        self.epochs = epochs
        self.N = G.number_of_nodes()
        self.edges = list(G.edges())

        # degree^0.75 distribution for negative sampling
        degs = np.array([G.degree(i) for i in range(self.N)], dtype=float)
        probs = np.power(degs, 0.75)
        self.neg_dist = probs / probs.sum()

        # initialize embeddings
        self.emb_target = np.random.rand(self.N, self.dim) / self.dim  # u
        self.emb_context = np.random.rand(self.N, self.dim) / self.dim  # u'
        self.emb_first = np.random.rand(self.N, self.dim) / self.dim  # for 1st-order

    def _sigmoid(self, x):
        return 1 / (1 + math.exp(-x)) if x > -6 else math.exp(x) / (1 + math.exp(x))  # numerical stable

    def train_first_order(self):
        print("Training 1st-order proximity...")
        for epoch in range(1, self.epochs + 1):
            print(f"Epoch {epoch}/{self.epochs}")
            random.shuffle(self.edges)
            for u, v in self.edges:
                eu, ev = self.emb_first[u], self.emb_first[v]
                score = eu.dot(ev)
                sig = self._sigmoid(score)
                grad = 1 - sig
                self.emb_first[u] += self.lr * grad * ev
                self.emb_first[v] += self.lr * grad * eu
                # negative samples
                for _ in range(self.neg):
                    w = np.random.choice(self.N, p=self.neg_dist)
                    if w == u: continue
                    ew = self.emb_first[w]
                    score_n = eu.dot(ew)
                    sig_n = self._sigmoid(-score_n)
                    grad_n = -sig_n
                    self.emb_first[u] += self.lr * grad_n * ew
                    self.emb_first[w] += self.lr * grad_n * eu

    def train_second_order(self):
        print("Training 2nd-order proximity...")
        for epoch in range(1, self.epochs + 1):
            print(f"Epoch {epoch}/{self.epochs}")
            random.shuffle(self.edges)
            for u, v in self.edges:
                eu, ev = self.emb_target[u], self.emb_context[v]
                score = eu.dot(ev)
                sig = self._sigmoid(score)
                grad = 1 - sig
                self.emb_target[u] += self.lr * grad * ev
                self.emb_context[v] += self.lr * grad * eu
                # negative samples
                for _ in range(self.neg):
                    w = np.random.choice(self.N, p=self.neg_dist)
                    if w == u: continue
                    ew = self.emb_context[w]
                    score_n = eu.dot(ew)
                    sig_n = self._sigmoid(-score_n)
                    grad_n = -sig_n
                    self.emb_target[u] += self.lr * grad_n * ew
                    self.emb_context[w] += self.lr * grad_n * eu

    def train(self):
        self.train_first_order()
        self.train_second_order()
        # final embedding = concat(first, target, context)
        return np.hstack([self.emb_first, self.emb_target, self.emb_context])

if __name__ == "__main__":
    # === parameters ===
    # file_name = 'lol222324-24'
    # file_name = 'email-Eu-core'
    # file_name = 'tree'
    # file_name = 'com-dblp_largest_deliso'
    file_name = 'com-youtube_largest_deliso'
    # file_name = 'com-amazon_largest_deliso'
    node_file = f"norm_dataset/{file_name}/{file_name}_nodes.txt"
    edge_file = f"norm_dataset/{file_name}/{file_name}_edges.txt"
    print(node_file, edge_file)
    G = load_graph_with_attributes(node_file, edge_file)
    
    cutoff = 2
    if cutoff is not None:
        G = extract_hop_subgraph(G, cutoff=cutoff)
    
    # train LINE embeddings
    model = LINE(G, dim=128, neg_samples=5, lr=0.025, epochs=10)
    X = model.train()
    # clustering
    true_coms = [G.nodes[n]['actual_community'] for n in G.nodes()]
    K = len(set(true_coms))
    kmeans = KMeans(n_clusters=K)
    preds = kmeans.fit_predict(X)
    # evaluation
    q = cal_q(G, preds)
    print(f"Modularity: {q:.4f}")