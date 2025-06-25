import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch
from torch_geometric.data import Data
import networkx as nx
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from torch_geometric.utils import from_networkx
from networkx.algorithms.community.quality import modularity
from torch_geometric.nn import GATConv, GCNConv
from sklearn.model_selection import KFold
import torch.nn as nn
import copy
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm


def cal_q(G, pred_labels):
    pred_dict = {}
    for idx, label in enumerate(pred_labels):
        pred_dict.setdefault(label.item(), []).append(idx)

    q = modularity(G, pred_dict.values())
    return q

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

# define GCN
class GCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=8):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels) 
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = torch.nn.Dropout(p=0.2)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        return x
    

def five_fold_predict_batch(data, model_class, epochs=100, batch_size=1024, num_neighbors=[10, 10]):
    device = data.x.device
    num_nodes = data.num_nodes
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    all_preds = torch.full((num_nodes,), -1, dtype=torch.long, device=device)

    for fold, (train_idx, test_idx) in enumerate(kf.split(np.arange(num_nodes))):
        print(f"Fold {fold + 1}/5")
        train_idx = torch.tensor(train_idx, dtype=torch.long, device=device)
        test_idx = torch.tensor(test_idx, dtype=torch.long, device=device)

        model = model_class(data.num_node_features, data.y.max().item()+1).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        criterion = torch.nn.CrossEntropyLoss()

        # NeighborLoader for training
        train_loader = NeighborLoader(
            data,
            input_nodes=train_idx,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            shuffle=True
        )

        # Training loop
        for epoch in tqdm(range(epochs), desc=f"Epoch (Fold {fold+1})"):
            model.train()
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                out = model(batch.x, batch.edge_index)
                loss = criterion(out[:batch.batch_size], batch.y[:batch.batch_size])
                loss.backward()
                optimizer.step()

        # NeighborLoader for inference
        test_loader = NeighborLoader(
            data,
            input_nodes=test_idx,
            num_neighbors=[-1], 
            batch_size=batch_size,
            shuffle=False
        )
        model.eval()
        preds = []
        node_ids = []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index)
                pred = out[:batch.batch_size].argmax(dim=1)
                preds.append(pred.cpu())
                node_ids.append(batch.input_id.cpu())
        preds = torch.cat(preds, dim=0)
        node_ids = torch.cat(node_ids, dim=0)
        all_preds[node_ids] = preds.to(device)

    return all_preds

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

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # file_name = "email-Eu-core"
    # file_name = 'lol222324-24'
    # file_name = 'com-dblp_largest_deliso'
    # file_name = 'com-amazon_largest_deliso'
    # file_name = 'tree'
    file_name = 'com-youtube_largest_deliso'
    node_path = f"norm_dataset/{file_name}/{file_name}_nodes.txt"
    edge_path = f"norm_dataset/{file_name}/{file_name}_edges.txt"

    G = load_graph_with_attributes(node_path, edge_path)
    
    cutoff = 2
    if cutoff is not None:
        G = extract_hop_subgraph(G, cutoff=cutoff)
    
    labels = [G.nodes[n]["actual_community"] for n in G.nodes()]

    data = from_networkx(G)
    
    for key in list(data.keys()):
        if key not in ['x', 'y', 'edge_index']:
            del data[key]
    
    data.y = torch.tensor(labels, dtype=torch.long)  
    node_feat = [G.degree(n) for n in G.nodes()]
    data.x = torch.tensor(node_feat, dtype=torch.float32).unsqueeze(1)  

    data.x = data.x.to(device)
    data.y = data.y.to(device)
    data.edge_index = data.edge_index.to(device)
  
    gcn_preds = five_fold_predict_batch(copy.deepcopy(data), GCN, batch_size=1024)
    
    gcn_q = cal_q(G, gcn_preds.cpu().numpy())

    print(f"GCN -> Modularity: {gcn_q:.4f}")