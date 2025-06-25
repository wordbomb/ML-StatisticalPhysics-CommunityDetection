# csea_pipeline.py  ----------------------------------------------------------
import numpy as np
import networkx as nx
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from networkx.algorithms.community.quality import modularity
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans
import scipy.sparse as sp
from torch.cuda.amp import autocast, GradScaler
torch.backends.cudnn.benchmark = True

# ---------------------------------------------------------------------------#
# 0. read network                                #
# ---------------------------------------------------------------------------#

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

# ---------------------------------------------------------------------------#
# 1. K-truss                                                         #
# ---------------------------------------------------------------------------#
def k_truss_subgraph(G: nx.Graph, k: int) -> nx.Graph:
    print(f" k-truss is extracting ... k={k}")
    H = G.copy()
    tri_cnt = {e: 0 for e in H.edges()}
    for u in H:
        nbrs = set(H[u])
        for v in nbrs:
            if u < v:
                tri_cnt[(u, v)] = len(nbrs & set(H[v]))
    changed = True
    while changed:
        changed = False
        for (u, v), t in list(tri_cnt.items()):
            if t < k - 2 and H.has_edge(u, v):
                H.remove_edge(u, v); changed = True
                for w in set(H[u]) & set(H[v]):
                    p1, p2 = tuple(sorted((u, w))), tuple(sorted((v, w)))
                    tri_cnt[p1] = max(0, tri_cnt.get(p1, 1) - 1)
                    tri_cnt[p2] = max(0, tri_cnt.get(p2, 1) - 1)
    return H


# ---------------------------------------------------------------------------#
# 2. VAE                                                              #
# ---------------------------------------------------------------------------#

class DynamicGraphVAE(nn.Module):
    def __init__(self, layer_dims):
        """
        layer_dims: list of ints, e.g. [4096, 2048, 1024, 512]
                     4096 is input dimension, 512 is latent dimension。
        """
        super().__init__()
        assert len(layer_dims) >= 2, "layer_dims must have at least two elements (input and latent dimensions)."

        latent_dim = layer_dims[-1]
        enc_dims   = layer_dims[:-1]

        # --- Encoder  ---
        enc_blocks = []
        for in_d, out_d in zip(enc_dims[:-1], enc_dims[1:]):
            enc_blocks += [
                nn.Linear(in_d, out_d),
                nn.ReLU(inplace=True),
            ]
        self.encoder = nn.Sequential(*enc_blocks)

        # mu / logv 
        last_hidden = enc_dims[-1]
        self.fc_mu   = nn.Linear(last_hidden, latent_dim)
        self.fc_logv = nn.Linear(last_hidden, latent_dim)

        # --- Decoder ---
        dec_dims = [latent_dim] + list(reversed(enc_dims))
        dec_blocks = []
        for in_d, out_d in zip(dec_dims[:-1], dec_dims[1:]):
            dec_blocks += [
                nn.Linear(in_d, out_d),
                nn.ReLU(inplace=True),
            ]
        # Sigmoid
        dec_blocks[-1] = nn.Sigmoid()
        self.decoder = nn.Sequential(*dec_blocks)

    def encode(self, x):
        h = self.encoder(x)
        mu   = self.fc_mu(h)
        logv = self.fc_logv(h)
        return mu, logv

    def reparam(self, mu, logv):
        std = torch.exp(0.5 * logv)
        return mu + std * torch.randn_like(std)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logv = self.encode(x)
        z        = self.reparam(mu, logv)
        recon    = self.decode(z)
        return recon, mu, logv, z


class SVaeDataset(torch.utils.data.Dataset):
    def __init__(self, S_sp):
        self.S_sp = S_sp
        self.n     = S_sp.shape[0]

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        row = self.S_sp.getrow(idx).toarray().ravel().astype(np.float32)
        norm = np.linalg.norm(row)
        if norm>0: row /= norm
        return torch.from_numpy(row)  # 1×n dense
    
def train_vae_minibatch_stream(S_sp, layer_dims, epochs=100, batch=4096, lr=1e-4):
    print("miniVAE is training...")
    device = 'cuda'
    dataset = SVaeDataset(S_sp)
    loader  = torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=True)
    
    layer_dims = [S_sp.shape[1]] + layer_dims
    model = DynamicGraphVAE(layer_dims).to(device)

    # model = GraphVAE(S_sp.shape[1], hidden, latent).to(device).float()
    opt   = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in tqdm(range(epochs), desc="miniVAE"):
        for xb in loader:           # xb.shape = (batch_size, n)
            xb = xb.to(device)
            opt.zero_grad()

            recon, mu, logv, _ = model(xb)
            # 1) construct reconstruction loss
            recon_loss = F.binary_cross_entropy(recon, xb, reduction='sum')
            # 2) KL 
            kl_loss    = -0.5 * torch.sum(1 + logv - mu.pow(2) - logv.exp())
            # 3) merge
            loss = recon_loss + kl_loss

            loss.backward()
            opt.step()

    # collect z
    zs = []
    with torch.no_grad():
        for xb in torch.utils.data.DataLoader(dataset, batch_size=batch):
            xb = xb.to(device)
            _,_,_, z = model(xb)
            zs.append(z.cpu())
    return torch.cat(zs, dim=0).numpy()

# ---------------------------------------------------------------------------#
# 3. K-means                                                    #
# ---------------------------------------------------------------------------#
def kmeans_cluster(z, k):
    labels = KMeans(n_clusters=k).fit_predict(z)
    return labels


def propagate_labels(G, core_nodes, core_labels):
    lab = {i: -1 for i in G}; lab.update(dict(zip(core_nodes, core_labels)))
    unlabeled = {n for n in G if lab[n] == -1}
    while unlabeled:
        progressed = False
        for n in list(unlabeled):
            neigh = [lab[v] for v in G[n] if lab[v] != -1]
            if neigh:
                lab[n] = int(np.bincount(neigh).argmax())
                unlabeled.remove(n); progressed = True
        if not progressed:  
            for n in unlabeled:
                lab[n] = max(lab.values()) + 1
            break
    return np.array([lab[i] for i in sorted(G.nodes())])

# ------------------------------------------------------------------#
#  construct similarity matrix             #
# ------------------------------------------------------------------#

def build_similarity_sparse_global(core: nx.Graph):
    # 1) construct index
    nodes = list(core.nodes())
    idx   = {v:i for i,v in enumerate(nodes)}
    n     = len(nodes)

    # 2) calcute support = |N(u) ∩ N(v)| 
    edges = list(core.edges())
    support = []
    for u, v in edges:
        nbrs_u = set(core[u])
        nbrs_v = set(core[v])
        support.append(len(nbrs_u & nbrs_v))
    sup_max = max(support) if support else 0

    # 3) calcilate similarity
    data = [ (s + 1) / (sup_max + 1) for s in support ]

    # 4) construct COO sparse matrix
    rows = [ idx[u] for u, v in edges ]
    cols = [ idx[v] for u, v in edges ]
    X_sp = sp.coo_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float32)

    # 5) to CSR
    X_sp = (X_sp + X_sp.T).tocsr()

    return X_sp


# ---------------------------------------------------------------------------#
# 4. CSEA main function                                                             #
# ---------------------------------------------------------------------------#
def run_csea(G, layer_dims,  k_truss=6, n_clusters=None, epochs=200, batch=4096):
    print(f"CSEA is running... k_truss={k_truss}   epochs={epochs}")
    core = k_truss_subgraph(G, k_truss)
    core_nodes = list(core.nodes())
    S_sp = build_similarity_sparse_global(core)

    z = train_vae_minibatch_stream(S_sp, layer_dims, epochs, batch=batch)
    if n_clusters is None:
        n_clusters = int(np.sqrt(len(core_nodes)))
        print(f" number of clusters: {n_clusters}")
    lab_core = kmeans_cluster(z, n_clusters)
    return propagate_labels(G, core_nodes, lab_core)


# ---------------------------------------------------------------------------#
# 5. evaluation metric                                                                #
# ---------------------------------------------------------------------------#
def cal_q(G, pred_labels):
    # modularity
    d = {}
    for idx, lab in enumerate(pred_labels):
        lab = int(lab)
        d.setdefault(lab, []).append(idx)
    q = modularity(G, [set(v) for v in d.values()])
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



# ---------------------------------------------------------------------------#
# 6. Example entry                                                                #1
# ---------------------------------------------------------------------------#
if __name__ == "__main__":
    
    # lol
    file_name = 'lol222324-24'
    node_file = f"norm_dataset/{file_name}/{file_name}_nodes.txt"
    edge_file = f"norm_dataset/{file_name}/{file_name}_edges.txt"
    G = load_graph_with_attributes(node_file, edge_file)
    cutoff = None
    if cutoff is not None:
        G = extract_hop_subgraph(G, cutoff=cutoff)
    layer_dims = [32, 16]
    k_truss=5
    
    # email
    # file_name = 'email-Eu-core'
    # node_file = f"norm_dataset/{file_name}/{file_name}_nodes.txt"
    # edge_file = f"norm_dataset/{file_name}/{file_name}_edges.txt"
    # G = load_graph_with_attributes(node_file, edge_file)
    # cutoff = None
    # if cutoff is not None:
    #     G = extract_hop_subgraph(G, cutoff=cutoff)
    # layer_dims = [64,48,32,28,12]
    # k_truss=5
    
    # tree
    # file_name = 'tree'
    # node_file = f"norm_dataset/{file_name}/{file_name}_nodes.txt"
    # edge_file = f"norm_dataset/{file_name}/{file_name}_edges.txt"
    # G = load_graph_with_attributes(node_file, edge_file)
    # cutoff = None
    # if cutoff is not None:
    #     G = extract_hop_subgraph(G, cutoff=cutoff)
    # layer_dims = [32, 16]
    # k_truss=3
    
    # dblp
    # file_name = 'com-dblp_largest_deliso'
    # node_file = f"norm_dataset/{file_name}/{file_name}_nodes.txt"
    # edge_file = f"norm_dataset/{file_name}/{file_name}_edges.txt"
    # G = load_graph_with_attributes(node_file, edge_file)
    # cutoff = 2
    # if cutoff is not None:
    #     G = extract_hop_subgraph(G, cutoff=cutoff)
    # layer_dims = [4096, 2048, 1024, 512, 256]
    # k_truss=3
    
    
    # youtube
    # file_name = 'com-youtube_largest_deliso'
    # node_file = f"norm_dataset/{file_name}/{file_name}_nodes.txt"
    # edge_file = f"norm_dataset/{file_name}/{file_name}_edges.txt"
    # G = load_graph_with_attributes(node_file, edge_file)
    # cutoff = 2
    # if cutoff is not None:
    #     G = extract_hop_subgraph(G, cutoff=cutoff)
    # layer_dims = [1024,512,256,128,96]
    # k_truss=3
    
    
    # amazon
    # file_name = 'com-amazon_largest_deliso'
    # node_file = f"norm_dataset/{file_name}/{file_name}_nodes.txt"
    # edge_file = f"norm_dataset/{file_name}/{file_name}_edges.txt"
    # G = load_graph_with_attributes(node_file, edge_file)
    # cutoff = 2
    # if cutoff is not None:
    #     G = extract_hop_subgraph(G, cutoff=cutoff)
    # layer_dims = [1024,512,256,128,96]
    # k_truss=3
    
    
    true_q = len(set(nx.get_node_attributes(G, 'actual_community').values()))
    pred = run_csea(G, layer_dims, n_clusters=true_q, k_truss=k_truss, epochs=20,batch=4096)

    q = cal_q(G, pred)
    print(f"CSEA result:  Q={q:.4f}")