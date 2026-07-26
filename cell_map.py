"""
Spatial perturbation visualization pipeline for PerturbLandscape.
Loads a trained CFlow model, transports single-cell drug responses onto
spatial transcriptomics spots via Optimal Transport + GNN force, and
generates a suite of spatial and network visualizations.
"""

import os
import scanpy as sc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial.distance import cdist
from sklearn.neighbors import kneighbors_graph
from sklearn.decomposition import PCA
from community import best_partition
import ot


# ----------------------------------------------------------------------
# Model Components
# ----------------------------------------------------------------------

class DisentangledEncoder(nn.Module):
    """Encoder that separates identity and perturbation latent factors."""

    def __init__(self, n_genes, latent_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_genes, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.SiLU()
        )
        self.fc_id = nn.Linear(256, latent_dim)
        self.fc_pert = nn.Linear(256, latent_dim)

    def forward(self, x):
        h = self.net(x)
        return self.fc_id(h), self.fc_pert(h)


class VelocityNet(nn.Module):
    """Time-conditioned velocity field conditioned on drug identity and dose."""

    def __init__(self, latent_dim, num_drugs, dose_emb_dim=16):
        super().__init__()
        self.drug_emb_dim = 128
        self.drug_emb = nn.Embedding(num_drugs + 1, self.drug_emb_dim)
        self.dose_emb1 = nn.Linear(1, dose_emb_dim)
        self.dose_emb2 = nn.Linear(1, dose_emb_dim)

        self.net = nn.Sequential(
            nn.Linear(latent_dim + 1 + self.drug_emb_dim + 2 * dose_emb_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, latent_dim)
        )

    def forward(self, z, t, d_ids, dose1=None, dose2=None):
        if d_ids.dim() == 1:
            d_vec = self.drug_emb(d_ids)
        else:
            d_vec = self.drug_emb(d_ids[:, 0]) + self.drug_emb(d_ids[:, 1])

        dose_vec = torch.zeros(z.size(0), 32, device=z.device)
        if dose1 is not None and dose2 is not None:
            dose_vec = torch.cat([self.dose_emb1(dose1), self.dose_emb2(dose2)], dim=-1)

        return self.net(torch.cat([z, t, d_vec, dose_vec], dim=-1))


class GeneDecoder(nn.Module):
    """Gene-aware decoder that reconstructs expression from latent codes."""

    def __init__(self, latent_dim, n_genes):
        super().__init__()
        self.hidden = nn.Sequential(
            nn.Linear(latent_dim * 2, 512),
            nn.SiLU(),
            nn.Linear(512, 256),
            nn.SiLU()
        )
        self.gene_emb = nn.Parameter(torch.randn(n_genes, 256))
        self.bias = nn.Parameter(torch.zeros(n_genes))

    def forward(self, z_id, z_pert):
        h = self.hidden(torch.cat([z_id, z_pert], dim=-1))
        return F.softplus(torch.matmul(h, self.gene_emb.T) + self.bias)


class CFlowModel(nn.Module):
    """Complete Continuous-Flow model."""

    def __init__(self, n_genes, num_drugs):
        super().__init__()
        self.encoder = DisentangledEncoder(n_genes)
        self.flow = VelocityNet(128, num_drugs)
        self.decoder = GeneDecoder(128, n_genes)


def pseudo_gnn_force(delta_base, adj):
    """Lightweight graph force that encourages spatial smoothness of deltas."""
    agg = adj @ delta_base
    force = agg - delta_base
    return force


# ----------------------------------------------------------------------
# Optimal Transport & Deconvolution
# ----------------------------------------------------------------------

def compute_ot_mapping(latent_sc, latent_st):
    """Compute Sinkhorn Optimal Transport plan between SC and ST latents."""
    M = cdist(latent_sc, latent_st, metric="cosine")
    M = M / (M.max() + 1e-8)
    a = np.ones(latent_sc.shape[0]) / latent_sc.shape[0]
    b = np.ones(latent_st.shape[0]) / latent_st.shape[0]
    pi = ot.sinkhorn(a, b, M, 0.05)
    return pi


def compute_deconvolution_weights(pi):
    """Convert OT plan into normalized deconvolution weights (spots × cells)."""
    W = pi.T
    W = W / (W.sum(axis=1, keepdims=True) + 1e-8)
    return W


def adaptive_sparsify_weights(W, min_k=10, max_k=100, scale=30):
    """Entropy-adaptive sparsification of deconvolution weights."""
    W_sparse = np.zeros_like(W)
    for i in range(W.shape[0]):
        w = W[i]
        entropy_norm = -np.sum(w * np.log(w + 1e-8)) / np.log(W.shape[1] + 1e-8)
        K = int(min_k + entropy_norm * scale)
        K = max(min_k, min(K, max_k))
        idx = np.argsort(w)[-K:]
        W_sparse[i, idx] = w[idx]
        W_sparse[i] /= (W_sparse[i].sum() + 1e-8)
    return W_sparse


# ----------------------------------------------------------------------
# Prediction & Transport
# ----------------------------------------------------------------------

def predict_sc_perturbation(model, sc_tensor, drug_id, device):
    """
    Integrate the velocity field from control single-cell latents
    under a given drug to obtain the perturbation delta.
    """
    model.eval()
    with torch.no_grad():
        n = sc_tensor.shape[0]
        _, z_pert = model.encoder(sc_tensor)
        z_pert = z_pert + 0.01 * torch.randn_like(z_pert)
        z0 = z_pert.clone()

        drug = torch.full((n,), drug_id, dtype=torch.long, device=device)
        steps = 25
        dt = 1.0 / steps

        for i in range(steps):
            t = torch.ones(n, 1, device=device) * (i * dt)
            v = model.flow(z_pert, t, drug)
            z_pert = z_pert + v * dt

        return (z_pert - z0).cpu().numpy()


def transport_with_deconv_and_force(delta_sc, W, adj, lam=0.1):
    """
    Map single-cell deltas onto spatial spots via deconvolution weights
    and refine them with a lightweight GNN force.
    """
    delta_base = W @ delta_sc
    force = pseudo_gnn_force(delta_base, adj)
    delta_refined = delta_base + lam * force
    return delta_refined


def generate_st_expression(model, z_id_st, z_pert_st, delta_st):
    """Apply a transported delta onto spatial latents and decode expression."""
    delta = torch.tensor(delta_st, dtype=torch.float32).to(z_id_st.device)
    with torch.no_grad():
        z_new = z_pert_st + delta
        return model.decoder(z_id_st, z_new)


# ----------------------------------------------------------------------
# Visualization Functions
# ----------------------------------------------------------------------

def plot_spatial_vector_field(adata_st, delta_st, drug_name, top_ratio=0.20):
    """Spatial perturbation vector field (PCA-reduced directions + magnitude)."""
    coords = adata_st.obsm["spatial"]
    pca = PCA(n_components=2)
    vec2d = pca.fit_transform(delta_st)
    strength = np.linalg.norm(vec2d, axis=1)

    plt.figure(figsize=(12, 10))
    sc_plot = plt.scatter(
        coords[:, 0], coords[:, 1], c=strength, cmap="magma",
        s=14, alpha=0.7, edgecolors="none"
    )

    threshold = np.percentile(strength, 100 * (1 - top_ratio))
    idx = strength >= threshold
    U, V = vec2d[idx, 0], vec2d[idx, 1]
    norm = np.sqrt(U ** 2 + V ** 2) + 1e-8
    U, V = U / norm, V / norm

    plt.quiver(
        coords[idx, 0], coords[idx, 1], U, V, strength[idx],
        cmap="turbo", scale=45, width=0.0035, alpha=0.95
    )

    plt.gca().invert_yaxis()
    cbar = plt.colorbar(sc_plot, shrink=0.75, label="Perturbation Magnitude")
    cbar.ax.tick_params(labelsize=10)
    plt.title(f"Spatial Perturbation Vector Field\nDrug: {drug_name}",
              fontsize=17, weight="bold", pad=20)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def plot_enhanced_spatial_comm(adata_st, delta_st, drug_name):
    """Spatial communication network based on perturbation similarity."""
    coords = adata_st.obsm["spatial"]
    adj = kneighbors_graph(coords, n_neighbors=8, mode="connectivity").toarray()

    node_strength = np.linalg.norm(delta_st, axis=1)
    node_strength = (node_strength - node_strength.min()) / (node_strength.max() - node_strength.min() + 1e-8)

    edge_list = []
    for i in range(adj.shape[0]):
        for j in range(i + 1, adj.shape[1]):
            if adj[i, j] > 0:
                sim = np.dot(delta_st[i], delta_st[j]) / (
                    np.linalg.norm(delta_st[i]) * np.linalg.norm(delta_st[j]) + 1e-8)
                weight = (sim + 1) / 2
                edge_list.append((i, j, weight))

    weights_all = np.array([e[2] for e in edge_list])
    threshold = np.percentile(weights_all, 85)

    G = nx.Graph()
    for i in range(len(coords)):
        G.add_node(i, strength=node_strength[i])
    for i, j, w in edge_list:
        if w >= threshold:
            G.add_edge(i, j, weight=w)

    pos = {i: coords[i] for i in range(len(coords))}

    plt.figure(figsize=(12, 10))
    ax = plt.gca()

    nx.draw_networkx_nodes(
        G, pos, node_size=node_strength * 40 + 8,
        node_color=node_strength, cmap="plasma",
        alpha=0.95, linewidths=0, ax=ax
    )

    edges = G.edges(data=True)
    weights = [d["weight"] * 3 for _, _, d in edges]
    nx.draw_networkx_edges(
        G, pos, width=weights, edge_color=weights,
        edge_cmap=plt.cm.coolwarm, alpha=0.35, ax=ax
    )

    plt.title(f"Spatial Communication Network\nDrug: {drug_name}",
              fontsize=17, weight="bold", pad=20)

    sm = plt.cm.ScalarMappable(cmap="plasma", norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.75, label="Perturbation Strength")
    cbar.ax.tick_params(labelsize=10)

    plt.gca().invert_yaxis()
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def plot_community_strength(adata_st, delta_st, drug_name):
    """Spatial community structure detected by Louvain clustering."""
    coords = adata_st.obsm["spatial"]
    adj = kneighbors_graph(coords, n_neighbors=6, mode="connectivity").toarray()

    G = nx.Graph()
    for i in range(len(coords)):
        G.add_node(i)
    for i in range(adj.shape[0]):
        for j in range(i + 1, adj.shape[1]):
            if adj[i, j] > 0:
                sim = np.dot(delta_st[i], delta_st[j]) / (
                    np.linalg.norm(delta_st[i]) * np.linalg.norm(delta_st[j]) + 1e-8)
                if sim > 0.3:
                    G.add_edge(i, j)

    partition = best_partition(G)
    communities = np.array([partition[i] for i in range(len(coords))])

    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(
        coords[:, 0], coords[:, 1], c=communities,
        cmap="tab20", s=18, alpha=0.9, edgecolors="none"
    )

    plt.gca().invert_yaxis()
    cbar = plt.colorbar(scatter, shrink=0.75)
    cbar.set_label("Community Module")
    cbar.ax.tick_params(labelsize=10)

    plt.title(f"Spatial Community Structure\nDrug: {drug_name}",
              fontsize=17, weight="bold", pad=20)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def plot_force_directed_comm(adata_st, delta_st, drug_name):
    """Force-directed communication network colored by Louvain communities."""
    coords = adata_st.obsm["spatial"]
    adj = kneighbors_graph(coords, n_neighbors=6, mode="connectivity").toarray()

    node_strength = np.linalg.norm(delta_st, axis=1)
    node_strength_norm = (node_strength - node_strength.min()) / (
        node_strength.max() - node_strength.min() + 1e-8)

    G = nx.Graph()
    for i in range(len(coords)):
        G.add_node(i, strength=node_strength[i])

    for i in range(adj.shape[0]):
        for j in range(i + 1, adj.shape[1]):
            if adj[i, j] > 0:
                sim = np.dot(delta_st[i], delta_st[j]) / (
                    np.linalg.norm(delta_st[i]) * np.linalg.norm(delta_st[j]) + 1e-8)
                if sim > 0.3:
                    G.add_edge(i, j, weight=sim)

    pos = nx.spring_layout(G, k=0.15, iterations=300, seed=42)
    partition = best_partition(G)

    plt.figure(figsize=(14, 12))
    ax = plt.gca()

    communities = list(set(partition.values()))
    cmap = plt.get_cmap("tab20", len(communities))
    node_colors = [cmap(partition[i]) for i in G.nodes()]

    nx.draw_networkx_nodes(
        G, pos,
        node_size=node_strength_norm * 120 + 30,
        node_color=node_colors,
        alpha=0.95,
        edgecolors="black",
        linewidths=0.8
    )

    edges = G.edges(data=True)
    weights = [d["weight"] * 6 for _, _, d in edges]
    nx.draw_networkx_edges(G, pos, edge_color="gold", width=weights, alpha=0.75)

    plt.title(
        f"Force-Directed Community Cell Communication Network\n"
        f"Drug: {drug_name}\n(Colored by Louvain Communities)",
        fontsize=16, pad=20
    )
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def plot_control_vs_drug(adata_st, delta_st, drug_name):
    """Side-by-side comparison of control baseline versus drug perturbation."""
    coords = adata_st.obsm["spatial"]
    strength = np.linalg.norm(delta_st, axis=1)

    fig, axs = plt.subplots(1, 3, figsize=(20, 7))

    # Control (baseline)
    axs[0].scatter(coords[:, 0], coords[:, 1], c="lightgray", s=18, alpha=0.85, edgecolors="none")
    axs[0].set_title("Control (Vehicle)", fontsize=14)
    axs[0].invert_yaxis()

    # Drug
    sc1 = axs[1].scatter(coords[:, 0], coords[:, 1], c=strength, cmap="inferno",
                         s=20, alpha=0.9, edgecolors="none")
    axs[1].set_title(f"Drug: {drug_name}", fontsize=14)
    plt.colorbar(sc1, ax=axs[1], label="Perturbation Strength")
    axs[1].invert_yaxis()

    # Difference
    sc2 = axs[2].scatter(coords[:, 0], coords[:, 1], c=strength, cmap="RdBu_r",
                         s=20, alpha=0.9, edgecolors="none")
    axs[2].set_title("Drug vs Control Difference", fontsize=14)
    plt.colorbar(sc2, ax=axs[2], label="Δ Perturbation")
    axs[2].invert_yaxis()

    plt.suptitle("Control vs Drug Spatial Perturbation Comparison", fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_delta_umap(adata_st, delta_st, drug_name):
    """UMAP embedding of the perturbation landscape colored by magnitude."""
    adata_delta = sc.AnnData(X=delta_st)
    sc.pp.pca(adata_delta, n_comps=50)
    sc.pp.neighbors(adata_delta, n_neighbors=20)
    sc.tl.umap(adata_delta, min_dist=0.5, spread=1.2)

    strength = np.linalg.norm(delta_st, axis=1)
    vmin = np.percentile(strength, 5)
    vmax = np.percentile(strength, 95)

    plt.figure(figsize=(10, 8))
    plt.scatter(
        adata_delta.obsm["X_umap"][:, 0],
        adata_delta.obsm["X_umap"][:, 1],
        c=strength,
        cmap="turbo",
        s=10,
        alpha=0.85,
        edgecolors="none",
        rasterized=True,
        vmin=vmin,
        vmax=vmax
    )

    cb = plt.colorbar(shrink=0.8)
    cb.set_label("Perturbation Strength", fontsize=12)
    plt.title(f"Perturbation Landscape in UMAP Space\nDrug: {drug_name}", fontsize=15)
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------
# Main Pipeline
# ----------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Load and align SC + ST data
    adata_sc = sc.read_h5ad("./dataset/split_5_subsets_with_control/subset_1.h5ad")
    adata_sc.var_names = adata_sc.var["feature_name"].astype(str)
    adata_sc.var_names_make_unique()

    adata_st = sc.read_visium(
        "dataset/Human_Breast_Cancer",
        count_file="filtered_feature_bc_matrix.h5"
    )
    adata_st.var_names_make_unique()

    common = [g for g in adata_sc.var_names if g in adata_st.var_names]
    adata_sc = adata_sc[:, common].copy()
    adata_st = adata_st[:, common].copy()

    sc.pp.normalize_total(adata_st)
    sc.pp.log1p(adata_st)

    tensor_sc = torch.tensor(
        adata_sc[adata_sc.obs["vehicle"] == True].X.toarray(),
        dtype=torch.float32
    ).to(device)
    tensor_st = torch.tensor(adata_st.X.toarray(), dtype=torch.float32).to(device)

    coords = adata_st.obsm["spatial"]
    adj = kneighbors_graph(coords, n_neighbors=8, mode="connectivity").toarray()
    adj = adj / (adj.sum(axis=1, keepdims=True) + 1e-8)

    # Load trained model
    ckpt = torch.load("best_model_sc_ST.pt", map_location=device)
    model = CFlowModel(len(common), ckpt["num_drugs"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    drug_to_id = ckpt["drug_to_id"]

    # Encode to latent space
    with torch.no_grad():
        _, z_pert_sc = model.encoder(tensor_sc)
        z_id_st, z_pert_st = model.encoder(tensor_st)

    z_sc = F.normalize(z_pert_sc, dim=1).cpu().numpy()
    z_st = F.normalize(z_pert_st, dim=1).cpu().numpy()

    # Optimal Transport mapping and deconvolution
    print("Computing Optimal Transport mapping...")
    pi = compute_ot_mapping(z_sc, z_st)
    W = compute_deconvolution_weights(pi)
    W = adaptive_sparsify_weights(W)
    print("Deconvolution finished.")

    # Select drug for visualization
    drug = "(+)-JQ1"
    if drug not in drug_to_id:
        print(f"Drug '{drug}' not found in the trained model.")
        return

    print(f"\nSimulating drug: {drug}")

    delta_sc = predict_sc_perturbation(model, tensor_sc, drug_to_id[drug], device)
    delta_st = transport_with_deconv_and_force(delta_sc, W, adj, lam=0.1)

    # Generate all visualizations
    print("\nGenerating spatial visualizations...")
    plot_spatial_vector_field(adata_st, delta_st, drug)
    plot_enhanced_spatial_comm(adata_st, delta_st, drug)
    plot_community_strength(adata_st, delta_st, drug)
    plot_delta_umap(adata_st, delta_st, drug)
    plot_force_directed_comm(adata_st, delta_st, drug)
    plot_control_vs_drug(adata_st, delta_st, drug)

    print("\nAll visualizations completed.")


if __name__ == "__main__":
    main()