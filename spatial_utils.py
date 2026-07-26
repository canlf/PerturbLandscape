"""
Spatial processing utilities for ST inference.
Includes deep residual matrix-factorization denoising,
Optimal Transport mapping, adaptive deconvolution,
transport + GNN force refinement, and evaluation metrics.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import ot
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.neighbors import kneighbors_graph


# ------------------------------------------------------------------
# Spatial Denoising
# ------------------------------------------------------------------

def deep_spatial_denoise(
    adata_st,
    coords,
    device=None,
    n_epochs=50,
    lr=8e-5,
    latent_dim=6,
    alpha=0.1,
    lambda_spatial=0.1,
    lambda_l2=1e-4
):
    """
    Extremely weak residual deep matrix factorization for spatial denoising.
    Learns a small residual correction that is regularized by spatial smoothness.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Starting residual deep matrix-factorization spatial denoising...")

    is_sparse = hasattr(adata_st.X, "toarray")
    X_orig_np = adata_st.X.copy()
    if is_sparse:
        X_orig_np = X_orig_np.toarray()

    X_orig = torch.tensor(X_orig_np, dtype=torch.float32, device=device)
    n_spots, n_genes = X_orig.shape
    coords = np.asarray(coords)

    # Build mild spatial adjacency
    A = kneighbors_graph(coords, n_neighbors=5, mode='connectivity').toarray()
    A = torch.tensor(A, dtype=torch.float32, device=device)
    A = A / (A.sum(dim=1, keepdim=True) + 1e-8)

    class ResidualMF(nn.Module):
        def __init__(self):
            super().__init__()
            self.spot_emb = nn.Parameter(torch.randn(n_spots, latent_dim, device=device) * 0.0005)
            self.gene_emb = nn.Parameter(torch.randn(n_genes, latent_dim, device=device) * 0.0005)

        def forward(self):
            delta = torch.matmul(self.spot_emb, self.gene_emb.T)
            delta = torch.tanh(delta) * 0.8  # further compress residual range
            return X_orig + alpha * delta, delta

    model = ResidualMF().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=lambda_l2)

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        X_denoised, delta = model()

        recon_loss = F.mse_loss(X_denoised, X_orig)
        residual_loss = torch.mean(delta ** 2)

        # Extremely weak spatial smoothness regularizer
        spot_diff = torch.matmul(A, model.spot_emb) - model.spot_emb
        spatial_loss = torch.mean(spot_diff ** 2)

        loss = recon_loss + 0.05 * residual_loss + lambda_spatial * spatial_loss

        loss.backward()
        optimizer.step()

        if epoch == 0 or (epoch + 1) % 10 == 0:
            mean_diff = torch.mean(torch.abs(X_denoised - X_orig)).item()
            print(f"Epoch {epoch + 1:2d} | Loss={loss.item():.8f} | MeanDiff={mean_diff:.8f}")

    with torch.no_grad():
        X_final, _ = model()
        X_final = X_final.cpu().numpy()

    diff_mean = np.abs(X_final - X_orig_np).mean()
    print(f"\nSpatial denoising finished. Mean absolute change: {diff_mean:.8f}")

    X_final = np.clip(X_final, 0, None)

    if is_sparse:
        from scipy.sparse import csr_matrix
        adata_st.X = csr_matrix(X_final)
    else:
        adata_st.X = X_final

    return adata_st


# ------------------------------------------------------------------
# Optimal Transport & Deconvolution
# ------------------------------------------------------------------

def compute_ot_mapping(latent_sc, latent_st):
    """
    Compute Sinkhorn Optimal Transport plan between
    single-cell and spatial latent representations.
    """
    M = cdist(latent_sc, latent_st, metric="cosine")
    M = M / (M.max() + 1e-8)

    a = np.ones(latent_sc.shape[0]) / latent_sc.shape[0]
    b = np.ones(latent_st.shape[0]) / latent_st.shape[0]

    pi = ot.sinkhorn(a, b, M, 0.05)
    return pi


def compute_deconvolution_weights(pi):
    """
    Convert OT plan into normalized deconvolution weights (spots x cells).
    """
    W = pi.T
    W = W / (W.sum(axis=1, keepdims=True) + 1e-8)
    return W


def adaptive_sparsify_weights(W, min_k=10, max_k=100, scale=30):
    """
    Adaptive sparsification of deconvolution weights based on
    local entropy. High-entropy spots keep more cells.
    """
    W_sparse = np.zeros_like(W)
    Ks = []

    for i in range(W.shape[0]):
        w = W[i]

        entropy = -np.sum(w * np.log(w + 1e-8))
        entropy_norm = entropy / np.log(W.shape[1] + 1e-8)

        K = int(min_k + entropy_norm * scale)
        K = max(min_k, min(K, max_k))
        Ks.append(K)

        idx = np.argsort(w)[-K:]
        W_sparse[i, idx] = w[idx]
        W_sparse[i] /= (W_sparse[i].sum() + 1e-8)

    print(f"[Adaptive K] mean={np.mean(Ks):.2f}, min={np.min(Ks)}, max={np.max(Ks)}")
    return W_sparse


def reconstruct_st_expression(W, X_sc):
    """Reconstruct spatial expression by weighted combination of SC profiles."""
    return W @ X_sc


# ------------------------------------------------------------------
# Perturbation prediction & Transport
# ------------------------------------------------------------------

def predict_sc_perturbation(model, sc_tensor, drug_id, device):
    """
    Integrate the velocity field from control single-cell latents
    under a given drug to obtain the perturbation delta.
    """
    model.eval()

    with torch.no_grad():
        n = sc_tensor.shape[0]

        z_id, z_pert = model.encoder(sc_tensor)
        z_pert = z_pert + 0.01 * torch.randn_like(z_pert)  # small noise for robustness

        z0 = z_pert.clone()
        drug = torch.full((n,), drug_id, dtype=torch.long, device=device)

        steps = 25
        dt = 1.0 / steps

        for i in range(steps):
            t = torch.ones(n, 1, device=device) * (i * dt)
            v = model.flow(z_pert, t, drug)
            z_pert = z_pert + v * dt

        delta = (z_pert - z0).cpu().numpy()

    return delta


def transport_with_deconv_and_force(delta_sc, W, adj, lam=0.1):
    """
    Map single-cell perturbation deltas onto spatial spots via
    deconvolution weights, then refine with a lightweight GNN force.
    """
    from models import pseudo_gnn_force

    delta_base = W @ delta_sc
    force = pseudo_gnn_force(delta_base, adj)
    delta_refined = delta_base + lam * force
    return delta_refined


def generate_st_expression(model, z_id_st, z_pert_st, delta_st):
    """
    Apply a transported perturbation delta onto spatial latent codes
    and decode back to gene expression space.
    """
    delta = torch.tensor(delta_st, dtype=torch.float32).to(z_id_st.device)
    z_new = z_pert_st + delta
    pred = model.decoder(z_id_st, z_new)
    return pred


# ------------------------------------------------------------------
# Evaluation Metrics
# ------------------------------------------------------------------

def compute_morans_I(delta_st, coords, k=8):
    """
    Compute average Moran's I across genes for a spatial perturbation matrix.
    """
    A = kneighbors_graph(coords, n_neighbors=k, mode='connectivity').toarray()
    A = 0.5 * (A + A.T)

    row_sum = A.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1
    A = A / row_sum

    W = A.sum()
    N = delta_st.shape[0]

    var = np.var(delta_st, axis=0)
    valid = var > 1e-6

    I_all = []
    for g in np.where(valid)[0]:
        x = delta_st[:, g]
        x_centered = x - x.mean()

        num = x_centered @ A @ x_centered
        den = np.sum(x_centered ** 2)

        if den < 1e-8:
            continue

        I = (N / W) * (num / den)
        I_all.append(I)

    return np.mean(I_all) if len(I_all) > 0 else np.nan


def get_top_spatial_genes(delta_st, coords, k=50, n_neighbors=8):
    """
    Rank genes by their individual Moran's I and return the top-k.
    """
    A = kneighbors_graph(coords, n_neighbors=n_neighbors, mode='connectivity').toarray()
    A = 0.5 * (A + A.T)

    row_sum = A.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1
    A = A / row_sum

    W = A.sum()
    N = delta_st.shape[0]
    var = np.var(delta_st, axis=0)

    results = []
    for g in range(delta_st.shape[1]):
        if var[g] < 1e-6:
            continue

        x = delta_st[:, g]
        x_centered = x - x.mean()
        den = np.sum(x_centered ** 2)
        if den < 1e-8:
            continue

        num = x_centered @ A @ x_centered
        I = (N / (W + 1e-8)) * (num / den)
        results.append((g, I))

    results = sorted(results, key=lambda x: x[1], reverse=True)
    return results[:k]


def compute_de_gene_corr(delta_sc, delta_st, W):
    """
    Spearman correlation between projected SC deltas and refined ST deltas.
    Measures how well the transport preserves differential expression patterns.
    """
    delta_proj = W @ delta_sc
    corr, _ = spearmanr(delta_proj.flatten(), delta_st.flatten())
    return corr


def topk_gene_overlap(delta_sc, delta_st, W, k=100):
    """
    Overlap of the top-k most changed genes between projected SC and ST deltas.
    """
    delta_proj = W @ delta_sc

    sc_score = np.mean(delta_proj, axis=0)
    st_score = np.mean(delta_st, axis=0)

    top_sc = set(np.argsort(np.abs(sc_score))[-k:])
    top_st = set(np.argsort(np.abs(st_score))[-k:])

    overlap = len(top_sc & top_st) / k
    return overlap


def compute_morans_I_for_signature(x, coords, k=8):
    """
    Compute Moran's I for a one-dimensional spatial signature vector.
    """
    A = kneighbors_graph(coords, n_neighbors=k, mode='connectivity').toarray()
    A = 0.5 * (A + A.T)
    row_sum = A.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1
    A = A / row_sum

    W = A.sum()
    N = len(x)

    x_centered = x - x.mean()
    num = x_centered @ A @ x_centered
    den = np.sum(x_centered ** 2)

    if den < 1e-8:
        return 0.0

    return (N / W) * (num / den)