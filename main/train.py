"""
CFlow training script for single-cell to spatial transcriptomics (SC-to-ST) transfer.
Includes disentangled encoding, continuous flow matching, gene-aware decoding,
orthogonal regularization, and drug-level contrastive learning.
"""

import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import scanpy as sc
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import pearsonr, wasserstein_distance
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity


def set_seed(seed=42):
    """Set random seeds for full reproducibility across Python, NumPy and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


class DisentangledEncoder(nn.Module):
    """
    Encoder that maps gene expression into two independent latent spaces:
    - z_id   : identity / cell-background factors
    - z_pert : perturbation-specific factors
    """

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
        z_id = self.fc_id(h)
        z_pert = self.fc_pert(h)
        # Remove possible singleton dimensions introduced by batching
        if z_id.dim() == 3:
            z_id = z_id.squeeze(1)
        if z_pert.dim() == 3:
            z_pert = z_pert.squeeze(1)
        return z_id, z_pert


class VelocityNet(nn.Module):
    """
    Time-conditioned velocity field that predicts the instantaneous
    change of the perturbation latent code given drug identity and dose.
    """

    def __init__(self, latent_dim, num_drugs, dose_emb_dim=16):
        super().__init__()
        self.drug_emb_dim = 128
        self.dose_emb_dim = dose_emb_dim
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
        # Sum of two drug embeddings (supports combination treatments)
        d_vec = self.drug_emb(d_ids[:, 0]) + self.drug_emb(d_ids[:, 1])

        dose_vec = torch.zeros(z.size(0), self.dose_emb_dim * 2, device=z.device)
        if dose1 is not None and dose2 is not None:
            d1_emb = self.dose_emb1(dose1)
            d2_emb = self.dose_emb2(dose2)
            dose_vec = torch.cat([d1_emb, d2_emb], dim=-1)

        # Guarantee consistent rank of all tensors before concatenation
        if z.dim() == 3:
            z = z.squeeze(1)
        if t.dim() == 3:
            t = t.squeeze(1)
        if d_vec.dim() == 3:
            d_vec = d_vec.squeeze(1)
        if dose_vec.dim() == 3:
            dose_vec = dose_vec.squeeze(1)

        x = torch.cat([z, t, d_vec, dose_vec], dim=-1)
        return self.net(x)


class GeneDecoder(nn.Module):
    """
    Decoder that reconstructs gene expression from the concatenation
    of identity and perturbation latent codes. Uses a shared gene embedding.
    """

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
        h = torch.cat([z_id, z_pert], dim=-1)
        h = self.hidden(h)
        expr = torch.matmul(h, self.gene_emb.T) + self.bias
        return F.softplus(expr)


class CFlowModel(nn.Module):
    """
    Full Continuous-Flow model combining disentangled encoder,
    velocity network and gene decoder, together with the training losses.
    """

    def __init__(self, n_genes, num_drugs, temp=0.1):
        super().__init__()
        self.encoder = DisentangledEncoder(n_genes)
        self.flow = VelocityNet(128, num_drugs)
        self.decoder = GeneDecoder(128, n_genes)
        self.temperature = temp

    def orthogonal_loss(self, z_id, z_pert):
        """Encourage identity and perturbation latents to be orthogonal."""
        if z_id.dim() == 3:
            z_id = z_id.squeeze(1)
        if z_pert.dim() == 3:
            z_pert = z_pert.squeeze(1)
        dot = (z_id * z_pert).sum(dim=1)
        return torch.mean(dot ** 2)

    def contrastive_perturbation_loss(self, z, drug_ids):
        """
        InfoNCE-style contrastive loss that pulls perturbation latents
        of the same drug together and pushes different drugs apart.
        z         : (B, D)
        drug_ids  : (B,)
        """
        if z.dim() == 3:
            z = z.squeeze(1)

        z = F.normalize(z, dim=1)
        sim = torch.matmul(z, z.T) / self.temperature

        # Positive pairs share the same drug label
        drug_ids = drug_ids.view(-1, 1)
        mask = (drug_ids == drug_ids.T).float()

        # Exclude self-similarity
        logits_mask = torch.ones_like(mask) - torch.eye(mask.size(0), device=mask.device)
        mask = mask * logits_mask

        exp_sim = torch.exp(sim) * logits_mask
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        loss = -mean_log_prob_pos.mean()
        return loss

    def get_loss(self, x0, x1, d, dose1=None, dose2=None):
        """
        Compute the total training objective consisting of:
        1. Reconstruction loss
        2. Flow-matching loss
        3. Identity consistency loss
        4. Orthogonal regularization
        5. Drug-level contrastive loss
        """
        z_id0, z_p0 = self.encoder(x0)
        z_id1, z_p1 = self.encoder(x1)

        if z_id0.dim() == 3:
            z_id0 = z_id0.squeeze(1)
        if z_id1.dim() == 3:
            z_id1 = z_id1.squeeze(1)
        if z_p0.dim() == 3:
            z_p0 = z_p0.squeeze(1)
        if z_p1.dim() == 3:
            z_p1 = z_p1.squeeze(1)

        # Identity codes of control and treated cells should stay close
        loss_id = F.mse_loss(z_id0, z_id1)

        # Identity and perturbation codes should be orthogonal
        loss_ortho = self.orthogonal_loss(z_id1, z_p1)

        # Continuous flow matching on the perturbation trajectory
        t = torch.rand(x0.shape[0], 1).to(x0.device)
        z_t = (1 - t) * z_p0 + t * z_p1
        v_pred = self.flow(z_t, t, d, dose1, dose2)
        loss_flow = F.mse_loss(v_pred, z_p1 - z_p0)

        # Reconstruct the treated expression profile
        x1_rec = self.decoder(z_id1, z_p1)
        loss_recon = F.mse_loss(x1_rec, x1)

        # Contrastive regularization on the primary drug
        drug_label = d[:, 0]
        loss_pert = self.contrastive_perturbation_loss(z_p1, drug_label)

        total_loss = (
            loss_recon
            + loss_flow
            + 0.1 * loss_id
            + 0.1 * loss_ortho
            + 0.1 * loss_pert
        )
        return total_loss


def parse_log_dose(dose_val):
    """Convert a dose value to a numeric pair; treat missing/zero as (0, 0)."""
    if pd.isna(dose_val) or float(dose_val) == 0:
        return 0.0, 0.0
    return float(dose_val), 0.0


class ComboDataset(Dataset):
    """
    Dataset that pairs a randomly sampled control cell (x0) with a
    treated cell (x1) together with its drug identity and dose.
    """

    def __init__(self, adata, drug_to_id, ctrl):
        self.x1 = torch.FloatTensor(adata.X.A)
        self.ctrl = torch.FloatTensor(ctrl)
        self.drug_ids = []
        self.dose1 = []
        self.dose2 = []

        for _, row in adata.obs.iterrows():
            d1 = str(row["product_name"])
            d2 = "Vehicle"
            d1_id = drug_to_id.get(d1, 0)
            d2_id = drug_to_id.get(d2, 0)
            self.drug_ids.append([d1_id, d2_id])
            dose1_val, dose2_val = parse_log_dose(row["dose"])
            self.dose1.append([dose1_val])
            self.dose2.append([dose2_val])

        self.drug_ids = torch.LongTensor(self.drug_ids)
        self.dose1 = torch.FloatTensor(self.dose1)
        self.dose2 = torch.FloatTensor(self.dose2)

    def __len__(self):
        return len(self.x1)

    def __getitem__(self, idx):
        x0 = self.ctrl[np.random.randint(len(self.ctrl))].squeeze()
        x1 = self.x1[idx].squeeze()
        return x0, x1, self.drug_ids[idx], self.dose1[idx], self.dose2[idx]


def run_prediction(model, drug_name, dose_val, drug_to_id, ctrl, device, steps=100):
    """
    Generate predicted treated expression by integrating the learned
    velocity field starting from control cells under a given drug/dose.
    """
    model.eval()
    with torch.no_grad():
        x_ctrl = torch.FloatTensor(ctrl[:500]).to(device)
        d1 = drug_name
        d2 = "Vehicle"
        dose1, dose2 = parse_log_dose(dose_val)
        id1 = drug_to_id.get(str(d1), 0)
        id2 = drug_to_id.get(str(d2), 0)

        d_ids = torch.LongTensor([[id1, id2]]).repeat(500, 1).to(device)
        dose1_tensor = torch.FloatTensor([[dose1]]).repeat(500, 1).to(device)
        dose2_tensor = torch.FloatTensor([[dose2]]).repeat(500, 1).to(device)

        z_id, z_p = model.encoder(x_ctrl)
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.ones(500, 1).to(device) * (i * dt)
            z_p = z_p + model.flow(z_p, t, d_ids, dose1_tensor, dose2_tensor) * dt

        pred = model.decoder(z_id, z_p)
        return pred.cpu().numpy()


def evaluate(true_cells, pred_cells):
    """
    Compute a suite of metrics comparing true and predicted
    population-level expression profiles.
    Returns: Pearson, R2, RMSE, Cosine similarity, mean Wasserstein distance.
    """
    true_mu = true_cells.mean(0)
    pred_mu = pred_cells.mean(0)

    pearson, _ = pearsonr(true_mu, pred_mu)
    r2 = r2_score(true_mu, pred_mu)
    rmse = np.sqrt(mean_squared_error(true_mu, pred_mu))
    cosine = cosine_similarity(true_mu.reshape(1, -1), pred_mu.reshape(1, -1))[0][0]
    wd = np.mean([
        wasserstein_distance(true_cells[:, i], pred_cells[:, i])
        for i in range(true_cells.shape[1])
    ])
    return pearson, r2, rmse, cosine, wd


def main():
    """
    End-to-end training and evaluation pipeline.
    Aligns SC genes with an ST reference, trains the CFlow model,
    saves the best checkpoint (including gene names for later ST inference),
    and reports performance on test / OOD splits.
    """
    DATA = "./dataset/split_5_subsets_with_control/subset_1.h5ad"
    ST_DIR = "dataset/Human_Breast_Cancer"
    adata = sc.read_h5ad(DATA)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Align SC gene names with the spatial transcriptomics reference
    print("Loading ST data to extract common genes...")
    adata_st = sc.read_visium(ST_DIR, count_file="filtered_feature_bc_matrix.h5")

    adata.var_names = adata.var["feature_name"].astype(str)
    adata.var_names_make_unique()

    common_genes = sorted(list(set(adata.var_names) & set(adata_st.var_names)))
    print(f"Extracted {len(common_genes)} common genes.")

    adata = adata[:, common_genes].copy()

    print("Cells:", adata.n_obs)
    print("Genes:", adata.n_vars)

    train = adata[adata.obs["split"] == "train"]
    val = adata[adata.obs["split"] == "val"]
    test = adata[adata.obs["split"] == "test"]
    ood = test

    print("Train:", len(train), "Val:", len(val), "OOD:", len(ood), "Test:", len(test))

    all_drugs = adata.obs["product_name"].astype(str).tolist()
    all_drugs.append("Vehicle")
    drugs = sorted(list(set(all_drugs)))
    drug_to_id = {d: i + 1 for i, d in enumerate(drugs)}

    ctrl = adata[adata.obs["vehicle"] == True].X.A

    train_loader = DataLoader(ComboDataset(train, drug_to_id, ctrl), batch_size=256, shuffle=True)
    val_loader = DataLoader(ComboDataset(val, drug_to_id, ctrl), batch_size=256)
    ood_loader = DataLoader(ComboDataset(ood, drug_to_id, ctrl), batch_size=256)
    test_loader = DataLoader(ComboDataset(test, drug_to_id, ctrl), batch_size=256)

    model = CFlowModel(adata.n_vars, len(drug_to_id)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    best_val = 1e9

    epochs = 80
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x0, x1, d, dose1, dose2 in train_loader:
            x0 = x0.to(device)
            x1 = x1.to(device)
            d = d.to(device)
            dose1 = dose1.to(device)
            dose2 = dose2.to(device)

            loss = model.get_loss(x0, x1, d, dose1, dose2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x0, x1, d, dose1, dose2 in val_loader:
                x0 = x0.to(device)
                x1 = x1.to(device)
                d = d.to(device)
                dose1 = dose1.to(device)
                dose2 = dose2.to(device)
                val_loss += model.get_loss(x0, x1, d, dose1, dose2).item()
        val_loss /= len(val_loader)

        print(f"Epoch {epoch}  Train Loss {train_loss / len(train_loader):.4f}  Val Loss {val_loss:.4f}")

        # Save the best validation checkpoint together with metadata
        if val_loss < best_val:
            best_val = val_loss
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "n_genes": adata.n_vars,
                "num_drugs": len(drug_to_id),
                "drug_to_id": drug_to_id,
                "gene_names": common_genes,
            }
            torch.save(checkpoint, "best_model_sc_ST.pt")
            print(f"Saved best model checkpoint to {os.getcwd()}/best_model_sc_ST.pt")

    # Reload the best model for final evaluation
    checkpoint = torch.load("best_model_sc_ST.pt")
    model.load_state_dict(checkpoint["model_state_dict"])

    # Test-set evaluation
    test_metrics = []
    for cond in tqdm(test.obs["product_name"].unique(), desc="Test Evaluation"):
        subset = test[test.obs["product_name"] == cond]
        if len(subset) < 10:
            continue
        true_cells = subset.X.A
        d1 = subset.obs["product_name"].iloc[0]
        dose1 = subset.obs["dose"].iloc[0]
        pred = run_prediction(model, d1, dose1, drug_to_id, ctrl, device)
        test_metrics.append(evaluate(true_cells, pred))

    df_test = pd.DataFrame(test_metrics, columns=["Pearson", "R2", "RMSE", "Cosine", "Wasserstein"])
    print("\n=== Test Set ===")
    print(df_test.mean())

    # OOD evaluation (currently identical to the test split)
    ood_metrics = []
    for cond in tqdm(ood.obs["product_name"].unique(), desc="OOD Evaluation"):
        subset = ood[ood.obs["product_name"] == cond]
        if len(subset) < 10:
            continue
        true_cells = subset.X.A
        d1 = subset.obs["product_name"].iloc[0]
        dose1 = subset.obs["dose"].iloc[0]
        pred = run_prediction(model, d1, dose1, drug_to_id, ctrl, device)
        ood_metrics.append(evaluate(true_cells, pred))

    df_ood = pd.DataFrame(ood_metrics, columns=["Pearson", "R2", "RMSE", "Cosine", "Wasserstein"])
    print("\n=== OOD Set ===")
    print(df_ood.mean())


if __name__ == "__main__":
    main()