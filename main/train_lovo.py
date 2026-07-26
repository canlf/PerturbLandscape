"""
Training script for the CFlow model aligned with LoVo gene symbols.
Performs gene matching between the training single-cell dataset and LoVo.h5ad,
trains a disentangled continuous-flow model, and saves the best checkpoint.
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
        # Remove possible singleton dimensions
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

        # Guarantee consistent tensor ranks before concatenation
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
    of identity and perturbation latent codes using a shared gene embedding.
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
        self.x1 = torch.FloatTensor(adata.X.A if hasattr(adata.X, "A") else adata.X)
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

            dose1_val, dose2_val = parse_log_dose(row.get("dose", 0))
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


def main():
    """
    End-to-end training pipeline aligned with LoVo gene symbols.
    1. Load training SC data and LoVo reference.
    2. Perform intelligent gene matching and optionally limit to 5000 genes.
    3. Train the CFlow model.
    4. Save the best checkpoint (including gene names) for later inference.
    """
    DATA = "./dataset/split_5_subsets_with_control/subset_1.h5ad"
    LOVO_PATH = "./dataset/jeizhichangai/LoVo.h5ad"

    adata = sc.read_h5ad(DATA)
    adata_lovo = sc.read_h5ad(LOVO_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------- Gene matching with LoVo --------------------
    print("Performing gene matching with LoVo.h5ad ...")

    # Prefer feature_name column if available
    if "feature_name" in adata.var.columns:
        adata.var_names = adata.var["feature_name"].astype(str)
    adata.var_names_make_unique()

    adata_lovo.var_names = adata_lovo.var_names.astype(str)
    adata_lovo.var_names_make_unique()

    common_genes = sorted(list(set(adata.var_names) & set(adata_lovo.var_names)))
    print(f"Extracted {len(common_genes)} common genes.")

    if len(common_genes) == 0:
        raise ValueError("No common genes found. Please check the gene naming conventions.")

    # Optionally limit to the first 5000 genes to control memory / compute
    if len(common_genes) > 5000:
        common_genes = common_genes[:5000]
        print("Limited to the first 5000 common genes.")

    adata = adata[:, common_genes].copy()
    print(f"Final number of genes used for training: {adata.n_vars}")
    # -----------------------------------------------------------------

    train = adata[adata.obs["split"] == "train"]
    val = adata[adata.obs["split"] == "val"]
    test = adata[adata.obs["split"] == "test"]
    ood = test

    all_drugs = adata.obs["product_name"].astype(str).tolist()
    all_drugs.append("Vehicle")
    drugs = sorted(list(set(all_drugs)))
    drug_to_id = {d: i + 1 for i, d in enumerate(drugs)}

    ctrl = adata[adata.obs["vehicle"] == True].X
    if hasattr(ctrl, "A"):
        ctrl = ctrl.A

    train_loader = DataLoader(
        ComboDataset(train, drug_to_id, ctrl),
        batch_size=256,
        shuffle=True
    )
    val_loader = DataLoader(
        ComboDataset(val, drug_to_id, ctrl),
        batch_size=256
    )

    model = CFlowModel(adata.n_vars, len(drug_to_id)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    best_val = 1e9

    epochs = 60
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

        print(f"Epoch {epoch} | Train Loss {train_loss / len(train_loader):.4f} | Val Loss {val_loss:.4f}")

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
            torch.save(checkpoint, "best_model_LoVo_aligned.pt")
            print("→ Saved best model checkpoint.")

    print("Training finished successfully.")


if __name__ == "__main__":
    main()