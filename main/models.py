"""
Neural network components for the CFlow model.
Contains the disentangled encoder, velocity network, gene decoder,
main CFlowModel wrapper, and a lightweight spatial force operator.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DisentangledEncoder(nn.Module):
    """
    Encoder that separates identity (cell-type / background) and
    perturbation latent factors from gene expression.
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
        return self.fc_id(h), self.fc_pert(h)


class VelocityNet(nn.Module):
    """
    Continuous-time velocity field conditioned on drug identity and dose.
    Used for flow-matching based perturbation prediction.
    """

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
        # Handle both single-drug and combination-drug inputs
        if d_ids.dim() == 1:
            d_vec = self.drug_emb(d_ids)
        else:
            d_vec = self.drug_emb(d_ids[:, 0]) + self.drug_emb(d_ids[:, 1])

        dose_vec = torch.zeros(z.size(0), 32, device=z.device)

        if dose1 is not None and dose2 is not None:
            dose_vec = torch.cat([
                self.dose_emb1(dose1),
                self.dose_emb2(dose2)
            ], dim=-1)

        return self.net(torch.cat([z, t, d_vec, dose_vec], dim=-1))


class GeneDecoder(nn.Module):
    """
    Gene-aware decoder that reconstructs expression from
    identity and perturbation latent codes.
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
        h = self.hidden(torch.cat([z_id, z_pert], dim=-1))
        return F.softplus(torch.matmul(h, self.gene_emb.T) + self.bias)


class CFlowModel(nn.Module):
    """
    Complete Continuous Flow model that combines
    the disentangled encoder, velocity network and gene decoder.
    """

    def __init__(self, n_genes, num_drugs):
        super().__init__()
        self.encoder = DisentangledEncoder(n_genes)
        self.flow = VelocityNet(128, num_drugs)
        self.decoder = GeneDecoder(128, n_genes)


def pseudo_gnn_force(delta_base, adj):
    """
    Lightweight graph-based spatial force.
    Aggregates neighboring deltas and computes the residual force
    that encourages spatial smoothness.
    """
    agg = adj @ delta_base
    force = agg - delta_base
    return force