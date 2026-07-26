"""
Inference and analysis script for the CFlow model on LoVo cells.
Loads a trained checkpoint, predicts drug effects on key functional modules
(proliferation, apoptosis, cell cycle, DNA damage), ranks drugs, and generates
publication-ready plots and tables.
"""

import torch
import scanpy as sc
import numpy as np
import pandas as pd
import warnings
from scipy.sparse import issparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import umap

warnings.filterwarnings("ignore")


# ----------------------------------------------------------------------
# 1. Model Definition
# ----------------------------------------------------------------------
class DisentangledEncoder(torch.nn.Module):
    """
    Encoder that maps gene expression into two independent latent spaces:
    identity (z_id) and perturbation (z_pert).
    """

    def __init__(self, n_genes, latent_dim=128):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_genes, 512),
            torch.nn.LayerNorm(512),
            torch.nn.SiLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(512, 256),
            torch.nn.SiLU()
        )
        self.fc_id = torch.nn.Linear(256, latent_dim)
        self.fc_pert = torch.nn.Linear(256, latent_dim)

    def forward(self, x):
        h = self.net(x)
        return self.fc_id(h), self.fc_pert(h)


class VelocityNet(torch.nn.Module):
    """
    Time-conditioned velocity field conditioned on drug identity and dose.
    Predicts the instantaneous change of the perturbation latent code.
    """

    def __init__(self, latent_dim, num_drugs, dose_emb_dim=16):
        super().__init__()
        self.drug_emb = torch.nn.Embedding(num_drugs + 1, 128)
        self.dose_emb1 = torch.nn.Linear(1, dose_emb_dim)
        self.dose_emb2 = torch.nn.Linear(1, dose_emb_dim)

        self.net = torch.nn.Sequential(
            torch.nn.Linear(latent_dim + 1 + 128 + 2 * dose_emb_dim, 256),
            torch.nn.SiLU(),
            torch.nn.Linear(256, 256),
            torch.nn.SiLU(),
            torch.nn.Linear(256, latent_dim)
        )

    def forward(self, z, t, d_ids, dose1=None, dose2=None):
        d_vec = self.drug_emb(d_ids[:, 0]) + self.drug_emb(d_ids[:, 1])

        if dose1 is not None and dose2 is not None:
            dose_vec = torch.cat([self.dose_emb1(dose1), self.dose_emb2(dose2)], dim=-1)
        else:
            dose_vec = torch.zeros(z.size(0), 32, device=z.device)

        if z.dim() == 3:
            z = z.squeeze(1)
        if t.dim() == 3:
            t = t.squeeze(1)
        if d_vec.dim() == 3:
            d_vec = d_vec.squeeze(1)

        x = torch.cat([z, t, d_vec, dose_vec], dim=-1)
        return self.net(x)


class GeneDecoder(torch.nn.Module):
    """
    Decoder that reconstructs gene expression from the concatenation
    of identity and perturbation latent codes.
    """

    def __init__(self, latent_dim, n_genes):
        super().__init__()
        self.hidden = torch.nn.Sequential(
            torch.nn.Linear(latent_dim * 2, 512),
            torch.nn.SiLU(),
            torch.nn.Linear(512, 256),
            torch.nn.SiLU()
        )
        self.gene_emb = torch.nn.Parameter(torch.randn(n_genes, 256))
        self.bias = torch.nn.Parameter(torch.zeros(n_genes))

    def forward(self, z_id, z_pert):
        h = torch.cat([z_id, z_pert], dim=-1)
        h = self.hidden(h)
        return torch.nn.functional.softplus(h @ self.gene_emb.T + self.bias)


class CFlowModel(torch.nn.Module):
    """
    Complete Continuous-Flow model consisting of
    disentangled encoder, velocity network and gene decoder.
    """

    def __init__(self, n_genes, num_drugs):
        super().__init__()
        self.encoder = DisentangledEncoder(n_genes)
        self.flow = VelocityNet(128, num_drugs)
        self.decoder = GeneDecoder(128, n_genes)


# ----------------------------------------------------------------------
# 2. Load Trained Model
# ----------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

checkpoint = torch.load("best_model_LoVo_aligned.pt", map_location=device)

train_adata = sc.read_h5ad("./dataset/split_5_subsets_with_control/subset_1.h5ad")
drugs = sorted(set(train_adata.obs["product_name"].astype(str).tolist() + ["Vehicle"]))
drug_to_id = {d: i + 1 for i, d in enumerate(drugs)}

n_genes = checkpoint["n_genes"]
gene_names = checkpoint.get("gene_names", None)

model = CFlowModel(n_genes, len(drug_to_id)).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print(f"Model loaded successfully | Genes: {n_genes} | Drugs: {len(drugs) - 1}")


# ----------------------------------------------------------------------
# 3. Load and Align LoVo Data
# ----------------------------------------------------------------------
adata = sc.read_h5ad("./dataset/jeizhichangai/LoVo.h5ad")
adata.var_names = adata.var_names.astype(str)
adata.var_names_make_unique()

# Keep only genes that were used during training
if gene_names is not None:
    common_genes = [g for g in gene_names if g in adata.var_names]
    adata = adata[:, common_genes].copy()

# Pad with zeros if the number of genes is still smaller than expected
if adata.shape[1] < n_genes:
    pad_width = n_genes - adata.shape[1]
    X_raw = adata.X.toarray() if issparse(adata.X) else np.array(adata.X)
    pad = np.zeros((adata.n_obs, pad_width), dtype=np.float32)
    X_padded = np.hstack([X_raw, pad])
    adata = sc.AnnData(X=X_padded, obs=adata.obs.copy(), var=pd.DataFrame(index=range(n_genes)))

X = adata.X.toarray() if issparse(adata.X) else np.array(adata.X)
X = np.log1p(X)
X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

print(f"Final input shape: {X.shape}")


# ----------------------------------------------------------------------
# 4. Functional Gene Sets
# ----------------------------------------------------------------------
def normalize_gene_name(g):
    """Normalize gene names for robust matching (upper-case, remove - and _)."""
    return g.upper().replace("-", "").replace("_", "")


gene_to_idx = {normalize_gene_name(str(g)): i for i, g in enumerate(adata.var_names)}


def build_mask(gene_list):
    """
    Build a boolean mask for a list of genes and return both the mask
    and the list of successfully matched gene names.
    """
    mask = np.zeros(n_genes, dtype=bool)
    hit = []
    for g in gene_list:
        ng = normalize_gene_name(g)
        if ng in gene_to_idx:
            mask[gene_to_idx[ng]] = True
            hit.append(g)
    return mask, hit


prolif_genes = [
    "MKI67", "PCNA", "TOP2A", "CCNB1", "AURKA", "PLK1", "MCM2", "CDK1", "CCNE1", "BIRC5",
    "MCM3", "MCM4", "MCM5", "MCM6", "MCM7", "CDC20", "CDC25A", "CDC6",
    "CCNA2", "CCNB2", "CCNE2", "UBE2C", "UBE2S", "TYMS", "RRM2"
]
apoptosis_genes = [
    "CASP3", "CASP7", "CASP8", "CASP9", "BAX", "BAK1", "BCL2", "BCL2L1", "BCL2L11",
    "TP53", "FAS", "FASLG", "BAD", "BID", "BBC3", "XIAP", "DIABLO",
    "TNFRSF10A", "TNFRSF10B"
]
cell_cycle_genes = [
    "CCND1", "CCND2", "CCND3", "CCNE1", "CCNE2", "CDK1", "CDK2", "CDK4", "CDK6",
    "E2F1", "E2F2", "E2F3", "CDKN1A", "CDKN1B", "CDKN2A",
    "CDC25A", "CDC25B", "CDC25C", "CCNA2", "CCNB1", "CCNB2", "SKP2", "WEE1"
]
dna_damage_genes = [
    "BRCA1", "BRCA2", "ATM", "ATR", "CHEK1", "CHEK2", "RAD51", "RAD51C", "RAD51D",
    "PARP1", "PARP2", "XRCC1", "XRCC2", "XRCC5", "XRCC6", "MRE11", "NBN", "RAD50",
    "TP53", "H2AFX", "POLQ", "FANCD2", "FANCI"
]

m_prolif, hit_prolif = build_mask(prolif_genes)
m_apop, hit_apop = build_mask(apoptosis_genes)
m_cycle, hit_cycle = build_mask(cell_cycle_genes)
m_dna, hit_dna = build_mask(dna_damage_genes)

print("\n=== Gene-set hit rates on LoVo data ===")
print(f"Proliferation : {len(hit_prolif)} / {len(prolif_genes)}")
print(f"Apoptosis     : {len(hit_apop)} / {len(apoptosis_genes)}")
print(f"Cell Cycle    : {len(hit_cycle)} / {len(cell_cycle_genes)}")
print(f"DNA Damage    : {len(hit_dna)} / {len(dna_damage_genes)}")
print("-----------------------------------")


# ----------------------------------------------------------------------
# 5. Prediction Function
# ----------------------------------------------------------------------
@torch.no_grad()
def predict(drug, dose=3.0, steps=50, dt=0.02):
    """
    Integrate the learned velocity field under a given drug and dose
    to obtain predicted treated expression and the corresponding latents.
    """
    batch_size = 256
    idx = np.random.choice(len(X_tensor), batch_size, replace=True)
    x = X_tensor[idx]

    d1 = drug_to_id.get(str(drug), 0)
    d2 = drug_to_id["Vehicle"]
    d_ids = torch.tensor([[d1, d2]] * batch_size, device=device)

    dose1 = torch.full((batch_size, 1), float(dose), device=device)
    dose2 = torch.zeros_like(dose1)

    z_id, z_p = model.encoder(x)

    for _ in range(steps):
        t = torch.ones(batch_size, 1, device=device)
        v = model.flow(z_p, t, d_ids, dose1, dose2)
        z_p = z_p + v * dt

    pred = model.decoder(z_id, z_p).cpu().numpy()
    return pred, idx, z_id.cpu().numpy(), z_p.cpu().numpy()


# ----------------------------------------------------------------------
# 6. Main Analysis Loop
# ----------------------------------------------------------------------
print("\nPredicting effects for 100 drugs ...")
results = []

test_drugs = [d for d in drugs if d != "Vehicle"][:100]

latent_list = []
drug_labels = []

for i, drug in enumerate(test_drugs, 1):
    try:
        pred, idx, z_id, z_p = predict(drug, dose=3.0)
        orig = X_tensor[idx].cpu().numpy()
        logfc = pred - orig

        results.append({
            "rank": len(results) + 1,
            "drug": drug,
            "proliferation": float(logfc[:, m_prolif].mean()) if m_prolif.sum() > 0 else np.nan,
            "apoptosis": float(logfc[:, m_apop].mean()) if m_apop.sum() > 0 else np.nan,
            "cell_cycle": float(logfc[:, m_cycle].mean()) if m_cycle.sum() > 0 else np.nan,
            "dna_damage": float(logfc[:, m_dna].mean()) if m_dna.sum() > 0 else np.nan,
            "global_inhibition": float(-logfc.mean())
        })

        # Keep a subset of latents for optional UMAP visualization
        if i <= 20:
            latent_list.append(z_p[:50])
            drug_labels.extend([drug] * 50)

        if i % 10 == 0 or i == len(test_drugs):
            print(f"Completed {i:3d}/100 drugs")
    except Exception as e:
        print(f"Failed for {drug}: {e}")

df = pd.DataFrame(results)

if not df.empty:
    # Rank by pure proliferation inhibition
    df_pro = df.sort_values("proliferation", ascending=True).reset_index(drop=True)
    df_pro["rank"] = df_pro.index + 1

    # Composite score (proliferation receives the highest weight)
    df_pro["composite_score"] = (
        df_pro["proliferation"] * 0.70 +
        df_pro["apoptosis"] * 0.05 +
        df_pro["cell_cycle"] * 0.20 +
        df_pro["dna_damage"] * 0.05
    )

    pd.set_option("display.max_rows", 150)
    pd.set_option("display.max_columns", 10)
    pd.set_option("display.width", 140)
    pd.set_option("display.float_format", "{:.4f}".format)

    print("\n" + "=" * 100)
    print("Top 100 Drug Prediction Results")
    print("=" * 100)
    print(df_pro[["rank", "drug", "proliferation", "composite_score", "global_inhibition",
                  "apoptosis", "cell_cycle", "dna_damage"]].round(4))

    # ----- Two Top-10 lists -----
    print("\n" + "=" * 20)
    print("[1] Top 10 strongest inhibitors ranked by Proliferation only")
    print(df_pro.head(10)[["rank", "drug", "proliferation", "composite_score", "global_inhibition"]].to_string(index=False))

    df_comp = df_pro.sort_values("composite_score", ascending=True).reset_index(drop=True)
    df_comp["comp_rank"] = df_comp.index + 1

    print("\n[2] Top 10 strongest inhibitors ranked by Composite Score")
    print(df_comp.head(10)[["comp_rank", "drug", "composite_score", "proliferation", "global_inhibition"]].to_string(index=False))
    print("=" * 20)


    # ----------------------------------------------------------------------
    # 7. Visualization
    # ----------------------------------------------------------------------
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    sns.set_style("whitegrid")

    top20 = df_pro.head(20).copy()
    top10_comp = df_comp.head(10).copy()

    # 1. Heatmap of functional modules for Top-20 drugs
    heatmap_data = top20[["proliferation", "cell_cycle", "apoptosis", "dna_damage"]].set_index(top20["drug"])

    plt.figure(figsize=(11, 8))
    sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="RdBu_r", center=0, linewidths=0.5)
    plt.title("Top 20 Drugs - Effect on Key Functional Modules", fontsize=15, pad=20)
    plt.ylabel("Drug")
    plt.xlabel("Functional Module")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    # 2. Bar plot of Top-10 drugs by composite score
    plt.figure(figsize=(12, 7))
    sns.barplot(data=top10_comp, x="drug", y="composite_score", palette="viridis")
    plt.title("Top 10 Drugs Ranked by Composite Inhibition Score", fontsize=15)
    plt.xlabel("Drug")
    plt.ylabel("Composite Score")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    # 3. Average effect across functional modules
    module_means = top20[["proliferation", "cell_cycle", "apoptosis", "dna_damage"]].mean()

    plt.figure(figsize=(9, 6))
    ax = sns.barplot(x=module_means.index, y=module_means.values, palette="RdBu_r")
    ax.set_title("Average Drug Effect Across Functional Modules", fontsize=15)
    ax.set_ylabel("Mean Effect Score")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.show()

    # 4. Distribution of global inhibition scores
    plt.figure(figsize=(10, 7))
    sns.histplot(df_pro["global_inhibition"], kde=True, color="#2ca02c", bins=20)
    plt.title("Distribution of Global Inhibition Effects Across Drugs", fontsize=15)
    plt.xlabel("Global Inhibition Score")
    plt.ylabel("Number of Drugs")
    plt.tight_layout()
    plt.show()

    # 5. Bubble plot overview of drug-module effects
    plot_df = top20.melt(
        id_vars=["drug"],
        value_vars=["proliferation", "cell_cycle", "apoptosis", "dna_damage"],
        var_name="module",
        value_name="logFC"
    )
    plot_df["abs_effect"] = np.abs(plot_df["logFC"])

    plt.figure(figsize=(13, 9))
    sns.scatterplot(
        data=plot_df,
        x="module",
        y="drug",
        size="abs_effect",
        hue="logFC",
        palette="RdBu_r",
        sizes=(80, 1100),
        alpha=0.8,
        edgecolor="black",
        linewidth=0.6
    )
    plt.title("Drug-Specific Effects on Functional Modules", fontsize=15)
    plt.xlabel("Functional Module")
    plt.ylabel("Drug")
    plt.legend(title="Effect Direction", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()

    print("\nAll plots generated successfully.")


    # ----------------------------------------------------------------------
    # 8. Publication-ready Tables
    # ----------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("Publication Tables")
    print("=" * 100)

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 50)
    pd.set_option("display.float_format", "{:.4f}".format)

    # Table 1: Top 15 by composite score
    df_top15 = df_comp.head(15).copy()
    df_top15 = df_top15[["comp_rank", "drug", "composite_score", "proliferation",
                         "cell_cycle", "apoptosis", "dna_damage", "global_inhibition"]].round(4)

    print("\n" + "=" * 80)
    print("Table 1: Top 15 Drugs by Composite Inhibition Score")
    print("=" * 80)
    print(df_top15.to_string(index=False))

    # Table 2: Correlation matrix among functional modules
    module_cols = ["proliferation", "cell_cycle", "apoptosis", "dna_damage"]
    corr_matrix = df[module_cols].corr().round(3)

    print("\n" + "=" * 80)
    print("Table 2: Correlation Matrix between Functional Modules")
    print("=" * 80)
    print(corr_matrix.to_string())

    # Table 3: Detailed effects of representative drugs
    rep_drugs = df_comp.head(8)["drug"].tolist() + ["Oxaliplatin", "Irinotecan", "5-FU", "Cisplatin"]
    rep_drugs = [d for d in rep_drugs if d in df["drug"].values]

    rep_df = df[df["drug"].isin(rep_drugs)][["drug", "proliferation", "cell_cycle",
                                             "apoptosis", "dna_damage", "global_inhibition"]].round(4)

    print("\n" + "=" * 80)
    print("Table 3: Detailed Effects of Representative Drugs")
    print("=" * 80)
    print(rep_df.to_string(index=False))

    # Table 4: Top 5 drugs for each functional module
    print("\n" + "=" * 80)
    print("Table 4: Top 5 Drugs for Each Functional Module")
    print("=" * 80)

    for module in ["proliferation", "cell_cycle", "apoptosis", "dna_damage"]:
        top5 = df.nsmallest(5, module)[["drug", module]].round(4)
        print(f"\n--- Top 5 for {module.replace('_', ' ').title()} ---")
        print(top5.to_string(index=False))

    # Table 5: Summary statistics
    summary = df[["proliferation", "cell_cycle", "apoptosis", "dna_damage", "global_inhibition"]].describe().round(4)

    print("\n" + "=" * 80)
    print("Table 5: Summary Statistics of Drug Effects")
    print("=" * 80)
    print(summary.to_string())

    print("\nAll tables printed successfully.")