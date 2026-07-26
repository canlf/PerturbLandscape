"""
Main script for spatial transcriptomics drug-perturbation inference.
Loads a trained CFlow model, performs OT-based deconvolution,
transports single-cell perturbations onto spatial spots with GNN force,
evaluates spatial metrics, and generates visualizations.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import scanpy as sc
import matplotlib.pyplot as plt
from sklearn.neighbors import kneighbors_graph
from anndata import AnnData, concat

from models import CFlowModel
from spatial_utils import (
    deep_spatial_denoise,
    compute_ot_mapping,
    compute_deconvolution_weights,
    adaptive_sparsify_weights,
    predict_sc_perturbation,
    transport_with_deconv_and_force,
    generate_st_expression,
    compute_morans_I,
    compute_de_gene_corr,
    topk_gene_overlap,
    compute_morans_I_for_signature,
)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ------------------------------------------------------------------
    # Load and align data
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Spatial denoising
    # ------------------------------------------------------------------
    adata_st = deep_spatial_denoise(
        adata_st,
        coords=adata_st.obsm["spatial"],
        device=device
    )

    # ------------------------------------------------------------------
    # Prepare tensors and spatial graph
    # ------------------------------------------------------------------
    sc_ctrl = adata_sc[adata_sc.obs["vehicle"] == True]
    tensor_sc = torch.tensor(sc_ctrl.X.toarray(), dtype=torch.float32).to(device)
    tensor_st = torch.tensor(adata_st.X.toarray(), dtype=torch.float32).to(device)

    coords = adata_st.obsm["spatial"]
    adj = kneighbors_graph(coords, n_neighbors=8, mode='connectivity').toarray()
    adj = adj / (adj.sum(axis=1, keepdims=True) + 1e-8)

    # ------------------------------------------------------------------
    # Load trained model
    # ------------------------------------------------------------------
    ckpt = torch.load("best_model_sc_ST.pt", map_location=device)
    model = CFlowModel(len(common), ckpt["num_drugs"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    drug_to_id = ckpt["drug_to_id"]

    # ------------------------------------------------------------------
    # Encode to latent space
    # ------------------------------------------------------------------
    with torch.no_grad():
        z_id_sc, z_pert_sc = model.encoder(tensor_sc)
        z_id_st, z_pert_st = model.encoder(tensor_st)

    z_sc = F.normalize(z_pert_sc, dim=1).cpu().numpy()
    z_st = F.normalize(z_pert_st, dim=1).cpu().numpy()

    # ------------------------------------------------------------------
    # Optimal Transport & deconvolution
    # ------------------------------------------------------------------
    print("Computing Optimal Transport mapping...")
    pi = compute_ot_mapping(z_sc, z_st)
    W = compute_deconvolution_weights(pi)
    W = adaptive_sparsify_weights(W)
    print("Deconvolution finished.")

    # ------------------------------------------------------------------
    # Drug simulation loop
    # ------------------------------------------------------------------
    drugs = [d for d in drug_to_id if d.lower() != "vehicle"][:4]

    all_moran = []
    all_decorr = []
    all_topk = []

    for drug in drugs:
        print("\n" + "=" * 40)
        print("Simulating", drug)

        delta_sc = predict_sc_perturbation(
            model, tensor_sc, drug_to_id[drug], device
        )

        delta_st = transport_with_deconv_and_force(
            delta_sc, W, adj, lam=0.1
        )

        # Evaluation metrics
        try:
            moran = compute_morans_I(delta_st, coords)
        except Exception:
            moran = np.nan

        try:
            de_corr = compute_de_gene_corr(delta_sc, delta_st, W)
        except Exception:
            de_corr = np.nan

        try:
            topk = topk_gene_overlap(delta_sc, delta_st, W, k=20)
        except Exception:
            topk = np.nan

        print(f"[{drug}] Moran's I     : {moran:.4f}")
        print(f"[{drug}] DE Correlation: {de_corr:.4f}")
        print(f"[{drug}] Top-K Overlap : {topk:.4f}")

        all_moran.append(moran)
        all_decorr.append(de_corr)
        all_topk.append(topk)

        # Decode perturbed spatial expression
        pred = generate_st_expression(model, z_id_st, z_pert_st, delta_st)
        adata_st.layers[f"drug_{drug}"] = pred.detach().cpu().numpy()

    # ------------------------------------------------------------------
    # Summary metrics
    # ------------------------------------------------------------------
    print("\n========= Overall Metrics =========")
    print(f"Mean Moran's I     : {np.nanmean(all_moran):.4f}")
    print(f"Mean DE Correlation: {np.nanmean(all_decorr):.4f}")
    print(f"Mean Top-K Overlap : {np.nanmean(all_topk):.4f}")

    # ------------------------------------------------------------------
    # Visualization – Top-N gene signature
    # ------------------------------------------------------------------
    print("\n=== Starting Visualization – Top-N Genes Summed Signature ===")

    top_n = 100  # freely adjustable

    control_X = adata_st.X.copy()
    if hasattr(control_X, "toarray"):
        control_X = control_X.toarray()
    control_X = np.asarray(control_X, dtype=np.float32)

    for drug in drugs:
        print(f"\n{'=' * 80}")
        print(f"Drug: {drug}  –  Top {top_n} Perturbed Genes Summed Signature")
        print(f"{'=' * 80}")

        pred_layer = adata_st.layers[f"drug_{drug}"]
        if hasattr(pred_layer, "toarray"):
            pred_np = pred_layer.toarray().astype(np.float32)
        else:
            pred_np = np.asarray(pred_layer, dtype=np.float32)

        pred_np = np.log1p(pred_np)

        gene_change = np.mean(np.abs(pred_np - control_X), axis=0).flatten()
        top_n_idx = np.argsort(gene_change)[-top_n:][::-1]

        print(f"Top {top_n} Genes:")
        for rank, i in enumerate(top_n_idx, 1):
            print(f"  #{rank:2d}  {common[i]:<15}  |Δ| = {gene_change[i]:.4f}")

        top_n_control = control_X[:, top_n_idx].sum(axis=1)
        top_n_perturbed = pred_np[:, top_n_idx].sum(axis=1)

        print(f"Summed signature range → Control  : [{top_n_control.min():.2f}, {top_n_control.max():.2f}]")
        print(f"Summed signature range → Perturbed: [{top_n_perturbed.min():.2f}, {top_n_perturbed.max():.2f}]")

        print("\nComputing Moran's I for Top-N Signature...")
        moran_control = compute_morans_I_for_signature(top_n_control, coords)
        moran_perturbed = compute_morans_I_for_signature(top_n_perturbed, coords)

        print(f"[Control]  Top{top_n} Signature Moran's I = {moran_control:.4f}")
        print(f"[{drug}]   Top{top_n} Signature Moran's I = {moran_perturbed:.4f}")
        print(f"Δ Moran's I = {moran_perturbed - moran_control:.4f}")

        # Spatial plots
        fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))

        adata_ctrl = AnnData(
            X=top_n_control.reshape(-1, 1),
            obs=adata_st.obs.copy(),
            obsm=adata_st.obsm.copy(),
            uns=adata_st.uns.copy()
        )
        adata_ctrl.var_names = [f"Top{top_n}_Signature"]
        adata_ctrl.obsm["spatial"] = adata_st.obsm["spatial"]

        sc.pl.spatial(
            adata_ctrl,
            color=[f"Top{top_n}_Signature"],
            title=" ",
            ax=axes[0],
            show=False,
            vmax="p99",
            vmin="p1",
            cmap="viridis",
            spot_size=180,
            na_color="white",
            colorbar_loc=None,
            alpha_img=0.1,
        )

        adata_pert = AnnData(
            X=top_n_perturbed.reshape(-1, 1),
            obs=adata_st.obs.copy(),
            obsm=adata_st.obsm.copy(),
            uns=adata_st.uns.copy()
        )
        adata_pert.var_names = [f"Top{top_n}_Signature"]
        adata_pert.obsm["spatial"] = adata_st.obsm["spatial"]

        sc.pl.spatial(
            adata_pert,
            color=[f"Top{top_n}_Signature"],
            title=" ",
            ax=axes[1],
            show=False,
            vmax="p99",
            vmin="p1",
            cmap="viridis",
            spot_size=180,
            na_color="#f5f5f5",
            alpha_img=0.1,
        )

        gene_list = ", ".join([common[i] for i in top_n_idx[:8]]) + ("..." if top_n > 8 else "")
        plt.suptitle(
            f"Drug: {drug}  |  Top {top_n} Most Perturbed Genes Summed Signature\n"
            f"Genes: {gene_list}",
            fontsize=13, y=1.02
        )
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Heatmap & Dotplot visualization
    # ------------------------------------------------------------------
    print("\n=== Starting Gene Heatmap Visualization ===")

    combined_list = []

    # Control
    adata_ctrl = adata_st.copy()
    adata_ctrl.X = control_X.copy() if hasattr(control_X, "copy") else control_X
    adata_ctrl.obs["condition"] = "Control"
    adata_ctrl.obs_names = [f"Control_{i}" for i in range(adata_ctrl.n_obs)]
    combined_list.append(adata_ctrl)

    for drug in drugs:
        pred_layer = adata_st.layers[f"drug_{drug}"]
        pred_np = pred_layer.toarray() if hasattr(pred_layer, "toarray") else pred_layer
        pred_np = np.log1p(pred_np.astype(np.float32))

        adata_drug = adata_st.copy()
        adata_drug.X = pred_np
        adata_drug.obs["condition"] = drug
        adata_drug.obs_names = [f"{drug}_{i}" for i in range(adata_drug.n_obs)]
        combined_list.append(adata_drug)

    adata_combined = concat(combined_list, join="outer", label=None, index_unique=None)
    adata_combined.obs["condition"] = adata_combined.obs["condition"].astype("category")

    print("Conditions in combined data:", adata_combined.obs["condition"].cat.categories.tolist())

    # Select highly perturbed genes
    all_top_genes = set()
    for drug in drugs:
        pred_np = np.log1p(
            adata_st.layers[f"drug_{drug}"].toarray()
            if hasattr(adata_st.layers[f"drug_{drug}"], "toarray")
            else adata_st.layers[f"drug_{drug}"]
        )
        gene_change = np.mean(np.abs(pred_np - control_X), axis=0)
        top_idx = np.argsort(gene_change)[-60:]
        top_gene_names = [common[i] for i in top_idx]
        all_top_genes.update(top_gene_names[:40])

    top_genes = list(all_top_genes)[:50]
    print(f"Selected {len(top_genes)} highly perturbed genes for heatmap")

    sc.pp.scale(adata_combined, max_value=10)

    # Heatmap
    plt.figure(figsize=(18, 11))
    sc.pl.heatmap(
        adata_combined,
        var_names=top_genes,
        groupby="condition",
        swap_axes=True,
        show_gene_labels=True,
        show=False,
        cmap="RdBu_r",
        vmin=-2.5,
        vmax=2.5,
        dendrogram=True,
        standard_scale=None,
        figsize=(18, 11)
    )
    plt.tight_layout()
    plt.show()

    # Dotplot
    plt.figure(figsize=(16, 10))
    sc.pl.dotplot(
        adata_combined,
        var_names=top_genes,
        groupby="condition",
        standard_scale="var",
        show=False,
        cmap="RdBu_r",
        dendrogram=True
    )
    plt.title("Dotplot – Control vs Drug Perturbations")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()