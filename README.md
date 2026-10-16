# PerturbLandscape

PerturbLandscape is a computational framework for transferring drug-perturbation responses learned from single-cell RNA sequencing (scRNA-seq) data into spatial transcriptomics (ST) samples. It combines conditional latent-state modeling with optimal transport (OT) and spatial graph refinement to generate spatially resolved perturbation profiles for downstream analysis.

## Method at a glance

The model encodes expression profiles into latent representations that distinguish cellular identity from perturbation-associated state. A conditional flow model uses drug identity and dose to model transitions from control to treated states. For spatial transfer, shared genes define a common input space; OT estimates correspondences between single cells and spatial locations. The resulting transport weights project single-cell perturbation changes onto tissue locations, where neighborhood-based refinement incorporates local spatial structure. Predicted profiles can then be examined for spatial organization and drug-associated gene programs.

Data:https://drive.google.com/drive/folders/1L2HjGmnYI-nsPW3Ui8qqqJS_S6XCOOWG?usp=drive_links
### 1. PerturbLandscape framework

<img width="968" height="815" alt="image" src="https://github.com/user-attachments/assets/306ed25c-a793-45a4-9673-bd7254edb534" />

*PerturbLandscape transfers single-cell perturbation dynamics into tissue contexts. (A) Control and drug-treated single-cell states are encoded in a disentangled latent space representing cellular identity and perturbation-associated information; conditional flow matching models the transition between states. (B) Drug identity and dose condition perturbation generation through learnable drug embeddings. (C) Shared genes support a common representation for scRNA-seq and spatial transcriptomics; optimal transport maps perturbation states to spatial locations, followed by graph-based refinement. (D) Predicted spatial perturbation landscapes are examined through perturbation similarity, spatial organization, and biological response characterization.*

### 2. OT-based cross-modal transfer

<img width="861" height="404" alt="image" src="https://github.com/user-attachments/assets/2580d799-732b-4687-8d20-6857dfc37773" />


*Overview of the OT-based transfer procedure. Single-cell and spatial samples are represented in a shared perturbation latent space. A latent-space cost matrix and entropy-regularized Sinkhorn OT yield a coupling matrix. The coupling is transposed and normalized to obtain spot-level deconvolution weights; entropy-guided sparsification retains informative cell-to-spot connections. These weights project single-cell perturbation changes to an initial spatial perturbation representation.*

### 3. Correspondence between cell states and tissue regions

<img width="841" height="538" alt="image" src="https://github.com/user-attachments/assets/71479945-4b39-4f77-9b82-da0cdb5b64ce" />

*OT mapping between single-cell clusters and spatial transcriptomic regions. Curves show major transport trajectories, illustrating correspondences between single-cell transcriptional states and spatial tissue regions during perturbation transfer.*

### 4. Spatial graph refinement

<img width="859" height="254" alt="image" src="https://github.com/user-attachments/assets/6a6bd55a-28f7-499b-9ac7-a1e92fd4b342" />


*Cross-modal perturbation transfer and spatial refinement. OT weights transfer single-cell perturbation representations to spatial locations. A neighborhood graph built from spatial coordinates estimates local residual differences, which are combined with the transported representation to produce spatially refined perturbation states.*

### 5. Predicted spatial perturbation landscapes

<img width="593" height="634" alt="image" src="https://github.com/user-attachments/assets/0e2de68c-9517-4ff3-a937-e4b8277404ca" />

*Predicted perturbation profiles for sixteen representative drugs in lung cancer spatial transcriptomics data. Color intensity indicates the predicted response magnitude at each spatial location. The different spatial patterns illustrate heterogeneous tissue-level responses across drugs.*

### 6. Drug–gene program associations

<img width="844" height="613" alt="image" src="https://github.com/user-attachments/assets/e911c6ae-1b85-4a51-ab0e-44088f15d876" />


*Drug–gene program association network. Edges connect drugs to their major predicted responsive genes. Genes shared across drugs indicate potentially common response programs and biological mechanisms.*

## Repository contents

- `code/main/train.py` — trains the single-cell perturbation model and saves a checkpoint.
- `code/main/models.py` — neural network components used by spatial inference.
- `code/main/spatial_utils.py` — spatial denoising, OT mapping, deconvolution, perturbation transfer, graph refinement, and evaluation utilities.
- `code/main/main_st_inference.py` — spatial inference and visualization workflow.
- `code/main/cell_map.py` — additional spatial mapping visualizations.
- `code/main/train_lovo.py` and `code/main/Lovo_ref.py` — LoVo-aligned training and downstream analysis scripts.
- `dataset/` — curated public datasets included with this project.

## Software requirements

The analysis scripts use Python and the following packages: PyTorch, Scanpy, AnnData, NumPy, pandas, SciPy, scikit-learn, POT (Python Optimal Transport), tqdm, and Matplotlib. Some visualization scripts also use additional plotting or dimensionality-reduction packages. Package versions and a reproducible environment specification should be added when finalizing a release.

## Data use and citation

This collection brings together data from multiple public sources. Cite both this curated collection and the original source(s) for the data used in any analysis. Check the source-specific license and terms before redistribution or reuse. The project authors are responsible for the curation represented by this record; original dataset authors should be credited through citations to their source publications and repositories.

## Citation

If you use this collection, cite its Zenodo record using the DOI assigned upon publication. Also cite the original publications or repositories for the individual datasets used.
