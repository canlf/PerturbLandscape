## PerturbLandscape
PerturbLandscape provides a practical strategy for integrating single-cell perturbation knowledge with spatial transcriptomics, enabling tissue-scale perturbation response prediction and offering computational support for drug discovery and precision medicine.

## Overview
PerturbLandscape is a deep generative framework that transfers drug perturbation knowledge learned from single-cell RNA-seq (scRNA-seq) data to spatial transcriptomics (ST) data. It predicts spatially resolved gene expression changes under drug treatments.

The model learns a continuous flow in a disentangled latent space that separates:
Identity factors (`z_id`) — cell-type / background identity
Perturbation factors(`z_pert`) — drug-specific response

During inference, single-cell perturbation trajectories are transported onto spatial spots via Optimal Transport (OT) deconvolution and refined with a lightweight graph neural force, enabling spatially coherent predictions.






PerturbLandscape
├── train_sc_to_st.py              # SC model training script 

├── models.py                      # Neural network definitions

├── spatial_utils.py               # Denoising, OT, deconvolution, metrics 

├── main_st_inference.py           # ST inference & visualization 

├── best_model_sc_ST.pt            # Trained checkpoint (generated after training) 



## Requirements
bash

torch >= 2.0

scanpy >= 1.9

anndata

numpy

pandas

scipy

scikit-learn

tqdm
pot                    # Python Optimal Transport
matplotlib



pip install torch scanpy anndata numpy pandas scipy scikit-learn tqdm pot matplotlib
