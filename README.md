# Auto-Encoders for Focal Adhesion Type Discovery 

This project uses **auto-encoders (AEs)** to discover and analyze **focal adhesion (FA) types** through **unsupervised learning** on local 2D patches extracted from within segmented cell regions. The input is **multi-channel fluorescence microscopy data**, where:

* **Focal adhesion (FA) channel** is used for **patch extraction and embedding**
* **Actin channel** (phalloidin staining) is used **only for cell segmentation**

We train an AE to learn compact, discriminative representations of local FA morphology. The resulting **latent space** is analyzed to reveal phenotypic structure, variability, and potential FA subtypes.

## Project Summary
The project pipeline includes:
1. **Cell segmentation** using the **actin channel**
2. **Grid-based patch extraction** from the **FA channel** within cell masks
3. **Auto-encoder training** on FA patches to learn latent features
4. **Latent space analysis** (UMAP/t-SNE) to explore FA heterogeneity and clustering

This approach enables **segmentation-free analysis** of focal adhesion diversity across single cells and experimental conditions.

## Data Description

* **Input**: multi-channel fluorescence microscopy images
  * Channel 1~3: Focal adhesion markers (e.g., vinculin, paxillin, zyxin, etc.)
  * Channel last: Actin (phalloidin) for segmentation only

* **Patch Sampling**:
  * Grid-based with small random translation and rotation
  * Patch size: typically 64×64 or 32x32 pixels
  * Only patches with **≥ 50% of area inside cell mask** are kept
  * Only **FA channel** is used for AE input; actin is discarded post-mask generation
   
<img src="https://github.com/user-attachments/assets/0be90e3c-86bf-4122-9bcb-d3df7c20dfff" style="width:40%;"/>  

## Goals
* Learn **compressed latent representations** of focal adhesion patches
* Capture **morphological and spatial variation** across FAs
* Identify FA subtypes through **clustering in latent space**
* Enable comparisons across cell types, treatments, or timepoints
* Build a data-driven foundation for FA classification and phenotyping

## Requirements
* Python ≥ 3.9
* PyTorch or TensorFlow
* NumPy, scikit-image (for mask generation and transformations)
* tifffile, imageio, or h5py (for multichannel microscopy images)
* OpenCV or PIL (for patch extraction and affine transforms)
* scikit-learn, umap-learn (for dimensionality reduction and clustering)
* matplotlib, seaborn 

## Pipeline Summary  
1. **Segmentation**: 
   * Generate cell masks using actin channel  
2. **Patch Extraction**:  
   * Apply grid with jitter & rotation  
   * Retain patches where ≥50% lies within cell mask  
   * Extract **FA channel only** from each patch       
3. **Auto-Encoder Training**:  
   * Convolutional AE on FA-only patches  
   * Loss: MSE or SSIM  
4. **Latent Space Analysis**:  
   * Extract latent codes from encoder  
   * Visualize with UMAP, PCA  
   * Perform clustering to identify FA subtypes  

## Outputs  
* Binary cell masks from actin channel  
* Coordinates and metadata for each FA patch  
* Trained auto-encoder model   
* Patch embeddings  
* UMAP/t-SNE plots of FA morphology in latent space  
* Cluster assignments and montages of patch exemplars  

<img src="https://github.com/user-attachments/assets/bbcb1bd6-2f28-44b9-927a-dbd8aa560bfe" style="width:60%;"/>  

<img src="https://github.com/user-attachments/assets/71da5f56-a41e-4be0-bf3f-30bcad17d878" style="width:40%;"/>  

