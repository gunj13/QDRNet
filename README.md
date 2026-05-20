# QDRNet: A Hybrid Quantum-Classical Framework for Diabetic Retinopathy Grading with Lesion-Driven Attention

A 5-class DR grading model that combines an EfficientNet-B0 backbone with CBAM attention, UniverSeg-based lesion segmentation masks (MA, HE, HEX, SEX, vessel), and a quantum-classical interaction layer (Hadamard or VQC mode).

---

## Project Structure

```
.
├── config.py                    # All hyperparameters and paths
├── train.py                     # Training entry point
├── evaluate.py                  # Test-set evaluation on APTOS
├── evaluate_for_bootstrap.py    # Saves raw outputs for CI computation
├── bootstrap_ci.py              # 95% bootstrap confidence intervals
├── messidor_inference.py        # Zero-shot evaluation on Messidor-2
├── messidor_precompute_masks.py # Precompute UniverSeg masks for Messidor-2
├── precompute_masks.py          # Precompute UniverSeg masks for APTOS
├── visualize.py                 # Preprocessing / segmentation / GradCAM plots
├── visualize_aptos_cache.py     # Visual sanity-check of APTOS mask cache
├── visualize_messidor_cache.py  # Visual sanity-check of Messidor mask cache
├── visualize_seg_masks.py       # Standalone segmentation mask visualizer
├── plots.py                     # Training curve helpers
├── utils.py                     # Metric computation utilities
├── data/
│   ├── dataset.py               # PyTorch Dataset + DataLoader factory
│   └── mask_cache/              # Precomputed .pt mask tensors (generated, not uploaded)
│       ├── aptos/{train,val,test}/
│       └── messidor/
├── model/
│   ├── backbone.py              # EfficientNet-B0 feature extractor
│   ├── cbam.py                  # CBAM channel + spatial attention
│   ├── classifier.py            # Full DRNet model
│   └── quantum_layer.py         # Hadamard / VQC interaction layer
├── segmentation/
│   ├── support_loader.py        # Loads IDRiD / DRIVE support images
│   └── universeg_wrapper.py     # UniverSeg inference wrapper
├── checkpoints/                 # Saved model weights (generated, not uploaded)
└── outputs/                     # Evaluation results and plots (generated, not uploaded)
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download and place the datasets

The following datasets need to be downloaded manually and placed under `data/raw/`:

| Dataset | Purpose | Expected path |
|---|---|---|
| [APTOS 2019](https://www.kaggle.com/datasets/mariaherrerot/aptos2019) | Primary training / evaluation | `data/raw/APTOS/` |
| [IDRiD](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid) | UniverSeg lesion support images | `data/raw/IDRID/` |
| [DRIVE](https://www.kaggle.com/datasets/andrewmvd/drive-digital-retinal-images-for-vessel-extraction) | UniverSeg vessel support images | `data/raw/DRIVE/` |
| [Messidor-2](https://www.kaggle.com/datasets/mariaherrerot/messidor2preprocess) | Zero-shot evaluation and benchmarking | `data/raw/MESSIDOR/` |


---

## Precomputing Segmentation Masks

Masks must be generated **once** before training. This runs UniverSeg over every image and caches the 5-channel lesion tensors to disk so there is no segmentation overhead during training.

**APTOS masks:**
```bash
python precompute_masks.py
```
Output: `data/mask_cache/aptos/{train,val,test}/<filename>.pt` — each file is a `float32` tensor of shape `[5, 128, 128]`.

**Messidor-2 masks** (only needed for cross-dataset evaluation):
```bash
python messidor_precompute_masks.py
```
Output: `data/mask_cache/messidor/<filename>.pt`

---

## Training

All configuration (image size, batch size, epochs, quantum mode, etc.) is set in `config.py`.

| Command | Mode |
|---|---|
| `python train.py` | Default — Hadamard product interaction |
| `python train.py --mode vqc` | VQC (variational quantum circuit) interaction |
| `python train.py --ablation` | Classical linear layer (no quantum interaction) |
| `python train.py --resume` | Resume training from `checkpoints/last_checkpoint.pt` |

Checkpoints are saved to `checkpoints/`:
- `best_model.pt` — best validation F1 checkpoint
- `last_checkpoint.pt` — latest epoch (used for `--resume`)
- `training_log.pt` — per-epoch train/val metrics

---

## Evaluation

### APTOS test set

```bash
python evaluate.py
```

Loads `checkpoints/best_model.pt`, runs inference on the test split, prints per-class metrics, and saves plots to `outputs/`.

### Bootstrap confidence intervals

Run after `evaluate.py` (or `eval_for_bootstrap.py`) has written the numpy arrays:

```bash
python bootstrap_ci.py
```

Reads `outputs/test_labels.npy`, `outputs/test_preds.npy`, `outputs/test_probs.npy` and prints 95% CIs for per-class AUC and accuracy.

### Messidor-2 zero-shot evaluation

```bash
python messidor_inference.py
# or with a specific checkpoint:
python messidor_inference.py --ckpt checkpoints/best_model.pt
```

---

## Outputs

After evaluation, `outputs/` will contain:

- `confusion_matrix.png` — normalised confusion matrix
- `roc_curves.png` — per-class ROC curves
- `training_curves.png` — loss curve, F1 vs. epoch curves
- `test_labels.npy`, `test_preds.npy`, `test_probs.npy` — raw arrays for further analysis

---

## Visualizations

### Preprocessing and segmentation masks

```bash
python visualize.py --mode preprocess    # before/after CLAHE comparison
python visualize_seg_masks.py            # UniverSeg mask overlays on IDRID/DRIVE test set
python visualize_aptos_cache.py          # to display actual masks from APTOS images .pt files
```

### GradCAM

```bash
python visualize.py --mode gradcam
```

This requires `checkpoints/best_model.pt` to exist. It generates GradCAM saliency maps for representative images from each DR grade (DR1–DR4) and saves them to `outputs/`.

