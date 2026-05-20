# config.py
import os

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_ROOT      = "data/raw"
APTOS_DIR      = os.path.join(DATA_ROOT, "APTOS")
IDRID_DIR      = os.path.join(DATA_ROOT, "IDRID")
DRIVE_DIR      = os.path.join(DATA_ROOT, "DRIVE")
MASK_CACHE_DIR = "data/mask_cache"
CHECKPOINT_DIR = "checkpoints"
OUTPUT_DIR     = "outputs"

# ── Image sizes ────────────────────────────────────────────────────────────
IMG_SIZE  = 512   # changed from 224 — preserves small lesion structure
MASK_SIZE = 256   # recompute masks at 512 via precompute_masks.py

# ── CLAHE ──────────────────────────────────────────────────────────────────
CLAHE_CLIP = 2.0
CLAHE_TILE = (8, 8)

# ── Segmentation preprocessing ─────────────────────────────────────────────
SEG_PREPROCESS = "gray_l"   # PIL L-mode grayscale — preserves red channel

# ── Support sets ───────────────────────────────────────────────────────────
SUPPORT_SHOTS = {"MA": 54, "HE": 53, "HEX": 54, "SEX": 26, "vessel": 20}

# ── Model ──────────────────────────────────────────────────────────────────
BACKBONE_FREEZE = False
AUX_IN_CHANNELS = 5       # MA, HE, HEX, SEX, vessel — all 5 channels active
CLASSIFIER_DIM  = 64      # latent classifier width before the interaction layer
BOTTLENECK_DIM  = 4       # must equal NUM_QUBITS
NUM_CLASSES     = 5
CBAM_REDUCTION  = 16
DROPOUT         = 0.4

# ── Quantum ────────────────────────────────────────────────────────────────
INTERACTION_MODE = "hadamard" # hadamard / vqc
NUM_QUBITS       = 4
NUM_Q_LAYERS     = 2

# ── Training ───────────────────────────────────────────────────────────────
EPOCHS          = 60
BATCH_SIZE      = 16       # reduced from 16 — 512px images need more VRAM
NUM_WORKERS     = 0       # 0 on Windows to avoid multiprocess spawn issues
LEARNING_RATE   = 1e-4
WEIGHT_DECAY    = 1e-4
WARMUP_EPOCHS   = 5
PATIENCE        = 12      # early stopping: stop if no val F1 improvement
LABEL_SMOOTHING = 0.1
FOCAL_GAMMA     = 2.0  # penalize hard cases
ORDINAL_WEIGHT  = 0.3
RANDOM_SEED     = 42

ZERO_MASKS = False   # set True to run no-segmentation ablation
