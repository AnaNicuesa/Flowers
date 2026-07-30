"""Central configuration constants shared across data, model, and training code."""

from pathlib import Path

SEED = 42

IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 102

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models" / "checkpoints"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
SPLIT_CACHE_DIR = PROJECT_ROOT / "data" / "processed" / "split_cache"

# The official TFDS split allocates exactly 10 train + 10 validation images per
# class (a few-shot benchmark design) and dumps all natural class imbalance into
# the test split. This project instead pools all 8,189 images and re-splits them
# with these fractions, stratified by class, so imbalance is real and analyzable
# and every split gets a realistic amount of data (see src/data/dataset.py).
TRAIN_FRAC = 0.8
VAL_FRAC = 0.1

BASELINE_MODEL_NAME = "baseline_cnn"
TRANSFER_MODEL_NAME = "efficientnetb0_transfer"
TRANSFER_BACKBONE = "EfficientNetB0"
