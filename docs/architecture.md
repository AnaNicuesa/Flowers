# Architecture & Technical Decisions

This document explains the *why* behind the major technical choices in `src/`.
For *what* the code does, read the code — it's kept intentionally thin and
readable.

## Dataset & splits

- **Source:** Oxford Flowers 102 via `tensorflow_datasets` (`oxford_flowers102`),
  not a manually downloaded copy — this keeps the pipeline reproducible from a
  single `pip install` with no manual data-wrangling steps.
- **Custom stratified 80/10/10 split, not the official one.** The dataset's
  official TFDS split allocates exactly 10 images per class to training and 10
  to validation (a deliberate few-shot benchmark design), and dumps the
  dataset's entire natural class imbalance — 40 to 258 images per class — into
  the test split. That hides the imbalance from any EDA run against the
  official train split, and leaves far too little data (10 images/class) to
  meaningfully train a CNN from scratch or to do a real class-imbalance
  analysis, both of which this project's brief specifically calls for.
  `src/data/dataset.py: load_raw_combined` instead pools all three official
  splits (8,189 images total) and re-splits them with a stratified 80/10/10
  train/validation/test split (`sklearn.model_selection.train_test_split`,
  fixed `SEED`), giving 6,551 / 819 / 819 images respectively, with every
  class's natural imbalance preserved proportionally in each split. The
  trade-off: results are no longer directly comparable to the dataset's
  original published few-shot benchmark numbers — this project isn't
  attempting that comparison.
- **Split caching:** because building the split requires one decode pass over
  the full 8,189-image pool to extract labels for stratification, each
  resulting split is cached to disk (`data/processed/split_cache/`, gitignored)
  via `tf.data.Dataset.cache()` so repeated epochs — and repeated notebook runs
  — don't re-decode and re-filter the whole pool every time.

## Preprocessing (`src/data/dataset.py`)

- Images are resized to `224x224` (`src/config.py: IMG_SIZE`) — the native input
  size EfficientNetB0 was pretrained on, avoiding any accuracy loss from a
  mismatched resolution.
- Pixel values are cast to `float32` but **not rescaled to [0,1]** in the data
  pipeline. `tf.keras.applications.EfficientNetB0` includes its own internal
  rescaling/normalization layer and expects raw `[0, 255]` input — adding a
  second rescale step would double-normalize and hurt accuracy. The baseline CNN
  (which has no such built-in layer) applies its own explicit
  `Rescaling(1./255)` as its first layer.

## Data augmentation (`src/data/augmentation.py`)

Implemented as Keras preprocessing **layers** inside the model graph
(`RandomFlip`, `RandomRotation`, `RandomZoom`, `RandomContrast`) rather than as a
`tf.data.Dataset.map()` transform. Reasoning: Keras automatically runs these
layers only when `training=True`, so validation/test data is never accidentally
augmented — there is no separate branch to keep in sync, and the same model
object is correct for both training and inference.

Augmentation choices are intentionally mild (10% rotation, 15% zoom, 10%
contrast) because flower species are partly defined by color and shape — overly
aggressive color/geometric distortion risks destroying the very features that
distinguish visually similar species.

## Models (`src/models/build_model.py`)

Two models are trained, specifically to produce an honest **baseline vs.
transfer learning** comparison rather than assuming transfer learning helps:

1. **Baseline CNN** — a small 4-block Conv2D/BatchNorm/MaxPool stack trained
   *from scratch* on the ~6,551 training images. Even with a realistic amount
   of data, 102 fine-grained, imbalanced classes is still not much per class
   for a model with no prior visual knowledge — it exists to quantify, in this
   project's own numbers, how much transfer learning actually helps.
2. **Transfer learning model** — `EfficientNetB0` pretrained on ImageNet as a
   frozen feature extractor, with a new `GlobalAveragePooling2D -> Dropout ->
   Dense(102, softmax)` head.

**Why EfficientNetB0:** it offers a strong accuracy/parameter-count trade-off
among ImageNet backbones available in `tf.keras.applications`, keeping the
resulting `.keras` file reasonably small (relevant to the "model size" metric
and to eventual mobile/edge deployment discussed in the business case) without
sacrificing the accuracy fine-grained classification needs.

## Training strategy (`src/models/train.py`)

Two-stage transfer learning:

1. **Stage 1 — frozen backbone:** only the new head is trained, backbone weights
   untouched. This lets the head adapt to the 102-class output without
   destroying the pretrained features.
2. **Stage 2 — fine-tuning:** the top N layers of the backbone are unfrozen
   (`unfreeze_top_layers`) and trained at a **much lower learning rate**, so the
   pretrained weights are nudged toward this dataset rather than overwritten.
   `BatchNormalization` layers are kept frozen even inside the unfrozen block:
   with 6,551 training images spread across 102 imbalanced classes (as few as
   32 images for the rarest class), updating BN running statistics during
   fine-tuning is still a small-sample-size risk, so this project keeps that
   part conservative rather than testing it unfrozen.

**Reproducibility:** `set_global_seed()` fixes the Python, NumPy, and TensorFlow
random seeds (default `42`, see `src/config.py`) before any model is built or
trained.

**Callbacks** (`build_callbacks`): `EarlyStopping` (restores best weights, avoids
overfitting past the point validation accuracy stops improving),
`ModelCheckpoint` (persists only the best epoch), `ReduceLROnPlateau` (backs off
the learning rate when validation accuracy plateaus rather than requiring a
manual LR schedule).

## Evaluation (`src/evaluation/metrics.py`)

Reports **macro** precision/recall/F1 (not just overall accuracy), because with
102 classes and an uneven number of images per class, macro-averaging weights
every class equally and surfaces poor performance on rare classes that a
micro/weighted average would hide.

For error analysis, rather than rendering a 102x102 confusion matrix (unreadable
at any reasonable figure size), `most_confused_pairs()` extracts only the
highest-count off-diagonal (true, predicted) pairs, and `per_class_f1()` isolates
the best/worst-performing classes — both are the useful signal a full matrix
would otherwise bury.
