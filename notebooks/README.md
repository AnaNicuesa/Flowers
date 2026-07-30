# Notebooks

All four notebooks are implemented and have been executed end-to-end. They are
meant to be run **in numerical order** — each one depends on artifacts (a
dataset split, a trained model) produced by an earlier notebook, not just on
being listed after it.

## 1. `01_data_exploration.ipynb`

- **Input:** the Oxford Flowers 102 dataset, loaded directly via
  `tensorflow-datasets` (no local files required).
- **What it does:** builds this project's custom stratified 80/10/10 split,
  analyzes real class imbalance (40-258 images/class), and audits raw image
  properties (format, resolution, aspect ratio, pixel statistics, integrity
  checks, visual examples).
- **Output:** EDA figures in `reports/figures/01_*.png`. No model or data
  files are produced — later notebooks re-derive the same split from
  `src/data/dataset.py`, they don't read anything from this notebook's output.

## 2. `02_preprocessing_augmentation.ipynb`

- **Input:** the same dataset split (via `src/data/dataset.py`), no files from
  notebook 01 required.
- **What it does:** demonstrates and visually verifies the resize/cast
  pipeline and the augmentation layers used inside the models.
- **Output:** figures in `reports/figures/02_*.png`.

## 3. `03_model_training.ipynb`

- **Input:** the dataset split (via `src/data/dataset.py`); no files from
  notebooks 01-02 are required to run it.
- **What it does:** builds and trains the baseline CNN and the EfficientNetB0
  transfer-learning model (frozen backbone, then fine-tuned).
- **Output:** **this is the notebook that produces the model files** —
  `models/checkpoints/baseline_cnn.keras`,
  `models/checkpoints/efficientnetb0_transfer.keras`, and
  `models/checkpoints/efficientnetb0_transfer_finetuned.keras` — plus training
  history JSON in `reports/` and training-curve figures in
  `reports/figures/03_*.png`. CPU-only training takes roughly 2.5 hours.

## 4. `04_evaluation.ipynb`

- **Input:** **requires the three `.keras` checkpoints produced by
  `03_model_training.ipynb` to already exist in `models/checkpoints/`** — it
  loads them rather than training anything itself, and will error if they're
  missing.
- **What it does:** evaluates all three checkpoints on the held-out test
  split, then runs a deeper error analysis (confusion pairs, per-class F1,
  top-k accuracy, calibration, confidence-threshold trade-off) on the
  best-performing one.
- **Output:** figures in `reports/figures/04_*.png`.

## 5. `05_project_summary.ipynb`

- **Input:** none (a short markdown-only close-out, no code, no computation).
- **What it does:** summarizes the project's main findings, limitations, and
  future work in one place, without repeating the detailed analysis already
  in notebooks 01-04.
- **Output:** none — reference-only.

## Running them

```bash
jupyter notebook notebooks/
```

or headless, in order, from the command line:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_data_exploration.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_preprocessing_augmentation.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/03_model_training.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/04_evaluation.ipynb
```

Each notebook saves reusable logic into `src/` rather than duplicating it inline.
