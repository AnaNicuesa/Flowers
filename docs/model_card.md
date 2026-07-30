# Model Card: Flower Species Classifier (EfficientNetB0, fine-tuned)

Following the format proposed in [Mitchell et al., 2019, "Model Cards for Model Reporting"](https://arxiv.org/abs/1810.03993).

## Model Details

- **Developed by:** Ana Nicuesa, as a portfolio/coursework project (Masterschool).
- **Model type:** Image classifier. `EfficientNetB0` (ImageNet-pretrained) backbone,
  fine-tuned, with a `GlobalAveragePooling2D -> Dropout(0.3) -> Dense(102, softmax)` head.
- **Model file:** `models/checkpoints/efficientnetb0_transfer_finetuned.keras`
  (not tracked in git — see the repository README for how to obtain it: retrain
  locally via `notebooks/03_model_training.ipynb`, or download it from the
  project's [GitHub Releases](https://github.com/AnaNicuesa/Flowers/releases)).
- **Input:** a single RGB image, resized to 224x224, `float32`, `[0, 255]` range
  (no manual rescaling — see `docs/architecture.md`).
- **Output:** a 102-way softmax over Oxford Flowers 102 species names
  (`models/checkpoints/class_names.json`).
- **License:** the code is MIT-licensed (see `LICENSE`); the underlying Oxford
  Flowers 102 dataset has its own terms (Nilsback & Zisserman, 2008) — see
  "Licensing" under Ethical Considerations below.
- **Two other checkpoints exist for comparison, not for production use:** a
  from-scratch baseline CNN (`baseline_cnn.keras`) and the same EfficientNetB0
  backbone before fine-tuning (`efficientnetb0_transfer.keras`) — see Evaluation
  Results below for why the fine-tuned checkpoint was selected.

## Intended Use

- **Primary intended use:** demonstrating an end-to-end transfer-learning
  workflow, and as the model behind the interactive demo app (`app/app.py`) for
  identifying one of the 102 trained flower species from a photo.
- **Intended users:** portfolio reviewers, coursework graders, and (per the
  business framing in `docs/business_case.md`) as a prototype for consumer
  plant-ID apps, e-commerce cataloging, or citizen-science tools.
- **Out of scope:** any use where a wrong species prediction has a safety,
  medical, legal, or financial consequence (e.g., identifying whether a plant
  is edible or toxic). This model has no notion of "poisonous" or "safe" — it
  only ranks visual similarity to its 102 training classes.

## Training Data

- **Source:** Oxford Flowers 102 (Nilsback & Zisserman, 2008), loaded via
  `tensorflow-datasets`. 102 species, 8,189 images total, naturally imbalanced
  (40-258 images per species).
- **Split used:** a **custom stratified 80/10/10** split built by this project
  (`src/data/dataset.py: load_raw_combined`) — 6,551 train / 819 validation /
  819 test images — not the dataset's official few-shot benchmark split (which
  allocates a uniform 10 images/class to train/validation). See
  `docs/architecture.md` for the full rationale. **Consequence:** metrics
  below are not directly comparable to published Oxford Flowers 102 benchmark
  results, which use the official split. The split is stratified and
  reproducible (identical on rerun with the same seed), and the train/
  validation/test index sets are disjoint by construction (`tests/test_data.py`).
- **Preprocessing:** resize to 224x224, no manual rescaling (EfficientNetB0's
  built-in normalization handles it). Mild augmentation during training only
  (horizontal flip, ±10% rotation, ±15% zoom, ±10% contrast).

## Evaluation Results

Test set: 819 held-out images (custom split, never used in training or for
early-stopping/checkpoint selection).

| Model | Test accuracy | Macro precision | Macro recall | Macro F1 |
|---|---|---|---|---|
| Baseline CNN (from scratch) | 60.0% | 60.6% | 55.9% | 55.0% |
| EfficientNetB0 (frozen backbone) | 94.4% | 95.1% | 93.2% | 93.6% |
| **EfficientNetB0 (fine-tuned) — this model** | **95.8%** | **96.5%** | **94.9%** | **95.2%** |

Top-3 accuracy is 99.0% and Expected Calibration Error is 0.020 (well
calibrated) — see `notebooks/04_evaluation.ipynb`. The demo app's 0.7
confidence threshold was selected on validation data alone (92.1% coverage,
98.9% accuracy) and confirmed once, not tuned, on test (91.9% coverage,
98.5% accuracy).

Full confusion analysis, per-class F1 extremes, and sample predictions
(correct and incorrect) are in `reports/figures/04_*.png` and
`notebooks/04_evaluation.ipynb`.

**Caveat on precision:** the 819-image test set averages only ~8 images per
class. Per-class metrics (per-class F1, individual confusion counts) should be
read as indicative, not statistically precise, for any single species — the
macro-averaged headline numbers above are more reliable than any one class's
breakdown.

## Ethical Considerations

- **No personal or sensitive data.** The dataset contains photographs of
  flowers, not people; there is no privacy or demographic-fairness axis to
  this task in the way there would be for a model classifying people.
- **Data quality varies by class.** Class imbalance (40-258 images/species)
  means rarer species are learned from less evidence and are more likely to
  be misclassified — this is a data-availability issue, not a demographic
  bias issue, but it does mean the model is not equally reliable across all
  102 species (see per-class F1 in `notebooks/04_evaluation.ipynb`).
- **Licensing:** Oxford Flowers 102 images were collected from various online
  sources by the dataset's creators for research use; check the
  [dataset's own terms](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)
  before any commercial deployment of a model trained on it.

## Caveats and Recommendations

- **Closed-set assumption.** The model always outputs one of its 102 trained
  species — it has no built-in mechanism to say "none of these" for an
  out-of-distribution flower. The demo app (`app/app.py`) mitigates this with
  a confidence warning threshold (0.7, chosen on validation data — see
  Evaluation Results); a production deployment should validate that threshold
  against its own real-world traffic rather than reuse it as-is.
- **Distribution shift.** Training photos are relatively clean, centered,
  single-flower shots (see `notebooks/01_data_exploration.ipynb`, "Image
  Characteristics"). Real user photos (phone cameras, poor lighting, multiple
  flowers, occlusion) are not represented in training data, and accuracy on
  such photos is unvalidated.
- **Not benchmark-comparable.** Because this project uses a custom split (see
  Training Data above), reported accuracy should not be quoted alongside
  official Oxford Flowers 102 leaderboard results without noting the
  methodology difference.
- **Before production use:** see `docs/business_case.md` ("What would be
  required before production deployment") for the full list — confidence
  thresholding, broader/noisier training data, monitoring for confidence
  drift, latency/footprint validation, and a licensing review.
