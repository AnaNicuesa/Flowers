# Demo App

A lightweight Streamlit app for interactive inference: upload a flower image and get
the top-5 predicted species with confidence, using the fine-tuned EfficientNetB0
transfer-learning model from `models/checkpoints/` (the best-performing checkpoint
on the held-out test set — see `notebooks/04_evaluation.ipynb`).

## Run

```bash
source .venv/bin/activate
streamlit run app/app.py
```

Requires `models/checkpoints/efficientnetb0_transfer_finetuned.keras` (produced
by `notebooks/03_model_training.ipynb`) and `models/checkpoints/class_names.json`
(the 102 Oxford Flowers species names, in label-index order).
