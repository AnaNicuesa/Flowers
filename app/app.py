"""Streamlit demo: upload a flower photo, get a predicted species with confidence.

Run with: streamlit run app/app.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import IMG_SIZE, MODELS_DIR, TRANSFER_MODEL_NAME

MODEL_PATH = MODELS_DIR / f"{TRANSFER_MODEL_NAME}_finetuned.keras"
CLASS_NAMES_PATH = MODELS_DIR / "class_names.json"
# 0.7 is not a guess: chosen from notebooks/04_evaluation.ipynb's
# coverage/selective-accuracy trade-off computed on VALIDATION data (92.1%
# coverage, 98.9% accuracy on the kept predictions there), then confirmed —
# once, never used to pick the threshold — on the held-out test set (91.9%
# coverage, 98.5% accuracy). A 0.4 threshold barely filters anything (98.4%
# coverage) because this model's confidence is well-calibrated (ECE 0.02) and
# rarely lands in the 0.4-0.7 range at all.
TOP_K = 5
LOW_CONFIDENCE_THRESHOLD = 0.7


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_class_names():
    with open(CLASS_NAMES_PATH) as f:
        return json.load(f)


def preprocess(image: Image.Image):
    image = image.convert("RGB")
    array = tf.convert_to_tensor(np.array(image), dtype=tf.float32)
    array = tf.image.resize(array, (IMG_SIZE, IMG_SIZE))
    return tf.expand_dims(array, axis=0)


st.set_page_config(page_title="Flower Species Classifier", page_icon="🌸")
st.title("🌸 Flower Species Classifier")
st.caption(
    "EfficientNetB0 transfer-learning model (fine-tuned) trained on the Oxford "
    "Flowers 102 dataset (102 species, ~96% test accuracy). Upload a photo of a "
    "single flower."
)

model = load_model()
class_names = load_class_names()

uploaded_file = st.file_uploader("Upload a flower image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns([1, 1.4])

    with col1:
        st.image(image, caption="Uploaded image", use_container_width=True)

    with col2:
        batch = preprocess(image)
        probs = model.predict(batch, verbose=0)[0]
        top_indices = np.argsort(probs)[::-1][:TOP_K]

        best_prob = probs[top_indices[0]]
        if best_prob < LOW_CONFIDENCE_THRESHOLD:
            st.warning(
                f"Low confidence ({best_prob:.0%}) — this photo may not match any of "
                "the 102 trained species well. Treat the prediction below as a guess."
            )

        st.subheader("Top predictions")
        for idx in top_indices:
            name = class_names[idx].title()
            st.write(f"**{name}** — {probs[idx]:.1%}")
            st.progress(float(probs[idx]))

st.divider()
st.caption(
    "This model only recognizes the 102 species it was trained on (a closed-set "
    "classifier) — see `docs/business_case.md` for known limitations."
)
