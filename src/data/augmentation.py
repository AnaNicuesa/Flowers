"""Data augmentation as Keras preprocessing layers.

Implemented as layers (rather than tf.data.map transforms) so they can be placed
directly inside the model graph: Keras automatically disables them in inference
mode (model.predict/evaluate) and during validation, while keeping them active
in training mode — no manual train/eval branching required.
"""

from tensorflow import keras
from tensorflow.keras import layers

from src.config import SEED


def build_augmentation_layer():
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal", seed=SEED),
            layers.RandomRotation(0.1, seed=SEED),
            layers.RandomZoom(0.15, seed=SEED),
            layers.RandomContrast(0.1, seed=SEED),
        ],
        name="augmentation",
    )
