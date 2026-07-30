"""Training utilities: reproducible seeding, standard callbacks, and a thin
`fit` wrapper that also records wall-clock training time.
"""

import json
import os
import platform
import random
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.config import MODELS_DIR, REPORTS_DIR, SEED


def set_global_seed(seed=SEED, deterministic_ops=True):
    """Fix Python/NumPy/TensorFlow seeds. `deterministic_ops=True` also asks
    TensorFlow to use deterministic kernel implementations where available
    (`enable_op_determinism`) — stricter than seeding alone, since some GPU ops
    are non-deterministic by default regardless of seed. Costs a little
    throughput; worth it for a reproducible experiment record.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    if deterministic_ops:
        tf.config.experimental.enable_op_determinism()


def environment_snapshot(seed=SEED):
    """Metadata worth recording alongside any training run, so results can be
    traced back to the exact environment that produced them.
    """
    return {
        "python_version": platform.python_version(),
        "tensorflow_version": tf.__version__,
        "platform": platform.platform(),
        "seed": seed,
        "gpus_visible": [d.name for d in tf.config.list_physical_devices("GPU")],
    }


def build_callbacks(model_name, monitor="val_accuracy", patience=8, min_lr=1e-6):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = MODELS_DIR / f"{model_name}.keras"
    return [
        keras.callbacks.EarlyStopping(
            monitor=monitor, patience=patience, restore_best_weights=True, mode="max"
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path), monitor=monitor, save_best_only=True, mode="max"
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor=monitor, factor=0.5, patience=3, min_lr=min_lr, mode="max"
        ),
    ]


def fit_and_time(model, train_ds, val_ds, epochs, callbacks, model_name):
    start = time.time()
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)
    elapsed_seconds = time.time() - start

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    history_path = REPORTS_DIR / f"{model_name}_history.json"
    payload = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    payload["_training_time_seconds"] = elapsed_seconds
    payload["_environment"] = environment_snapshot()
    with open(history_path, "w") as f:
        json.dump(payload, f, indent=2)

    return history, elapsed_seconds
