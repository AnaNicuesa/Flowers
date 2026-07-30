"""Loading and preprocessing of the Oxford Flowers 102 dataset via tensorflow-datasets."""

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split

from src.config import BATCH_SIZE, IMG_SIZE, SEED, SPLIT_CACHE_DIR, TRAIN_FRAC, VAL_FRAC

AUTOTUNE = tf.data.AUTOTUNE


def load_raw_splits(data_dir=None):
    """Load the three official TFDS train/validation/test splits and dataset info, as-is.

    The official Oxford Flowers 102 split allocates only 10 images per class to
    train and 10 to validation, with the remaining ~6,149 images in test — a
    deliberate few-shot benchmark design. This project does not train on that
    split directly (see `load_raw_combined`); this function is kept as the raw
    building block and for inspecting the official split's composition.
    """
    (train_ds, val_ds, test_ds), info = tfds.load(
        "oxford_flowers102",
        split=["train", "validation", "test"],
        with_info=True,
        as_supervised=True,
        data_dir=data_dir,
    )
    return train_ds, val_ds, test_ds, info


def get_class_names(info):
    return info.features["label"].names


def _stratified_split_assignment(labels, train_frac=TRAIN_FRAC, val_frac=VAL_FRAC, seed=SEED):
    """Assign each example (by position) to train(0) / val(1) / test(2), stratified
    by class, so every split preserves the natural per-class image counts.
    """
    idx = np.arange(len(labels))
    train_idx, rest_idx = train_test_split(
        idx, train_size=train_frac, stratify=labels, random_state=seed
    )
    val_share_of_rest = val_frac / (1 - train_frac)
    val_idx, test_idx = train_test_split(
        rest_idx,
        train_size=val_share_of_rest,
        stratify=labels[rest_idx],
        random_state=seed,
    )
    assignment = np.zeros(len(labels), dtype=np.int64)
    assignment[val_idx] = 1
    assignment[test_idx] = 2
    return assignment


def load_raw_combined(data_dir=None):
    """Pool all three official TFDS splits into one 8,189-image dataset and
    re-split it into a stratified 80/10/10 train/val/test split.

    Rationale: the official split's train/validation sets are artificially
    uniform (10 images/class each), which hides this dataset's real class
    imbalance (40-258 images/class) and leaves too little data to meaningfully
    train a CNN from scratch. Pooling and re-splitting surfaces that imbalance
    for EDA and gives every split a realistic amount of data, at the cost of
    no longer being directly comparable to the dataset's original benchmark
    numbers.
    """
    train_raw, val_raw, test_raw, info = load_raw_splits(data_dir=data_dir)
    combined = train_raw.concatenate(val_raw).concatenate(test_raw)

    labels = np.array([label.numpy() for _, label in combined])
    assignment = tf.constant(_stratified_split_assignment(labels))

    SPLIT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _make_split(target, cache_name):
        def _in_split(i, _xy):
            return tf.equal(tf.gather(assignment, i), target)

        return (
            combined.enumerate()
            .filter(_in_split)
            .map(lambda _i, xy: xy, num_parallel_calls=AUTOTUNE)
            .cache(str(SPLIT_CACHE_DIR / cache_name))
        )

    train_ds = _make_split(0, "train")
    val_ds = _make_split(1, "val")
    test_ds = _make_split(2, "test")
    return train_ds, val_ds, test_ds, info


def _resize_and_cast(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.cast(image, tf.float32)
    return image, label


def build_pipeline(ds, training=False, batch_size=BATCH_SIZE, shuffle_buffer=1024):
    """Resize/cast/batch a raw (image, label) dataset.

    Augmentation is intentionally NOT applied here: it is added as Keras
    preprocessing layers inside the model itself (see src/data/augmentation.py),
    so it is automatically active only during training and inactive at
    inference/evaluation time without any extra bookkeeping here.
    """
    ds = ds.map(_resize_and_cast, num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.shuffle(shuffle_buffer, seed=SEED)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def load_datasets(data_dir=None, batch_size=BATCH_SIZE):
    """Convenience wrapper returning ready-to-train tf.data pipelines + metadata,
    built on this project's custom stratified 80/10/10 split.
    """
    train_raw, val_raw, test_raw, info = load_raw_combined(data_dir=data_dir)
    train_ds = build_pipeline(train_raw, training=True, batch_size=batch_size)
    val_ds = build_pipeline(val_raw, training=False, batch_size=batch_size)
    test_ds = build_pipeline(test_raw, training=False, batch_size=batch_size)
    class_names = get_class_names(info)
    return train_ds, val_ds, test_ds, class_names, info
