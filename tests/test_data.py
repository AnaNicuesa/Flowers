import numpy as np
import tensorflow as tf

from src.config import IMG_SIZE
from src.data.augmentation import build_augmentation_layer
from src.data.dataset import _stratified_split_assignment, build_pipeline


def _fake_raw_dataset(n=4, size=96):
    images = tf.random.uniform((n, size, size, 3), maxval=255, dtype=tf.float32)
    labels = tf.range(n)
    return tf.data.Dataset.from_tensor_slices((images, labels))


def test_build_pipeline_resizes_and_batches():
    ds = _fake_raw_dataset(n=6)
    pipeline = build_pipeline(ds, training=False, batch_size=2)
    batch_images, batch_labels = next(iter(pipeline))
    assert batch_images.shape == (2, IMG_SIZE, IMG_SIZE, 3)
    assert batch_images.dtype == tf.float32
    assert batch_labels.shape == (2,)


def test_augmentation_layer_preserves_shape():
    layer = build_augmentation_layer()
    images = tf.random.uniform((2, IMG_SIZE, IMG_SIZE, 3), maxval=255)
    out = layer(images, training=True)
    assert out.shape == images.shape


def test_augmentation_layer_is_inactive_at_inference():
    layer = build_augmentation_layer()
    images = tf.random.uniform((2, IMG_SIZE, IMG_SIZE, 3), maxval=255, seed=0)
    out = layer(images, training=False)
    np.testing.assert_allclose(out.numpy(), images.numpy())


def _fake_labels(counts):
    return np.concatenate([np.full(c, i) for i, c in enumerate(counts)])


def test_stratified_split_assignment_covers_every_example_once():
    labels = _fake_labels([50, 30, 100])
    assignment = _stratified_split_assignment(labels, train_frac=0.8, val_frac=0.1, seed=0)
    assert assignment.shape == labels.shape
    assert set(np.unique(assignment)).issubset({0, 1, 2})


def test_stratified_split_assignment_keeps_every_class_in_every_split():
    labels = _fake_labels([50, 30, 100])
    assignment = _stratified_split_assignment(labels, train_frac=0.8, val_frac=0.1, seed=0)
    for split in (0, 1, 2):
        classes_present = set(labels[assignment == split].tolist())
        assert classes_present == {0, 1, 2}


def test_stratified_split_assignment_matches_requested_fractions():
    labels = _fake_labels([200, 200, 200])
    assignment = _stratified_split_assignment(labels, train_frac=0.8, val_frac=0.1, seed=0)
    counts = np.bincount(assignment, minlength=3)
    proportions = counts / len(labels)
    np.testing.assert_allclose(proportions, [0.8, 0.1, 0.1], atol=0.02)
