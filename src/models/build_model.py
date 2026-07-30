"""Model architectures: a from-scratch baseline CNN and an EfficientNetB0 transfer
learning model, used to demonstrate the value transfer learning adds on a
low-data, fine-grained classification problem.
"""

from tensorflow import keras
from tensorflow.keras import layers

from src.config import IMG_SIZE, NUM_CLASSES
from src.data.augmentation import build_augmentation_layer


def build_baseline_cnn(num_classes=NUM_CLASSES, img_size=IMG_SIZE):
    """A small CNN trained from scratch. Serves as a baseline to quantify how
    much transfer learning improves accuracy on this ~6,551-image training set.
    """
    inputs = keras.Input(shape=(img_size, img_size, 3))
    x = build_augmentation_layer()(inputs)
    x = layers.Rescaling(1.0 / 255)(x)

    for filters in (32, 64, 128, 256):
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs, name="baseline_cnn")


def build_transfer_model(num_classes=NUM_CLASSES, img_size=IMG_SIZE, dropout=0.3):
    """EfficientNetB0 backbone pretrained on ImageNet, with a new classification
    head for the 102 flower classes. The backbone starts fully frozen; use
    `unfreeze_top_layers` for the fine-tuning stage.

    EfficientNetB0 expects raw [0, 255] float input — its Keras implementation
    includes an internal Rescaling/Normalization step, so no separate
    preprocess_input call is needed here.
    """
    base_model = keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(img_size, img_size, 3),
        pooling=None,
    )
    base_model.trainable = False

    inputs = keras.Input(shape=(img_size, img_size, 3))
    x = build_augmentation_layer()(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="efficientnetb0_transfer")
    return model, base_model


def unfreeze_top_layers(base_model, num_layers):
    """Unfreeze the top `num_layers` layers of the backbone for fine-tuning.

    BatchNormalization layers are kept frozen even when "unfrozen" by depth:
    with 102 imbalanced classes and as few as ~32 training images for the
    rarest one, updating BN running statistics during fine-tuning is still a
    small-sample-size risk, so this project keeps that part conservative.
    """
    base_model.trainable = True
    freeze_until = len(base_model.layers) - num_layers
    for layer in base_model.layers[:freeze_until]:
        layer.trainable = False
    for layer in base_model.layers[freeze_until:]:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
