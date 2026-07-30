from src.models.build_model import build_baseline_cnn, build_transfer_model, unfreeze_top_layers


def test_baseline_cnn_output_shape():
    model = build_baseline_cnn(num_classes=10, img_size=64)
    assert model.output_shape == (None, 10)


def test_transfer_model_output_shape_and_frozen_backbone():
    model, base_model = build_transfer_model(num_classes=10, img_size=96)
    assert model.output_shape == (None, 10)
    assert base_model.trainable is False


def test_unfreeze_top_layers_only_unfreezes_requested_depth():
    _, base_model = build_transfer_model(num_classes=10, img_size=96)
    unfreeze_top_layers(base_model, num_layers=5)
    assert base_model.trainable is True
    trainable_count = sum(1 for layer in base_model.layers if layer.trainable)
    assert trainable_count <= 5
