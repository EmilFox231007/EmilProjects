import os
import tempfile
import numpy as np
import cv2
import pytest
import tensorflow as tf
from tensorflow.keras import layers, models
from axonrooter.evaluate import evaluate, load_patched_data_no_filter


def create_dummy_model(save_path, input_shape=(256, 256, 3)):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=[])
    model.save(os.path.join(save_path, "model.keras"))


@pytest.fixture
def dummy_test_data(tmp_path):
    # Create dirs
    test_dir = tmp_path
    image_dir = test_dir / "test_images"
    mask_dir = test_dir / "test_masks"
    image_dir.mkdir()
    mask_dir.mkdir()

    # Create dummy image and mask
    image = (np.random.rand(512, 512) * 255).astype(np.uint8)
    mask = (np.random.rand(512, 512) > 0.5).astype(np.uint8) * 255

    image_path = image_dir / "sample_01.png"
    mask_path = mask_dir / "sample_01_mask.png"

    cv2.imwrite(str(image_path), image)
    cv2.imwrite(str(mask_path), mask)

    return test_dir


def test_evaluate_runs(dummy_test_data, tmp_path):
    # Arrange
    model_dir = tmp_path / "model_dir"
    model_dir.mkdir()
    create_dummy_model(str(model_dir))

    output_dir = tmp_path / "results"

    # Act
    evaluate(
        model_path=str(model_dir),
        test_data_dir=str(dummy_test_data),
        output_path=str(output_dir),
        patch_size=256,
    )

    # Assert
    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "pred_visuals").is_dir()


def test_load_patched_data_no_filter_returns_expected_shapes(dummy_test_data):
    img_dir = os.path.join(dummy_test_data, "test_images")
    mask_dir = os.path.join(dummy_test_data, "test_masks")

    X, y = load_patched_data_no_filter(img_dir, mask_dir, patch_size=256)
    assert X.shape[0] > 0
    assert X.shape[1:] == (256, 256, 3)
    assert y.shape[1:] == (256, 256, 1)


def test_evaluate_raises_on_empty_dataset(tmp_path):
    # Arrange: model and empty dataset
    model_dir = tmp_path / "model_dir"
    model_dir.mkdir()
    create_dummy_model(str(model_dir))

    empty_test_dir = tmp_path / "empty_data"
    (empty_test_dir / "test_images").mkdir(parents=True)
    (empty_test_dir / "test_masks").mkdir(parents=True)

    with pytest.raises(ValueError, match="No test patches found"):
        evaluate(
            model_path=str(model_dir),
            test_data_dir=str(empty_test_dir),
            output_path=str(tmp_path / "results"),
            patch_size=256,
        )
