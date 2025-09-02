'''import numpy as np
import pytest
from unittest.mock import patch, MagicMock, mock_open
from axonrooter.train import load_data, run_training


@patch("axonrooter.train.os.listdir")
@patch("axonrooter.train.cv2.imread")
@patch("axonrooter.train.patchify")
def test_load_data_success(mock_patchify, mock_imread, mock_listdir):
    mock_listdir.side_effect = [
        ["img1.png", "img2.png"],  # images
        ["mask1.tif", "mask2.tif"]  # masks
    ]

    # Simulate two grayscale images and two masks
    image = np.ones((256, 256), dtype=np.uint8) * 128
    mask = np.ones((256, 256), dtype=np.uint8)

    # imread should return images in order: image1, mask1, image2, mask2
    mock_imread.side_effect = [image, mask, image, mask]

    def patchify_side_effect(arr, patch_shape, step):
        # Always return a single patch per image, shape (1, 256, 256, 3)
        arr3 = np.stack([arr]*3, axis=-1)
        return np.ones((1, 256, 256, 3), dtype=np.uint8)

    mock_patchify.side_effect = patchify_side_effect

    x, y = load_data("fake/img/dir", "fake/mask/dir", patch_size=256)

    assert x.shape[1:] == (256, 256, 3)
    assert x.shape[0] == 2
    assert y.shape == (2, 256, 256, 1)


@patch("axonrooter.train.os.listdir", return_value=[])
def test_load_data_empty_dir(mock_listdir):
    x, y = load_data("empty/img/dir", "empty/mask/dir", patch_size=256)
    assert x.size == 0
    assert y.size == 0


@patch("axonrooter.train.load_model")
@patch("axonrooter.train.load_data")
@patch("axonrooter.train.EarlyStopping")
@patch("axonrooter.train.os.makedirs")
def test_run_training_success(mock_makedirs, mock_early, mock_loaddata, mock_load_model):
    mock_loaddata.side_effect = [
        (np.random.rand(2, 256, 256, 3), np.random.rand(2, 256, 256, 1)),  # train
        (np.random.rand(1, 256, 256, 3), np.random.rand(1, 256, 256, 1)),  # val
    ]

    mock_model = MagicMock()
    mock_model.fit.return_value = None
    mock_model.save.return_value = None
    mock_load_model.return_value = mock_model

    run_training(
        model_path="fake_model.keras",
        data_root="fake_data",
        model_dir="out_dir",
        patch_size=256,
        batch_size=2,
        epochs=1
    )

    assert mock_model.fit.called
    assert mock_model.save.called


@patch("axonrooter.train.load_data", return_value=(np.array([]), np.array([])))
@patch("axonrooter.train.load_model")
def test_run_training_raises_if_no_data(mock_load_model, mock_loaddata):
    with pytest.raises(ValueError, match="No training or validation data found"):
        run_training(
            model_path="model.keras",
            data_root="no_data",
            model_dir="output",
            patch_size=256,
            batch_size=2,
            epochs=1
        )'''
