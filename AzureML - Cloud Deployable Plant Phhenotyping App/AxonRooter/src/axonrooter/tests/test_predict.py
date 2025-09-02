import pytest
import numpy as np
import cv2
from unittest.mock import patch, MagicMock
from axonrooter.predict import predict_root_tips


@patch("axonrooter.predict.os.path.exists", return_value=False)
def test_file_not_found(mock_exists):
    with pytest.raises(FileNotFoundError):
        predict_root_tips("fake_path.png")


@patch("axonrooter.predict.os.path.exists", return_value=True)
@patch("axonrooter.predict.cv2.imread", return_value=None)
def test_image_load_failure(mock_imread, mock_exists):
    with pytest.raises(ValueError):
        predict_root_tips("bad_image.png")


@patch("axonrooter.predict.os.path.exists", return_value=True)
@patch("axonrooter.predict.cv2.imread")
@patch("axonrooter.predict.crop_initial")
@patch("axonrooter.predict.format_crop")
@patch("axonrooter.predict.padding_function")
@patch("axonrooter.predict.run_inference")
@patch("axonrooter.predict.preprocess_single_mask")
@patch("axonrooter.predict.apply_strict_bboxes_with_top_cutoff_and_proximity")
@patch("axonrooter.predict.extract_bottom_tips")
def test_predict_root_tips_success(
    mock_extract,
    mock_bboxes,
    mock_preprocess,
    mock_inference,
    mock_pad,
    mock_format,
    mock_crop,
    mock_imread,
    mock_exists,
):
    mock_imread.return_value = np.ones((100, 100), dtype=np.uint8)
    mock_crop.return_value = np.ones((80, 80), dtype=np.uint8)
    mock_format.return_value = np.ones((64, 64), dtype=np.uint8)
    mock_pad.return_value = (np.ones((64, 64), dtype=np.uint8), 0, 0, 0, 0)
    mock_inference.return_value = np.ones((64, 64), dtype=np.uint8)
    mock_preprocess.return_value = np.ones((64, 64), dtype=np.uint8)
    mock_bboxes.return_value = (np.ones((64, 64), dtype=np.uint8), [(10, 10, 20, 20)])
    mock_extract.return_value = [(15, 63)]

    result = predict_root_tips("mock_image.png")
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0] == (15, 63)
