from unittest.mock import patch

import matplotlib
import numpy as np
import pytest

from axonrooter.models.inference import run_inference

matplotlib.use("Agg")


def test_run_inference_output_shape_and_type():
    PATCH_SIZE = 256
    padded_input = np.ones((PATCH_SIZE, PATCH_SIZE), dtype=np.uint8)
    pad_top = pad_bottom = pad_left = pad_right = 0

    with patch("axonrooter.models.inference.patchify") as mock_patchify, patch(
        "axonrooter.models.inference.unpatchify"
    ) as mock_unpatchify, patch("axonrooter.models.inference.MODEL") as mock_model, patch(
        "matplotlib.pyplot.show"
    ):
        # patchify returns 1 patch of the right shape
        mock_patchify.return_value = np.ones((1, 1, PATCH_SIZE, PATCH_SIZE, 3))
        # model returns an array like the output of predict
        mock_model.predict.return_value = np.ones((1, PATCH_SIZE, PATCH_SIZE, 1))
        # unpatchify returns a mask of the right shape
        mock_unpatchify.return_value = np.ones((PATCH_SIZE, PATCH_SIZE), dtype=np.uint8)

        output = run_inference(padded_input, pad_top, pad_bottom, pad_left, pad_right)
        mask = output if isinstance(output, np.ndarray) else output[0]
        assert isinstance(mask, np.ndarray)
        assert mask.shape == (PATCH_SIZE, PATCH_SIZE)


def test_run_inference_handles_padding():
    PATCH_SIZE = 256
    padded_input = np.ones((PATCH_SIZE * 2, PATCH_SIZE * 2), dtype=np.uint8)
    pad_top = pad_bottom = pad_left = pad_right = 16

    with patch("axonrooter.models.inference.patchify") as mock_patchify, patch(
        "axonrooter.models.inference.unpatchify"
    ) as mock_unpatchify, patch("axonrooter.models.inference.MODEL") as mock_model, patch(
        "matplotlib.pyplot.show"
    ):
        mock_patchify.return_value = np.ones((2, 2, PATCH_SIZE, PATCH_SIZE, 3))
        mock_model.predict.return_value = np.ones((4, PATCH_SIZE, PATCH_SIZE, 1))
        mock_unpatchify.return_value = np.ones((PATCH_SIZE * 2, PATCH_SIZE * 2), dtype=np.uint8)
        output = run_inference(padded_input, pad_top, pad_bottom, pad_left, pad_right)
        # Should have padding cropped from each side
        expected_shape = (
            PATCH_SIZE * 2 - pad_top - pad_bottom,
            PATCH_SIZE * 2 - pad_left - pad_right,
        )
        if isinstance(output, tuple):
            mask = output[0]
        else:
            mask = output
        assert mask.shape == expected_shape


def test_run_inference_runs_with_non_square_input():
    PATCH_SIZE = 256
    padded_input = np.ones((PATCH_SIZE, PATCH_SIZE * 2), dtype=np.uint8)
    pad_top = pad_bottom = pad_left = pad_right = 0

    with patch("axonrooter.models.inference.patchify") as mock_patchify, patch(
        "axonrooter.models.inference.unpatchify"
    ) as mock_unpatchify, patch("axonrooter.models.inference.MODEL") as mock_model, patch(
        "matplotlib.pyplot.show"
    ):
        # 1 row, 2 columns of patches
        mock_patchify.return_value = np.ones((1, 2, PATCH_SIZE, PATCH_SIZE, 3))
        mock_model.predict.return_value = np.ones((2, PATCH_SIZE, PATCH_SIZE, 1))
        mock_unpatchify.return_value = np.ones((PATCH_SIZE, PATCH_SIZE * 2), dtype=np.uint8)

        # ✅ Unpack the output from run_inference
        output_mask, _ = run_inference(padded_input, pad_top, pad_bottom, pad_left, pad_right)

        assert output_mask.shape == (PATCH_SIZE, PATCH_SIZE * 2)


def test_run_inference_plotting_does_not_crash():
    PATCH_SIZE = 256
    padded_input = np.ones((PATCH_SIZE, PATCH_SIZE), dtype=np.uint8)
    pad_top = pad_bottom = pad_left = pad_right = 0

    with patch("axonrooter.models.inference.patchify") as mock_patchify, patch(
        "axonrooter.models.inference.unpatchify"
    ) as mock_unpatchify, patch("axonrooter.models.inference.MODEL") as mock_model, patch(
        "matplotlib.pyplot.show"
    ):
        mock_patchify.return_value = np.ones((1, 1, PATCH_SIZE, PATCH_SIZE, 3))
        mock_model.predict.return_value = np.ones((1, PATCH_SIZE, PATCH_SIZE, 1))
        mock_unpatchify.return_value = np.ones((PATCH_SIZE, PATCH_SIZE), dtype=np.uint8)
        run_inference(padded_input, pad_top, pad_bottom, pad_left, pad_right)


def test_run_inference_invalid_input_raises():
    # For example, a string input should raise an error
    with pytest.raises(Exception):
        run_inference("not an array", 0, 0, 0, 0)
