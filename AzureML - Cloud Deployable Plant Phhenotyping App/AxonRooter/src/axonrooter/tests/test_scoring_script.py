import base64
import json
import os
import tempfile
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
from PIL import Image
import axonrooter.scoring as scoring


# ---------- Fixtures ----------

@pytest.fixture
def dummy_grayscale_image_base64():
    """Create a dummy grayscale image and return it as base64 string."""
    img = Image.fromarray(np.ones((128, 128), dtype=np.uint8) * 255)
    buffer = tempfile.SpooledTemporaryFile()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode("utf-8")
    return encoded


# ---------- init() tests ----------

def test_init_loads_model_successfully(monkeypatch):
    mock_model = MagicMock()
    dummy_path = tempfile.mkdtemp()
    keras_model_path = os.path.join(dummy_path, "model.keras")
    open(keras_model_path, "w").close()  # simulate model file

    monkeypatch.setenv("AZUREML_MODEL_DIR", dummy_path)
    monkeypatch.setattr(scoring.tf.keras.models, "load_model", lambda *args, **kwargs: mock_model)

    scoring.init()
    assert scoring.model is not None


def test_init_raises_if_no_model(monkeypatch):
    monkeypatch.setenv("AZUREML_MODEL_DIR", tempfile.mkdtemp())
    with pytest.raises(FileNotFoundError):
        scoring.init()


# ---------- run() tests ----------

@patch("axonrooter.scoring.model")
def test_run_success(mock_model, dummy_grayscale_image_base64):
    # Mock prediction shape: 1 patch => 1 prediction (1, 256, 256, 1)
    mock_pred = np.ones((1, 256, 256, 1), dtype=np.float32)
    mock_model.predict.return_value = mock_pred

    input_json = json.dumps({"data": dummy_grayscale_image_base64})
    result_json = scoring.run(input_json)
    result = json.loads(result_json)

    assert "predicted_mask_base64" in result
    decoded = base64.b64decode(result["predicted_mask_base64"])
    assert decoded.startswith(b'\x89PNG')  # PNG magic number


@patch("axonrooter.scoring.model")
def test_run_with_invalid_base64_returns_error(mock_model):
    bad_input = json.dumps({"data": "this_is_not_base64"})
    result_json = scoring.run(bad_input)
    result = json.loads(result_json)

    assert "error" in result
    assert "Incorrect padding" in result["error"] or "base64" in result["error"]


def test_run_with_missing_key_returns_error():
    bad_input = json.dumps({"wrong_key": "value"})
    result_json = scoring.run(bad_input)
    result = json.loads(result_json)

    assert "error" in result
    assert "data" in result["error"] or "KeyError" in result["error"]
