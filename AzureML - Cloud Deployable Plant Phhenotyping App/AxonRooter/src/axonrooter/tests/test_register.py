'''import pytest
import json
import sys
from unittest.mock import patch, mock_open, MagicMock

# Mock Azure modules early
sys.modules["azure.ai.ml"] = MagicMock()
sys.modules["azure.ai.ml.entities"] = MagicMock()
sys.modules["azure.identity"] = MagicMock()
sys.modules["azureml.core"] = MagicMock()

from axonrooter.register import register_model_if_accuracy_good


@patch("azure.ai.ml.MLClient")
@patch("azure.identity.ClientSecretCredential")
@patch("axonrooter.register.subprocess.check_call")
def test_register_with_valid_metrics(mock_check_call, mock_credential, mock_mlclient):
    # Fully control the model registration mock chain
    mock_create_or_update = MagicMock()
    mock_models = MagicMock(create_or_update=mock_create_or_update)
    mock_instance = MagicMock(models=mock_models)
    mock_mlclient.return_value = mock_instance

    m = mock_open(read_data='{"f1_score": 0.85}')
    with patch("builtins.open", m), patch("os.path.exists", return_value=True):
        from axonrooter.register import register_model_if_accuracy_good
        register_model_if_accuracy_good(
            model_path="some/model/path",
            accuracy_path="metrics",
            tags_json='{"source": "test"}',
            threshold=0.01,
        )

    mock_create_or_update.assert_called_once()
    assert m.called


@patch("axonrooter.register.os.path.exists", return_value=False)
def test_missing_metrics_file(mock_path_exists):
    with pytest.raises(FileNotFoundError):
        register_model_if_accuracy_good("some/model/path", "missing/path")


@patch("axonrooter.register.os.path.exists", return_value=True)
@patch("axonrooter.register.open", new_callable=mock_open, read_data="{bad json}")
def test_corrupted_metrics_file(mock_open_file, mock_path_exists):
    with pytest.raises(json.JSONDecodeError):
        register_model_if_accuracy_good("some/model/path", "some/metrics/path")


@patch("azure.ai.ml.MLClient")
@patch("azure.identity.ClientSecretCredential")
def test_register_with_no_metrics_keys(mock_credential, mock_mlclient):
    # Mock full MLClient chain
    mock_create_or_update = MagicMock()
    mock_models = MagicMock(create_or_update=mock_create_or_update)
    mock_instance = MagicMock(models=mock_models)
    mock_mlclient.return_value = mock_instance

    m = mock_open(read_data="{}")
    with patch("builtins.open", m), patch("os.path.exists", return_value=True):
        from axonrooter.register import register_model_if_accuracy_good
        register_model_if_accuracy_good("some/model/path", "metrics", tags_json="{}")

    mock_create_or_update.assert_called_once()
    assert m.called'''

