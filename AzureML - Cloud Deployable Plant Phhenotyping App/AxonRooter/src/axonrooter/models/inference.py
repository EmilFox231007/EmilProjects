"""
This module handles inference on padded grayscale input images of the seedlings
using a pre-trained U-Net model.
The image is split into patches, normalized, predicted, and reassembled.
Postprocessing includes binarization and removal of padding.
"""

import logging
import os
import tempfile
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    load_dotenv()
    logging.info("Environment variables loaded from .env file")
except ImportError:
    logging.warning(
        "python-dotenv not available. Environment variables from .env file will not be loaded."
    )

import numpy as np
from keras.models import load_model
from patchify import patchify, unpatchify

# === Logging Setup ===
from axonrooter.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

# === Constants ===
PATCH_SIZE = 256
MODEL_PATH = os.path.join(os.path.dirname(__file__), "michal_232079_unet_model_v3_256px.h5")

# === Azure ML Integration ===
try:
    from azure.ai.ml import MLClient
    from azure.identity import ClientSecretCredential

    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False
    logger.warning("Azure ML SDK not available. Only local model loading supported.")

# === Global Model Variable ===
MODEL = None
MODEL_INFO = {"source": "none", "name": "none", "version": "none"}


def load_model_from_azure():
    """
    Load the latest model from Azure Machine Learning workspace.

    Retrieves Azure credentials from environment variables, connects to the
    Azure ML workspace, finds the latest version of the specified model,
    downloads it locally, and loads it into memory using TensorFlow/Keras.

    Returns
    -------
    model : tensorflow.keras.Model or None
        The loaded Keras model instance if successful, or None if any error occurs.

    Notes
    -----
    - Supports loading models in .keras, TensorFlow SavedModel, or .h5 formats.
    - Requires Azure ML SDK and Azure credentials set in environment variables:
      AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP, AZURE_WORKSPACE_NAME,
      AZURE_TENANT_ID, AZURE_CLIENT_ID, AZURE_CLIENT_SECRET.
    - Logs progress and errors via the configured logger.
    """
    if not AZURE_AVAILABLE:
        logger.error("Azure ML SDK not available")
        return None

    try:
        # Get Azure credentials from environment
        subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
        resource_group = os.getenv("AZURE_RESOURCE_GROUP")
        workspace_name = os.getenv("AZURE_WORKSPACE_NAME")
        tenant_id = os.getenv("AZURE_TENANT_ID")
        client_id = os.getenv("AZURE_CLIENT_ID")
        client_secret = os.getenv("AZURE_CLIENT_SECRET")
        model_name = os.getenv("AZURE_MODEL_NAME", "axonrooter-finetuned")

        if not all(
            [subscription_id, resource_group, workspace_name, tenant_id, client_id, client_secret]
        ):
            logger.error("Missing Azure credentials in environment variables")
            return None

        # Create Azure ML client
        credential = ClientSecretCredential(tenant_id, client_id, client_secret)
        ml_client = MLClient(credential, subscription_id, resource_group, workspace_name)

        # Get latest model
        models = list(ml_client.models.list(name=model_name))
        if not models:
            logger.error(f"No models found with name: {model_name}")
            return None

        latest_model = max(models, key=lambda m: int(m.version))
        logger.info(f"Found latest model: {model_name} version {latest_model.version}")

        # Download model to temp directory
        temp_dir = Path(tempfile.gettempdir()) / "azure_models" / model_name
        temp_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Downloading model to: {temp_dir}")
        ml_client.models.download(
            name=model_name, version=latest_model.version, download_path=str(temp_dir)
        )

        # Find the model file - Updated logic to handle nested Azure ML structure
        model_file = None

        # First, let's search for any model files recursively and log what we find
        logger.info(f"Contents of download directory: {list(temp_dir.rglob('*'))}")

        # Check for .keras files first (newest format)
        keras_files = list(temp_dir.rglob("*.keras"))
        if keras_files:
            model_file = keras_files[0]
            logger.info(f"Found .keras model file at: {model_file}")
        else:
            # Check for TensorFlow SavedModel format
            possible_savedmodel_paths = [
                temp_dir / "INPUT_model",  # Direct path
                temp_dir / model_name / "INPUT_model",  # With model name subdirectory
            ]

            for savedmodel_path in possible_savedmodel_paths:
                if savedmodel_path.exists() and (savedmodel_path / "saved_model.pb").exists():
                    model_file = savedmodel_path
                    logger.info(f"Found TensorFlow SavedModel format at: {model_file}")
                    break

            if not model_file:
                # Check for .h5 format (older models)
                h5_files = list(temp_dir.rglob("*.h5"))
                if h5_files:
                    model_file = h5_files[0]
                    logger.info(f"Found .h5 model file at: {model_file}")

        if not model_file:
            logger.error(f"Model file not found in {temp_dir}")
            return None

        # Load the model - Updated to handle .keras files
        logger.info(f"Loading model from: {model_file}")

        try:
            if model_file.suffix == ".keras":
                # .keras format (newest TensorFlow format)
                import tensorflow as tf

                try:
                    model = tf.keras.models.load_model(str(model_file), compile=False)
                    logger.info("Loaded model in .keras format (without compilation)")
                except Exception as e:
                    logger.warning(f"Failed to load .keras model without compilation: {e}")
                    # Try with custom objects as fallback
                    model = tf.keras.models.load_model(
                        str(model_file), custom_objects={"f1": lambda y_true, y_pred: y_pred}
                    )
                    logger.info("Loaded model in .keras format (with custom objects)")
            elif model_file.name == "INPUT_model" or (model_file / "saved_model.pb").exists():
                # TensorFlow SavedModel format
                import tensorflow as tf

                try:
                    model = tf.keras.models.load_model(str(model_file), compile=False)
                    logger.info("Loaded model in TensorFlow SavedModel format (without compilation)")
                except Exception as e:
                    logger.warning(f"Failed to load SavedModel without compilation: {e}")
                    # Try with custom objects as fallback
                    model = tf.keras.models.load_model(
                        str(model_file), custom_objects={"f1": lambda y_true, y_pred: y_pred}
                    )
                    logger.info("Loaded model in TensorFlow SavedModel format (with custom objects)")
            else:
                # .h5 format
                model = load_model(
                    str(model_file), custom_objects={"f1": lambda y_true, y_pred: y_pred}
                )
                logger.info("Loaded model in .h5 format")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return None

        global MODEL_INFO
        MODEL_INFO = {
            "source": "azure",
            "name": model_name,
            "version": latest_model.version,
            "path": str(model_file),
        }

        logger.info(f"Successfully loaded Azure model: {model_name} v{latest_model.version}")
        return model

    except Exception as e:
        logger.error(f"Failed to load model from Azure: {e}")
        return None


def load_local_model():
    """Load model from local file"""
    try:
        if not os.path.exists(MODEL_PATH):
            logger.error(f"Local model not found: {MODEL_PATH}")
            return None

        logger.info(f"Loading local model from: {MODEL_PATH}")
        model = load_model(MODEL_PATH, custom_objects={"f1": lambda y_true, y_pred: y_pred})

        global MODEL_INFO
        MODEL_INFO = {
            "source": "local",
            "name": "local_model",
            "version": "local",
            "path": MODEL_PATH,
        }

        logger.info("Successfully loaded local model")
        return model

    except Exception as e:
        logger.error(f"Failed to load local model: {e}")
        return None


def initialize_model():
    """Initialize model - try Azure first, then local"""
    global MODEL

    if MODEL is not None:
        return MODEL

    # Try Azure first if enabled
    use_azure = os.getenv("USE_AZURE_MODEL", "true").lower() == "true"
    if use_azure:
        logger.info("Attempting to load model from Azure ML...")
        MODEL = load_model_from_azure()

    # Fallback to local model
    if MODEL is None:
        logger.info("Loading local model as fallback...")
        MODEL = load_local_model()

    if MODEL is None:
        logger.error("Failed to load any model!")
        raise RuntimeError("No model could be loaded")

    return MODEL


def get_model():
    """Get the current model, loading if necessary"""
    if MODEL is None:
        return initialize_model()
    return MODEL


def get_model_info():
    """Get information about the current model"""
    return MODEL_INFO.copy()


def refresh_model():
    """Refresh model from Azure (if Azure is enabled)"""
    global MODEL
    use_azure = os.getenv("USE_AZURE_MODEL", "false").lower() == "true"

    if not use_azure:
        logger.info("Azure model loading not enabled")
        return False

    logger.info("Refreshing model from Azure...")
    new_model = load_model_from_azure()

    if new_model is not None:
        MODEL = new_model
        logger.info("Model refreshed successfully")
        return True
    else:
        logger.error("Failed to refresh model")
        return False


# === Initialize model on import ===
try:
    initialize_model()
    logger.info(f"Model initialized: {MODEL_INFO}")
except Exception as e:
    logger.error(f"Failed to initialize model on import: {e}")


def run_inference(padded_input, pad_top, pad_bottom, pad_left, pad_right, remove_padding=True):
    """
    Run inference on padded input image.

    Parameters:
    -----------
    padded_input : numpy.ndarray
        Padded grayscale input image
    pad_top, pad_bottom, pad_left, pad_right : int
        Padding values used to create the padded image
    remove_padding : bool
        Whether to remove padding from the result (default: True)

    Returns:
    --------
    numpy.ndarray
        Predicted mask, either padded or unpadded based on remove_padding parameter
    """
    logger.info("Starting inference...")

    # Get model
    model = get_model()
    if model is None:
        raise RuntimeError("No model available for inference")

    padded_input = np.repeat(padded_input[..., np.newaxis], 3, axis=-1) / 255.0
    logger.info(f"Input shape after RGB repeat: {padded_input.shape}")

    patches = patchify(padded_input, (PATCH_SIZE, PATCH_SIZE, 3), step=PATCH_SIZE)
    logger.info(f"Patches shape: {patches.shape}")

    patches_flat = patches.reshape(-1, PATCH_SIZE, PATCH_SIZE, 3)
    logger.info(f"Flattened patches shape: {patches_flat.shape}")

    # === Run Inference ===
    preds = model.predict(patches_flat, verbose=0)
    logger.info(f"Prediction completed on {len(patches_flat)} patches.")

    preds_bin = (
        (preds > 0.5)
        .astype(np.uint8)
        .reshape(patches.shape[0], patches.shape[1], PATCH_SIZE, PATCH_SIZE)
    )

    pred_mask = unpatchify(preds_bin, padded_input.shape[:2])
    logger.info(f"Mask shape after unpatchify: {pred_mask.shape}")

    if remove_padding:
        # === Remove Padding ===
        row_end = -pad_bottom if pad_bottom != 0 else None
        col_end = -pad_right if pad_right != 0 else None
        predicted_mask = pred_mask[pad_top:row_end, pad_left:col_end]
        logger.info(
            f"Removed padding. Final mask shape: {predicted_mask.shape} "
            f"(Top: {pad_top}, Bottom: {pad_bottom}, Left: {pad_left}, Right: {pad_right})"
        )
        return predicted_mask, MODEL_INFO["source"]
    else:
        logger.info(f"Keeping padding. Final mask shape: {pred_mask.shape}")
        return pred_mask, MODEL_INFO["source"]
