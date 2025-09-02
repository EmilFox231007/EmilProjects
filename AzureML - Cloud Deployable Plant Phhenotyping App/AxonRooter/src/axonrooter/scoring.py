"""
Azure ML-compatible script to load a Keras image segmentation model and serve predictions.

This script performs the following:
1. Loads a `.keras` model with a custom F1 metric from the environment path.
2. Accepts a base64-encoded grayscale image.
3. Pads, normalizes, and splits the image into 256x256 RGB patches.
4. Uses the model to predict segmentation masks per patch.
5. Reconstructs the full-sized binary mask from patches and removes padding.
6. Returns a base64-encoded PNG mask.
"""

import base64
import io
import json
import os

import numpy as np
from patchify import patchify, unpatchify
from PIL import Image
from tensorflow.keras.models import load_model


# Custom F1 function used in training
def f1(y_true, y_pred):
    """
    Compute F1 score metric used during model training.

    Parameters
    ----------
    y_true : tensor
        Ground truth binary labels.
    y_pred : tensor
        Predicted binary labels.

    Returns
    -------
    tensor
        Computed F1 score.
    """
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    pos = K.sum(K.round(K.clip(y_true, 0, 1)))
    pre = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = tp / (pre + K.epsilon())
    recall = tp / (pos + K.epsilon())
    return 2 * (precision * recall) / (precision + recall + K.epsilon())


# Global model variable
import tensorflow as tf
from tensorflow.keras import backend as K

PATCH_SIZE = 256
model = None


def init():
    """
    Initialize the global segmentation model by loading a `.keras` file
    from the AzureML model directory specified in the `AZUREML_MODEL_DIR`
    environment variable.

    Raises
    ------
    FileNotFoundError
        If no model file is found.
    RuntimeError
        If model loading fails.
    """
    global model

    base_path = os.getenv("AZUREML_MODEL_DIR", ".")
    print(f"[INFO] AZUREML_MODEL_DIR: {base_path}")

    model_extensions = [".keras"]
    model_path = None

    for root, dirs, files in os.walk(base_path):
        for file in files:
            if any(file.endswith(ext) for ext in model_extensions):
                potential_path = os.path.join(root, file)
                print(f"[INFO] Found potential model file: {potential_path}")

                if file.endswith(".keras"):
                    model_path = potential_path
                    break
        if model_path:
            break

    if model_path:
        try:
            model = tf.keras.models.load_model(model_path, custom_objects={"f1": f1})
            print(f"[INFO] Model loaded successfully from: {model_path}")
        except Exception as e:
            print(f"[ERROR] Failed to load model from {model_path}: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
    else:
        raise FileNotFoundError("No .keras model file found in AZUREML_MODEL_DIR.")


def run(raw_data):
    """
    Run segmentation inference on a base64-encoded grayscale image.

    Parameters
    ----------
    raw_data : str
        JSON-formatted string with a single key `"data"` containing a base64-encoded image.

    Returns
    -------
    str
        JSON-formatted string with the key `"predicted_mask_base64"` or `"error"` if failed.
    """
    try:
        print("[INFO] run() called")
        input_data = json.loads(raw_data)
        base64_image = input_data["data"]

        # Decode image
        image_bytes = base64.b64decode(base64_image)
        image = Image.open(io.BytesIO(image_bytes)).convert("L")
        image_np = np.array(image, dtype=np.uint8)
        print(f"[INFO] Original image shape: {image_np.shape}")

        # Pad image to fit 256x256
        height, width = image_np.shape
        pad_bottom = (PATCH_SIZE - height % PATCH_SIZE) % PATCH_SIZE
        pad_right = (PATCH_SIZE - width % PATCH_SIZE) % PATCH_SIZE
        pad_top, pad_left = 0, 0

        padded = np.pad(
            image_np,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=0,
        )
        print(f"[INFO] Padded shape: {padded.shape}")

        # Convert to RGB and normalize
        rgb_input = np.repeat(padded[..., np.newaxis], 3, axis=-1) / 255.0

        # Patchify
        patches = patchify(rgb_input, (PATCH_SIZE, PATCH_SIZE, 3), step=PATCH_SIZE)
        patches_flat = patches.reshape(-1, PATCH_SIZE, PATCH_SIZE, 3)

        # Predict
        preds = model.predict(patches_flat, verbose=0)
        preds_bin = (
            (preds > 0.5)
            .astype(np.uint8)
            .reshape(patches.shape[0], patches.shape[1], PATCH_SIZE, PATCH_SIZE)
        )
        mask = unpatchify(preds_bin, padded.shape[:2])

        # Remove padding
        row_end = -pad_bottom if pad_bottom != 0 else None
        col_end = -pad_right if pad_right != 0 else None
        final_mask = mask[pad_top:row_end, pad_left:col_end]
        print(f"[INFO] Final mask shape: {final_mask.shape}")

        # Convert to PNG and base64
        out_img = Image.fromarray(final_mask * 255)
        buffer = io.BytesIO()
        out_img.save(buffer, format="PNG")
        encoded_mask = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return json.dumps({"predicted_mask_base64": encoded_mask})

    except Exception as e:
        print("[ERROR] Exception occurred in run():", str(e))
        return json.dumps({"error": str(e)})
