"""
Script for evaluating a patch-based image segmentation model.

This script loads a pre-trained TensorFlow/Keras model, applies it to a test dataset
of grayscale images and masks, evaluates performance using F1 score and accuracy,
and saves prediction visualizations and metrics. It includes:
- A custom F1 metric for Keras
- A data loader that splits images and masks into non-overlapping patches
- Evaluation logic including metric computation and result saving
"""

import argparse
import glob
import json
import logging
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import backend as K
from patchify import patchify
from sklearn.metrics import accuracy_score, f1_score

from axonrooter.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


# ----------------------------- Custom F1 Metric -----------------------------
def f1(y_true, y_pred):
    """
    Computes the F1 score as a custom Keras metric.

    Parameters
    ----------
    y_true : tensor
        Ground truth labels.
    y_pred : tensor
        Predicted labels.

    Returns
    -------
    tensor
        F1 score.
    """
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    pos = K.sum(K.round(K.clip(y_true, 0, 1)))
    pre = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = tp / (pre + K.epsilon())
    recall = tp / (pos + K.epsilon())
    return 2 * (precision * recall) / (precision + recall + K.epsilon())


# ----------------------------- Data Loader -----------------------------
def load_patched_data_no_filter(image_dir, mask_dir, patch_size=256):
    """
    Loads grayscale images and masks from disk, applies patching, and returns arrays of patches.

    Parameters
    ----------
    image_dir : str
        Directory containing the input images.
    mask_dir : str
        Directory containing the corresponding masks.
    patch_size : int, optional
        Size of the square patches to extract, by default 256.

    Returns
    -------
    tuple of np.ndarray
        A tuple containing:
        - images (np.ndarray): Array of image patches with shape (N, patch_size, patch_size, 3)
        - masks (np.ndarray): Array of mask patches with shape (N, patch_size, patch_size, 1)
    """
    images = []
    masks = []
    image_filenames = sorted(
        [f for f in os.listdir(image_dir) if f.endswith((".png", ".jpg", ".jpeg", ".tif"))]
    )

    for fname in image_filenames:
        base = Path(fname).stem
        image_path = os.path.join(image_dir, fname)

        # FLEXIBLE MASK MATCHING
        mask_matches = glob.glob(os.path.join(mask_dir, base + "*"))
        if not mask_matches:
            logging.warning(f"No matching mask found for {fname}")
            continue
        root_mask_path = mask_matches[0]

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        root_mask = cv2.imread(root_mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None or root_mask is None:
            logging.warning(f"Skipping file due to load failure: {fname}")
            continue

        image = np.stack([image] * 3, axis=-1).astype(np.float32) / 255.0
        combined_mask = (root_mask > 0).astype(np.float32)

        image_patches = patchify(image, (patch_size, patch_size, 3), step=patch_size)
        mask_patches = patchify(combined_mask, (patch_size, patch_size), step=patch_size)

        for i in range(image_patches.shape[0]):
            for j in range(image_patches.shape[1]):
                images.append(image_patches[i, j, 0])
                masks.append(np.expand_dims(mask_patches[i, j], axis=-1))

    return np.array(images), np.array(masks)


# ----------------------------- Evaluation -----------------------------
def evaluate(model_path, test_data_dir, output_path, patch_size=256):
    """
    Loads a model and evaluates it on the test dataset by computing F1 and accuracy scores.

    Parameters
    ----------
    model_path : str
        Path to the trained model directory (expects `model.keras` inside).
    test_data_dir : str
        Directory containing 'test_images' and 'test_masks' subdirectories.
    output_path : str
        Directory where evaluation metrics and prediction visualizations will be saved.
    patch_size : int, optional
        Size of the patches for splitting test images, by default 256.

    Raises
    ------
    ValueError
        If no test patches are found.
    """
    logging.info(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(
        os.path.join(model_path, "model.keras"), custom_objects={"f1": f1}
    )

    test_image_dir = os.path.join(test_data_dir, "test_images")
    test_mask_dir = os.path.join(test_data_dir, "test_masks")

    X_test, y_test = load_patched_data_no_filter(test_image_dir, test_mask_dir, patch_size)

    logging.info(f"Loaded {len(X_test)} test patches.")
    if len(X_test) == 0:
        raise ValueError("No test patches found.")

    y_pred = model.predict(X_test)
    logging.info(f"Model output: min={y_pred.min()}, max={y_pred.max()}, mean={y_pred.mean():.4f}")

    y_pred_binary = (y_pred > 0.5).astype(np.uint8)
    y_true_binary = (y_test > 0.5).astype(np.uint8)

    f1_val = f1_score(y_true_binary.flatten(), y_pred_binary.flatten(), zero_division=0)
    accuracy_val = accuracy_score(y_true_binary.flatten(), y_pred_binary.flatten())

    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, "metrics.json"), "w") as f:
        json.dump({"f1_score": float(f1_val), "accuracy": float(accuracy_val)}, f)

    logging.info(f"F1 score: {f1_val:.4f}")
    logging.info(f"Accuracy: {accuracy_val:.4f}")
    logging.info(f"Saved evaluation results to {output_path}/metrics.json")

    # Save visualizations
    vis_dir = os.path.join(output_path, "pred_visuals")
    os.makedirs(vis_dir, exist_ok=True)
    for i in range(min(5, len(X_test))):
        plt.imsave(os.path.join(vis_dir, f"pred_patch_{i}.png"), y_pred[i].squeeze(), cmap="gray")
        plt.imsave(os.path.join(vis_dir, f"true_patch_{i}.png"), y_test[i].squeeze(), cmap="gray")


# ----------------------------- CLI -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--test_data_dir", type=str, required=True)
    parser.add_argument("--accuracy_path", type=str, required=True)
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument(
        "--use_uri", action="store_true", help="Unused placeholder for compatibility"
    )

    args = parser.parse_args()

    evaluate(
        model_path=args.model_path,
        test_data_dir=args.test_data_dir,
        output_path=args.accuracy_path,
        patch_size=args.patch_size,
    )
