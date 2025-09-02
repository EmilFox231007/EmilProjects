#!/usr/bin/env python
"""
Train a patch-based segmentation model using grayscale root images and binary masks.

This script:
- Loads a pre-trained Keras model with a custom F1 metric.
- Loads image and mask data from directory structures.
- Converts images and masks into 256x256 patches.
- Trains the model using early stopping on validation loss.
- Saves the fine-tuned model to the specified directory.

Directory structure expected under --data:
    ├── train_images/
    ├── train_masks/
    ├── val_images/
    └── val_masks/

"""

import argparse
import glob
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.models import load_model
from patchify import patchify


# ----------------------------- Custom F1 Metric -----------------------------
def f1(y_true, y_pred):
    """
    Compute the F1 score between predicted and true labels.

    Parameters
    ----------
    y_true : tensor
        Ground truth binary mask tensor.
    y_pred : tensor
        Predicted binary mask tensor.

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


CUSTOM_OBJECTS = {"f1": f1}


# ----------------------------- Data Loader -----------------------------
def load_data(image_dir, mask_dir, patch_size):
    """
    Load grayscale images and binary masks from directories and convert to patches.

    Parameters
    ----------
    image_dir : str
        Directory containing grayscale root images.
    mask_dir : str
        Directory containing corresponding binary masks.
    patch_size : int
        Size of each patch (e.g. 256).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Arrays of image patches and mask patches.
    """
    x_patches, y_patches = [], []

    # Collect matching image and mask file paths
    image_files = sorted(
        [
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif"))
        ]
    )
    mask_files = sorted(
        [
            os.path.join(mask_dir, f)
            for f in os.listdir(mask_dir)
            if f.lower().endswith(".tif")  # Adjust if masks may be in other formats
        ]
    )

    pair_count = min(len(image_files), len(mask_files))
    if pair_count == 0:
        print("[WARN] No image-mask pairs found for training.")

    for i in range(pair_count):
        img_path = image_files[i]
        mask_path = mask_files[i]

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        root_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None or root_mask is None:
            print(
                f"[WARN] Skipping file pair due to load failure:\n  image: {img_path}\n  mask: {mask_path}"
            )
            continue

        image_rgb = np.stack([image] * 3, axis=-1).astype(np.float32) / 255.0
        combined_mask = (root_mask > 0).astype(np.float32)

        print(f"[INFO] Image shape: {image_rgb.shape}, Mask shape: {combined_mask.shape}")
        print(f"[DEBUG] Mask pixel values: {np.unique(combined_mask)}")
        print(f"[DEBUG] Visible mask pixels: {np.sum(combined_mask)}")

        image_patches = patchify(image_rgb, (patch_size, patch_size, 3), step=patch_size)
        mask_patches = patchify(combined_mask, (patch_size, patch_size), step=patch_size)

        for i in range(image_patches.shape[0]):
            for j in range(image_patches.shape[1]):
                img_patch = image_patches[i, j, 0]
                msk_patch = mask_patches[i, j]

                x_patches.append(img_patch)
                y_patches.append(np.expand_dims(msk_patch, axis=-1))

    return np.array(x_patches), np.array(y_patches)


# ----------------------------- Training -----------------------------
def run_training(model_path, data_root, model_dir, patch_size, batch_size, epochs):
    """
    Load and fine-tune a pre-trained model on root segmentation data.

    Parameters
    ----------
    model_path : str
        Path to the pre-trained `.keras` model.
    data_root : str
        Root directory containing training and validation images/masks.
    model_dir : str
        Directory to save the trained model.
    patch_size : int
        Size of the patches (e.g., 256).
    batch_size : int
        Batch size for training.
    epochs : int
        Maximum number of epochs to train.

    Raises
    ------
    ValueError
        If no training or validation data is found.
    """
    print("[INFO] Loading model...")
    model = load_model(model_path, custom_objects=CUSTOM_OBJECTS)

    train_img_dir = os.path.join(data_root, "train_images")
    train_mask_dir = os.path.join(data_root, "train_masks")
    val_img_dir = os.path.join(data_root, "val_images")
    val_mask_dir = os.path.join(data_root, "val_masks")

    print("[INFO] Loading and processing training data...")
    x_train, y_train = load_data(train_img_dir, train_mask_dir, patch_size)
    print(f"[INFO] Training samples: {len(x_train)}")

    print("[INFO] Loading and processing validation data...")
    x_val, y_val = load_data(val_img_dir, val_mask_dir, patch_size)
    print(f"[INFO] Validation samples: {len(x_val)}")

    if len(x_train) == 0 or len(x_val) == 0:
        raise ValueError(
            "[ERROR] No training or validation data found. Check paths and mask structure."
        )

    early = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)

    print("[INFO] Starting training...")
    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[early],
    )

    os.makedirs(model_dir, exist_ok=True)
    model.save(os.path.join(model_dir, "model.keras"), save_format="keras")
    print(f"[INFO] Model saved to {model_dir}")


# ----------------------------- CLI -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train on raw images and combined masks")
    parser.add_argument("--pretrained-model", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    run_training(
        model_path=args.pretrained_model,
        data_root=args.data_root,
        model_dir=args.model_dir,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )


# test CI/CD triggerrr
