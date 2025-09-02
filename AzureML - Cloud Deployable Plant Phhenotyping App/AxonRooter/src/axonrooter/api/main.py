import base64
import io
import logging

import numpy as np
from fastapi import FastAPI, File, UploadFile
from PIL import Image

from axonrooter import (
    apply_strict_bboxes,
    extract_bottom_tips,
    load_and_preprocess_image,
    preprocess_closing,
    run_inference,
)

# === Logging Setup ===
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# === NumPy to Python Converter ===
# Convert all tips to regular Python ints or lists
def convert_numpy(obj):
    """
    Recursively convert NumPy data types to native Python data types.

    Parameters
    ----------
    obj : Any
        The input object which may be a numpy.ndarray, numpy.integer, numpy.floating,
        list, dict, or other types.

    Returns
    -------
    obj : Any
        Converted object with NumPy types replaced by native Python types.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    else:
        return obj


# === FastAPI App ===
app = FastAPI()


@app.get("/health")
async def health():
    """
    Health check endpoint to verify the API is responsive.

    Returns
    -------
    dict
        Dictionary indicating service status.
    """
    logger.info("Health check ping received.")
    return {"status": "healthy"}


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Accept an uploaded image file, run root tip detection inference pipeline,
    and return detected root tips along with the processed images.

    Parameters
    ----------
    file : UploadFile
        Image file uploaded by the client.

    Returns
    -------
    dict
        Dictionary containing:
        - 'tips': List of detected root tip coordinates (adjusted for padding).
        - 'padded_image': Base64-encoded PNG string of the padded input image.
        - 'pred_mask': Base64-encoded PNG string of the predicted mask.
        - 'backend': String indicating which backend was used for inference.
        - 'error' (optional): Error message if prediction fails.
    """
    logger.info(f"Received file for prediction: {file.filename}")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Convert PIL image to numpy array (OpenCV format)
        import cv2

        image_np = np.array(image)
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # Run the image processing and inference pipeline
        (
            padded_image,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            crop_left,
            crop_top,
        ) = load_and_preprocess_image(image_cv)
        logger.info("Image preprocessing completed.")
        logger.info(
            f"Padding applied: top={pad_top}, bottom={pad_bottom}, left={pad_left}, right={pad_right}"
        )

        # Get both padded and unpadded masks for consistency
        predicted_mask_padded, backend_used = run_inference(
            padded_image, pad_top, pad_bottom, pad_left, pad_right, remove_padding=False
        )
        predicted_mask_unpadded, backend_used = run_inference(
            padded_image, pad_top, pad_bottom, pad_left, pad_right, remove_padding=True
        )
        logger.info("Inference completed.")

        # Use unpadded mask for processing
        closed_mask = preprocess_closing(predicted_mask_unpadded)
        logger.info("Postprocessing (closing) completed.")

        final_labeled_mask, _ = apply_strict_bboxes(
            closed_mask,
            predicted_mask_unpadded,
            iou_threshold=0.001,
            size_threshold=2000,
            bottom_y_cutoff_ratio=0.85,
            top_y_cutoff=100,
            min_x_distance=20,
        )
        logger.info("Bounding box filtering applied.")

        tips = extract_bottom_tips(final_labeled_mask)
        logger.info(f"Extracted {len(tips)} root tips from unpadded mask.")

        # Adjust tip coordinates to match padded image coordinates
        # Tips are extracted from unpadded mask, need to add padding offsets
        tips_adjusted = []
        for tip in tips:
            if isinstance(tip, (list, tuple)) and len(tip) >= 2:
                # Tips from extract_bottom_tips are in (y, x) format
                original_y, original_x = tip[0], tip[1]

                # Add padding offsets to align with padded image
                adjusted_y = original_y + pad_top
                adjusted_x = original_x + pad_left

                # Preserve any additional elements in the tip data
                if len(tip) > 2:
                    adjusted_tip = [adjusted_y, adjusted_x] + list(tip[2:])
                else:
                    adjusted_tip = [adjusted_y, adjusted_x]

                tips_adjusted.append(adjusted_tip)
                logger.debug(
                    f"Adjusted tip from ({original_y}, {original_x}) to ({adjusted_y}, {adjusted_x})"
                )
            else:
                # If tip format is unexpected, keep original
                tips_adjusted.append(tip)
                logger.warning(f"Unexpected tip format: {tip}")

        logger.info(
            f"Adjusted {len(tips_adjusted)} tip coordinates by padding offset (top: {pad_top}, left: {pad_left})"
        )

        tips_clean = convert_numpy(tips_adjusted)

        # Convert padded_image (NumPy array) to PIL Image if needed
        if isinstance(padded_image, np.ndarray):
            padded_image_pil = Image.fromarray(padded_image)
        else:
            padded_image_pil = padded_image

        # Convert padded_image (PIL Image) to base64 string
        buffered = io.BytesIO()
        padded_image_pil.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Use the padded mask to match the padded image dimensions
        if isinstance(predicted_mask_padded, np.ndarray):
            predicted_mask_img = Image.fromarray(predicted_mask_padded)
        else:
            predicted_mask_img = predicted_mask_padded

        # Convert to base64 PNG
        pred_mask_buffered = io.BytesIO()
        predicted_mask_img.save(pred_mask_buffered, format="PNG")
        pred_mask_str = base64.b64encode(pred_mask_buffered.getvalue()).decode("utf-8")
        logger.info("Converted predicted mask to base64 string.")
        logger.info("padded image shape: %s", padded_image.shape)
        logger.info("predicted mask shape: %s", predicted_mask_padded.shape)
        logger.info(f"Returning {len(tips_adjusted)} tips with adjusted coordinates")

        return {
            "tips": tips_clean,
            "padded_image": img_str,
            "pred_mask": pred_mask_str,
            "backend": backend_used,
        }

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return {"error": str(e)}
