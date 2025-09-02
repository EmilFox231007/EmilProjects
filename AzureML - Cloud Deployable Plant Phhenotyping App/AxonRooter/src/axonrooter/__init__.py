# Contents of src/axonrooter/__init__.py

# --- Data preprocessing ---
from .data.preprocessing import (
    crop_initial,
    find_edges,
    format_crop,
    load_and_preprocess_image,
    padding_function,
)

# --- Models ---
from .models.inference import run_inference
from .models.tips import extract_bottom_tips

# --- Utilities ---
from .utils.bboxes import apply_strict_bboxes_with_top_cutoff_and_proximity as apply_strict_bboxes
from .utils.bboxes import calculate_iou
from .utils.morphology import preprocess_single_mask as preprocess_closing

__all__ = [
    # Models
    "run_inference",
    "extract_bottom_tips",
    # Data
    "crop_initial",
    "find_edges",
    "format_crop",
    "padding_function",
    "load_and_preprocess_image",
    # Utils
    "calculate_iou",
    "apply_strict_bboxes",
    "preprocess_closing",
]
__version__ = "0.1.0"
