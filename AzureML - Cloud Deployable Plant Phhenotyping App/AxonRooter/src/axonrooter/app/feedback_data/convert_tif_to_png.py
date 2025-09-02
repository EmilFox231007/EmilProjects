import os

import numpy as np
import tifffile
from PIL import Image


def convert_tif_to_png(
    tif_path, output_dir=None, remove_padding=False, pad_top=0, pad_bottom=0, pad_left=0, pad_right=0
):
    """
    Convert a .tif file back to PNG format.

    Parameters:
    -----------
    tif_path : str
        Path to the input .tif file
    output_dir : str, optional
        Directory to save the PNG file. If None, saves in same directory as input
    remove_padding : bool
        Whether to remove padding that was added during preprocessing
    pad_top, pad_bottom, pad_left, pad_right : int
        Padding values to remove (only used if remove_padding=True)

    Returns:
    --------
    str : Path to the saved PNG file
    """
    # Load the TIFF image
    tif_image = tifffile.imread(tif_path)

    # Remove padding if specified
    if remove_padding and any([pad_top, pad_bottom, pad_left, pad_right]):
        height, width = tif_image.shape
        row_end = height - pad_bottom if pad_bottom > 0 else height
        col_end = width - pad_right if pad_right > 0 else width
        tif_image = tif_image[pad_top:row_end, pad_left:col_end]
        print(f"Removed padding. New shape: {tif_image.shape}")

    # Convert to PIL Image
    if tif_image.dtype != np.uint8:
        # Normalize if needed
        if tif_image.max() <= 1.0:
            tif_image = (tif_image * 255).astype(np.uint8)
        else:
            tif_image = tif_image.astype(np.uint8)

    pil_image = Image.fromarray(tif_image)

    # Determine output path
    if output_dir is None:
        output_dir = os.path.dirname(tif_path)

    # Create output filename
    base_name = os.path.splitext(os.path.basename(tif_path))[0]
    if remove_padding:
        png_filename = f"{base_name}_unpadded.png"
    else:
        png_filename = f"{base_name}.png"

    png_path = os.path.join(output_dir, png_filename)

    # Save as PNG
    pil_image.save(png_path, "PNG")
    print(f"Converted {tif_path} -> {png_path}")

    return png_path


convert_tif_to_png("test1_20250617_134206_input.tif", output_dir=".")
