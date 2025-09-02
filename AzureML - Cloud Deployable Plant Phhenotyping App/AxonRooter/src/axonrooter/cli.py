"""
Command-line interface for AxonRooter root tip detection.

Supports inference from real images or synthetic simulations, and can optionally
output the detected root tips as JSON.
"""
import logging
import os

import matplotlib.pyplot as plt
import typer

from axonrooter.predict import predict_root_tips

plt.show = lambda: None


app = typer.Typer(
    help="Run inference on root images to detect root tip coordinates.\n\nExample structure:\n  python -m axonrooter image.png --output result.json"
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


@app.command()
def infer(
    infer: str = typer.Option(..., help="Path to the input image."),
    # output: str = typer.Option(None, "--output", "-o", help="Optional path to save output JSON."),
):
    """
    Run inference on a given image to detect root tip coordinates.

    Parameters
    ----------
    infer : str
        Path to the input image file. Supported formats: PNG, JPG, JPEG, TIFF.

    # output : str, optional
    #     Optional path to save the output results as JSON. (Currently commented out)

    Raises
    ------
    typer.Exit
        Exits with code 1 if the input file is missing, has unsupported file type,
        or an unexpected error occurs during processing.
        Exits with code 0 if no root tips are detected.

    Notes
    -----
    Logs the detected root tip coordinates at INFO level.
    """
    logger.info(f"Starting inference on: {infer}")

    try:
        # Validate file existence
        if not os.path.exists(infer):
            logger.error(f"File not found: {infer}")
            raise typer.Exit(code=1)

        # Validate file type
        valid_extensions = [".png", ".jpg", ".jpeg", ".tif", ".tiff"]
        if not any(infer.lower().endswith(ext) for ext in valid_extensions):
            logger.error(f"Unsupported file type: {infer}")
            raise typer.Exit(code=1)

        # Run prediction
        tips = predict_root_tips(infer)

        if not tips:
            logger.warning("No root tips detected in the image.")
            raise typer.Exit(code=0)

        logger.info(f"Detected {len(tips)} root tips.")
        for i, (x, y) in enumerate(tips):
            logger.info(f"  Tip {i+1}: (x={x:.2f}, y={y:.2f})")

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise typer.Exit(code=1)

    """if output:
        with open(output, "w") as f:
            json.dump(tips, f)
        print(f"[INFO] Saved results to {output}")"""
