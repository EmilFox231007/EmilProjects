"""This script configures the Streamlit application used for predicting root tip masks, integrated with FastAPI.
It provides functionality for visualizing the predictions and editing them within a dedicated interactive session.
"""

import os
import sys

# Add the src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "..", "..")
sys.path.insert(0, os.path.abspath(src_dir))

import base64
import csv
import io
import json
import logging
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
import yaml
from PIL import Image
from streamlit_image_zoom import image_zoom

from axonrooter.streamlit_drawable_canvas import st_canvas

# === Constants ===
SUBSCRIPTION_ID = os.getenv("AZURE_SUBSCRIPTION_ID")
RESOURCE_GROUP = os.getenv("AZURE_RESOURCE_GROUP")
WORKSPACE_NAME = os.getenv("AZURE_WORKSPACE_NAME")
DATASTORE_NAME = os.getenv("AZURE_DATASTORE_NAME")
BLOB_SUBDIR = os.getenv("AZURE_BLOB_SUBDIR")
DATA_ASSET_NAME = os.getenv("AZURE_DATA_ASSET_NAME")

# Storage account details
STORAGE_ACCOUNT_NAME = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
BLOB_CONTAINER = os.getenv("AZURE_BLOB_CONTAINER")
STORAGE_URI = os.getenv("AZURE_STORAGE_URI")

# Add the src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "..", "..")
sys.path.insert(0, os.path.abspath(src_dir))

from axonrooter.logging_config import setup_logging  # noqa: E402

# Set up logging
setup_logging()
logger = logging.getLogger(__name__)


# Page configuration
st.set_page_config(
    page_title="AxonRooter", page_icon="🌱", layout="wide", initial_sidebar_state="expanded"
)

# Apply Poppins font across the app
st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap" rel="stylesheet">
    <style>
        /* Apply to most elements */
        html, body, div, p, span, input, label, textarea, button {
            font-family: 'Poppins', sans-serif !important;
        }

        /* Streamlit markdown and widget containers */
        .stMarkdown, .stTextInput, .stTextArea, .stButton, .stRadio, .stSelectbox, .stMultiSelect, .stSlider, .stNumberInput {
            font-family: 'Poppins', sans-serif !important;
        }
    </style>
""",
    unsafe_allow_html=True,
)


# Load and encode local image
def get_base64_image(image_path):
    """
    Convert a local image to its base64-encoded string for proper display in streamlit.

    Parameters
    ----------
    image_path : str
        Path to the image file to be encoded.

    Returns
    -------
    str
        Base64-encoded string representation of the image content.
    """
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()


if not os.environ.get("SPHINX_BUILD"):  # Only run this if not building docs
    img_base64 = get_base64_image("greenhouse.jpg")
else:
    img_base64 = None  # or a dummy string


def initialize_session_state():
    """
    Initialize Streamlit session state variables.

    This function checks for the existence of specific keys in the
    `st.session_state` object and initializes them with default values
    if they are not already set. These variables are used to manage
    navigation, prediction state, image data, and user feedback
    across different pages of the Streamlit application.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Home"
    if "prediction_completed" not in st.session_state:
        st.session_state.prediction_completed = False
    if "last_prediction_filename" not in st.session_state:
        st.session_state.last_prediction_filename = ""
    if "tips" not in st.session_state:
        st.session_state.tips = []
    if "pad_img_np" not in st.session_state:
        st.session_state.pad_img_np = None
    if "pred_mask" not in st.session_state:
        st.session_state.pred_mask = None
    if "uploaded_filename" not in st.session_state:
        st.session_state.uploaded_filename = None
    if "frontend_logs" not in st.session_state:
        st.session_state.frontend_logs = []
    if "feedback_submitted" not in st.session_state:
        st.session_state.feedback_submitted = False


def add_frontend_log(level, message):
    """
    Add a log entry to the frontend logs in the Streamlit session state.

    This function creates a structured log entry with a timestamp, log level,
    module name, and message, and appends it to the `frontend_logs` list
    stored in `st.session_state`. It maintains a maximum of 100 log entries
    to limit memory usage.

    Parameters
    ----------
    level : str
        The severity level of the log (e.g., 'INFO', 'WARNING', 'ERROR').

    message : str
        The log message describing the event or action.

    Returns
    -------
    None
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = {
        "timestamp": timestamp,
        "level": level,
        "module": "streamlit_frontend",
        "message": message,
    }
    st.session_state.frontend_logs.append(log_entry)

    # Keep only last 100 logs to prevent memory issues
    if len(st.session_state.frontend_logs) > 100:
        st.session_state.frontend_logs = st.session_state.frontend_logs[-100:]


def render_homepage():
    """
    Render the homepage layout for the Streamlit application.

    This function creates a structured homepage with two main sections:
    a top section displaying the application logo and a descriptive title
    side by side, and a bottom section (not shown) reserved for additional
    information or UI components. The layout uses Streamlit columns for
    horizontal alignment and custom HTML/CSS for styling.

    The homepage is intended to introduce users to the application,
    emphasizing its purpose of AI-powered root tip detection for plant research.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """

    # TOP SECTION: Logo and Intro side by side
    st.markdown('<div class="top-section">', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("<div style='display: flex; justify-content: center;'>", unsafe_allow_html=True)
        st.image("assets/logo.png", width=480)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown(
            """
            <div style="display: flex; justify-content: center; align-items: center;">
                <p style="font-size: 2.8rem; color: #FFFFFF; font-weight: 480; margin: 0;
                        text-shadow: 3px 3px 10px rgba(0 ,0 ,0 ,0.8);">
                    AI-powered root tip detection for plant research
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # BOTTOM SECTION: Light grey background with additional info
    st.markdown(
        """
        <div class="bottom-section">
            <div class="info-content">
                <h3>Getting Started</h3>
                <div class="info-grid">
                    <div class="info-item">
                        <strong>Upload Images</strong>
                        <p>Support for JPG, JPEG, and PNG formats</p>
                        <div class="custom-text">
                            Click <span style="font-weight: bold;">Predict & Analysis</span> in the left sidebar
                        </div>
                    </div>
                    <div class="info-item">
                        <strong>AI Processing</strong>
                        <p>Advanced machine learning for root tip detection</p>
                        <div class="custom-text">
                            Click <span style="font-weight: bold;">Predict & Analysis</span> in the left sidebar
                        </div>
                    </div>
                    <div class="info-item">
                        <strong>Monitor System</strong>
                        <p>Real-time performance tracking and logs</p>
                        <div class="custom-text">
                            Click <span style="font-weight: bold;">System Logs</span> in the left sidebar
                        </div>
                    </div>
                    <div class="info-item">
                        <strong>Submit Feedback</strong>
                        <p>Help us improve by sharing your experience and corrected masks</p>
                        <div class="custom-text">
                            Submit <span style="font-weight: bold;">Feedback form</span> in the left sidebar and
                            Click <span style="font-weight: bold;">Mask Submit</span> to draw your corrected masks
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <style>
        .top-section {
            display: flex;
            align-items: center;
            margin-bottom: 0;
        }

        .intro-content {
            display: flex;
            flex-direction: column;
            align-items: flex-end;  /* aligns the text container to right */
            text-align: right;      /* aligns the text inside to right */
            width: 100%;            /* ensure it fills the column */
            margin-top: 0;
        }

        .intro-text {
            font-size: 2.5rem; /* increased size */
            color: #ffffff; /* white text */
            margin-bottom: 2rem;
            font-weight: 600; /* bolder for emphasis */
            line-height: 1.4;
        }

        .action-buttons {
            display: flex;
            gap: 1rem;
            flex-wrap: wrap;
        }

        .action-btn {
            padding: 0.75rem 1.5rem;
            border-radius: 25px;
            text-decoration: none;
            font-weight: 500;
            transition: all 0.3s ease;
            border: 1px solid #ddd;
        }

        .action-btn.primary {
            background: #228B22;
            color: white;
            border: 1px solid #228B22;
        }

        .action-btn.primary:hover {
            background: #1f7a1f;
            transform: translateY(-2px);
        }

        .action-btn.secondary {
            background: #f8f9fa;
            color: #333;
            border: 1px solid #ddd;
        }

        .action-btn.secondary:hover {
            background: #e9ecef;
            transform: translateY(-2px);
        }

        .bottom-section {
            background: #f8f9fa;
            padding: 1rem 1rem;
            border-radius: 10px;
            margin-top: 0;
        }

        .info-content h3 {
            color: #228B22;
            text-align: center;
            margin-bottom: 2rem;
            font-size: 3rem;
            font-family: 'Poppins', sans-serif;
        }

        .info-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 2rem;
            margin-bottom: 2rem;
        }

        .info-item {
            background: white;
            padding: 2.4rem;
            border-radius: 8px;
            box-shadow: 0 3px 6px rgba(0, 0, 0, 0.1);
            text-align: center;
        }

        .info-item strong {
            color: #228B22;
            font-size: 1.1rem;
            display: block;
            margin-bottom: 0.5rem;
        }

        .info-item p {
            color: #666;
            margin: 0.5rem 0 1rem 0;
            font-size: 1.5rem;
        }

        .help-section {
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            text-align: center;
            border-left: 4px solid #228B22;
        }

        .help-section p {
            margin: 0;
            color: #555;
        }

        /* Ensure proper column alignment */
        .stColumn {
            display: flex;
            align-items: center;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def update_progress_with_details(progress_bar, status_text, step, total_steps, message, detail=""):
    """
    Update a Streamlit progress bar and status message with detailed information.

    This function updates the visual progress indicator and displays a formatted
    status message to reflect the current step in a multi-step process. It also logs
    the progress message to the frontend logs and introduces a short delay for
    better visual feedback.

    Parameters
    ----------
    progress_bar : st.delta_generator.DeltaGenerator
        The Streamlit progress bar object to be updated.

    status_text : st.delta_generator.DeltaGenerator
        The Streamlit text element for showing status messages.

    step : int
        The current step number in the progress sequence.

    total_steps : int
        The total number of steps in the process.

    message : str
        The main message to display alongside the progress.

    detail : str, optional
        Additional information or context to show under the main message. Default is an empty string.

    Returns
    -------
    None
    """
    progress = int((step / total_steps) * 100)
    progress_bar.progress(progress)

    if detail:
        status_text.info(f"**Step {step}/{total_steps}:** {message}\n\n*{detail}*")
    else:
        status_text.info(f"**Step {step}/{total_steps}:** {message}")

    add_frontend_log("INFO", f"Progress: {message} ({progress}%)")
    time.sleep(0.3)  # Visual feedback delay


def create_azure_ml_yml(dataset_name, version, description, datastore_path):
    """
    Generate a YAML configuration dictionary for an Azure ML dataset.

    This function constructs a dictionary representing the contents of a
    dataset YAML file used to register a dataset in Azure Machine Learning.
    The dataset is of type `uri_folder` and is described using a custom name,
    version, description, and a reference to a path in an Azure ML datastore.

    Parameters
    ----------
    dataset_name : str
        The name of the dataset to be registered in Azure ML.

    version : str or int
        The version number or label for the dataset.

    description : str
        A human-readable description of the dataset.

    datastore_path : str
        The path to the dataset within the Azure ML datastore (e.g., 'azureml://...').

    Returns
    -------
    dict
        A dictionary representing the dataset YAML configuration compatible
        with Azure ML's data registration schema.
    """
    yml_content = {
        "$schema": "https://azuremlschemas.azureedge.net/latest/data.schema.json",
        "type": "uri_folder",
        "name": dataset_name,
        "version": version,
        "description": description,
        "path": datastore_path,
    }

    return yml_content


def find_azure_cli_path():
    """
    Locate the Azure CLI executable on the system.

    This function checks a list of common installation paths for the Azure CLI
    (`az` or `az.cmd`) and attempts to locate it using the system `PATH` as a fallback.
    It logs and returns the path if found. If not found, it returns `None`.

    Returns
    -------
    str or None
        The absolute path to the Azure CLI executable if found; otherwise, `None`.
    """
    possible_paths = [
        r"C:\Program Files (x86)\Microsoft SDKs\Azure\CLI2\wbin\az.cmd",
        r"C:\Program Files\Microsoft SDKs\Azure\CLI2\wbin\az.cmd",
        r"C:\Users\{}\AppData\Local\Programs\Microsoft\Azure CLI\wbin\az.cmd".format(
            os.getenv("USERNAME")
        ),
        "az.cmd",  # If in PATH
        "az",  # If in PATH (Linux style)
    ]

    for path in possible_paths:
        try:
            if os.path.exists(path):
                logger.info(f"Found Azure CLI at: {path}")
                return path
        except (FileNotFoundError, PermissionError):
            continue

    # Try to find in PATH
    try:
        result = subprocess.run(["where", "az"], capture_output=True, text=True, check=True)
        az_path = result.stdout.strip().split("\n")[0]
        logger.info(f"Found Azure CLI in PATH: {az_path}")
        return az_path
    except (subprocess.CalledProcessError, FileNotFoundError, IndexError):  # Specify exception types
        pass
    return None


def test_azure_cli_connection():
    """
    Verify that the Azure CLI is correctly installed, configured, and authenticated.

    This function performs a series of checks to validate the Azure CLI setup:
    1. Ensures the Azure CLI executable is found.
    2. Confirms that the CLI is functioning by checking its version.
    3. Verifies that the Azure ML extension is installed.
    4. Checks whether the user is logged into Azure.
    5. Confirms that Azure ML workspace access is available.

    Returns
    -------
    tuple of (bool, str)
        A tuple where the first element is a boolean indicating success (`True`)
        or failure (`False`), and the second element is a message describing the result.

    Examples
    --------
    >>> success, message = test_azure_cli_connection()
    >>> print(success, message)
    True Azure CLI is working properly
    """
    az_path = find_azure_cli_path()

    if not az_path:
        return False, "Azure CLI not found"

    try:
        # Test basic CLI
        result = subprocess.run(
            [az_path, "--version"], capture_output=True, text=True, check=True, timeout=30
        )
        logger.info(f"Azure CLI version: {result.stdout.strip()}")

        # Test ML extension
        result = subprocess.run(
            [az_path, "extension", "list"], capture_output=True, text=True, check=True, timeout=30
        )
        if "ml" not in result.stdout:
            return False, "Azure ML extension not installed. Run: az extension add -n ml"

        # Test login status
        result = subprocess.run(
            [az_path, "account", "show"], capture_output=True, text=True, check=True, timeout=30
        )
        logger.info("Azure CLI login verified")

        # Test ML workspace access
        result = subprocess.run(
            [az_path, "ml", "workspace", "list"],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
        logger.info("Azure ML workspace access verified")

        return True, "Azure CLI is working properly"

    except subprocess.TimeoutExpired:
        return False, "Azure CLI commands are timing out"
    except subprocess.CalledProcessError as e:
        if "Please run 'az login'" in e.stderr:
            return False, "Not logged in to Azure. Run: az login"
        return False, f"Azure CLI error: {e.stderr}"
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"


def upload_files_to_azure_datastore(uploaded_images, uploaded_masks, canvas_json=None):
    """
    Upload image, mask, and optional correction files to an Azure ML datastore using workspace credentials.

    This function performs the following steps:
    - Verifies Azure CLI connection and workspace context.
    - Creates a temporary directory structure to organize uploaded files.
    - Saves uploaded images, masks, and optional canvas correction JSON files locally.
    - Adds metadata related to the submission.
    - Executes an Azure CLI command to create a new Azure ML data asset with the uploaded files.
    - Cleans up temporary files after upload.

    Parameters
    ----------
    uploaded_images : list of images or None
        List of image files uploaded via the Streamlit interface or similar.
    uploaded_masks : list of images or None
        List of mask files uploaded via the Streamlit interface or similar.
    canvas_json : dict, optional
        Optional JSON object containing corrections or annotations from a canvas tool (default is None).

    Returns
    -------
    tuple
        - data_asset_name (str or None): The name of the created Azure ML data asset, or None if upload failed.
        - saved_files (dict): Dictionary listing saved file paths by category ('images', 'masks', 'corrections').
        - cli_output (str or None): The standard output from the Azure CLI upload command, or an error message.

    Examples
    --------
    >>> data_asset_name, saved_files, output = upload_files_to_azure_datastore(
    ...     uploaded_images=[uploaded_img1, uploaded_img2],
    ...     uploaded_masks=[uploaded_mask1],
    ...     canvas_json={"corrected": True, "details": []}
    ... )
    >>> if data_asset_name is not None:
    ...     print(f"Upload successful: {data_asset_name}")
    ... else:
    ...     print("Upload failed:", output)
    """
    try:
        # First, test Azure CLI connection with workspace context
        cli_working, cli_message = test_azure_cli_connection_with_workspace()
        if not cli_working:
            logger.error(f"Azure CLI test failed: {cli_message}")
            return None, {}, f"Azure CLI issue: {cli_message}"

        az_path = find_azure_cli_path()
        logger.info(f"Using Azure CLI at: {az_path}")

        # Create temporary directory structure
        temp_dir = tempfile.mkdtemp(prefix="axonrooter_student_upload_")
        temp_path = Path(temp_dir)

        # Create ML-ready directory structure
        train_images_dir = temp_path / "train_images"
        train_masks_dir = temp_path / "train_masks"
        val_images_dir = temp_path / "val_images"
        val_masks_dir = temp_path / "val_masks"

        train_images_dir.mkdir(exist_ok=True)
        train_masks_dir.mkdir(exist_ok=True)
        val_images_dir.mkdir(exist_ok=True)
        val_masks_dir.mkdir(exist_ok=True)

        saved_files = {"train_images": [], "train_masks": [], "val_images": [], "val_masks": []}

        # Process images and masks with simple 80/20 split
        if uploaded_images and uploaded_masks:
            # Ensure we have matching pairs
            min_files = min(len(uploaded_images), len(uploaded_masks))
            if min_files == 0:
                raise ValueError("No matching image-mask pairs found")

            # Simple 80/20 split: first 80% go to train, last 20% go to val
            split_point = int(min_files * 0.8)

            logger.info(
                f"Splitting {min_files} files: {split_point} train, {min_files - split_point} validation"
            )

            # --- Start safe pairing split ---
            pairs = match_image_mask_pairs(uploaded_images, uploaded_masks)
            if not pairs:
                raise ValueError("No matching image-mask pairs found.")

            split_point = int(len(pairs) * 0.8)
            train_pairs, val_pairs = pairs[:split_point], pairs[split_point:]

            # Save training files
            for img_file, mask_file in train_pairs:
                # Save training image
                img_path = train_images_dir / img_file.name
                with open(img_path, "wb") as f:
                    f.write(img_file.getbuffer())
                saved_files["train_images"].append(str(img_path))
                logger.info(f"Saved training image: {img_path}")

                # Save training mask
                mask_path = train_masks_dir / mask_file.name
                with open(mask_path, "wb") as f:
                    f.write(mask_file.getbuffer())
                saved_files["train_masks"].append(str(mask_path))
                logger.info(f"Saved training mask: {mask_path}")

            # Save validation files
            for img_file, mask_file in val_pairs:
                # Save validation image
                img_path = val_images_dir / img_file.name
                with open(img_path, "wb") as f:
                    f.write(img_file.getbuffer())
                saved_files["val_images"].append(str(img_path))
                logger.info(f"Saved validation image: {img_path}")

                # Save validation mask
                mask_path = val_masks_dir / mask_file.name
                with open(mask_path, "wb") as f:
                    f.write(mask_file.getbuffer())
                saved_files["val_masks"].append(str(mask_path))
                logger.info(f"Saved validation mask: {mask_path}")

        elif uploaded_images:
            # Handle case where only images are provided
            split_point = int(len(uploaded_images) * 0.8)

            logger.info(
                f"Splitting {len(uploaded_images)} images: {split_point} train, {len(uploaded_images) - split_point} validation"
            )

            # Save training images
            for i in range(split_point):
                img_file = uploaded_images[i]
                img_path = train_images_dir / img_file.name
                with open(img_path, "wb") as f:
                    f.write(img_file.getbuffer())
                saved_files["train_images"].append(str(img_path))
                logger.info(f"Saved training image: {img_path}")

            # Save validation images
            for i in range(split_point, len(uploaded_images)):
                img_file = uploaded_images[i]
                img_path = val_images_dir / img_file.name
                with open(img_path, "wb") as f:
                    f.write(img_file.getbuffer())
                saved_files["val_images"].append(str(img_path))
                logger.info(f"Saved validation image: {img_path}")

        # Generate unique upload directory name with student prefix
        upload_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        student_id = os.getenv("USERNAME", "student")  # Use Windows username as student ID

        # Create data asset with proper workspace context
        data_asset_name = f"{DATA_ASSET_NAME}_{student_id}_{upload_timestamp}"

        upload_cmd = [
            az_path,
            "ml",
            "data",
            "create",
            "--name",
            data_asset_name,
            "--path",
            str(temp_path),
            "--type",
            "uri_folder",
            "--description",
            f"AxonRooter ML dataset by {student_id} - Train: {len(saved_files['train_images'])} images/{len(saved_files['train_masks'])} masks, Val: {len(saved_files['val_images'])} images/{len(saved_files['val_masks'])} masks",
            "--resource-group",
            RESOURCE_GROUP,
            "--workspace-name",
            WORKSPACE_NAME,
            "--subscription",
            SUBSCRIPTION_ID,
        ]

        logger.info(f"Creating data asset with command: {' '.join(upload_cmd)}")

        # Set environment variables for subprocess
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"

        # Set Azure ML workspace context
        env["AZUREML_ARM_SUBSCRIPTION"] = SUBSCRIPTION_ID
        env["AZUREML_ARM_RESOURCEGROUP"] = RESOURCE_GROUP
        env["AZUREML_ARM_WORKSPACE_NAME"] = WORKSPACE_NAME

        result = subprocess.run(
            upload_cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=300,
            env=env,
            cwd=str(temp_path),
        )

        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)

        logger.info(f"Successfully created data asset: {data_asset_name}")
        return data_asset_name, saved_files, result.stdout

    except FileNotFoundError as e:
        error_msg = f"File not found error: {str(e)}. Azure CLI path: {az_path if 'az_path' in locals() else 'Not found'}"
        logger.error(error_msg)
        if "temp_dir" in locals() and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
        return None, {}, error_msg
    except subprocess.CalledProcessError as e:
        error_msg = f"Azure CLI command failed: {e.stderr}"
        logger.error(error_msg)
        logger.error(f"Command output: {e.stdout}")
        if "temp_dir" in locals() and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
        return None, {}, error_msg
    except subprocess.TimeoutExpired:
        error_msg = "Upload command timed out after 5 minutes"
        logger.error(error_msg)
        if "temp_dir" in locals() and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
        return None, {}, error_msg
    except Exception as e:
        error_msg = f"Unexpected error during upload: {str(e)}"
        logger.error(error_msg)
        if "temp_dir" in locals() and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
        return None, {}, error_msg


def test_azure_cli_connection_with_workspace():
    """
    Test if Azure CLI is installed, logged in, and configured with the specified Azure ML workspace.

    This function verifies:
    - Azure CLI is installed and accessible.
    - Azure ML extension is installed.
    - User is logged in with the expected subscription.
    - Access to the specified ML workspace.
    - Access to the specified datastore within the workspace.

    Returns
    -------
    tuple
        A tuple (success, message) where `success` is a boolean indicating
        if the test passed, and `message` provides details or error info.

    Examples
    --------
    >>> success, msg = test_azure_cli_connection_with_workspace()
    >>> if success:
    ...     print("Azure CLI setup verified.")
    ... else:
    ...     print(f"Setup issue: {msg}")
    """
    az_path = find_azure_cli_path()

    if not az_path:
        return False, "Azure CLI not found"

    try:
        # Test basic CLI
        result = subprocess.run(
            [az_path, "--version"], capture_output=True, text=True, check=True, timeout=30
        )
        logger.info(f"Azure CLI version: {result.stdout.strip()}")

        # Test ML extension
        result = subprocess.run(
            [az_path, "extension", "list"], capture_output=True, text=True, check=True, timeout=30
        )
        if "ml" not in result.stdout:
            return False, "Azure ML extension not installed. Run: az extension add -n ml"

        # Test login status
        result = subprocess.run(
            [az_path, "account", "show"], capture_output=True, text=True, check=True, timeout=30
        )
        account_info = json.loads(result.stdout)
        current_subscription = account_info.get("id", "")

        if current_subscription != SUBSCRIPTION_ID:
            return (
                False,
                f"Wrong subscription. Current: {current_subscription}, Expected: {SUBSCRIPTION_ID}. Run: az account set --subscription {SUBSCRIPTION_ID}",
            )

        logger.info("Azure CLI login and subscription verified")

        # Test ML workspace access with specific workspace
        result = subprocess.run(
            [
                az_path,
                "ml",
                "workspace",
                "show",
                "--name",
                WORKSPACE_NAME,
                "--resource-group",
                RESOURCE_GROUP,
                "--subscription",
                SUBSCRIPTION_ID,
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
        logger.info(f"Azure ML workspace access verified for {WORKSPACE_NAME}")

        # Test datastore access
        result = subprocess.run(
            [
                az_path,
                "ml",
                "datastore",
                "show",
                "--name",
                DATASTORE_NAME,
                "--workspace-name",
                WORKSPACE_NAME,
                "--resource-group",
                RESOURCE_GROUP,
                "--subscription",
                SUBSCRIPTION_ID,
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
        logger.info(f"Datastore access verified for {DATASTORE_NAME}")

        return True, "Azure CLI is working properly with your workspace"

    except subprocess.TimeoutExpired:
        return False, "Azure CLI commands are timing out"
    except subprocess.CalledProcessError as e:
        if "Please run 'az login'" in e.stderr:
            return False, "Not logged in to Azure. Run: az login"
        elif "not found" in e.stderr.lower():
            return (
                False,
                f"Workspace or resource not found. Check: {WORKSPACE_NAME} in {RESOURCE_GROUP}",
            )
        return False, f"Azure CLI error: {e.stderr}"
    except json.JSONDecodeError:
        return False, "Failed to parse Azure CLI response"
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"


# Helper functions for matching images and masks
def get_base_id(filename):
    # Remove _mask or extension
    return re.sub(r"_root_mask$", "", Path(filename).stem)


def match_image_mask_pairs(images, masks):
    image_dict = {get_base_id(img.name): img for img in images}
    mask_dict = {get_base_id(mask.name): mask for mask in masks}

    common_ids = sorted(set(image_dict.keys()) & set(mask_dict.keys()))
    unmatched_images = set(image_dict.keys()) - set(mask_dict.keys())
    unmatched_masks = set(mask_dict.keys()) - set(image_dict.keys())

    if unmatched_images:
        print(f"⚠️ Unmatched images: {unmatched_images}")
    if unmatched_masks:
        print(f"⚠️ Unmatched masks: {unmatched_masks}")

    pairs = [(image_dict[iid], mask_dict[iid]) for iid in common_ids]
    return pairs


def split_pairs(pairs, split_ratio=0.8):
    split_index = int(len(pairs) * split_ratio)
    return pairs[:split_index], pairs[split_index:]


def upload_with_azcopy(local_path, storage_url, sas_token):
    """
    Uploads files from a local directory to Azure Blob Storage using AzCopy.

    Parameters:
    - local_path: Local directory containing files to upload.
    - storage_url: Azure Blob Storage URL (e.g., 'https://<account>.blob.core.windows.net/<container>').
    - sas_token: Shared Access Signature token for authentication.
    """
    try:
        # Construct the AzCopy command
        command = [
            "azcopy",
            "copy",
            f"{local_path}/*",
            f"{storage_url}?{sas_token}",
            "--recursive=true",
        ]
        # Execute the command
        subprocess.run(command, check=True)
        print(f"Successfully uploaded files from {local_path} to {storage_url}")
    except subprocess.CalledProcessError as e:
        print(f"Error during AzCopy upload: {e}")


#  Register dataset with Azure ML using CLI
def register_dataset_with_azure_ml(dataset_name, version, description, datastore_path):
    """
    Register a dataset in Azure ML workspace using a datastore path.

    This function creates a YAML configuration for the dataset and uses the Azure CLI
    to register it as a data asset in Azure Machine Learning.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset to register.
    version : str
        Version string for the dataset.
    description : str
        Description of the dataset.
    datastore_path : str
        Path to the dataset location in the Azure ML datastore.

    Returns
    -------
    tuple
        A tuple (success, message) where `success` is a boolean indicating whether
        the registration succeeded, and `message` contains the Azure CLI output
        or error message.

    Raises
    ------
    subprocess.CalledProcessError
        If the Azure CLI command fails during execution.
    Exception
        For any other unexpected errors.

    Notes
    -----
    The function creates a temporary YAML configuration file for the dataset registration.
    The file is cleaned up automatically after the command completes or fails.

    Examples
    --------
    >>> success, msg = register_dataset_with_azure_ml(
    ...     "root_tips_dataset", "1.0", "Root tips images and masks",
    ...     "azureml://datastores/workspaceblobstore/paths/root_tips/"
    ... )
    >>> if success:
    ...     print("Dataset registered successfully")
    ... else:
    ...     print(f"Dataset registration failed: {msg}")
    """
    yml_path = None
    try:
        # Find Azure CLI path
        az_path = find_azure_cli_path()
        if not az_path:
            return False, "Azure CLI not found"

        # Test Azure CLI connection first
        cli_working, cli_message = test_azure_cli_connection_with_workspace()
        if not cli_working:
            return False, f"Azure CLI issue: {cli_message}"

        # Create Azure ML dataset YAML configuration
        yml_content = create_azure_ml_yml(dataset_name, version, description, datastore_path)

        # Create temporary YAML file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as yml_file:
            yaml.dump(yml_content, yml_file, default_flow_style=False)
            yml_path = yml_file.name

        logger.info(f"Created temporary YAML file: {yml_path}")

        # Run Azure ML CLI command with workspace context
        cmd = [
            az_path,
            "ml",
            "data",
            "create",
            "-f",
            yml_path,
            "--resource-group",
            RESOURCE_GROUP,
            "--workspace-name",
            WORKSPACE_NAME,
            "--subscription",
            SUBSCRIPTION_ID,
        ]

        logger.info(f"Running Azure ML CLI command: {' '.join(cmd)}")

        # Set environment variables for subprocess
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["AZUREML_ARM_SUBSCRIPTION"] = SUBSCRIPTION_ID
        env["AZUREML_ARM_RESOURCEGROUP"] = RESOURCE_GROUP
        env["AZUREML_ARM_WORKSPACE_NAME"] = WORKSPACE_NAME

        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True, timeout=300, env=env
        )

        logger.info(f"Successfully registered dataset: {dataset_name} v{version}")
        add_frontend_log("SUCCESS", f"Dataset registered: {dataset_name} v{version}")

        return True, result.stdout

    except subprocess.TimeoutExpired:
        error_msg = "Dataset registration timed out after 5 minutes"
        logger.error(error_msg)
        add_frontend_log("ERROR", error_msg)
        return False, error_msg
    except subprocess.CalledProcessError as e:
        error_msg = f"Azure ML CLI command failed: {e.stderr}"
        logger.error(error_msg)
        add_frontend_log("ERROR", f"Dataset registration failed: {e.stderr}")
        return False, e.stderr
    except Exception as e:
        error_msg = f"Error registering dataset: {str(e)}"
        logger.error(error_msg)
        add_frontend_log("ERROR", error_msg)
        return False, str(e)
    finally:
        # Clean up YAML file if it exists
        if yml_path and os.path.exists(yml_path):
            try:
                os.unlink(yml_path)
                logger.info(f"Cleaned up temporary YAML file: {yml_path}")
            except OSError as e:
                logger.warning(f"Failed to clean up YAML file {yml_path}: {e}")


def render_prediction_page():
    """
    Render the main prediction and analysis page in the Streamlit app.

    This function provides an interactive UI for users to upload plant root images,
    run predictions via a backend API, track progress, handle errors, and display results.
    It manages session state variables for prediction results and feedback.

    Features
    --------
    - Image upload (JPG, JPEG, PNG up to 200MB)
    - Run prediction button triggering backend analysis via HTTP POST
    - Real-time progress bar and status updates
    - Handling connection, HTTP, and unexpected errors gracefully
    - Storing prediction results and metadata in `st.session_state`
    - Clearing previous data on new predictions
    - Logging key events for frontend and backend diagnostics

    Returns
    -------
    None
        This function directly renders Streamlit components and updates session state.

    Examples
    --------
    >>> import streamlit as st
    >>> from app import render_prediction_page
    >>>
    >>> def main():
    ...     st.title("Root Analysis App")
    ...     render_prediction_page()
    ...
    >>> if __name__ == "__main__":
    ...     main()
    """
    st.markdown('<div class="main-header">Prediction & Analysis</div>', unsafe_allow_html=True)

    # Input section
    with st.expander("Image Upload", expanded=True):
        st.markdown("Upload your plant root image to begin analysis")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=["jpg", "jpeg", "png"],
            help="Supported formats: JPG, JPEG, PNG. Max file size: 200MB",
        )

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            run_pred_bt = st.button(
                "Run Prediction", key="run_prediction", use_container_width=True, type="primary"
            )

        # Prediction processing
        if uploaded_file is not None and run_pred_bt:
            add_frontend_log("INFO", f"Starting prediction for file: {uploaded_file.name}")
            logger.info(f"Starting prediction for uploaded file: {uploaded_file.name}")

            st.success(f"Image '{uploaded_file.name}' uploaded successfully!")

            # Create progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                # Step 1: Image preprocessing
                update_progress_with_details(
                    progress_bar,
                    status_text,
                    1,
                    10,
                    "Initializing image preprocessing",
                    "Loading and validating image format",
                )

                original_image = Image.open(uploaded_file).convert("RGBA")
                width, height = original_image.size

                update_progress_with_details(
                    progress_bar,
                    status_text,
                    2,
                    10,
                    "Processing image dimensions",
                    f"Original size: {width} x {height} pixels",
                )

                st.session_state.uploaded_filename = uploaded_file.name
                logger.info(f"Image loaded successfully - Dimensions: {width}x{height}")

                # Step 3: Preparing server request
                update_progress_with_details(
                    progress_bar,
                    status_text,
                    3,
                    10,
                    "Preparing server communication",
                    "Encoding image data for transmission",
                )

                # Step 4: Sending to backend
                update_progress_with_details(
                    progress_bar,
                    status_text,
                    4,
                    10,
                    "Connecting to prediction server",
                    "Establishing connection to localhost:8000",
                )

                # Azure ML endpoint configuration
                url = "https://adsai2.ngrok.dev/port-30526/api/v1/endpoint/cv9-endpoint-06251752354033/score"
                api_key = "6eg7gJAM95s4ZtT5sOH9mIGdmqS1m3N2O0e7S3ZoPQwg4768yJVJJQQJ99BFAAAAAAAAAAAAINFRAZML3BI9"

                # Headers as specified in Azure ML consumption code
                headers = {
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "Authorization": f"Bearer {api_key}",
                }

                # Convert image to base64 for JSON payload
                import base64

                image_b64 = base64.b64encode(uploaded_file.getvalue()).decode("utf-8")

                # Data format expected by scoring.py: {"data": base64_image}
                data = {"data": image_b64}

                logger.info(f"Sending POST request to Azure ML endpoint: {url}")
                logger.info(f"Request payload keys: {list(data.keys())}")
                add_frontend_log("INFO", "Sending request to Azure ML prediction server")

                # Step 5: Server processing
                update_progress_with_details(
                    progress_bar,
                    status_text,
                    5,
                    10,
                    "Server processing image",
                    "AI model analyzing root structures via Azure ML",
                )

                response = requests.post(url, headers=headers, json=data)
                response.raise_for_status()

                # Step 6: Receiving results
                update_progress_with_details(
                    progress_bar,
                    status_text,
                    6,
                    10,
                    "Receiving prediction results",
                    f"Server response: {response.status_code}",
                )  # Parse the response - handle Azure ML scoring script format
                try:
                    # Log raw response for debugging
                    raw_response = response.text
                    logger.info(f"Raw response (first 200 chars): {raw_response[:200]}")

                    result = response.json()
                    logger.info(f"Received JSON response: {type(result)}")

                    # Handle the case where Azure ML returns a string that contains JSON
                    if isinstance(result, str):
                        logger.info("Response is a string, attempting to parse as JSON")
                        logger.info(f"String content (first 200 chars): {result[:200]}")
                        try:
                            result = json.loads(result)
                            logger.info(f"Successfully parsed string as JSON: {type(result)}")
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to parse string response as JSON: {e}")
                            logger.error(f"String content: {result[:200]}...")
                            st.error(
                                f"Azure ML returned a string that isn't valid JSON: {result[:200]}..."
                            )
                            add_frontend_log("ERROR", f"Invalid JSON string from Azure ML: {str(e)}")
                            return

                    # Now check the actual content
                    if isinstance(result, dict):
                        # Check if this is the Azure ML scoring script format
                        if "predicted_mask_base64" in result:
                            # This is the Azure ML format - extract the mask
                            pred_mask_str = result.get("predicted_mask_base64")
                            backend_used = "Azure ML Scoring Script"

                            # Since the scoring script doesn't detect tips, we'll need to extract them from the mask
                            # For now, provide empty tips - you may want to add tip detection logic later
                            tips = []
                            padded_image = 0  # This will be calculated below from the uploaded image

                            logger.info(
                                "Received Azure ML scoring script response with predicted mask"
                            )
                            add_frontend_log("INFO", "Received prediction mask from Azure ML")

                        elif "tips" in result and "pred_mask" in result:
                            # This is the expected full format (if you have a different endpoint)
                            tips, padded_image, pred_mask_str, backend_used = (
                                result.get("tips", []),
                                result.get("padded_image", 0),
                                result.get("pred_mask", None),
                                result.get("backend", "Azure ML"),
                            )
                            logger.info(f"Received full format response - Tips found: {len(tips)}")
                            add_frontend_log("INFO", f"Received {len(tips)} predicted tips")

                        elif "error" in result:
                            # Handle error response from scoring script
                            error_msg = result.get("error", "Unknown error from Azure ML")
                            logger.error(f"Azure ML endpoint returned error: {error_msg}")
                            st.error(f"Prediction failed: {error_msg}")
                            add_frontend_log("ERROR", f"Azure ML error: {error_msg}")
                            return

                        else:
                            # Unknown format
                            logger.warning(f"Unexpected response format: {result}")
                            st.error(f"Unexpected response format from Azure ML endpoint: {result}")
                            add_frontend_log(
                                "WARNING", f"Unexpected response format: {type(result)}"
                            )
                            return
                    else:
                        # If result is still not a dict after parsing attempts
                        logger.error(f"Response is not a dictionary after parsing: {type(result)}")
                        st.error(f"Azure ML returned unexpected data type: {type(result)}")
                        add_frontend_log("ERROR", f"Unexpected response type: {type(result)}")
                        return

                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON response: {e}")
                    logger.error(f"Raw response content: {response.text}")
                    add_frontend_log("ERROR", f"Invalid JSON response from Azure ML: {str(e)}")
                    st.error(f"Azure ML endpoint returned invalid JSON: {response.text[:200]}...")
                    return

                except Exception as e:
                    logger.error(f"Unexpected error parsing response: {str(e)}")
                    logger.error(f"Exception type: {type(e).__name__}")
                    logger.error(f"Exception args: {e.args}")
                    add_frontend_log("ERROR", f"Response parsing error: {str(e)}")
                    st.error(f"Error processing Azure ML response: {str(e)}")
                    return

                # Step 7: Processing image data for visualization
                update_progress_with_details(
                    progress_bar,
                    status_text,
                    7,
                    10,
                    "Processing image data",
                    "Preparing image for visualization",
                )

                # For Azure ML scoring script, create pad_img_np from original uploaded image
                # since the scoring script doesn't return a padded image
                if padded_image:
                    # If a padded image is provided (from other endpoints)
                    image_bytes = base64.b64decode(padded_image)
                    mask_img = Image.open(io.BytesIO(image_bytes)).convert("L")
                    pad_img_np = np.array(mask_img)
                    logger.info("Converted padded image to numpy array")
                else:
                    # Create pad_img_np from the original uploaded image for Azure ML scoring script
                    original_img = Image.open(uploaded_file).convert("L")
                    pad_img_np = np.array(original_img)
                    logger.info(
                        f"Created pad_img_np from original uploaded image - Shape: {pad_img_np.shape}"
                    )
                    logger.info(
                        "Note: Azure ML scoring script doesn't return padded image, using original"
                    )

                # Step 8: Processing prediction mask
                update_progress_with_details(
                    progress_bar,
                    status_text,
                    8,
                    10,
                    "Processing prediction mask",
                    "Converting mask overlay data",
                )

                if pred_mask_str:
                    pred_mask_bytes = base64.b64decode(pred_mask_str)
                    pred_mask_img = Image.open(io.BytesIO(pred_mask_bytes)).convert("L")
                    pred_mask = np.array(pred_mask_img)
                    logger.info("Converted prediction mask to numpy array")
                else:
                    pred_mask = None
                    logger.warning("No prediction mask data received")

                # Step 9: Finalizing results
                update_progress_with_details(
                    progress_bar,
                    status_text,
                    9,
                    10,
                    "Finalizing prediction results",
                    "Storing results in session state",
                )

                # Store results in session state
                st.session_state.tips = tips
                st.session_state.pad_img_np = pad_img_np
                st.session_state.pred_mask = pred_mask
                st.session_state.padded_image = padded_image
                st.session_state.backend_used = backend_used

                # ==== NEW ADDITION: Set prediction completion flags ====
                st.session_state.prediction_completed = True
                st.session_state.last_prediction_filename = uploaded_file.name
                # Reset feedback state for new prediction
                st.session_state.feedback_submitted = False

                logger.info(
                    f"Session state updated - prediction_completed: {st.session_state.prediction_completed}"
                )
                add_frontend_log("INFO", f"Session state updated for feedback system")

                # Step 10: Complete
                update_progress_with_details(
                    progress_bar,
                    status_text,
                    10,
                    10,
                    "Prediction completed successfully!",
                    f"Analysis complete: {len(tips)} root tips detected",
                )

                logger.info(f"Prediction pipeline completed successfully")
                add_frontend_log("SUCCESS", f"Prediction completed: {len(tips)} tips found")

                if "canvas_initial_drawing" in st.session_state:
                    del st.session_state.canvas_initial_drawing

                # Clear progress indicators
                time.sleep(1.5)
                progress_bar.empty()
                status_text.empty()

                # st.success(f"Analysis Complete! Detected **{len(tips)}** root tips") Commented out for this version without tips implemented

            except requests.exceptions.ConnectionError:
                progress_bar.empty()
                status_text.empty()
                error_msg = "Connection failed - Unable to connect to prediction server"
                logger.error(error_msg)
                add_frontend_log("ERROR", error_msg)
                st.error(
                    "**Connection Error:** Unable to connect to the prediction server. Please ensure the FastAPI backend is running on localhost:8000"
                )
                st.session_state.tips = []
                st.session_state.pad_img_np = None
                st.session_state.pred_mask = None

            except requests.exceptions.HTTPError as e:
                progress_bar.empty()
                status_text.empty()
                error_msg = f"HTTP error occurred - Status code: {e.response.status_code}"
                logger.error(error_msg)
                add_frontend_log("ERROR", error_msg)
                st.error(f"**HTTP Error:** Server returned error {e.response.status_code}")
                st.session_state.tips = []
                st.session_state.pad_img_np = None
                st.session_state.pred_mask = None

            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                error_msg = f"Unexpected error during prediction: {str(e)}"
                logger.error(error_msg)
                add_frontend_log("ERROR", error_msg)
                st.error(f"**Prediction Failed:** {str(e)}")
                st.info("Please check that your image is valid and the server is running correctly.")
                st.session_state.tips = []
                st.session_state.pad_img_np = None
                st.session_state.pred_mask = None

    # Display results
    render_results_section()


def render_results_section():
    """
    Render the results visualization and details section for root image predictions.

    This function retrieves prediction results and related data from Streamlit's session state,
    including detected root tips, padded grayscale image, and prediction mask, then displays:

    - An interactive visualization of the padded image with overlayed prediction mask and detected tips,
      including sliders for adjusting mask and tip opacities.
    - Zoomable visualization using mouse wheel.
    - Detailed metrics and coordinates of detected root tips.
    - Informative messages guiding the user if no predictions exist.

    The visualization uses matplotlib to create an overlay image combining grayscale image,
    mask transparency, and red tip markers.

    Returns
    -------
    None
        Renders Streamlit UI components directly and relies on session state variables:
        - 'tips': list of (y, x) coordinates of detected root tips
        - 'pad_img_np': numpy array of padded grayscale root image
        - 'pred_mask': numpy array of prediction mask overlay
        - 'backend_used': optional string naming backend model or service
        - 'uploaded_filename': uploaded image filename for status checks

    Notes
    -----
    The function expects 'tips' and 'pad_img_np' to be present in the session state for rendering
    the visualization and details. If they are missing, appropriate warnings or info messages are shown.

    Examples
    --------
    >>> import streamlit as st
    >>> from your_module import render_results_section
    >>>
    >>> # Assume prediction results stored in session state beforehand:
    >>> st.session_state.tips = [(10, 20), (15, 25)]
    >>> st.session_state.pad_img_np = np.zeros((100, 100), dtype=np.uint8)
    >>> st.session_state.pred_mask = np.zeros((100, 100), dtype=np.uint8)
    >>> st.session_state.backend_used = "ExampleModel"
    >>>
    >>> render_results_section()
    """
    shift_x, shift_y = 0, 0
    tips = st.session_state.get("tips", [])
    pad_img_np = st.session_state.get("pad_img_np", None)
    pred_mask = st.session_state.get("pred_mask", None)

    # Visualization section
    with st.expander("Predictions View", expanded=True):
        if pad_img_np is not None:
            logger.info("Rendering predictions view with mask overlay")
            st.info("Adjust the sliders below to customize the visualization")
            opacity = st.slider("Tip Opacity", min_value=0.0, max_value=1.0, value=1.0, step=0.01)
            mask_alpha = st.slider(
                "Mask Overlay Opacity", min_value=0.0, max_value=1.0, value=0.9, step=0.01
            )

            logger.info(
                f"Visualization settings - Tip opacity: {opacity}, Mask opacity: {mask_alpha}"
            )

            # Show processing status for visualization
            with st.spinner("Generating visualization..."):
                logger.info("Starting visualization generation process")

                # Pad the predicted mask
                if pred_mask is not None:
                    logger.info(f"Padding prediction mask with shift_x={shift_x}, shift_y={shift_y}")
                    pad_width = (
                        ((shift_y, 0), (shift_x, 0))
                        if shift_y >= 0 and shift_x >= 0
                        else ((0, -shift_y), (0, -shift_x))
                    )
                    pred_mask_padded = np.pad(
                        pred_mask, pad_width, mode="constant", constant_values=0
                    )
                    # Crop to original size if needed
                    pred_mask_padded = pred_mask_padded[: pad_img_np.shape[0], : pad_img_np.shape[1]]
                    logger.info(f"Prediction mask padded to shape: {pred_mask_padded.shape}")
                else:
                    pred_mask_padded = None
                    logger.warning("No prediction mask available for padding")

                logger.info("Creating matplotlib figure for visualization")
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.imshow(pad_img_np, cmap="gray", origin="upper")

                # Convert to RGBA where only the mask is visible and background is transparent
                if pred_mask_padded is not None:
                    rgba_mask = np.zeros((*pred_mask_padded.shape, 4), dtype=np.float32)
                    rgba_mask[..., 1] = 1  # Green channel for the mask
                    rgba_mask[..., 3] = (pred_mask_padded > 0).astype(
                        np.float32
                    ) * mask_alpha  # Alpha (opacity)

                    # Overlay the padded predicted mask
                    ax.imshow(rgba_mask, origin="upper", alpha=mask_alpha)
                    logger.info("Applied prediction mask overlay to visualization")
                else:
                    logger.warning("No prediction mask available for overlay")

                # Overlay the shifted tips (if any)
                if tips:
                    ys, xs = zip(*tips)
                    ax.scatter(
                        np.array(xs) + shift_x,
                        np.array(ys) + shift_y,
                        color="red",
                        s=20,
                        alpha=opacity,
                        label="Tips",
                    )

                    logger.info(f"Added {len(tips)} tip markers to visualization")
                else:
                    logger.info("No tips to display - showing mask overlay only")

                ax.axis("off")
                buf = io.BytesIO()
                plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
                plt.close(fig)
                buf.seek(0)
                overlay_img = Image.open(buf)
                logger.info("Visualization image generated successfully")

            st.success("Visualization ready! Use mouse wheel to zoom in/out")
            image_zoom(overlay_img, mode="scroll", zoom_factor=5)

        elif pad_img_np is None and st.session_state.get("uploaded_filename"):
            logger.warning("No prediction results available for visualization")
            st.warning("No prediction results available. Please run a prediction first.")
        else:
            st.info("Upload an image and click 'Run Prediction' to see results here.")

    # Details section
    with st.expander("Prediction Details", expanded=False):
        if pad_img_np is not None:
            col1, col2 = st.columns(2)

            with col1:
                st.metric("Root Tips Detected", len(tips))
                st.write("**Image Dimensions:**")
                st.write(f"• Width: {pad_img_np.shape[1]} pixels")
                st.write(f"• Height: {pad_img_np.shape[0]} pixels")

            with col2:
                if pred_mask is not None:
                    st.write("**Prediction Mask:**")
                    st.write(f"• Shape: {pred_mask.shape}")
                    st.write(f"• Data type: {pred_mask.dtype}")
                    st.write(f"• Backend: {st.session_state.get('backend_used', None)}")

            if tips:
                st.write("**Root Tip Coordinates (y, x):**")
                tips_df_data = []
                for idx, (y, x) in enumerate(tips, 1):
                    tips_df_data.append({"Tip #": idx, "Y Coordinate": y, "X Coordinate": x})

                st.dataframe(tips_df_data, use_container_width=True)
            else:
                st.info(
                    "No individual root tips detected. The mask overlay shows the predicted root regions."
                )
        elif st.session_state.get("uploaded_filename"):
            st.info("Prediction details will appear here after running analysis.")


def parse_log_line(line):
    """
    Parse a single log line string into its components.

    The expected log line format is:
    'YYYY-MM-DD HH:MM:SS | LEVEL | module_name | message'

    Parameters
    ----------
    line : str
        A single line from a log file/string in the expected format.

    Returns
    -------
    dict or None
        A dictionary with keys:
            - 'timestamp' (str): Timestamp string in 'YYYY-MM-DD HH:MM:SS' format.
            - 'level' (str): Log level (e.g., INFO, ERROR).
            - 'module' (str): Name of the module generating the log.
            - 'message' (str): The log message text.
        Returns None if the line does not match the expected format.
    """
    # Pattern: timestamp | level | module | message
    pattern = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \| (\w+) \| ([^|]+) \| (.+)"
    match = re.match(pattern, line.strip())

    if match:
        return {
            "timestamp": match.group(1),
            "level": match.group(2),
            "module": match.group(3).strip(),
            "message": match.group(4),
        }
    return None


def read_backend_logs():
    """
    Read and parse backend logs from the configured backend log file.

    The function attempts to locate the backend log file at:
    '<project_root>/src/logs/app.log' relative to this script's location.
    Each log line is parsed into a dictionary using `parse_log_line`.

    Parameters
    ----------
    None

    Returns
    -------
    logs : List[Dict[str, str]]
        A list of parsed log entries. Each entry contains keys:
        'timestamp', 'level', 'module', and 'message'.
        Returns an empty list if the log file is missing or unreadable.
    """
    # Get the project root directory and construct the correct path
    current_dir = Path(__file__).resolve().parent  # Current file location
    project_root = current_dir.parent.parent.parent  # Go up to AxonRooter/
    backend_log_path = project_root / "src" / "axonrooter" / "app" / "logs" / "app.log"

    logs = []

    try:
        if backend_log_path.exists():
            with open(backend_log_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                for line in lines:
                    parsed = parse_log_line(line)
                    if parsed:
                        logs.append(parsed)
        else:
            print(f"Backend log file not found at: {backend_log_path}")
        return logs
    except Exception as e:
        st.error(f"Error reading backend log file: {e}")
        print(f"Error reading backend log file from {backend_log_path}: {e}")
        return []


def render_logging_page():
    """
    Render the system logging page with backend and frontend logs, filters, and statistics.

    This page provides interactive filtering for backend logs based on module and level,
    displays frontend logs stored in session state, and summarizes log statistics.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    st.markdown('<div class="main-header">System Logs</div>', unsafe_allow_html=True)

    backend_logs = read_backend_logs()

    if backend_logs:
        with st.expander("Filters", expanded=True):
            # Backend Logs Section
            st.markdown('<div class="sub-header">Backend Logs</div>', unsafe_allow_html=True)

            # Module filter for backend logs
            all_modules = list(set([log["module"] for log in backend_logs]))
            selected_modules = st.multiselect(
                "Filter by Module",
                options=all_modules,
                default=all_modules,
                help="Select modules to display logs from",
            )

            # Level filter
            levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            selected_levels = st.multiselect(
                "Filter by Level",
                options=levels,
                default=["INFO", "WARNING", "ERROR"],
                help="Select log levels to display",
            )

            # Filter logs
            filtered_logs = [
                log
                for log in backend_logs
                if log["module"] in selected_modules and log["level"] in selected_levels
            ]

            # Display logs
            st.write(f"**Showing {len(filtered_logs)} of {len(backend_logs)} backend log entries**")

            # Control options
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("Refresh Logs"):
                    st.rerun()
            with col2:
                max_logs = st.selectbox("Max logs to show", [10, 20, 50, 100], index=0)

        # Create scrollable container
        with st.expander("View logs", expanded=True):
            # Start scrollable container HTML
            scroll_html = f"""
                <style>
                    .scroll-box {{
                        background-color: #ffffff;
                        padding: 10px;
                        border-radius: 5px;
                        height: 300px;
                        overflow-y: auto;
                        font-family: monospace;
                    }}
                    .log-entry {{
                        margin-bottom: 10px;
                    }}
                </style>
                <div class="scroll-box">
                """

            # Append log entries
            for log in filtered_logs[-max_logs:]:
                level = log["level"].lower()
                color = (
                    "#F44336" if level == "error" else "#FF9800" if level == "warning" else "#2196F3"
                )

                log_html = f"""
                <div class="log-entry">
                    <strong>{log['timestamp']}</strong> |
                    <span style="color: {color};">{log['level']}</span> |
                    <em>{log['module']}</em><br>
                    {log['message']}
                </div>
                """
                scroll_html += log_html

            # Close scrollable div
            scroll_html += "</div>"

            # Render all at once
            st.markdown(scroll_html, unsafe_allow_html=True)

    else:
        st.info("No backend logs found. Ensure the backend is running and generating logs.")

    with st.expander("View frontend logs", expanded=True):
        # Frontend Logs Section
        st.markdown('<div class="sub-header">Frontend Logs</div>', unsafe_allow_html=True)

        frontend_logs = st.session_state.get("frontend_logs", [])

        # Ensure max_logs is defined (default to 10 if not set above)
        max_logs = 10

        if frontend_logs:
            st.write(f"**Showing {len(frontend_logs)} frontend log entries**")

            with st.container():
                for log in reversed(frontend_logs[-max_logs:]):  # Show latest first
                    level = log["level"].lower()
                    css_class = (
                        f"log-{level}"
                        if level in ["info", "error", "warning", "debug"]
                        else "log-info"
                    )

                    st.markdown(
                        f"""
                    <div class="log-entry {css_class}">
                        <strong>{log['timestamp']}</strong> |
                        <span style="color: {'#4CAF50' if level == 'success' else '#F44336' if level == 'error' else '#FF9800' if level == 'warning' else '#2196F3'}">
                            {log['level']}
                        </span> |
                        <em>{log['module']}</em><br>
                        {log['message']}
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )
        else:
            st.info(
                "No frontend logs available. Frontend logs will appear here as you use the application."
            )

    # Log statistics
    with st.expander("View Logs Statistics", expanded=True):
        if backend_logs or frontend_logs:
            st.divider()
            st.markdown('<div class="sub-header">Log Statistics</div>', unsafe_allow_html=True)

            with st.container():
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Backend Logs", len(backend_logs))
                with col2:
                    st.metric("Frontend Logs", len(frontend_logs))
                with col3:
                    error_count = len([log for log in backend_logs if log["level"] == "ERROR"])
                    st.metric("Backend Errors", error_count)
                with col4:
                    frontend_errors = len([log for log in frontend_logs if log["level"] == "ERROR"])
                    st.metric("Frontend Errors", frontend_errors)


stroke_opacity = 0.7  # Default stroke opacity for canvas drawing
opacity = 0.7  # Default opacity for mask overlay in visualization


def create_drawing_canvas():
    """
    Create a drawable canvas configuration dictionary for mask correction.

    Returns
    -------
    dict
        A dictionary containing settings for the drawing canvas, including:
        - drawing_mode (str): The current drawing mode (e.g., 'freedraw').
        - stroke_width (int): The width of the brush stroke.
        - stroke_color (str): The color of the brush stroke in RGBA format.
        - fill_color (str): The fill color for shapes in RGBA format.
        - point_display_radius (int): The radius for displaying points in point mode.
        - realtime_update (bool): Whether the canvas updates in real time.
        - display_toolbar (bool): Whether to show the drawing toolbar.
    """
    # Canvas configuration
    # st.sidebar.subheader("Drawing Tools")

    # Drawing tool selector
    drawing_mode = "freedraw"

    # Tool parameters
    stroke_width = 3
    stroke_color = f"rgba(0, 255, 0, {stroke_opacity})"
    fill_color = f"rgba(0, 255, 0, 0)"

    # Point display radius for point mode
    point_display_radius = 0
    if drawing_mode == "point":
        point_display_radius = 3

    # Canvas settings
    realtime_update = True
    display_toolbar = True

    return {
        "drawing_mode": drawing_mode,
        "stroke_width": stroke_width,
        "stroke_color": stroke_color,
        "fill_color": fill_color,
        "point_display_radius": point_display_radius,
        "realtime_update": realtime_update,
        "display_toolbar": display_toolbar,
    }


def create_inline_drawing_controls():
    """
    Create inline drawing controls for mask correction on the Mask Submit page.

    Returns
    -------
    dict
        A dictionary containing the current drawing control settings:
        - drawing_mode (str): Selected drawing tool ('freedraw', 'line', 'eraser', etc.).
        - stroke_width (int): Width of the drawing stroke.
        - opacity (float): Opacity of the stroke.
        - point_display_radius (int): Radius for point display (only if drawing_mode is 'point').
        - realtime_update (bool): Whether to update drawing in real time.
        - display_toolbar (bool): Whether to display the drawing toolbar.
    """

    (
        col1,
        col2,
        col3,
        col4,
        col5,
    ) = st.columns(5)

    with col1:
        drawing_mode = st.selectbox(
            "Drawing tool:",
            ("freedraw", "line", "eraser"),
            help="Select the drawing tool to correct or annotate the mask",
            key="drawing_mode",
        )

        # Check if drawing mode has changed and automatically update objects
        if "previous_drawing_mode" not in st.session_state:
            st.session_state.previous_drawing_mode = drawing_mode

    with col2:
        stroke_width = st.slider("Stroke width:", 1, 25, 3)

    with col3:
        opacity = st.slider("Stroke opacity:", 0.1, 1.0, 0.7, 0.1)

        # Check if opacity has changed and automatically update objects
        if "previous_opacity" not in st.session_state:
            st.session_state.previous_opacity = opacity

    with col4:
        realtime_update = True

    with col5:
        display_toolbar = True

    # Point display radius for point mode (only show if needed)
    point_display_radius = 0
    if drawing_mode == "point":
        point_display_radius = st.slider("Point display radius:", 1, 25, 3)

    return {
        "drawing_mode": drawing_mode,
        "stroke_width": stroke_width,
        "opacity": opacity,
        "point_display_radius": point_display_radius,
        "realtime_update": realtime_update,
        "display_toolbar": display_toolbar,
    }


def convert_mask_to_canvas_json(pred_mask, canvas_width=600, canvas_height=600, opacity=0.7):
    """
    Convert a prediction mask to Fabric.js JSON format for canvas initialDrawing.

    Parameters
    ----------
    pred_mask : np.ndarray
        Numpy array of the prediction mask.
    canvas_width : int, optional
        Target width of the canvas (default is 600).
    canvas_height : int, optional
        Target height of the canvas (default is 600).
    opacity : float, optional
        Opacity for the mask stroke (0.0 to 1.0, default is 0.7).

    Returns
    -------
    dict
        Fabric.js compatible JSON object representing the mask as canvas objects.
    """

    if pred_mask is None:
        return {"version": "5.2.4", "objects": []}

    try:
        # Get mask dimensions
        mask_height, mask_width = pred_mask.shape

        # Calculate scaling factors to fit canvas
        scale_x = canvas_width / mask_width
        scale_y = canvas_height / mask_height

        # Find contours in the mask
        import cv2

        # Ensure mask is binary (0 or 255)
        binary_mask = (pred_mask > 0).astype(np.uint8) * 255

        # Find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        objects = []

        for i, contour in enumerate(contours):
            # Skip very small contours
            if cv2.contourArea(contour) < 10:
                continue

            # Convert contour to path points
            points = []
            for point in contour:
                x, y = point[0]
                # Scale coordinates to canvas size
                scaled_x = x * scale_x
                scaled_y = y * scale_y
                points.append([scaled_x, scaled_y])

            if len(points) < 3:  # Need at least 3 points for a polygon
                continue

            # Create a Fabric.js path object
            path_string = f"M {points[0][0]} {points[0][1]}"
            for point in points[1:]:
                path_string += f" L {point[0]} {point[1]}"
            path_string += " Z"  # Close the path

            fabric_object = {
                "type": "path",
                "version": "5.2.4",
                "originX": "left",
                "originY": "top",
                "left": min(p[0] for p in points),
                "top": min(p[1] for p in points),
                "width": max(p[0] for p in points) - min(p[0] for p in points),
                "height": max(p[1] for p in points) - min(p[1] for p in points),
                "fill": f"rgba(0, 255, 0, 0)",  # No fill, always transparent
                "stroke": f"rgba(0, 255, 0, {opacity})",  # Use dynamic opacity
                "strokeWidth": 2,
                "strokeDashArray": None,
                "strokeLineCap": "butt",
                "strokeDashOffset": 0,
                "strokeLineJoin": "miter",
                "strokeUniform": False,
                "strokeMiterLimit": 4,
                "scaleX": 1,
                "scaleY": 1,
                "angle": 0,
                "flipX": False,
                "flipY": False,
                "opacity": 1.0,  # CHANGED: Set to 1.0 since opacity is handled by stroke color
                "shadow": None,
                "visible": True,
                "backgroundColor": "",
                "fillRule": "nonzero",
                "paintFirst": "fill",
                "globalCompositeOperation": "source-over",
                "skewX": 0,
                "skewY": 0,
                "path": path_string,
                "pathOffset": {"x": 0, "y": 0},
            }

            objects.append(fabric_object)

        canvas_json = {"version": "5.2.4", "objects": objects}

        logger.info(
            f"Converted mask to canvas JSON with {len(objects)} objects at opacity {opacity}"
        )
        add_frontend_log("INFO", f"Converted prediction mask to {len(objects)} canvas objects")

        return canvas_json

    except Exception as e:
        logger.error(f"Error converting mask to canvas JSON: {str(e)}")
        add_frontend_log("ERROR", f"Mask conversion failed: {str(e)}")
        return {"version": "5.2.4", "objects": []}


def update_canvas_with_prediction_mask(opacity=0.7):
    """
    Update session state with canvas-compatible prediction mask.

    Parameters
    ----------
    opacity : float, optional
        Opacity value for the mask objects (0.0 to 1.0), by default 0.7

    Returns
    -------
    None
    """
    pred_mask = st.session_state.get("pred_mask", None)

    if pred_mask is not None:
        # Convert mask to canvas JSON format with dynamic opacity
        canvas_json = convert_mask_to_canvas_json(pred_mask, opacity=opacity)

        # Store in session state for canvas initialization
        st.session_state.canvas_initial_drawing = canvas_json

        logger.info(f"Updated canvas initial drawing with prediction mask at opacity {opacity}")
        add_frontend_log("INFO", f"Canvas initialized with prediction mask (opacity: {opacity})")
    else:
        # Empty canvas if no mask
        st.session_state.canvas_initial_drawing = {"version": "5.2.4", "objects": []}


def render_mask_submit_page():
    """Render the Mask Submit page with drawable canvas"""
    # Page header
    st.markdown('<div class="main-header">Mask Submit</div>', unsafe_allow_html=True)

    # Check if prediction results are available
    pad_img_np = st.session_state.get("pad_img_np", None)
    pred_mask = st.session_state.get("pred_mask", None)
    tips = st.session_state.get("tips", [])
    uploaded_filename = st.session_state.get("uploaded_filename", None)

    if pad_img_np is None or uploaded_filename is None:
        st.markdown(
            """
            <div class="info-box">
                <div class="sub-header">No Prediction Data Available</div>
                <p>Please complete a prediction first before submitting mask corrections.</p>
                <p>Go to <strong>Prediction & Analysis</strong> to upload and analyze an image.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # Page description
    st.markdown(
        f"""
        <div class="info-box">
            <div class="sub-header">Correct Prediction Mask</div>
            <p>Use the drawing tools to correct the predicted mask for: <strong>{uploaded_filename}</strong></p>
            <p>• The predicted mask is already loaded on the canvas</p>
            <p>• Draw missing root tips or areas</p>
            <p>• Use eraser to remove incorrect predictions</p>
            <p>• Use different tools to annotate specific regions</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Drawing Controls Section BEFORE canvas creation
    with st.expander("Drawing Controls", expanded=True):
        st.markdown("### Tool Settings")

        # Get inline drawing controls
        drawing_controls = create_inline_drawing_controls()

    # Get canvas configuration from sidebar (now only colors) and combine with drawing controls
    sidebar_config = create_drawing_canvas()

    # MODIFIED: Create stroke color with opacity from controls
    opacity = drawing_controls.get("opacity", 0.7)
    stroke_color = f"rgba(0, 255, 0, {opacity})"  # Use dynamic opacity from controls
    fill_color = f"rgba(0, 255, 0, 0)"  # Keep fill transparent

    # Initialize canvas initial drawing only once per prediction
    canvas_init_key = f"canvas_init_{uploaded_filename}"
    if canvas_init_key not in st.session_state and pred_mask is not None:
        # First time for this prediction - initialize with prediction mask
        update_canvas_with_prediction_mask(opacity=opacity)
        st.session_state[canvas_init_key] = True
        logger.info(f"Initialized canvas for {uploaded_filename}")

    # Combine all canvas configurations
    canvas_config = {
        **sidebar_config,
        **drawing_controls,
        "stroke_color": stroke_color,  # Override with opacity-adjusted color
        "fill_color": fill_color,
    }

    # Main canvas section
    with st.expander("Mask Correction Canvas", expanded=True):
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("### Draw Your Corrections")

            # Prepare background image
            background_image = None
            if pad_img_np is not None:
                # Convert grayscale to RGB for canvas
                if len(pad_img_np.shape) == 2:
                    background_rgb = np.stack([pad_img_np] * 3, axis=-1)
                else:
                    background_rgb = pad_img_np

                # Convert to PIL Image
                background_image = Image.fromarray(background_rgb.astype("uint8"))
                logger.info(f"Prepared background image with shape: {background_rgb.shape}")

            # FIXED: Save current canvas state when drawing mode changes
            current_drawing_mode = drawing_controls.get("drawing_mode", "freedraw")

            # Initialize previous_drawing_mode if not exists
            if "previous_drawing_mode" not in st.session_state:
                st.session_state.previous_drawing_mode = current_drawing_mode

            # Check if drawing mode changed and preserve canvas state
            if st.session_state.previous_drawing_mode != current_drawing_mode:
                # Get the current canvas result from a special session state key
                canvas_state_key = f"canvas_state_{uploaded_filename}"
                if canvas_state_key in st.session_state:
                    # Preserve the current canvas drawings
                    current_canvas_data = st.session_state[canvas_state_key]
                    if current_canvas_data and current_canvas_data.get("objects"):
                        st.session_state.canvas_initial_drawing = current_canvas_data
                        logger.info(
                            f"Preserved canvas state when changing from {st.session_state.previous_drawing_mode} to {current_drawing_mode}"
                        )

                st.session_state.previous_drawing_mode = current_drawing_mode

            # Use initial drawing but don't force reinit
            initial_drawing = st.session_state.get(
                "canvas_initial_drawing", {"version": "5.2.4", "objects": []}
            )

            # FIXED: Use stable canvas key that doesn't change with drawing mode
            canvas_key = f"mask_correction_canvas_{uploaded_filename}"  # Removed drawing_mode

            # Ensure background_image is valid for st_canvas
            if background_image is None:
                # Create a default white background if no image is available
                background_image = Image.new("RGB", (600, 600), (255, 255, 255))
                logger.warning("No background image available, using default white background")

            canvas_result = st_canvas(
                fill_color=canvas_config["fill_color"],
                stroke_width=canvas_config["stroke_width"],
                stroke_color=stroke_color,  # This will affect NEW drawings only
                background_image=background_image,
                update_streamlit=canvas_config["realtime_update"],
                height=600,
                width=600,
                drawing_mode=canvas_config["drawing_mode"],
                point_display_radius=canvas_config["point_display_radius"],
                display_toolbar=canvas_config["display_toolbar"],
                initial_drawing=initial_drawing,
                key=canvas_key,  # Stable key - doesn't change with drawing mode
            )

            # ADDED: Always save the current canvas state after each interaction
            if canvas_result and canvas_result.json_data is not None:
                canvas_state_key = f"canvas_state_{uploaded_filename}"
                st.session_state[canvas_state_key] = canvas_result.json_data

            # Show opacity change notice
            st.info(
                f"💡 **Current opacity setting: {opacity:.1f}** - This affects new drawings only. Existing drawings keep their original opacity."
            )

        with col2:
            # Display canvas statistics
            if canvas_result.json_data is not None:
                objects = canvas_result.json_data.get("objects", [])
                st.metric("Drawing Objects", len(objects))

                if objects:
                    object_types = {}
                    for obj in objects:
                        obj_type = obj.get("type", "unknown")
                        object_types[obj_type] = object_types.get(obj_type, 0) + 1

                    st.write("**Object Types:**")
                    for obj_type, count in object_types.items():
                        st.write(f"• {obj_type}: {count}")

            # Show current drawing tool settings
            st.markdown("### Current Tool Settings")
            st.write(f"**Mode:** {canvas_config['drawing_mode']}")
            st.write(f"**Stroke Width:** {canvas_config['stroke_width']}")
            st.write(f"**New Drawing Opacity:** {opacity:.1f}")
            st.write(f"**Stroke Color:** `{stroke_color}`")

            # Add opacity management options
            st.markdown("### Opacity Management")

            # IMPROVED: Handle opacity changes with current canvas state
            if st.session_state.previous_opacity != opacity:
                if canvas_result.json_data is not None:
                    # Update all existing objects to use current opacity
                    updated_objects = []
                    for obj in canvas_result.json_data.get("objects", []):
                        # Create a copy of the object with updated opacity
                        updated_obj = obj.copy()
                        if obj.get("type") in ["path", "line", "circle", "rect"]:
                            # Update stroke color to use current opacity
                            current_stroke = obj.get("stroke", "rgba(0, 255, 0, 0.7)")
                            if current_stroke.startswith("rgba"):
                                # Extract RGB values and apply new opacity
                                updated_obj["stroke"] = f"rgba(0, 255, 0, {opacity})"
                            updated_obj["opacity"] = 1.0  # Keep object opacity at 1.0
                        updated_objects.append(updated_obj)

                    # Update session state with modified objects
                    st.session_state.canvas_initial_drawing = {
                        "version": "5.2.4",
                        "objects": updated_objects,
                    }
                    st.session_state.previous_opacity = opacity
                    logger.info(f"Updated all canvas objects to opacity {opacity}")
                    add_frontend_log("INFO", f"Canvas opacity updated to {opacity}")
                    st.rerun()
                else:
                    st.session_state.previous_opacity = opacity

            if st.button("🔄 Reset to Original Prediction", use_container_width=True):
                # Reset canvas to original prediction mask
                if pred_mask is not None:
                    update_canvas_with_prediction_mask(opacity=opacity)
                    # Clear the saved canvas state
                    canvas_state_key = f"canvas_state_{uploaded_filename}"
                    if canvas_state_key in st.session_state:
                        del st.session_state[canvas_state_key]
                    st.success("Canvas reset to original prediction")
                    st.rerun()
                else:
                    st.warning("No original prediction to reset to.")

            # ... rest of the function remains the same (Drawing Details, Original Prediction Reference, etc.)

            st.markdown("Upload your plant root image to begin analysis")
            uploaded_ct_images = st.file_uploader(
                "Choose an image file",
                type=["jpg", "jpeg", "png"],
                help="Supported formats: JPG, JPEG, PNG. Max file size: 200MB",
                accept_multiple_files=True,
            )
            uploaded_ct_masks = st.file_uploader(
                "Upload a mask file",
                type=["tif"],
                help="Upload a mask file in JSON or CSV format to correct the prediction",
                accept_multiple_files=True,
            )

            if st.button(
                "Submit Corrections",
                use_container_width=True,
                help="Submit your mask corrections for further processing",
            ):
                if canvas_result.json_data is not None:
                    # Convert canvas JSON to mask format
                    logger.info("Starting dataset submission process")
                    add_frontend_log("INFO", "Starting dataset submission process")

                    # Show progress
                    with st.spinner("Processing submission..."):
                        try:
                            # Step 1: Validate uploads
                            if (
                                not uploaded_ct_images
                                and not uploaded_ct_masks
                                and canvas_result.json_data.get("objects", []) == []
                            ):
                                st.warning(
                                    "Please upload files or draw corrections before submitting."
                                )
                                return

                            # Step 2: Test Azure CLI first
                            st.info("Checking Azure CLI connection...")
                            cli_working, cli_message = test_azure_cli_connection_with_workspace()
                            if not cli_working:
                                st.error(f"❌ Azure CLI Issue: {cli_message}")
                                st.markdown(
                                    """
                                **Troubleshooting Steps:**
                                1. Install Azure CLI: https://aka.ms/installazurecliwindows
                                2. Install ML extension: `az extension add -n ml`
                                3. Login: `az login`
                                4. Restart Streamlit app
                                """
                                )
                                add_frontend_log("ERROR", f"Azure CLI issue: {cli_message}")
                                return

                            st.success(f"✅ Azure CLI connection verified")

                            # Step 3: Upload files to Azure ML
                            st.info("Creating Azure ML data asset...")
                            (
                                data_asset_name,
                                saved_files,
                                upload_result,
                            ) = upload_files_to_azure_datastore(
                                uploaded_ct_images, uploaded_ct_masks, canvas_result.json_data
                            )

                            if data_asset_name is None:
                                st.error(f"❌ Failed to create data asset")
                                st.error(f"**Error:** {upload_result}")
                                add_frontend_log(
                                    "ERROR", f"Data asset creation failed: {upload_result}"
                                )
                                return

                            # SUCCESS - Show results without nested expanders
                            st.success(f"🎉 Data asset successfully created!")

                            # Create info box with results
                            st.markdown(
                                f"""
                            <div style="background: #f0f8ff; padding: 1rem; border-radius: 8px; border-left: 4px solid #228B22;">
                                <h4 style="color: #228B22; margin-top: 0;">✅ Submission Successful</h4>
                                <p><strong>Data Asset Name:</strong> {data_asset_name}</p>
                                <p><strong>Files Processed:</strong></p>
                                <ul>
                                    <li>� Training Images: {len(saved_files['train_images'])}</li>
                                    <li>🎭 Training Masks: {len(saved_files['train_masks'])}</li>
                                    <li>🧪 Validation Images: {len(saved_files['val_images'])}</li>
                                    <li>🎯 Validation Masks: {len(saved_files['val_masks'])}</li>
                                </ul>
                            </div>
                            """,
                                unsafe_allow_html=True,
                            )

                            add_frontend_log("SUCCESS", f"Data asset created: {data_asset_name}")

                            # Show additional details in separate sections (not nested)
                            st.markdown("---")
                            st.markdown("### 📋 Submission Details")

                            # Azure ML details
                            st.markdown("**Azure ML Registration:**")
                            st.code(upload_result, language="text")

                            # File details
                            st.markdown("**Uploaded Files:**")
                            for category, files in saved_files.items():
                                if files:
                                    st.markdown(f"**{category.title()}:**")
                                    for file_path in files:
                                        file_name = Path(file_path).name
                                        st.markdown(f"- {file_name}")

                        except Exception as e:
                            error_msg = f"Error during submission: {str(e)}"
                            logger.error(error_msg)
                            add_frontend_log("ERROR", error_msg)
                            st.error(f"❌ Submission failed: {str(e)}")

                            # Show debugging information (not in expander)
                            st.markdown("---")
                            st.markdown("### 🔧 Debugging Information")
                            st.markdown(
                                f"""
                            **Target Datastore:** {DATASTORE_NAME}
                            **Target Directory:** {BLOB_SUBDIR}
                            **Data Asset Name:** {DATA_ASSET_NAME}
                            **Error:** {str(e)}
                            """
                            )

                else:
                    st.warning("No canvas data available. Please draw corrections first.")

    # Show original prediction for reference
    with st.expander("Original Prediction Reference", expanded=False):
        if pred_mask is not None and tips:
            st.markdown("### Original Prediction vs Your Corrections")

            col_orig, col_corrected = st.columns(2)

            with col_orig:
                st.markdown("**Original Prediction:**")

                # Create visualization of original prediction
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.imshow(pad_img_np, cmap="gray", origin="upper")

                # Overlay original mask
                if pred_mask is not None:
                    rgba_mask = np.zeros((*pred_mask.shape, 4), dtype=np.float32)
                    rgba_mask[..., 1] = 1  # Green channel
                    rgba_mask[..., 3] = (pred_mask > 0).astype(np.float32) * 0.7
                    ax.imshow(rgba_mask, origin="upper")

                # Overlay original tips
                if tips:
                    ys, xs = zip(*tips)
                    ax.scatter(xs, ys, color="red", s=20, alpha=0.8, label="Original Tips")

                ax.axis("off")

                buf = io.BytesIO()
                plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
                plt.close(fig)
                buf.seek(0)
                orig_img = Image.open(buf)
                image_zoom(orig_img, mode="scroll", zoom_factor=5)

            with col_corrected:
                st.markdown("**Your Corrections:**")
                if canvas_result.image_data is not None:
                    image_zoom(canvas_result.image_data, mode="scroll", zoom_factor=5)
                else:
                    st.info("Your corrections will appear here as you draw.")

    # Drawing details section
    with st.expander("Drawing Details", expanded=False):
        if canvas_result.json_data is not None:
            objects = canvas_result.json_data.get("objects", [])

            if objects:
                st.write(f"**Canvas contains {len(objects)} drawing objects:**")

                # Convert objects to DataFrame for display
                objects_df_data = []
                for idx, obj in enumerate(objects, 1):
                    # Extract opacity from stroke color if available
                    stroke_opacity = "N/A"
                    stroke_color_val = obj.get("stroke", "none")
                    if stroke_color_val.startswith("rgba"):
                        try:
                            # Extract opacity from rgba string
                            opacity_match = re.search(
                                r"rgba\(\d+,\s*\d+,\s*\d+,\s*([\d.]+)\)", stroke_color_val
                            )
                            if opacity_match:
                                stroke_opacity = float(opacity_match.group(1))
                        except (ValueError, AttributeError) as e:
                            logger.error(
                                f"Failed to parse stroke color: {stroke_color_val}, error: {e}"
                            )
                            pass

                    obj_data = {
                        "Object #": idx,
                        "Type": obj.get("type", "unknown"),
                        "Left": round(obj.get("left", 0), 1),
                        "Top": round(obj.get("top", 0), 1),
                        "Width": round(obj.get("width", 0), 1) if obj.get("width") else "N/A",
                        "Height": round(obj.get("height", 0), 1) if obj.get("height") else "N/A",
                        "Stroke": obj.get("stroke", "none"),
                        "Fill": obj.get("fill", "none"),
                        "Opacity": stroke_opacity,
                    }
                    objects_df_data.append(obj_data)

                st.dataframe(objects_df_data, use_container_width=True)

                # Export options
                st.markdown("### Export Options")
                col_json, col_csv = st.columns(2)

                with col_json:
                    if st.button("Export as JSON", use_container_width=True):
                        json_str = json.dumps(canvas_result.json_data, indent=2)
                        st.download_button(
                            label="Download JSON",
                            data=json_str,
                            file_name=f"mask_corrections_{uploaded_filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json",
                        )

                with col_csv:
                    if st.button("Export as CSV", use_container_width=True):
                        df = pd.DataFrame(objects_df_data)
                        csv_str = df.to_csv(index=False)
                        st.download_button(
                            label="Download CSV",
                            data=csv_str,
                            file_name=f"mask_corrections_{uploaded_filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                        )
            else:
                st.info("No drawing objects yet. Start drawing on the canvas to see details here.")
        else:
            st.info("Canvas data will appear here once you start drawing.")


def main():
    """
    Main application function to render the Streamlit web app interface.

    This function initializes session state, sets up the fonts and layout,
    renders the sidebar navigation and feedback form,
    and routes to the appropriate page content based on user selection.
    It handles feedback form submission and saves the
    data to a CSV file, displaying success or error messages accordingly.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Raises
    ------

    Notes
    -----
    - Relies on `st.session_state` for managing UI state and data persistence.
    - Sidebar includes navigation radio buttons and a feedback form with checkboxes.
    - Feedback is saved to a local CSV file named `feedback.csv`.
    - Uses several helper functions to render specific pages:
      `render_homepage()`, `render_prediction_page()`, `render_logging_page()`, and
      `render_mask_submit_page()`.
    - Feedback form submission state is tracked to display confirmation and allow reset.
    """
    # ====== ALL CSS STYLING IN ONE PLACE ======
    st.markdown(
        f"""
        <style>
        /* ===== GLOBAL BACKGROUND AND FOUNDATION STYLES ===== */
        .stApp {{
            background:
                linear-gradient(to right, rgba(255, 255, 255, 0.5) 0%, rgba(255, 255, 255, 0.0) 100%),
                url("data:image/jpg;base64,{img_base64}") no-repeat center center fixed;
            background-size: cover;
        }}

        /* ===== TYPOGRAPHY STYLES ===== */
        .main-header {{
            font-size: 3rem;
            color: #FFFFFF;
            text-align: center;
            margin-bottom: 2rem;
            text-shadow: 3px 3px 10px rgba(0, 0, 0, 0.8);
            font-weight: 480;
        }}
        .sub-header {{
            font-size: 1.5rem;
            color: #228B22;
            margin-bottom: 1rem;
        }}

        /* ===== CONTENT CONTAINER STYLES ===== */
        .info-box {{
            background-color: #F0F8FF;
            color: #000000;
            padding: 1rem;
            border-radius: 12px;
            border-left: 5px solid #228B22;
            margin: 2rem 0;
        }}
        .feature-box {{
            background-color: #F5F5DC;
            color: #000000;
            padding: 0;
            border-radius: 10px;
            margin: 0.5rem 0;
        }}
        .filters-container {{
            background-color: #F5F7F8;
            padding: 1.5rem;
            border-radius: 10px;
            border: 1px solid #DEE2E6;
            margin-bottom: 1rem;
        }}
        .log-metrics-box {{
            background-color: #F5F7F8;
            padding: 1.2rem;
            border-radius: 10px;
            border: 1px solid #DEE2E6;
            margin-top: 1rem;
        }}
        .custom-expander {{
            background-color: #F5F7F8;
            border: 1px solid #DEE2E6;
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
            font-family: 'Segoe UI', sans-serif;
        }}
        .custom-expander-title {{
            font-size: 1.4rem;
            font-weight: 700;
            color: #228B22;
            cursor: pointer;
            margin-bottom: 0.5rem;
        }}

        /* ===== LOG ENTRY STYLES ===== */
        .log-entry {{
            font-family: 'Courier New', monospace;
            font-size: 0.8rem;
            padding: 0rem;
            margin: 0.2rem 0;
            border-radius: 5px;
            color: #000000;
        }}

        /* ===== SIDEBAR STYLES ===== */
        section[data-testid="stSidebar"] {{
            background-color: #F5F7F8 !important;
            color: #000000 !important;
            border-right: 1px solid #DEE2E6;
        }}
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] .stMarkdown,
        section[data-testid="stSidebar"] * {{
            color: #416750 !important;
        }}
        .sidebar-feedback {{
            background-color: #F5F7F8;
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid #DEE2E6;
            margin: 1rem 0;
            color: #000000 !important;
        }}

        /* ===== STREAMLIT COMPONENT OVERRIDES ===== */
        div.stButton > button {{
            background-color: #228B22 !important;
            color: white !important;
            font-weight: bold !important;
            border-radius: 8px !important;
            padding: 0.6rem 1.2rem !important;
            border: none !important;
            transition: background-color 0.3s ease !important;
        }}
        div.stButton > button:hover {{
            background-color: #1a5e1a !important;
            color: white !important;
        }}

        /* ===== LAYOUT CONTAINER OVERRIDES ===== */
        div:has(> .sub-header) + div[data-testid="stHorizontalBlock"] {{
            background-color: #F5F7F8 !important;
            border-radius: 10px;
            padding: 2rem;
            border: 1px solid #DEE2E6;
            margin-bottom: 1rem;
        }}
        div[data-testid="stExpander"] {{
            background-color: #F5F7F8 !important;
            border: 1px solid #DEE2E6;
            border-radius: 8px;
            padding: 0.5rem;
            margin-bottom: 1rem;
        }}
        div[data-testid="stExpander"] > div {{
            background-color: #F5F7F8 !important;
            border-radius: 0 0 8px 8px;
            padding: 1rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    initialize_session_state()

    # Sidebar navigation
    st.sidebar.title("Navigation")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Choose a section:",
        ["Home", "Prediction & Analysis", "System Logs", "Mask Submit"],
        index=["Home", "Prediction & Analysis", "System Logs", "Mask Submit"].index(
            st.session_state.current_page
        ),
    )

    # Feedback form in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Feedback")

    pad_img_np = st.session_state.get("pad_img_np", None)
    tips = st.session_state.get("tips", [])

    if pad_img_np is not None and tips:
        with st.sidebar.form("feedback"):
            st.markdown("**Help us improve the model:**")
            st.info("Select all issues that apply:")

            fb_correct = st.checkbox("Tips are correct", key="tips_correct")
            fb_outofbounds = st.checkbox("Tips out of bounds", key="tips_oob")
            fb_rootoverlay = st.checkbox("Root overlay interferes", key="root_overlay")
            fb_thinroot = st.checkbox("Thin roots interfere", key="thin_root")
            fb_missing_tips = st.checkbox("Missing tips", key="missing_tips")
            fb_wrong_tips = st.checkbox("Wrong tips", key="wrong_tips")
            fb_obs_root = st.checkbox("Obscured roots", key="obscured_roots")

            submitted = st.form_submit_button("Submit Feedback", use_container_width=True)

            if submitted:
                logger.info("User submitted feedback form")
                add_frontend_log("INFO", "User submitted feedback form")

                with st.spinner("Saving feedback..."):
                    try:
                        feedback_file = os.path.join(os.path.dirname(__file__), "feedback.csv")
                        file_exists = os.path.isfile(feedback_file)

                        logger.info(f"Saving feedback to file: {feedback_file}")

                        with open(feedback_file, "a", newline="") as csvfile:
                            writer = csv.writer(csvfile)
                            if not file_exists:
                                logger.info("Creating new feedback CSV file with headers")
                                writer.writerow(
                                    [
                                        "file_name",
                                        "tips_correct",
                                        "tips_oob",
                                        "root_overlay",
                                        "thin_root",
                                        "missing_tips",
                                        "wrong_tips",
                                        "obscured_roots",
                                    ]
                                )

                            file_name = st.session_state.get("uploaded_filename", "")
                            feedback_data = [
                                file_name,
                                fb_correct,
                                fb_outofbounds,
                                fb_rootoverlay,
                                fb_thinroot,
                                fb_missing_tips,
                                fb_wrong_tips,
                                fb_obs_root,
                            ]
                            writer.writerow(feedback_data)
                            logger.info(f"Feedback saved for file: {file_name}")
                            add_frontend_log("SUCCESS", f"Feedback saved for: {file_name}")

                        st.session_state.feedback_submitted = True

                    except Exception as e:
                        logger.error(f"Error saving feedback: {str(e)}")
                        add_frontend_log("ERROR", f"Feedback save failed: {str(e)}")
                        st.error(f"Error saving feedback: {str(e)}")

        if st.session_state.feedback_submitted:
            st.sidebar.success("Thank you for your feedback!")
            # Reset feedback submitted state after showing success
            if st.sidebar.button("Reset Feedback Form"):
                st.session_state.feedback_submitted = False
                st.rerun()
    else:
        st.sidebar.info("Complete a prediction to provide feedback")

    st.sidebar.markdown("__Contact Us__")
    st.sidebar.markdown("__Email__: ")
    st.sidebar.markdown("230446@buas.nl (Gabriel Wang)")
    st.sidebar.markdown("234924@buas.nl (Endijs Kiršteins)")
    st.sidebar.markdown("231007@buas.nl (Emil Fox)")
    st.sidebar.markdown("232079@buas.nl (Michał Bątkowski)")
    st.sidebar.markdown(
        "__GitHub Page:__ https://github.com/BredaUniversityADSAI/2024-25d-fai2-adsai-group-cv9"
    )
    st.sidebar.markdown(
        "__Documentation Website:__ https://improved-adventure-7jp1e7g.pages.github.io/"
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Version:** 2.0.0")
    st.sidebar.markdown("**Status:** Online")
    st.sidebar.markdown("---")

    # Source references
    st.sidebar.markdown("**Sources:**")
    st.sidebar.markdown("[Drawable Canvas](https://github.com/andfanilo/streamlit-drawable-canvas)")
    st.sidebar.markdown(
        "[Canvas Enhancement](https://github.com/andfanilo/streamlit-drawable-canvas/pull/153)"
    )
    st.sidebar.markdown("[Image Zoom](https://github.com/vgilabert94/streamlit-image-zoom)")

    # Route to appropriate page
    if page == "Home":
        render_homepage()
    elif page == "Prediction & Analysis":
        render_prediction_page()
    elif page == "System Logs":
        render_logging_page()
    elif page == "Mask Submit":
        render_mask_submit_page()


if __name__ == "__main__":
    main()
