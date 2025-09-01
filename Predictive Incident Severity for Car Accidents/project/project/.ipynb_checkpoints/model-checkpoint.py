import os
import gzip
import logging
import pickle

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,  # Set logging level as needed
)

def load_random_forest_model(model_path=None):
    """
    Load the trained Random Forest model from a compressed file.

    Parameters:
    model_path (str): Path to the compressed model file. If None, defaults to 'random_forest_model.pkl.gz'.

    Returns:
    RandomForestClassifier: The loaded Random Forest model.
    """
    if model_path is None:
        model_path = os.path.join(os.path.dirname(__file__), "random_forest_model.pkl.gz")

    try:
        logging.info(f"Loading Random Forest model from '{model_path}'...")
        with gzip.open(model_path, "rb") as f:
            rf_model = pickle.load(f)
        logging.info("Random Forest model loaded successfully.")
        return rf_model
    except FileNotFoundError:
        logging.error(f"Model file '{model_path}' not found.")
        raise FileNotFoundError(f"Model file '{model_path}' not found.")
    except Exception as e:
        logging.error(f"Error loading model from '{model_path}': {str(e)}")
        raise RuntimeError(f"Error loading model from '{model_path}': {str(e)}")
