import pickle
import gzip

def load_random_forest_model(model_path='random_forest_model.pkl.gz'):
    # Load the trained Random Forest model from the compressed file
    with gzip.open(model_path, 'rb') as f:
        rf_model = pickle.load(f)
    return rf_model

# Usage example
model_path = 'random_forest_model.pkl.gz'
rf_model = load_random_forest_model(model_path)
print("Model loaded successfully.")
