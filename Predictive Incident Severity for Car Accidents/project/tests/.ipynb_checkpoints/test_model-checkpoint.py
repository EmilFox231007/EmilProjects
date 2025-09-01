import unittest
from unittest.mock import patch, mock_open
import logging
import gzip
import pickle
import sys
import os

# Adjust sys.path to include the 'project' directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'project')))

from project.model import load_random_forest_model  # Replace with your actual script name

class TestLoadRandomForestModel(unittest.TestCase):

    @patch('gzip.open', mock_open())
    @patch('pickle.load')
    def test_load_random_forest_model_success(self, mock_pickle_load):
        # Mock pickle.load to return a mock RandomForestClassifier object
        mock_rf_model = unittest.mock.Mock()
        mock_pickle_load.return_value = mock_rf_model

        # Call the function
        rf_model = load_random_forest_model(model_path="test_random_forest_model.pkl.gz")

        # Assertions
        self.assertEqual(rf_model, mock_rf_model)
        logging.info.assert_any_call("Loading Random Forest model from 'test_random_forest_model.pkl.gz'...")
        logging.info.assert_any_call("Random Forest model loaded successfully.")

    @patch('gzip.open', side_effect=FileNotFoundError("File not found"))
    def test_load_random_forest_model_file_not_found(self, mock_open):
        # Call the function and assert FileNotFoundError is raised
        with self.assertRaises(FileNotFoundError):
            load_random_forest_model(model_path="non_existing_model.pkl.gz")

        logging.error.assert_any_call("Model file 'non_existing_model.pkl.gz' not found.")

    @patch('gzip.open', side_effect=Exception("Unknown error"))
    def test_load_random_forest_model_exception(self, mock_open):
        # Call the function and assert RuntimeError is raised
        with self.assertRaises(RuntimeError):
            load_random_forest_model(model_path="corrupted_model.pkl.gz")

        logging.error.assert_any_call("Error loading model from 'corrupted_model.pkl.gz': Unknown error")

if __name__ == '__main__':
    unittest.main()

