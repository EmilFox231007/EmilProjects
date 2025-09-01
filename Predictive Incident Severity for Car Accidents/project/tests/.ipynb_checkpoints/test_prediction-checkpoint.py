import unittest
from unittest.mock import patch, MagicMock
from io import StringIO
import logging
import pandas as pd
import streamlit as st
import sys
import os

# Add the 'project' directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'project')))

# Now you can import modules from 'project'
from project.pages.prediction import prediction_page, predict_incident_severity
from project.model import load_random_forest_model


# Mocking the st.sidebar interactions
mock_sidebar = MagicMock()
st.sidebar = mock_sidebar

class TestPredictionPage(unittest.TestCase):

    def setUp(self):
        # Patching logging to capture log messages
        self.log_capture = StringIO()
        logging.basicConfig(stream=self.log_capture, format="%(asctime)s - %(message)s", level=logging.INFO)

    def tearDown(self):
        # Clean up after each test
        self.log_capture.close()

    @patch('streamlit.write')
    @patch('streamlit.subheader')
    @patch('streamlit.title')
    @patch('streamlit.error')
    @patch('project.pages.prediction.predict_incident_severity')
    def test_prediction_page_successful_prediction(self, mock_predict, mock_error, mock_title, mock_subheader, mock_write):
        # Mock user input
        mock_sidebar.number_input.side_effect = lambda label, min_value, value: 10
        mock_sidebar.selectbox.return_value = 'SPEED'

        # Mock predict_incident_severity function
        mock_predict.return_value = (1, [0.2, 0.3, 0.5])  # Example prediction and probabilities

        # Call prediction_page function
        prediction_page()

        # Check logging messages
        logs = self.log_capture.getvalue().strip().split('\n')
        self.assertIn("Predicting incident severity...", logs)
        self.assertIn("Prediction successful.", logs)

        # Check Streamlit outputs
        mock_title.assert_called_once_with("Predictions are waiting for you here 👇")
        mock_subheader.assert_called_once_with("Make your own prediction")
        mock_write.assert_any_call("The predicted incident severity bin is: 1")
        mock_write.assert_any_call("Probability of each incident severity class:")
        mock_write.assert_any_call("- Low Severity: 30.00%")
        mock_write.assert_any_call("- Medium Severity: 50.00%")
        mock_write.assert_any_call("- High Severity: 20.00%")
        mock_error.assert_not_called()

    @patch('streamlit.write')
    @patch('streamlit.subheader')
    @patch('streamlit.title')
    @patch('streamlit.error')
    @patch('project.pages.prediction.predict_incident_severity')
    def test_prediction_page_invalid_input(self, mock_predict, mock_error, mock_title, mock_subheader, mock_write):
        # Mock user input with negative values
        mock_sidebar.number_input.side_effect = lambda label, min_value, value: -10
        mo
