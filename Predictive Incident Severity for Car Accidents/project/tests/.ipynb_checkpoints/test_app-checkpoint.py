import unittest
from unittest.mock import patch, MagicMock
from io import StringIO
import logging
import sys
import os

# Add the 'project' directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'project')))

# Import the main function and other modules from app.py and pages
from project.app import main
from project.pages.heatmap import heatmap_page
from project.pages.information import information_page
from project.pages.prediction import prediction_page
class TestDashboard(unittest.TestCase):

    def setUp(self):
        # Patching logging to capture log messages
        self.log_capture = StringIO()
        logging.basicConfig(stream=self.log_capture, format="%(asctime)s - %(message)s", level=logging.INFO)

    def tearDown(self):
        # Clean up after each test
        self.log_capture.close()

    @patch('streamlit.set_page_config')
    @patch('streamlit.markdown')
    @patch('streamlit.info')
    @patch('streamlit.warning')
    @patch('streamlit.error')
    @patch('streamlit.exception')
    def test_main_information_page(self, mock_exception, mock_error, mock_warning, mock_info, mock_markdown, mock_set_page_config):
        # Mocking streamlit functions
        mock_set_page_config.return_value = None
        mock_markdown.return_value = None
        mock_info.return_value = None
        mock_warning.return_value = None
        mock_error.return_value = None
        mock_exception.return_value = None

        # Mock user selection
        with patch('streamlit.sidebar.selectbox', return_value='Information'):
            main()

        # Check logging messages
        logs = self.log_capture.getvalue().strip().split('\n')
        self.assertIn("Starting Incident Severity Prediction Dashboard...", logs)
        self.assertIn("Navigating to Information page...", logs)
        self.assertIn("Exiting Incident Severity Prediction Dashboard...", logs)

        # Check page function calls
        information_page.assert_called_once()
        prediction_page.assert_not_called()
        heatmap_page.assert_not_called()

    # Similar tests for Prediction and Heatmap pages can be added here

if __name__ == '__main__':
    unittest.main()

