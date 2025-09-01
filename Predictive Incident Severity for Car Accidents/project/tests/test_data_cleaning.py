import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import logging
import sys
import os

# Add the 'project' directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'project')))

from project.parameters import db_params


# Import functions from your script
from project.data_cleaning import (
    load_data,
    clean_data,
    detect_outliers_iqr,
    plot_boxplots,
    load_and_clean_data,
)


class TestLoadData(unittest.TestCase):

    @patch('sqlalchemy.create_engine')
    @patch('pandas.read_sql')
    def test_load_data_success(self, mock_read_sql, mock_create_engine):
        # Mocking create_engine and read_sql
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        mock_read_sql.return_value = pd.DataFrame({
            'event_start': pd.to_datetime(['2023-01-01', '2023-01-02']),
            'event_end': pd.to_datetime(['2023-01-01', '2023-01-02']),
            'category': ['A', 'B'],
            'incident_severity': ['Low', 'Medium'],
            'duration_seconds': [100, 200],
            'maxwaarde': [50, 100]
        })

        # Mock db_params
        mock_db_params = {
            'user': 'test_user',
            'password': 'test_password',
            'host': 'localhost',
            'port': '5432',
            'database': 'test_db'
        }

        # Call the function
        result = load_data(mock_db_params)

        # Assertions
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)  # Check number of rows returned
        self.assertListEqual(list(result.columns), ['event_start', 'event_end', 'category', 'incident_severity', 'duration_seconds', 'maxwaarde'])
        mock_create_engine.assert_called_once_with(
            f"postgresql+psycopg2://test_user:test_password@localhost:5432/test_db"
        )
        mock_read_sql.assert_called_once_with("SELECT * FROM data_lake.safe_driving", mock_engine)

    @patch('sqlalchemy.create_engine')
    def test_load_data_failure(self, mock_create_engine):
        # Mocking create_engine to raise an exception
        mock_create_engine.side_effect = Exception("Connection error")

        # Mock db_params
        mock_db_params = {
            'user': 'test_user',
            'password': 'test_password',
            'host': 'localhost',
            'port': '5432',
            'database': 'test_db'
        }

        # Call the function and assert that it raises an exception
        with self.assertRaises(Exception) as context:
            load_data(mock_db_params)

        self.assertEqual(str(context.exception), "Connection error")

        # Ensure logging.error was called
        self.assertIn("Error loading data", logging.error.call_args[0][0])

        # Reset mock
        logging.error.reset_mock()

class TestCleanData(unittest.TestCase):

    def setUp(self):
        # Create a sample DataFrame for testing
        self.df = pd.DataFrame({
            'event_start': pd.to_datetime(['2023-01-01', '2023-01-02']),
            'event_end': pd.to_datetime(['2023-01-01', '2023-01-02']),
            'category': ['A', 'B'],
            'incident_severity': ['HA1', 'HA2'],
            'duration_seconds': [100, 200],
            'maxwaarde': [50, 100]
        })

    def test_clean_data(self):
        # Test clean_data function
        cleaned_df = clean_data(self.df)

        # Assertions
        self.assertIsInstance(cleaned_df, pd.DataFrame)
        self.assertEqual(len(cleaned_df), 2)  # Check number of rows returned
        self.assertListEqual(list(cleaned_df.columns), ['event_start', 'event_end', 'category', 'incident_severity', 'duration_seconds', 'category_encoded', 'incident_severity_bin'])

    def test_detect_outliers_iqr(self):
        # Test detect_outliers_iqr function
        columns_to_check = ['duration_seconds', 'maxwaarde']
        outlier_summary = detect_outliers_iqr(self.df, columns_to_check)

        # Assertions
        self.assertIsInstance(outlier_summary, dict)
        self.assertEqual(len(outlier_summary), 2)  # Check number of columns in outlier summary
        self.assertIn('duration_seconds', outlier_summary)
        self.assertIn('maxwaarde', outlier_summary)

    def test_plot_boxplots(self):
        # Test plot_boxplots function (mocking plt.show)
        with patch('matplotlib.pyplot.show'):
            plot_boxplots(self.df, self.df.columns)

        # Assertions - since plot_boxplots is visual, we can't assert much programmatically

class TestLoadAndCleanData(unittest.TestCase):

    @patch('your_script_name.load_data')
    @patch('your_script_name.clean_data')
    def test_load_and_clean_data(self, mock_load_data, mock_clean_data):
        # Mock load_data and clean_data functions
        mock_load_data.return_value = self.df
        mock_clean_data.return_value = self.df

        # Call the function
        result = load_and_clean_data()

        # Assertions
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)  # Check number of rows returned

if __name__ == '__main__':
    unittest.main()

