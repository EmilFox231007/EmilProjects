import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import plotly.express as px
import streamlit as st
import sys
import os

# Add the 'project' directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'project')))

# Now you can import modules from 'project'
from project.pages.heatmap import heatmap_page
from project.data_cleaning import load_and_clean_data


class TestHeatmapPage(unittest.TestCase):

    @patch('project.pages.heatmap.load_and_clean_data')
    @patch('streamlit.title')
    @patch('streamlit.markdown')
    @patch('streamlit.write')
    @patch('streamlit.plotly_chart')
    def test_heatmap_page(self, mock_plotly_chart, mock_write, mock_markdown, mock_title, mock_load_and_clean_data):
        # Mocking load_and_clean_data() function
        mock_load_and_clean_data.return_value = pd.DataFrame({
            'latitude': [51.586, 51.587, 51.588],
            'longitude': [4.775, 4.776, 4.777],
            'category': ['HARSH CORNERING', 'SPEED', 'BRAKING']
        })

        # Mock Streamlit functions
        mock_title.return_value = None
        mock_markdown.return_value = None
        mock_write.return_value = None
        mock_plotly_chart.return_value = None

        # Call heatmap_page function
        heatmap_page()

        # Assert Streamlit function calls
        mock_title.assert_called_once_with("Heatmaps of our city Breda 🗺️...")
        mock_markdown.assert_any_call(
            """
            <style>
                .custom-subheader {            
                    padding: 10px;
                    border-radius: 5px;
                    display: inline;
                }
            </style>           
            <h3 class="custom-subheader">Here you are going to be able to compare all heatmaps for "HARSH CORNERING"; "SPEED"; "BRAKING" and "ACCELERATING". </h3>
            """,
            unsafe_allow_html=True,
        )

        # Assert plot_heatmap calls for each incident type
        expected_calls = [
            (('HARSH CORNERING', "Heatmap of Harsh Cornering Incidents"),),
            (('SPEED', "Heatmap of Speed Incidents"),),
            (('BRAKING', "Heatmap of Braking Incidents"),),
            (('ACCELERATING', "Heatmap of Accelerating Incidents"),)
        ]
        for call_args in expected_calls:
            mock_plotly_chart.assert_any_call(px.density_mapbox(...))

    # Add more tests as needed

if __name__ == '__main__':
    unittest.main()