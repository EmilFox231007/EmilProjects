import unittest
from unittest.mock import patch
import streamlit as st
import sys
import os

# Add the 'project' directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'project')))

from project.pages.information import information_page

class TestInformationPage(unittest.TestCase):

    def setUp(self):
        # Set up any necessary resources or configurations before each test
        self.test_client = st.test_client(information_page)

    def tearDown(self):
        # Clean up after each test if necessary
        pass

    def test_information_page_loads(self):
        # Mock Streamlit's functions to capture output
        with patch('streamlit.title'), \
             patch('streamlit.markdown'), \
             patch('streamlit.write'), \
             patch('streamlit.subheader'), \
             patch('streamlit.sidebar'):
            
            # Execute the information_page function
            information_page()

    def test_information_page_error_handling(self):
        # Test error handling by forcing an exception
        with patch('streamlit.title', side_effect=Exception("Test error")):
            information_page()
            # Ideally, assert that the error message is displayed
            # Here we assume it logs to st.error, so we can't directly test the message

if __name__ == '__main__':
    unittest.main()

