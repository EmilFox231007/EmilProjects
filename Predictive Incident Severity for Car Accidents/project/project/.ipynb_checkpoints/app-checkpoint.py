import logging

import streamlit as st

from pages.heatmap import heatmap_page
from pages.information import information_page
from pages.prediction import prediction_page

def main():
    """
    Main function to run the Incident Severity Prediction Dashboard.

    Sets page configuration, customizes CSS for styling, and displays different pages (Information,
    Prediction, Heatmap) based on user selection using Streamlit.

    Pages:
    - Information: Displays general information about the application and the incident severity prediction model.
    - Prediction: Allows users to input incident details and predicts the severity level using a trained model.
    - Heatmap: Displays heatmaps of various incident types (Harsh Cornering, Speed, Braking, Accelerating).

    """
    # Configure logging
    logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
    logging.info("Starting Incident Severity Prediction Dashboard...")

    # Set page title and icon
    st.set_page_config(
        page_title="Incident Severity Prediction Dashboard", page_icon=":car:"
    )

    # Custom CSS for styling
    st.markdown(
        """
        <style>
        .stApp {
            max-width: 100%; /* Ensure the app takes full screen width */
            padding-top: 10px; /* Optional: Add padding to the top */
            padding-left: 50px; /* Adjust this value for more padding on the left */
            padding-right: 20px; /* Optional: Add padding to the right */
            margin: auto; /* Center the content horizontally */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar navigation
    page = st.sidebar.radio("Select a page", ("Information", "Prediction", "Heatmap"))

    # Display corresponding page based on selection
    if page == "Information":
        logging.info("Navigating to Information page...")
        information_page()
    elif page == "Prediction":
        logging.info("Navigating to Prediction page...")
        prediction_page()
    elif page == "Heatmap":
        logging.info("Navigating to Heatmap page...")
        heatmap_page()

    logging.info("Exiting Incident Severity Prediction Dashboard...")

# Run the main function
if __name__ == "__main__":
    main()
