import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Function to display the information page
def information_page():
    st.title("Incident Severity Prediction Dashboard")
    st.subheader("General Information and Graphs")

    # Information about the model
    st.subheader("Incident Severity Prediction Model")
    st.write("""
    We have developed a model that can predict the severity of incidents. This model is designed to be a solution for improving road safety in Breda.
    By accurately predicting incident severity, we aim to:
    - Provide better resource allocation for emergency services.
    - Implement preventive measures in high-risk areas.
    - Enhance overall traffic management and safety protocols.
    Our model uses historical incident data and various predictive analytics techniques to provide accurate and actionable insights.
    """)

    # Show map of Breda
    st.subheader("Map of Breda")
    # Coordinates for Breda
    breda_coordinates = {'latitude': 51.5719, 'longitude': 4.7683}
    st.map(pd.DataFrame([breda_coordinates], columns=['latitude', 'longitude']))

    # Information about the dataset
    st.subheader("Information about the dataset")
    st.write("""
    We are using the ANWB dataset, which contains over a million data points.
    The dataset includes various features related to road incidents, such as duration, speed, end speed, maximum value, and categories of incidents.
    To prepare the dataset for our model, we performed the following steps:
    - **Data Cleaning**: We cleaned the data to remove any inconsistencies and errors.
    - **Outlier Removal**: We removed outliers to ensure the accuracy of our predictions.
    - **Creating Bins**: We created bins to categorize the severity of incidents into three levels: low, medium, and high.

    This processed dataset allows our model to accurately predict the severity of road incidents and provide actionable insights for improving road safety in Breda.
    """)


# Function to display the prediction page
def prediction_page():
    st.title("Incident Severity Prediction Dashboard")
    st.subheader("Make a Prediction")

    # Sidebar configuration for user input
    st.sidebar.header("User Input Parameters")

    def user_input_features():
        duration_seconds = st.sidebar.number_input('Duration (seconds)', min_value=0, value=10)
        if st.sidebar.button("ℹ️", key="duration_info"):
            st.sidebar.write("Duration: Duration of the incident in seconds")

        speed_kmh = st.sidebar.number_input('Speed (km/h)', min_value=0, value=50)
        if st.sidebar.button("ℹ️", key="speed_info"):
            st.sidebar.write("Speed: Driving speed at start of incident (km/h)")

        end_speed_kmh = st.sidebar.number_input('End Speed (km/h)', min_value=0, value=50)
        if st.sidebar.button("ℹ️", key="end_speed_info"):
            st.sidebar.write("End Speed: Driving speed at end of incident (km/h)")

        category_encoded = st.sidebar.selectbox('Category', ['SPEED', 'HARSH CORNERING', 'ACCELERATING', 'BRAKING'])
        if st.sidebar.button("ℹ️", key="category_info"):
            st.sidebar.write("Category: Incident type, (SPEED) excessive speed more than 10km above the allowed speed, (HARSH CORNERING) sharp steering excessive lateral force, (ACCELERATING) hard acceleration, (BRAKING) hard or abrupt braking")

        maxwaarde = st.sidebar.number_input('Max Value', min_value=0, value=100)
        if st.sidebar.button("ℹ️", key="max_value_info"):
            st.sidebar.write("Max Value: If category is (SPEED) then maximum recorded speed during the incident (km/h). If category is (HARSH CORNERING, ACCELERATING, BRAKING) then maximum recorded g-force value")

        return duration_seconds, speed_kmh, end_speed_kmh, category_encoded, maxwaarde

    duration_seconds, speed_kmh, end_speed_kmh, category_encoded, maxwaarde = user_input_features()

    # Display user inputs
    st.subheader("User Input:")
    st.write(f"Duration (seconds): {duration_seconds}")
    st.write(f"Speed (km/h): {speed_kmh}")
    st.write(f"End Speed (km/h): {end_speed_kmh}")
    st.write(f"Category: {category_encoded}")
    st.write(f"Max Value: {maxwaarde}")

    # Placeholder for prediction output
    st.subheader("Prediction Output")
    st.write("The prediction will be displayed here.")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Select a page", ["Information", "Prediction"])

# Show the selected page
if page == "Information":
    information_page()
elif page == "Prediction":
    prediction_page()
