import pandas as pd
import streamlit as st
import logging
from model import load_random_forest_model
from sklearn.preprocessing import LabelEncoder

# Load the trained Random Forest model
rf_model = load_random_forest_model()

# Define the label encoder for 'category_encoded'
label_encoder = LabelEncoder()
label_encoder.fit(["SPEED", "HARSH CORNERING", "ACCELERATING", "BRAKING"])

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO  # Set the desired logging level
)

def predict_incident_severity(features):
    """
    Predicts the severity of an incident based on input features using a trained Random Forest model.

    Args:
    - features (list): A list containing the input features in the following order:
        1. duration_seconds (int): Duration of the incident in seconds.
        2. speed_kmh (float): Driving speed at start of incident in kilometers per hour.
        3. end_speed_kmh (float): Driving speed at end of incident in kilometers per hour.
        4. maxwaarde (float): Maximum recorded value relevant to the incident.
        5. category_encoded (str): Encoded category of the incident ('SPEED', 'HARSH CORNERING', 'ACCELERATING', 'BRAKING').

    Returns:
    - prediction (int): Predicted severity level (0 for High, 1 for Medium, 2 for Low).
    - prediction_proba (array): Probabilities for each severity level as predicted by the model.
    """
    try:
        # Ensure the input features have the correct column names and order
        feature_names = [
            "duration_seconds",
            "speed_kmh",
            "end_speed_kmh",
            "maxwaarde",
            "category_encoded",
        ]
        features_df = pd.DataFrame([features], columns=feature_names)

        # Encode 'category_encoded' using the label encoder
        features_df["category_encoded"] = label_encoder.transform(
            features_df["category_encoded"]
        )

        prediction = rf_model.predict(features_df)
        prediction_proba = rf_model.predict_proba(features_df)[0]  # Extract probabilities for the first instance
        return prediction[0], prediction_proba

    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return None, None


def prediction_page():
    """
    Defines the Streamlit page for predicting incident severity based on user input.

    Displays input fields for user to enter incident details such as duration, speed, category,
    and maximum recorded value. On clicking the 'Predict' button, it uses the trained Random Forest
    model to predict the severity level and displays the predicted severity and probabilities.
    """
    try:
        st.title("Predictions are waiting for you here 👇")
        st.subheader("Make your own prediction")

        # Additional CSS for overall styling
        st.markdown(
            """
            <style>
                .stApp {
                    background-color: #06141B;
                }
                .stApp h1 {
                    color: #CCD0CF;
                }
                .stApp h3 {
                    color: #CCD0CF;
                }
            </style>
            """,
            unsafe_allow_html=True,
        )

        st.sidebar.markdown(
            "<h1 style='color: #CCD0CF;'>Input Parameters</h1>", unsafe_allow_html=True
        )

        def user_input_features():
            """
            Function to collect user input for incident features via Streamlit sidebar.

            Returns:
            - duration_seconds (int): Duration of the incident in seconds.
            - speed_kmh (float): Driving speed at start of incident in kilometers per hour.
            - end_speed_kmh (float): Driving speed at end of incident in kilometers per hour.
            - category_encoded (str): Encoded category of the incident ('SPEED', 'HARSH CORNERING', 'ACCELERATING', 'BRAKING').
            - maxwaarde (float): Maximum recorded value relevant to the incident.
            """
            duration_seconds = st.sidebar.number_input(
                "Duration (seconds)", min_value=0, value=10
            )
            if st.sidebar.button("ℹ️", key="duration_info"):
                st.sidebar.write("Duration: Duration of the incident in seconds")

            speed_kmh = st.sidebar.number_input("Speed (km/h)", min_value=0, value=50)
            if st.sidebar.button("ℹ️", key="speed_info"):
                st.sidebar.write(
                    "Speed: Driving speed at start of incident (km/h)")

            end_speed_kmh = st.sidebar.number_input(
                "End Speed (km/h)", min_value=0, value=50)
            if st.sidebar.button("ℹ️", key="end_speed_info"):
                st.sidebar.write(
                    "End Speed: Driving speed at end of incident (km/h)")

            category_encoded = st.sidebar.selectbox(
                "Category", ["SPEED", "HARSH CORNERING", "ACCELERATING", "BRAKING"])
            if st.sidebar.button("ℹ️", key="category_info"):
                st.sidebar.write(
                    "Category: Incident type, (SPEED) excessive speed more than 10km above the allowed speed, (HARSH CORNERING) sharp steering excessive lateral force, (ACCELERATING) hard acceleration, (BRAKING) hard or abrupt braking")

            maxwaarde = st.sidebar.number_input("Max Value", min_value=0, value=100)
            if st.sidebar.button("ℹ️", key="max_value_info"):
                st.sidebar.write(
                    "Max Value: If category is (SPEED) then maximum recorded speed during the incident (km/h). If category is (HARSH CORNERING, ACCELERATING, BRAKING) then maximum recorded g-force value")

            return duration_seconds, speed_kmh, end_speed_kmh, category_encoded, maxwaarde

        duration_seconds, speed_kmh, end_speed_kmh, category_encoded, maxwaarde = (
            user_input_features()
        )

        # Display user inputs
        st.subheader("User Input:")
        st.write(f"Duration (seconds): {duration_seconds}")
        st.write(f"Speed (km/h): {speed_kmh}")
        st.write(f"End Speed (km/h): {end_speed_kmh}")
        st.write(f"Category: {category_encoded}")
        st.write(f"Max Value: {maxwaarde}")

        st.write(
            """
            <style>
                p {
                    color: #9BA8AB;
                }
            </style>
        """,
            unsafe_allow_html=True,
        )

        # Placeholder for prediction output
        st.subheader("Your result:")

        if st.button("Predict"):
            # Make prediction
            features = [
                duration_seconds,
                speed_kmh,
                end_speed_kmh,
                maxwaarde,
                category_encoded,
            ]
            prediction, prediction_proba = predict_incident_severity(features)

            if prediction is not None and prediction_proba is not None:
                # Display the predicted incident severity bin
                st.write(f"The predicted incident severity bin is: {prediction}")

                # Display the probabilities for each incident severity class in percentages
                st.write("Probability of each incident severity class:")
                st.write(
                    "- Low Severity: {:.2f}%".format(prediction_proba[1] * 100)
                )  # Corresponds to Low
                st.write(
                    "- Medium Severity: {:.2f}%".format(prediction_proba[2] * 100)
                )  # Corresponds to Medium
                st.write(
                    "- High Severity: {:.2f}%".format(prediction_proba[0] * 100)
                )  # Corresponds to High
            else:
                st.error("Failed to make predictions. Please check your inputs and try again.")

    except Exception as e:
        logging.error(f"Error in prediction_page: {str(e)}")
        st.error("An error occurred during prediction. Please try again later.")


if __name__ == "__main__":
    prediction_page()
