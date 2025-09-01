import streamlit as st
import pandas as pd
from model import load_random_forest_model
from data_cleaning import main as clean_data
from sklearn.preprocessing import LabelEncoder

# Load and clean the data
df = clean_data()

# Define features (X) and target (y)
X = df[['duration_seconds', 'speed_kmh', 'end_speed_kmh', 'maxwaarde', 'category_encoded']]
y = df['incident_severity_bin']

# Load the trained Random Forest model
rf_model = load_random_forest_model()

# Function to make predictions
def predict_incident_severity(features):
    feature_names = ['duration_seconds', 'speed_kmh', 'end_speed_kmh', 'maxwaarde', 'category_encoded']
    features_df = pd.DataFrame([features], columns=feature_names)
    label_encoder = LabelEncoder()
    features_df['maxwaarde'] = label_encoder.fit_transform(features_df['maxwaarde'])
    features_df['category_encoded'] = label_encoder.transform(features_df['category_encoded'])
    prediction = rf_model.predict(features_df)
    prediction_proba = rf_model.predict_proba(features_df)
    return prediction[0], max(prediction_proba[0])

# Streamlit UI
st.title("Incident Severity Prediction")

# Input features
duration_seconds = st.slider("Duration (seconds)", 0, 1000, 10)
speed_kmh = st.slider("Speed (km/h)", 0, 200, 50)
end_speed_kmh = st.slider("End Speed (km/h)", 0, 200, 50)
maxwaarde = st.selectbox("Max Value", ['SPEED', 'HARSH CORNERING', 'ACCELERATING', 'BRAKING'])  # Assuming these are the possible values
category_encoded = st.slider("Category", 0, 200, 100)

# Predict button
if st.button("Predict"):
    try:
        # Make prediction
        prediction, prediction_proba = predict_incident_severity([duration_seconds, speed_kmh, end_speed_kmh, maxwaarde, category_encoded])
        st.write(f"The predicted incident severity bin is: {prediction}")
        st.write(f"Prediction confidence: {prediction_proba * 100:.2f}%")
    except Exception as e:
        st.write("Error occurred during prediction:", e)
