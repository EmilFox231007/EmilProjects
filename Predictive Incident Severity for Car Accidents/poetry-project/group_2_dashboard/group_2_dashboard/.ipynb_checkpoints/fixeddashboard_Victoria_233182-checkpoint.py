import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from streamlit_folium import st_folium
import folium
from sklearn.preprocessing import LabelEncoder
from model import load_random_forest_model


# Function to display the information page
def information_page():
    
    st.title("Welcome to our Incident Severity Prediction Dashboard!👋")


    # Adding the subheader with custom background
    st.markdown("""
    <style>
        .custom-subheader {
            background-color:CCD0CF;
            padding: 10px;
            border-radius: 5px;
            display: inline;
        }
    </style>
                
                
    <h3 class="custom-subheader">General Information about our application</h3>
    """, unsafe_allow_html=True)

    # Additional CSS for overall styling
    st.markdown("""
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
    """, unsafe_allow_html=True)


    st.write("""
        <style>
            p {
                color: #9BA8AB;
            }
        </style>
        <p>
             

        <p style="color: #CCD0CF;">What is our main goal❓</p>
             
        Our team has developed a model capable of predicting the severity of incidents in Breda, offering a promising solution to enhance road safety in Breda. By accurately forecasting the seriousness of incidents, our objectives are to:
        </p>
        <p style="color: #9BA8AB;">➤ Provide better resource allocation for emergency services.</p>
        <p style="color: #9BA8AB;">➤Implement preventive measures in high-risk areas.</p>
        <p style="color: #9BA8AB;">➤Enhance overall traffic management and safety protocols.</p>
        <p>
             
        <p style="color: #CCD0CF;"> What are the key features of the model...🔍</p>
             
        This predictive model which deployed into this streamlit application is built using comprehensive data sources, primarily from the ANWB Safe Driving App, which tracks critical driving behaviour metrics such as speed violations, harsh braking events, aggressive turns, and rapid acceleration incidents. Additionally, environmental data will be integrated such as weather conditions (rain, snow, fog), time of day (rush hours, nighttime), and road conditions, to ensure a multifactorial analysis.
        </p>
    """, unsafe_allow_html=True)

    st.markdown("""
    <style>
        .custom-subheader {
            background-color:CCD0CF;
            padding: 10px;
            border-radius: 5px;
            display: inline;
             .stApp h1 {
            color: #CCD0CF;
        }
    </style>
    <h3 class="custom-subheader">Let's see what our Incident Severity Prediction Model can show you...</h3>
    """, unsafe_allow_html=True)
    
   

   # Display the map and allow user to click on it with the pointing hand icon
    st.markdown("Click on the map to select your incident location 👉")


    #HERE WE HAVE TO PUT THE CODE FOR THE FOLIUM MAP
# Information about the dataset

    st.write("""
        <style>
            p {
                color: #9BA8AB;
            }
        </style>
        <p>
             
        <p style="color: #CCD0CF;">Information about the dataset we have used📊</p>
             
        <p style="color: #CCD0CF;We are using the ANWB dataset, which contains over a million data points.
        The dataset includes various features related to road incidents, such as duration, speed, end speed, maximum value, and categories of incidents.
        To prepare the dataset for our model, we performed the following steps:</p>
             
        </p>
        <p style="color: #9BA8AB;">➤ Data Cleaning: We cleaned the data to remove any inconsistencies and errors.</p>
        <p style="color: #9BA8AB;">➤ Outlier Removal: We removed outliers to ensure the accuracy of our predictions.</p>
        <p style="color: #9BA8AB;">➤ Creating Bins: We created bins to categorize the severity of incidents into three levels: low, medium, and high.</p>
        <p>
             
             
        This processed dataset allows our model to accurately predict the severity of road incidents and provide actionable insights for improving road safety in Breda. 
             
             
        Empower yourself by making predictions about the places you encounter daily in Breda. Simply click the 'Make predictions' button conveniently located in the sidebar.
        </p>
    """, unsafe_allow_html=True)

# Load the trained Random Forest model
rf_model = load_random_forest_model()

# Define the label encoder for 'category_encoded'
label_encoder = LabelEncoder()
label_encoder.fit(['SPEED', 'HARSH CORNERING', 'ACCELERATING', 'BRAKING'])

def predict_incident_severity(features):
    # Ensure the input features have the correct column names and order
    feature_names = ['duration_seconds', 'speed_kmh', 'end_speed_kmh', 'maxwaarde', 'category_encoded']
    features_df = pd.DataFrame([features], columns=feature_names)
    
    # Encode 'category_encoded' using the label encoder
    features_df['category_encoded'] = label_encoder.transform(features_df['category_encoded'])
    
    prediction = rf_model.predict(features_df)
    prediction_proba = rf_model.predict_proba(features_df)[0]  # Extracting the probabilities for the first instance
    return prediction[0], prediction_proba

def prediction_page():
    st.title("Predictions are waiting for you here 👇")
    
    st.subheader("Make your own prediction")
    
    # Additional CSS for overall styling
    st.markdown("""
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
    """, unsafe_allow_html=True)
    
    # Display the sidebar header with the specified color
    st.sidebar.markdown("<h1 style='color: #CCD0CF;'>Input Parameters</h1>", unsafe_allow_html=True)

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

    st.write("""
        <style>
            p {
                color: #9BA8AB;
            }
        </style>
    """, unsafe_allow_html=True)

    # Placeholder for prediction output
    st.subheader("Your result:")
    
    # Predict button
    if st.button("Predict"):
        # Make prediction
        features = [duration_seconds, speed_kmh, end_speed_kmh, maxwaarde, category_encoded]
        prediction, prediction_proba = predict_incident_severity(features)
        
        # Display the predicted incident severity bin
        st.write(f"The predicted incident severity bin is: {prediction}")
        
        # Display the probabilities for each incident severity class
        st.write("Probability of each incident severity class:")
        st.write(f"- Low Severity: {prediction_proba[1]*100:.2f}%")
        st.write(f"- Medium Severity: {prediction_proba[2]*100:.2f}%")
        st.write(f"- High Severity: {prediction_proba[0]*100:.2f}%")

# Set page configuration
st.set_page_config(layout="wide")

# Custom CSS to hide Streamlit's navigation bar, header, and footer
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Define custom CSS styles
custom_css = """
<style>
/* Style the sidebar */
[data-testid="stSidebar"] {
    background-color: #4A5C6A;
}

/* Style the select box title */
[data-testid="stSidebar"] h1 {
    color: #9BA8AB;
}

/* Style the select box */
[data-testid="stSelectbox"] > div {
    background-color: #9BA8AB;
    color: #06141B;
}

/* Style the options within the select box */
[data-testid="stSelectbox"] > div > div {
    color: #9BA8AB;
}
</style>
"""

# Inject the CSS styles into the Streamlit app
st.markdown(custom_css, unsafe_allow_html=True)

# Create the sidebar with the title and select boxes
st.sidebar.title("Navigation bar")

# Sidebar select box
page = st.sidebar.selectbox("Select a page", ["Information about the app", "Make predictions here"])

# Show the selected page
if page == "Information about the app":
    information_page()
elif page == "Make predictions here":
    prediction_page()