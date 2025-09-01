import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from streamlit_folium import st_folium
import folium
from data_cleaning import load_and_clean_data
from model import load_random_forest_model 
from sklearn.preprocessing import LabelEncoder
import plotly.express as px


# Function to display the information page
def information_page():

    st.title("Welcome to our Incident Severity Prediction Dashboard!👋")

    # Adding the subheader with custom background
    st.markdown(
        """
    <style>
        .custom-subheader {
            background-color:CCD0CF;
            padding: 10px;
            border-radius: 5px;
            display: inline;
        }
    </style>
                
                
    <h3 class="custom-subheader">General Information about our application</h3>
    """,
        unsafe_allow_html=True,
    )

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

    st.write(
        """
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
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
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
    """,
        unsafe_allow_html=True,
    )

    # Information about the dataset

    st.write(
        """
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
    """,
        unsafe_allow_html=True,
    )


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
        duration_seconds = st.sidebar.number_input(
            "Duration (seconds)", min_value=0, value=10
        )
        if st.sidebar.button("ℹ️", key="duration_info"):
            st.sidebar.write("Duration: Duration of the incident in seconds")

        speed_kmh = st.sidebar.number_input("Speed (km/h)", min_value=0, value=50)
        if st.sidebar.button("ℹ️", key="speed_info"):
            st.sidebar.write("Speed: Driving speed at start of incident (km/h)")

        end_speed_kmh = st.sidebar.number_input(
            "End Speed (km/h)", min_value=0, value=50
        )
        if st.sidebar.button("ℹ️", key="end_speed_info"):
            st.sidebar.write("End Speed: Driving speed at end of incident (km/h)")

        category_encoded = st.sidebar.selectbox(
            "Category", ["SPEED", "HARSH CORNERING", "ACCELERATING", "BRAKING"]
        )
        if st.sidebar.button("ℹ️", key="category_info"):
            st.sidebar.write(
                "Category: Incident type, (SPEED) excessive speed more than 10km above the allowed speed, (HARSH CORNERING) sharp steering excessive lateral force, (ACCELERATING) hard acceleration, (BRAKING) hard or abrupt braking"
            )

        maxwaarde = st.sidebar.number_input("Max Value", min_value=0, value=100)
        if st.sidebar.button("ℹ️", key="max_value_info"):
            st.sidebar.write(
                "Max Value: If category is (SPEED) then maximum recorded speed during the incident (km/h). If category is (HARSH CORNERING, ACCELERATING, BRAKING) then maximum recorded g-force value"
            )

        return duration_seconds, speed_kmh, end_speed_kmh, category_encoded, maxwaarde

    duration_seconds, speed_kmh, end_speed_kmh, category_encoded, maxwaarde = user_input_features()

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
        features = [duration_seconds, speed_kmh, end_speed_kmh, maxwaarde, category_encoded]
        prediction, prediction_proba = predict_incident_severity(features)
        
        # Display the predicted incident severity bin
        st.write(f"The predicted incident severity bin is: {prediction}")
        
        # Display the probabilities for each incident severity class in percentages
        st.write("Probability of each incident severity class:")
        st.write("- Low Severity: {:.2f}%".format(prediction_proba[1] * 100))  # Corresponds to Low
        st.write("- Medium Severity: {:.2f}%".format(prediction_proba[2] * 100))  # Corresponds to Medium
        st.write("- High Severity: {:.2f}%".format(prediction_proba[0] * 100))  # Corresponds to High

# Function to display the heatmap page
def heatmap_page():
    st.title("Heatmaps of our city Breda 🗺️...")

    # Adding the subheader with custom background
    st.markdown(
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

    # Load and clean data for heatmap only
    df_safe_driving = load_and_clean_data()  

    if df_safe_driving is None:
        st.write("Error loading data.")
        return

    def plot_heatmap(df, incident_type, title):
        df_filtered = df[df['category'] == incident_type]
        fig = px.density_mapbox(df_filtered,
                                lat='latitude',
                                lon='longitude',
                                z=None,
                                radius=5,  
                                center=dict(lat=df_filtered['latitude'].mean(), lon=df_filtered['longitude'].mean()),
                                zoom=10,  
                                mapbox_style="open-street-map",
                                title=title,
                                color_continuous_scale=px.colors.sequential.Plasma)
        fig.update_layout(
            coloraxis_colorbar=dict(title='Density'),
            margin={"r": 0, "t": 40, "l": 0, "b": 0}
        )
        st.plotly_chart(fig)

    # Harsh Cornering Heatmap
    st.markdown(
        """
    <style>
        .custom-subheader {            
            padding: 10px;
            border-radius: 5px;
            display: inline;
        }
    </style>                
    <h3 class="custom-subheader">1. Harsh Cornering Heatmap: </h3>
    """,
        unsafe_allow_html=True,
    )
    st.write(
        """
        <style>
            p {
                color: #9BA8AB;
            }
        </style>
        <p style="color: #CCD0CF;">Here is a brief explanation of what you are looking at➡️ </p>
        <p>This Harsh Cornering Heatmap visually represents areas where sharp turns or corners occur frequently. It uses color-coding to show the intensity or frequency of these maneuvers, helping to identify specific locations or routes where drivers are making abrupt turns. This information can be useful for improving driving habits, vehicle maintenance, and overall safety analysis.</p>
    """,
        unsafe_allow_html=True,
    )
    plot_heatmap(df_safe_driving, 'HARSH CORNERING', 'Heatmap of Harsh Cornering Incidents')

    # Speed Heatmap
    st.markdown(
        """
    <style>
        .custom-subheader {            
            padding: 10px;
            border-radius: 5px;
            display: inline;
        }
    </style>
    <h3 class="custom-subheader">2. Speed Heatmap: </h3>
    """,
        unsafe_allow_html=True,
    )
    st.write(
        """
        <style>
            p {
                color: #9BA8AB;
            }
        </style>
        <p style="color: #CCD0CF;">Here is a brief explanation of what you are looking at➡️ </p>
        <p>This Speed Heatmap visually illustrates variations in speed across a specific area or route. It uses colors to highlight where speeds are highest or lowest, providing insights into where drivers typically accelerate or decelerate. This information can help identify areas prone to traffic congestion, potential safety hazards, or where speed limits may need adjustment for smoother traffic flow.</p>
    """,
        unsafe_allow_html=True,
    )
    plot_heatmap(df_safe_driving, 'SPEED', 'Heatmap of Speed Incidents')

    # Braking Heatmap
    st.markdown(
        """
    <style>
        .custom-subheader {            ;
            padding: 10px;
            border-radius: 5px;
            display: inline;
        }
    </style>                
    <h3 class="custom-subheader">3. Braking Heatmap: </h3>
    """,
        unsafe_allow_html=True,
    )
    st.write(
        """
        <style>
            p {
                color: #9BA8AB;
            }
        </style>
        <p style="color: #CCD0CF;">Here is a brief explanation of what you are looking at➡️ </p>
        <p>This Braking Heatmap visually displays where frequent braking events occur along a given route or area. It uses color gradients to indicate the frequency or intensity of braking instances, helping to pinpoint locations where drivers frequently slow down or stop. This information is valuable for analyzing traffic patterns, identifying potential accident hotspots, and optimizing road design or signage to enhance traffic safety and efficiency.</p>
    """,
        unsafe_allow_html=True,
    )
    plot_heatmap(df_safe_driving, 'BRAKING', 'Heatmap of Braking Incidents')

    # Accelerating Heatmap
    st.markdown(
        """
    <style>
        .custom-subheader {            
            padding: 10px;
            border-radius: 5px;
            display: inline;
        }
    </style>                
    <h3 class="custom-subheader">4. Accelerating Heatmap:  </h3>
    """,
        unsafe_allow_html=True,
    )
    st.write(
        """
        <style>
            p {
                color: #9BA8AB;
            }
        </style>
        <p style="color: #CCD0CF;">Here is a brief explanation of what you are looking at➡️ </p>
        <p>Accelerating Heatmap visually represents areas where drivers frequently accelerate along a designated route or within a specific area. It uses color gradients to highlight the intensity or frequency of acceleration events, providing insights into where drivers typically increase their speed. This information is valuable for understanding traffic flow dynamics, optimizing road infrastructure, and identifying potential zones where improved traffic management or speed limit adjustments may be beneficial.</p>
    """,
        unsafe_allow_html=True,
    )
    plot_heatmap(df_safe_driving, 'ACCELERATING', 'Heatmap of Accelerating Incidents')



# Main function to run the app
def main():
    # Set page title and icon
    st.set_page_config(page_title="Incident Severity Prediction Dashboard", page_icon=":car:")

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
        unsafe_allow_html=True
    )

    # Sidebar navigation
    page = st.sidebar.radio("Select a page", ("Information", "Prediction", "Heatmap"))

    if page == "Information":
        information_page()
    elif page == "Prediction":
        prediction_page()
    elif page == "Heatmap":
        heatmap_page()

# Run the main function
if __name__ == "__main__":
    main()
    