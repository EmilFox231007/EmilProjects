import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from streamlit_folium import st_folium
import folium


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

    # Display the map and allow user to click on it with the pointing hand icon
    st.markdown("Click on the map to select your incident location 👉")

    # HERE WE HAVE TO PUT THE CODE FOR THE FOLIUM MAP
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


# Function to display the prediction page
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

    # Display the sidebar header with the specified color
    # Apply color to the sidebar header using HTML markup
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
    st.write("The prediction will be displayed here.")


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
    color: #06141B;
}

/* Style the select box */
[data-testid="stSelectbox"] > div {
    background-color: #06141B;
    color: #06141B;
}

/* Style the options within the select box */
[data-testid="stSelectbox"] > div > div {
    color: #06141B;
}
</style>
"""

# Inject the CSS styles into the Streamlit app
st.markdown(custom_css, unsafe_allow_html=True)

# Create the sidebar with the title and select boxes
st.sidebar.title("Navigation bar")

# Define custom CSS styles
custom_css = """
<style>
/* Style the sidebar */
.sidebar {
    background-color: #FFFFFF; /* Sidebar background color (white) */
    color: #000000; /* Sidebar text color (black) */
}

/* Style the sidebar title */
.sidebar .stSidebar {
    color: #000000; /* Sidebar title text color (black) */
}

/* Style the select box labels */
.sidebar .stSelectbox label {
    color: #000000; /* Select box label text color (black) */
}

/* Style the select box options */
.sidebar .stSelectbox div[data-baseweb="select"] {
    background-color: #FFFFFF; /* Select box background color (white) */
    color: #000000; /* Select box text color (black) */
}
</style>
"""


# Function to display the page where all heatmaps can be found
def heatmap_page():
    st.title("Heatmaps of our city Breda 🗺️...")

    # Adding the subheader with custom background
    st.markdown(
        """
    <style>
        .custom-subheader {
            background-color: CCD0CF;
            padding: 10px;
            border-radius: 5px;
            display: inline;
        }
    </style>           
    <h3 class="custom-subheader">Here you are going to be able to compare all heatmaps for "HARSH CORNERING"; "SPEED"; "BRAKING" and "ACCELERATING". </h3>
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
                

    <h3 class="custom-subheader">So, let's begin...</h3>
    """,
        unsafe_allow_html=True,
    )

    # heatmap itsefle is the code that will define all the things about harsh cornering hetmap: the title; the brief explanation and the
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
        <p>
             
        <p style="color: #CCD0CF;">Here is a brief explanation of what you are looking at➡️ </p>
        <p>
             
        This Harsh Cornering Heatmap visually represents areas where sharp turns or corners occur frequently. It uses color-coding to show the intensity or frequency of these maneuvers, helping to identify specific locations or routes where drivers are making abrupt turns. This information can be useful for improving driving habits, vehicle maintenance, and overall safety analysis
        </p>
    """,
        unsafe_allow_html=True,
    )

    # heatmap itsefle is the code that will define all the things about speed hetmap: the title; the brief explanation and the

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
        <p>
             
        <p style="color: #CCD0CF;">Here is a brief explanation of what you are looking at➡️ </p>
        <p>
             
         This Speed Heatmap visually illustrates variations in speed across a specific area or route. It uses colors to highlight where speeds are highest or lowest, providing insights into where drivers typically accelerate or decelerate. This information can help identify areas prone to traffic congestion, potential safety hazards, or where speed limits may need adjustment for smoother traffic flow.
        </p>
    """,
        unsafe_allow_html=True,
    )

    # here is everything about harsh braking hetmap: the title; the brief explanation and the heatmap withh all the data from ANWB

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
        <p>
             
        <p style="color: #CCD0CF;">Here is a brief explanation of what you are looking at➡️ </p>
        <p>
             
         This Braking Heatmap visually displays where frequent braking events occur along a given route or area. It uses color gradients to indicate the frequency or intensity of braking instances, helping to pinpoint locations where drivers frequently slow down or stop. This information is valuable for analyzing traffic patterns, identifying potential accident hotspots, and optimizing road design or signage to enhance traffic safety and efficiency. 
        </p>
    """,
        unsafe_allow_html=True,
    )

    # here is everything about harsh accelarating hetmap: the title; the brief explanation and the heatmap withh all the data from ANWB

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
        <p>
             
        <p style="color: #CCD0CF;">Here is a brief explanation of what you are looking at➡️ </p>
        <p>
             
         Accelerating Heatmap visually represents areas where drivers frequently accelerate along a designated route or within a specific area. It uses color gradients to highlight the intensity or frequency of acceleration events, providing insights into where drivers typically increase their speed. This information is valuable for understanding traffic flow dynamics, optimizing road infrastructure, and identifying potential zones where improved traffic management or speed limit adjustments may be beneficial.
        </p>
    """,
        unsafe_allow_html=True,
    )


# Sidebar select box
page = st.sidebar.selectbox(
    "Select a page",
    [
        "Information about the app",
        "Make predictions here",
        "Heatmap of Breda - overview",
    ],
)

# Show the selected page
if page == "Information about the app":
    information_page()
elif page == "Make predictions here":
    prediction_page()
elif page == "Heatmap of Breda - overview":
    heatmap_page()
