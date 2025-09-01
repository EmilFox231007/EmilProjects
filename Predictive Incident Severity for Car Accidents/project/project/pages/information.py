import logging

import streamlit as st

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,  # Set logging level as needed
)


def information_page():
    """
    Function to display the information page of the Incident Severity Prediction Dashboard.

    This function sets up the Streamlit page with title, subheaders, and detailed information
    about the application's goals, features of the predictive model, and dataset information.

    Displays:
    - Title and subheaders with custom styling.
    - Description of the application's main goal and objectives.
    - Features of the predictive model used in the application.
    - Information about the dataset used for training the model.
    """

    try:
        st.title("Welcome to our Incident Severity Prediction Dashboard!👋")

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
            <h3 class="custom-subheader">General Information about our application</h3>
            """,
            unsafe_allow_html=True,
        )

        # Additional CSS for overall styling
        st.markdown(
            """
            <style>
                .stApp {
                    
                }
                .stApp h1, .stApp h3 {
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
            <p style="color: #CCD0CF;">What is our main goal❓</p>
            Our team has developed a model capable of predicting the severity of incidents in Breda, offering a promising solution to enhance road safety in Breda. By accurately forecasting the seriousness of incidents, our objectives are to:
            </p>
            <p style="color: #9BA8AB;">➤ Provide better resource allocation for emergency services.</p>
            <p style="color: #9BA8AB;">➤ Implement preventive measures in high-risk areas.</p>
            <p style="color: #9BA8AB;">➤ Enhance overall traffic management and safety protocols.</p>
            <p style="color: #CCD0CF;"> What are the key features of the model...🔍</p>
            This predictive model which deployed into this streamlit application is built using comprehensive data sources, primarily from the ANWB Safe Driving App, which tracks critical driving behavior metrics such as speed violations, harsh braking events, aggressive turns, and rapid acceleration incidents. Additionally, environmental data will be integrated such as weather conditions (rain, snow, fog), time of day (rush hours, nighttime), and road conditions, to ensure a multifactorial analysis.
            </p>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <style>
                .custom-subheader {
                    padding: 10px;
                    border-radius: 5px;
                    display: inline;
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
            <p style="color: #CCD0CF;">Information about the dataset we have used📊</p>
            <p style="color: #CCD0CF;">We are using the ANWB dataset, which contains over a million data points. The dataset includes various features related to road incidents, such as duration, speed, end speed, maximum value, and categories of incidents. To prepare the dataset for our model, we performed the following steps:</p>
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

    except Exception as e:
        logging.error(f"Error in information_page: {str(e)}")
        st.error("An error occurred while displaying the information page.")


if __name__ == "__main__":
    information_page()
