import logging

import plotly.express as px
import streamlit as st
from data_cleaning import load_and_clean_data

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,  # Set logging level as needed
)


def heatmap_page():
    """
    Function to display multiple heatmaps for different driving incidents.

    This function loads cleaned data and generates interactive heatmaps using Plotly and Streamlit.
    It displays heatmaps for "Harsh Cornering", "Speed", "Braking", and "Accelerating" incidents.

    Displays:
    - Heatmap for Harsh Cornering incidents.
    - Heatmap for Speed incidents.
    - Heatmap for Braking incidents.
    - Heatmap for Accelerating incidents.
    """

    try:
        # Title of the page
        st.title("Heatmaps of our city Breda 🗺️...")

        # Subheader with description
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
        logging.info("Loading and cleaning data for heatmap...")
        df_safe_driving = load_and_clean_data()

        if df_safe_driving is None:
            logging.error("Error loading data.")
            st.write("Error loading data.")
            return

        def plot_heatmap(df, incident_type, title):
            """
            Function to plot a density heatmap based on incident type.

            Args:
            - df (DataFrame): The DataFrame containing incident data.
            - incident_type (str): Type of driving incident ('HARSH CORNERING', 'SPEED', 'BRAKING', 'ACCELERATING').
            - title (str): Title of the heatmap plot.

            Generates a density map using Plotly's density_mapbox function and displays it using Streamlit.
            """
            df_filtered = df[df["category"] == incident_type]
            fig = px.density_mapbox(
                df_filtered,
                lat="latitude",
                lon="longitude",
                z=None,
                radius=5,
                center=dict(
                    lat=df_filtered["latitude"].mean(),
                    lon=df_filtered["longitude"].mean(),
                ),
                zoom=10,
                mapbox_style="open-street-map",
                title=title,
                color_continuous_scale=px.colors.sequential.Plasma,
            )
            fig.update_layout(
                coloraxis_colorbar=dict(title="Density"),
                margin={"r": 0, "t": 40, "l": 0, "b": 0},
            )
            st.plotly_chart(fig)

        # Display Harsh Cornering Heatmap
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
        plot_heatmap(
            df_safe_driving, "HARSH CORNERING", "Heatmap of Harsh Cornering Incidents"
        )

        # Display Speed Heatmap
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
        plot_heatmap(df_safe_driving, "SPEED", "Heatmap of Speed Incidents")

        # Display Braking Heatmap
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
        plot_heatmap(df_safe_driving, "BRAKING", "Heatmap of Braking Incidents")

        # Display Accelerating Heatmap
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
        plot_heatmap(
            df_safe_driving, "ACCELERATING", "Heatmap of Accelerating Incidents"
        )

    except Exception as e:
        logging.error(f"Error in heatmap_page: {str(e)}")
        st.error("An error occurred while displaying the heatmaps.")


if __name__ == "__main__":
    heatmap_page()
