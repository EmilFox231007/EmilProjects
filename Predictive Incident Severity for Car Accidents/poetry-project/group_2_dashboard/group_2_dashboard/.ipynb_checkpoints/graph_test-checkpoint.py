import streamlit as st
import pandas as pd
import plotly.express as px
from data_cleaning import main  # Import your main function from data_cleaning.py

# Load and clean data
df_safe_driving = main()

# Limit to the first 2000 data points
df_safe_driving_subset = df_safe_driving.head(2000)

# Define Streamlit app
def main():
    # Set title and description
    st.title('Safe Driving Incidents in Breda')
    st.markdown('Explore safe driving incidents in Breda using a heatmap.')

    # Plotting using Plotly Express
    fig = px.density_heatmap(df_safe_driving_subset,
                             x='latitude',
                             y='longitude',
                             marginal_x="histogram",
                             marginal_y="histogram",
                             color_continuous_scale='inferno',  # Set heatmap color scale
                             title='Heatmap of Safe Driving Incidents',  # Set heatmap title
                             width=800,  # Set plot width
                             height=600)  # Set plot height

    # Update heatmap layout and show the plot
    fig.update_layout(
        xaxis_title="Latitude",  # Set x-axis title
        yaxis_title="Longitude",  # Set y-axis title
        coloraxis_colorbar=dict(title='Density'),  # Set colorbar title
        margin={"r": 0, "t": 40, "l": 0, "b": 0}  # Set plot margins
    )
    
    st.plotly_chart(fig)  # Display Plotly chart in Streamlit

if __name__ == '__main__':
    main()
