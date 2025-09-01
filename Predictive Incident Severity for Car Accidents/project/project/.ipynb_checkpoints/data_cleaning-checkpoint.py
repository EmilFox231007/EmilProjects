import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.preprocessing import LabelEncoder
import logging
from parameters import db_params

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

def load_data(db_params):
    """
    Load data from the PostgreSQL database.

    Parameters:
    db_params (dict): Database connection parameters including 'user', 'password', 'host', 'port', and 'database'.

    Returns:
    DataFrame: Loaded data from the 'data_lake.safe_driving' table.
    """
    logging.info("Loading data from the database...")
    try:
        conn_string = f"postgresql+psycopg2://{db_params['user']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/{db_params['database']}"
        engine = create_engine(conn_string)
        query = "SELECT * FROM data_lake.safe_driving"
        df_safe_driving = pd.read_sql(query, engine)
        engine.dispose()
        logging.info("Data loaded successfully.")
        return df_safe_driving
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        raise

def clean_data(df):
    """
    Clean and preprocess the data.

    Parameters:
    df (DataFrame): Raw data to be cleaned, expected to contain columns 'event_start', 'event_end', 'category', 'incident_severity', 'duration_seconds', and 'maxwaarde'.

    Returns:
    DataFrame: Cleaned data with columns 'event_start', 'event_end', 'category', 'incident_severity', 'duration_seconds', 'category_encoded', and 'incident_severity_bin'.
    """
    logging.info("Cleaning and preprocessing data...")
    try:
        df["event_start"] = pd.to_datetime(df["event_start"])
        df["event_end"] = pd.to_datetime(df["event_end"])
        df["road_number"] = df["road_number"].replace("", np.nan)
        df = df.drop("road_number", axis=1)

        df["category"] = df["category"].str.strip()
        df["incident_severity"] = df["incident_severity"].str.strip()

        # Remove outliers in 'duration_seconds'
        outlier_index = df[df["duration_seconds"] > 500].index
        df = df.drop(outlier_index)

        # Encode 'category' using LabelEncoder
        label_encoder = LabelEncoder()
        df["category_encoded"] = label_encoder.fit_transform(df["category"])

        # Classify 'incident_severity' into bins: Low, Medium, High, Unknown
        def classify_incident_severity(severity):
            low = ["HA1", "HB1", "HC1", "SP1", "HC4", "HC13"]
            medium = [
                "HA2", "HB2", "HC2", "HC5", "HC7", "HC14", "HC16", "SP2", "SP3",
                "HC6", "HC8", "HC10", "HC15", "HC17", "HC19"
            ]
            high = [
                "HA3", "HB3", "HC3", "HC9", "HC11", "HC12", "HC18", "HC20", "HC21",
                "SP4", "SP5"
            ]

            if severity in low:
                return "Low"
            elif severity in medium:
                return "Medium"
            elif severity in high:
                return "High"
            else:
                return "Unknown"

        df["incident_severity_bin"] = df["incident_severity"].apply(classify_incident_severity)

        # Sample data to balance classes
        sampled_dfs = []
        for bin_label, group in df.groupby("incident_severity_bin"):
            sampled_dfs.append(group.sample(n=63533, replace=True, random_state=42))
        df = pd.concat(sampled_dfs)

        logging.info("Data cleaned successfully.")
        return df
    except Exception as e:
        logging.error(f"Error cleaning data: {str(e)}")
        raise

def detect_outliers_iqr(df, columns):
    """
    Detect outliers in the DataFrame using the Interquartile Range (IQR) method.

    Parameters:
    df (DataFrame): DataFrame to analyze.
    columns (list): List of columns to check for outliers.

    Returns:
    dict: Summary of outliers for each numeric column in the DataFrame.
    """
    logging.info("Detecting outliers using IQR method...")
    try:
        outlier_summary = {}
        for column in columns:
            if pd.api.types.is_numeric_dtype(df[column]):
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
                outlier_summary[column] = outliers.shape[0]
            else:
                outlier_summary[column] = "Non-numeric data"
        
        logging.info("Outlier detection completed.")
        return outlier_summary
    except Exception as e:
        logging.error(f"Error detecting outliers: {str(e)}")
        raise

def plot_boxplots(df, columns):
    """
    Plot boxplots for numeric columns in the DataFrame.

    Parameters:
    df (DataFrame): DataFrame containing the data.
    columns (list): List of columns to plot boxplots for.
    """
    logging.info("Plotting boxplots...")
    try:
        numeric_columns = [column for column in columns if pd.api.types.is_numeric_dtype(df[column])]
        plt.figure(figsize=(15, 5 * len(numeric_columns)))
        for i, column in enumerate(numeric_columns, 1):
            plt.subplot(len(numeric_columns), 1, i)
            sns.boxplot(x=df[column].dropna())
            plt.title(f"Boxplot of {column}")
        plt.tight_layout()
        plt.show()
        logging.info("Boxplots plotted successfully.")
    except Exception as e:
        logging.error(f"Error plotting boxplots: {str(e)}")
        raise

def load_and_clean_data():
    """
    Load data from the PostgreSQL database, clean and preprocess it.

    Returns:
    DataFrame: Cleaned and preprocessed data from 'data_lake.safe_driving' table.
    """
    try:
        df = load_data(db_params)
        df = clean_data(df)
        return df
    except Exception as e:
        logging.error(f"Error loading and cleaning data: {str(e)}")
        raise