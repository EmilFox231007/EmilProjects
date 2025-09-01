import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
import missingno as msno
from sklearn.preprocessing import LabelEncoder
from parameters import db_params

def load_data(db_params):
    conn_string = f"postgresql+psycopg2://{db_params['user']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/{db_params['database']}"
    engine = create_engine(conn_string)
    query = "SELECT * FROM data_lake.safe_driving"
    df_safe_driving = pd.read_sql(query, engine)
    engine.dispose()
    return df_safe_driving

def clean_data(df):
    # Handle missing values and convert columns to appropriate types
    df['event_start'] = pd.to_datetime(df['event_start'])
    df['event_end'] = pd.to_datetime(df['event_end'])
    df['road_number'] = df['road_number'].replace("", np.nan)
    
    # Remove the 'road_number' column
    df = df.drop('road_number', axis=1)
    
    # Strip whitespace from string columns
    df['category'] = df['category'].str.strip()
    df['incident_severity'] = df['incident_severity'].str.strip()
    
    # Detect and remove outliers
    outlier_index = df[df['duration_seconds'] > 500].index
    df = df.drop(outlier_index)
    
    # Encode categorical columns
    label_encoder = LabelEncoder()
    df['category_encoded'] = label_encoder.fit_transform(df['category'])
    
    # Classify incident severity
    def classify_incident_severity(severity):
        low = ['HA1', 'HB1', 'HC1', 'SP1', 'HC4', 'HC13']
        medium = ['HA2', 'HB2', 'HC2', 'HC5', 'HC7', 'HC14', 'HC16', 'SP2', 'SP3', 'HC6', 'HC8', 'HC10', 'HC15', 'HC17', 'HC19']
        high = ['HA3', 'HB3', 'HC3', 'HC9', 'HC11', 'HC12', 'HC18', 'HC20', 'HC21', 'SP4', 'SP5']
        
        if severity in low:
            return 'Low'
        elif severity in medium:
            return 'Medium'
        elif severity in high:
            return 'High'
        else:
            return 'Unknown'
    
    df['incident_severity_bin'] = df['incident_severity'].apply(classify_incident_severity)
    
    # Random sampling with replacement
    sampled_dfs = []
    for bin_label, group in df.groupby('incident_severity_bin'):
        sampled_dfs.append(group.sample(n=63533, replace=True, random_state=42))
    df = pd.concat(sampled_dfs)
    
    return df

def detect_outliers_iqr(df, columns):
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
            outlier_summary[column] = 'Non-numeric data'
    return outlier_summary

def plot_boxplots(df, columns):
    numeric_columns = [column for column in columns if pd.api.types.is_numeric_dtype(df[column])]
    plt.figure(figsize=(15, 5 * len(numeric_columns)))
    for i, column in enumerate(numeric_columns, 1):
        plt.subplot(len(numeric_columns), 1, i)
        sns.boxplot(x=df[column].dropna())
        plt.title(f'Boxplot of {column}')
    plt.tight_layout()
    plt.show()

def load_and_clean_data():
    df = load_data(db_params)
    df = clean_data(df)
    return df

# Example usage
if __name__ == "__main__":
    df_safe_driving = load_and_clean_data(db_params)
