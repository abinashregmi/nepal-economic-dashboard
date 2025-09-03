import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import joblib
from io import BytesIO

# Page Configuration
st.set_page_config(
    page_title="Nepal Economic Resilience Analysis",
    page_icon="🇳🇵",
    layout="wide"
)

# 1. Caching Data Loading and Processing
@st.cache_data
def load_and_process_data():
    """Loads datasets from CSV files, cleans them, and merges them into a single DataFrame."""
    try:
        tourism_df_raw = pd.read_csv('tourism_revenue.csv.csv', skiprows=4)
        world_bank_df_raw = pd.read_csv('world_bank_data.csv.csv', skiprows=4)
        open_data_nepal_df_raw = pd.read_csv('open_data_nepal.csv.csv', skiprows=4)
    except FileNotFoundError:
        st.error("One or more CSV files not found. Please make sure `tourism_revenue.csv.csv`, `world_bank_data.csv.csv`, and `open_data_nepal.csv.csv` are in your GitHub repository.")
        return None

    # Data Cleaning and Reshaping
    def clean_and_reshape(df, column_map, indicator_name_col='Indicator Name'):
        df_clean = df.drop(columns=['Country Name', 'Country Code', 'Indicator Code']).set_index(indicator_name_col).T
        df_clean.index.name = 'Year'
        df_clean = df_clean.rename(columns=column_map)
        df_clean = df_clean.loc[:, list(column_map.values())]
        df_clean = df_clean.reset_index()
        df_clean['Year'] = pd.to_numeric(df_clean['Year'], errors='coerce')
        return df_clean

    tourism_col_map = {'International tourism, receipts (current US$)': 'Tourism Revenue (USD mn)'}
    world_bank_col_map = {
        'GDP (current US$)': 'GDP (current US$)',
        'Personal remittances, received (% of GDP)': 'Personal Remittances (% of GDP)'
    }
    unemployment_col_map = {'Unemployment, total (% of total labor force) (modeled ILO estimate)': 'Unemployment Rate'}

    tourism_df = clean_and_reshape(tourism_df_raw, tourism_col_map)
    world_bank_df = clean_and_reshape(world_bank_df_raw, world_bank_col_map)
    open_data_nepal_df = clean_and_reshape(open_data_nepal_df_raw, unemployment_col_map)

    #Merging DataFrames
    merged_df = pd.merge(tourism_df, world_bank_df, on='Year', how='outer')
    merged_df = pd.merge(merged_df, open_data_nepal_df, on='Year', how='outer')
    merged_df = merged_df.sort_values('Year').reset_index(drop=True)

    # 2. Data Cleaning and Preprocessing
    merged_df.ffill(inplace=True)
    merged_df.bfill(inplace=True)
    merged_df.drop_duplicates(inplace=True)

    #Feature Engineering
    merged_df['Remittance_to_Tourism_Ratio'] = merged_df['Personal Remittances (% of GDP)'] / merged_df['Tourism Revenue (USD mn)']
    merged_df['GDP_Growth_Rate'] = merged_df['GDP (current US$)'].pct_change() * 100
    merged_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    merged_df.dropna(inplace=True)
    
    return merged_df

# Train the Model
@st.cache_resource
def train_model(df):
    """Trains a Linear Regression model and returns it."""
    features = ['Tourism Revenue (USD mn)', 'Personal Remittances (% of GDP)', 'Unemployment Rate']
    target = 'GDP_Growth_Rate'
    X = df[features]
    y = df[target]
    model = LinearRegression()
    model.fit(X, y)
    return model

# Main Application 
st.title("🇳🇵 Data-Driven Analysis of Nepal's Economic Resilience")

st.markdown("""
This interactive dashboard analyzes key economic indicators for Nepal. You can explore historical data trends and use the predictive model on the left to forecast the future GDP growth rate.
""")

df = load_and_process_data()

if df is not None:
    model = train_model(df)

    # Sidebar for User Input
    st.sidebar.header("🔮 GDP Growth Rate Predictor")
    st.sidebar.markdown("Adjust the sliders to see the model's prediction.")

    tourism_input = st.sidebar.slider(
        'Tourism Revenue (million USD)',
        min_value=float(df['Tourism Revenue (USD mn)'].min()),
        max_value=float(df['Tourism Revenue (USD mn)'].max()),
        value=float(df['Tourism Revenue (USD mn)'].mean())
    )
    remittances_input = st.sidebar.slider(
        'Personal Remittances (% of GDP)',
        min_value=float(df['Personal Remittances (% of GDP)'].min()),
        max_value=float(df['Personal Remittances (% of GDP)'].max()),
        value=float(df['Personal Remittances (% of GDP)'].mean())
    )
    unemployment_input = st.sidebar.slider(
        'Unemployment Rate (%)',
        min_value=float(df['Unemployment Rate'].min()),
        max_value=float(df['Unemployment Rate'].max()),
        value=float(df['Unemployment Rate'].mean())
    )

    #Prediction Logic
    input_data = pd.DataFrame({
        'Tourism Revenue (USD mn)': [tourism_input],
        'Personal Remittances (% of GDP)': [remittances_input],
        'Unemployment Rate': [unemployment_input]
    })
    prediction = model.predict(input_data)

    st.sidebar.subheader('Predicted GDP Growth Rate:')
    st.sidebar.metric(label="Growth Rate", value=f"{prediction[0]:.2f}%")

    # 3. EDA Visualizations
    st.markdown("---")
    st.header("📈 Exploratory Data Analysis")

    # Time-series plot
    st.subheader("Nepal's Tourism Revenue and GDP Over Time")
    fig1, ax1 = plt.subplots(figsize=(14, 7))
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Tourism Revenue (USD mn)', color='tab:blue')
    ax1.plot(df['Year'], df['Tourism Revenue (USD mn)'], color='tab:blue', marker='o')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax2 = ax1.twinx()
    ax2.set_ylabel('GDP (current US$)', color='tab:green')
    ax2.plot(df['Year'], df['GDP (current US$)'], color='tab:green', marker='x', linestyle='--')
    ax2.tick_params(axis='y', labelcolor='tab:green')
    fig1.tight_layout()
    st.pyplot(fig1)

    # Scatter plot
    st.subheader('Relationship between Remittances and GDP Growth Rate')
    fig2, ax = plt.subplots(figsize=(12, 7))
    sns.regplot(x='Personal Remittances (% of GDP)', y='GDP_Growth_Rate', data=df, ax=ax)
    st.pyplot(fig2)

    #Correlation Heatmap
    st.subheader('Correlation Matrix of Economic Indicators')
    fig3, ax = plt.subplots(figsize=(12, 10))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='viridis', fmt=".2f", linewidths=.5, ax=ax)
    st.pyplot(fig3)

    # Displays Data
    with st.expander("View the Cleaned and Processed Data"):
        st.dataframe(df)


