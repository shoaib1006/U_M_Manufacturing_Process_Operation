
import streamlit as st
import pandas as pd
import plotly.express as px
import os
import requests

st.set_page_config(page_title='Manufacturing Operational Intelligence', layout='wide')

@st.cache_data
def load_data():
    FILE_PATH = 'smart_factory_cleaned.csv'
    # URL for the raw dataset if local cleaned file is missing
    SOURCE_URL = 'https://drive.google.com/uc?id=1DINKtcGlQfsGtPfYkPKgI3kqcjHv4Bml'
    
    if not os.path.exists(FILE_PATH):
        try:
            # If cleaned file is missing, we load the raw source and apply minimal cleaning
            df = pd.read_csv(SOURCE_URL)
            # Minimal recovery cleaning to match dashboard expectations
            df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Timestamp'], dayfirst=True, errors='coerce')
            # Calculate a dummy Machine_Health_Index if column is missing
            if 'Machine_Health_Index' not in df.columns:
                df['Machine_Health_Index'] = 100 - (df['Error_Rate_%'] * 5)
            if 'Efficiency_Status' not in df.columns:
                df['Efficiency_Status'] = 'Low'
            return df
        except Exception as e:
            st.error(f'Could not load data: {e}')
            return pd.DataFrame()
    
    return pd.read_csv(FILE_PATH)

df = load_data()

if not df.empty:
    st.title('ἳD Smart Factory Operational Intelligence')
    st.sidebar.header('Filter Controls')
    
    # Use machine IDs from the loaded dataframe
    m_list = sorted(df['Machine_ID'].unique())
    selected_machine = st.sidebar.multiselect('Select Machine ID', options=m_list, default=m_list[:5])
    
    modes = df['Operation_Mode'].unique()
    selected_mode = st.sidebar.multiselect('Operation Mode', options=modes, default=modes)

    filtered_df = df[(df['Machine_ID'].isin(selected_machine)) & (df['Operation_Mode'].isin(selected_mode))]

    # --- KPI Metrics ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Avg Machine Health', f"{filtered_df['Machine_Health_Index'].mean():.1f}")
    col2.metric('Avg Speed (units/hr)', f"{filtered_df['Production_Speed_units_per_hr'].mean():.1f}")
    col3.metric('Avg Defect Rate %', f"{filtered_df['Quality_Control_Defect_Rate_%'].mean():.2f}%")
    col4.metric('Avg Error Rate %', f"{filtered_df['Error_Rate_%'].mean():.2f}%")

    st.divider()
    c_left, c_right = st.columns(2)

    with c_left:
        st.subheader('Machine Health Index Distribution')
        fig_mhi = px.box(filtered_df, x='Operation_Mode', y='Machine_Health_Index', color='Operation_Mode')
        st.plotly_chart(fig_mhi, use_container_width=True)

    with c_right:
        st.subheader('Speed vs. Error Rate')
        sample_size = min(len(filtered_df), 2000)
        fig_scat = px.scatter(filtered_df.sample(sample_size), 
                              x='Production_Speed_units_per_hr', y='Error_Rate_%', 
                              color='Efficiency_Status', opacity=0.6)
        st.plotly_chart(fig_scat, use_container_width=True)

    st.subheader('Production Trend')
    trend_df = filtered_df.set_index(pd.to_datetime(filtered_df['DateTime'])).resample('H')['Production_Speed_units_per_hr'].mean().reset_index()
    fig_line = px.line(trend_df, x='DateTime', y='Production_Speed_units_per_hr')
    st.plotly_chart(fig_line, use_container_width=True)
else:
    st.warning('No data available to display.')
