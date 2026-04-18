
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title='Manufacturing Health & Efficiency', layout='wide')

@st.cache_data
def load_data():
    df = pd.read_csv('smart_factory_cleaned.csv')
    return df

df = load_data()

st.title('ἳD Smart Factory Operational Intelligence')
st.sidebar.header('Filter Controls')
selected_machine = st.sidebar.multiselect('Select Machine ID', options=sorted(df['Machine_ID'].unique()), default=df['Machine_ID'].unique()[:5])
selected_mode = st.sidebar.multiselect('Operation Mode', options=df['Operation_Mode'].unique(), default=df['Operation_Mode'].unique())

filtered_df = df[(df['Machine_ID'].isin(selected_machine)) & (df['Operation_Mode'].isin(selected_mode))]

# --- KPI Metrics ---
col1, col2, col3, col4 = st.columns(4)
col1.metric('Avg Machine Health', f"{filtered_df['Machine_Health_Index'].mean():.1f}")
col2.metric('Avg Speed (units/hr)', f"{filtered_df['Production_Speed_units_per_hr'].mean():.1f}")
col3.metric('Avg Defect Rate %', f"{filtered_df['Quality_Control_Defect_Rate_%'].mean():.2f}%")
col4.metric('Avg Error Rate %', f"{filtered_df['Error_Rate_%'].mean():.2f}%")

# --- Visuals ---
st.divider()
c_left, c_right = st.columns(2)

with c_left:
    st.subheader('Machine Health Index Distribution')
    fig_mhi = px.box(filtered_df, x='Operation_Mode', y='Machine_Health_Index', color='Operation_Mode')
    st.plotly_chart(fig_mhi, use_container_width=True)

with c_right:
    st.subheader('Speed vs. Error Rate')
    fig_scat = px.scatter(filtered_df.sample(min(len(filtered_df), 2000)), 
                          x='Production_Speed_units_per_hr', y='Error_Rate_%', 
                          color='Efficiency_Status', opacity=0.6)
    st.plotly_chart(fig_scat, use_container_width=True)

st.subheader('Production Trend over Time')
trend_df = filtered_df.set_index(pd.to_datetime(filtered_df['DateTime'])).resample('H')['Production_Speed_units_per_hr'].mean().reset_index()
fig_line = px.line(trend_df, x='DateTime', y='Production_Speed_units_per_hr', title='Hourly Production Speed')
st.plotly_chart(fig_line, use_container_width=True)
