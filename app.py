# ============================================================
#  Manufacturing Process Health & Operational Efficiency
#  6G-Enabled Smart Factories — Streamlit Dashboard
#  Unified Mentor × Thales Group
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler

# ── PAGE CONFIG ────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Factory Dashboard",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CUSTOM CSS ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main { background-color: #f0f2f6; }

.kpi-card {
    background: white;
    border-radius: 12px;
    padding: 20px 16px;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    border-top: 4px solid var(--accent, #1976D2);
    margin-bottom: 8px;
}
.kpi-value { font-size: 2rem; font-weight: 700; color: #1a1a2e; margin: 4px 0; }
.kpi-label { font-size: 0.78rem; color: #666; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em; }
.kpi-delta { font-size: 0.8rem; margin-top: 4px; }

.section-header {
    font-size: 1.15rem;
    font-weight: 700;
    color: #1a1a2e;
    border-left: 4px solid #1976D2;
    padding-left: 12px;
    margin: 24px 0 12px 0;
}
.alert-box {
    background: #fff3e0;
    border-left: 4px solid #ff9800;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 0.88rem;
}
.critical-box {
    background: #fce4ec;
    border-left: 4px solid #e53935;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 0.88rem;
}
.good-box {
    background: #e8f5e9;
    border-left: 4px solid #43a047;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 0.88rem;
}
[data-testid="stSidebar"] { background: #1a1a2e; }
[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stMultiSelect label { color: #b0bec5 !important; font-size: 0.8rem; text-transform: uppercase; }

.stTabs [data-baseweb="tab"] { font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── DATA LOADING & CACHING ─────────────────────────────────
@st.cache_data(show_spinner="Loading dataset…")
def load_data():
    import os, gdown
    path = "smart_factory.csv"
    if not os.path.exists(path):
        gdown.download(
            "https://drive.google.com/uc?id=1DINKtcGlQfsGtPfYkPKgI3kqcjHv4Bml",
            path, quiet=True
        )
    df = pd.read_csv(path)

    # Timestamp
    df["DateTime"] = pd.to_datetime(
        df["Date"] + " " + df["Timestamp"], dayfirst=True, errors="coerce"
    )
    df = df.sort_values("DateTime").reset_index(drop=True)

    # Time features
    df["Hour"]    = df["DateTime"].dt.hour
    df["Month"]   = df["DateTime"].dt.month
    df["DayName"] = df["DateTime"].dt.day_name()
    df["Shift"]   = pd.cut(
        df["Hour"], bins=[-1, 7, 15, 23],
        labels=["Night (00-07)", "Morning (08-15)", "Evening (16-23)"]
    )

    # Normalise categoricals
    df["Operation_Mode"]    = df["Operation_Mode"].str.strip().str.title()
    df["Efficiency_Status"] = df["Efficiency_Status"].str.strip().str.title()

    # Baselines & deviations
    baselines = df.groupby("Machine_ID")[
        ["Temperature_C", "Vibration_Hz", "Power_Consumption_kW"]
    ].median().rename(columns={
        "Temperature_C": "Temp_base",
        "Vibration_Hz":  "Vib_base",
        "Power_Consumption_kW": "Pwr_base"
    })
    df = df.merge(baselines, on="Machine_ID", how="left")
    df["Temp_dev"] = (df["Temperature_C"] - df["Temp_base"]).abs()
    df["Vib_dev"]  = (df["Vibration_Hz"]  - df["Vib_base"]).abs()
    df["Pwr_dev"]  = (df["Power_Consumption_kW"] - df["Pwr_base"]).abs()

    # Machine Health Index
    scaler = MinMaxScaler()
    dev_norm = pd.DataFrame(
        scaler.fit_transform(df[["Temp_dev", "Vib_dev", "Pwr_dev"]]),
        columns=["tn", "vn", "pn"]
    )
    df["Machine_Health_Index"] = (
        100 - (0.35 * dev_norm["tn"] + 0.40 * dev_norm["vn"] + 0.25 * dev_norm["pn"]) * 100
    ).round(2)

    # Defect density
    df["Defect_Density"] = (
        df["Quality_Control_Defect_Rate_%"] /
        df["Production_Speed_units_per_hr"].replace(0, np.nan)
    ).fillna(0).round(4)

    # Efficiency ordinal
    df["Eff_Num"] = df["Efficiency_Status"].map({"Low": 0, "Medium": 1, "High": 2})

    return df


df_full = load_data()

# ── SIDEBAR ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏭 Smart Factory")
    st.markdown("**6G-Enabled Operations**")
    st.markdown("---")

    st.markdown("### 🎛️ Filters")

    # Date range
    min_date = df_full["DateTime"].min().date()
    max_date = df_full["DateTime"].max().date()
    date_range = st.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    # Machine selector
    all_machines = sorted(df_full["Machine_ID"].unique())
    machine_sel = st.multiselect(
        "Machine IDs",
        options=all_machines,
        default=all_machines,
        help="Select one or more machines"
    )

    # Operation mode
    all_modes = sorted(df_full["Operation_Mode"].dropna().unique())
    mode_sel = st.multiselect(
        "Operation Mode",
        options=all_modes,
        default=all_modes
    )

    # Efficiency filter
    all_eff = ["High", "Medium", "Low"]
    eff_sel = st.multiselect(
        "Efficiency Status",
        options=all_eff,
        default=all_eff
    )

    # Shift
    all_shifts = df_full["Shift"].dropna().unique().tolist()
    shift_sel = st.multiselect(
        "Shift",
        options=all_shifts,
        default=all_shifts
    )

    st.markdown("---")
    st.markdown("### ⚙️ Metric Comparison")
    x_metric = st.selectbox("X-Axis Metric", [
        "Temperature_C", "Vibration_Hz", "Power_Consumption_kW",
        "Network_Latency_ms", "Packet_Loss_%",
        "Production_Speed_units_per_hr", "Predictive_Maintenance_Score"
    ])
    y_metric = st.selectbox("Y-Axis Metric", [
        "Quality_Control_Defect_Rate_%", "Error_Rate_%",
        "Production_Speed_units_per_hr", "Machine_Health_Index",
        "Temperature_C", "Vibration_Hz"
    ], index=0)

    st.markdown("---")
    st.caption("📊 Unified Mentor × Thales Group | April 2026")


# ── APPLY FILTERS ──────────────────────────────────────────
if len(date_range) == 2:
    start_dt = pd.Timestamp(date_range[0])
    end_dt   = pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)
else:
    start_dt = df_full["DateTime"].min()
    end_dt   = df_full["DateTime"].max()

df = df_full[
    (df_full["DateTime"] >= start_dt) &
    (df_full["DateTime"] <= end_dt) &
    (df_full["Machine_ID"].isin(machine_sel)) &
    (df_full["Operation_Mode"].isin(mode_sel)) &
    (df_full["Efficiency_Status"].isin(eff_sel)) &
    (df_full["Shift"].isin(shift_sel))
].copy()

if df.empty:
    st.warning("⚠️ No data matches the selected filters. Please widen your selection.")
    st.stop()

# ── HELPER COLOUR MAPS ─────────────────────────────────────
EFF_COLORS  = {"High": "#43A047", "Medium": "#FFA726", "Low": "#EF5350"}
MODE_COLORS = {"Active": "#1976D2", "Idle": "#78909C", "Maintenance": "#F57C00"}

# ═══════════════════════════════════════════════════════════
#  HEADER
# ═══════════════════════════════════════════════════════════
st.markdown("""
<div style='background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);
            border-radius:14px;padding:24px 32px;margin-bottom:24px;
            display:flex;align-items:center;gap:16px'>
  <span style='font-size:2.5rem'>🏭</span>
  <div>
    <div style='color:#ffffff;font-size:1.6rem;font-weight:700;line-height:1.2'>
        Manufacturing Process Health & Efficiency Dashboard</div>
    <div style='color:#90caf9;font-size:0.9rem;margin-top:4px'>
        6G-Enabled Smart Factory Analytics | Unified Mentor × Thales Group</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── ACTIVE FILTER SUMMARY ──────────────────────────────────
st.caption(
    f"🔎 Showing **{len(df):,}** records | "
    f"{len(machine_sel)} machines | "
    f"Modes: {', '.join(mode_sel)} | "
    f"Efficiency: {', '.join(eff_sel)}"
)

# ── TABS ───────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Factory Overview",
    "🔧 Machine Health",
    "⚡ Production & Quality",
    "📊 Efficiency Diagnostics",
    "🔗 Cross-Metric Analysis"
])

# ═══════════════════════════════════════════════════════════
#  TAB 1 – FACTORY OVERVIEW
# ═══════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">Factory-Wide KPI Cards</div>', unsafe_allow_html=True)

    mhi_avg     = df["Machine_Health_Index"].mean()
    speed_avg   = df["Production_Speed_units_per_hr"].mean()
    defect_avg  = df["Quality_Control_Defect_Rate_%"].mean()
    error_avg   = df["Error_Rate_%"].mean()
    latency_avg = df["Network_Latency_ms"].mean()
    pkt_avg     = df["Packet_Loss_%"].mean()
    pdm_avg     = df["Predictive_Maintenance_Score"].mean()
    high_pct    = (df["Efficiency_Status"] == "High").mean() * 100

    c1,c2,c3,c4 = st.columns(4)
    c5,c6,c7,c8 = st.columns(4)

    def kpi_html(label, value, accent="#1976D2", unit=""):
        return f"""<div class="kpi-card" style="--accent:{accent}">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}{unit}</div>
        </div>"""

    c1.markdown(kpi_html("Machine Health Index", f"{mhi_avg:.1f}", "#1976D2", "/100"), unsafe_allow_html=True)
    c2.markdown(kpi_html("Avg Production Speed", f"{speed_avg:.0f}", "#43A047", " u/hr"), unsafe_allow_html=True)
    c3.markdown(kpi_html("Avg Defect Rate", f"{defect_avg:.2f}", "#EF5350", "%"), unsafe_allow_html=True)
    c4.markdown(kpi_html("Avg Error Rate", f"{error_avg:.2f}", "#F57C00", "%"), unsafe_allow_html=True)
    c5.markdown(kpi_html("High Efficiency %", f"{high_pct:.1f}", "#43A047", "%"), unsafe_allow_html=True)
    c6.markdown(kpi_html("PdM Score", f"{pdm_avg:.3f}", "#7B1FA2"), unsafe_allow_html=True)
    c7.markdown(kpi_html("Network Latency", f"{latency_avg:.1f}", "#0288D1", " ms"), unsafe_allow_html=True)
    c8.markdown(kpi_html("Packet Loss", f"{pkt_avg:.2f}", "#E64A19", "%"), unsafe_allow_html=True)

    st.markdown("---")
    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.markdown('<div class="section-header">Efficiency Status Distribution</div>', unsafe_allow_html=True)
        eff_cnt = df["Efficiency_Status"].value_counts().reset_index()
        eff_cnt.columns = ["Status", "Count"]
        fig_pie = px.pie(
            eff_cnt, names="Status", values="Count",
            color="Status", color_discrete_map=EFF_COLORS,
            hole=0.42
        )
        fig_pie.update_traces(textinfo="label+percent", textfont_size=13)
        fig_pie.update_layout(
            height=360, margin=dict(t=20, b=20, l=20, r=20),
            showlegend=True, legend=dict(orientation="h", y=-0.1)
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-header">Efficiency by Operation Mode</div>', unsafe_allow_html=True)
        eff_mode = (df.groupby(["Operation_Mode", "Efficiency_Status"])
                      .size().reset_index(name="Count"))
        total_mode = eff_mode.groupby("Operation_Mode")["Count"].transform("sum")
        eff_mode["Pct"] = (eff_mode["Count"] / total_mode * 100).round(1)
        fig_mode = px.bar(
            eff_mode, x="Operation_Mode", y="Pct", color="Efficiency_Status",
            color_discrete_map=EFF_COLORS, barmode="stack",
            labels={"Pct": "%", "Operation_Mode": ""},
        )
        fig_mode.update_layout(height=360, margin=dict(t=20, b=20, l=20, r=20),
                                legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig_mode, use_container_width=True)

    # Alerts
    st.markdown('<div class="section-header">⚠️ Automated Alerts</div>', unsafe_allow_html=True)
    mhi_machine = df.groupby("Machine_ID")["Machine_Health_Index"].mean()
    critical_machines = mhi_machine[mhi_machine < 35].index.tolist()
    warn_machines     = mhi_machine[(mhi_machine >= 35) & (mhi_machine < 50)].index.tolist()

    if critical_machines:
        st.markdown(
            f'<div class="critical-box">🚨 <b>CRITICAL:</b> Machines {critical_machines} '
            f'have Machine Health Index below 35 — immediate inspection recommended.</div>',
            unsafe_allow_html=True
        )
    if warn_machines:
        st.markdown(
            f'<div class="alert-box">⚠️ <b>WARNING:</b> Machines {warn_machines} '
            f'have Machine Health Index between 35–50 — schedule preventive checks.</div>',
            unsafe_allow_html=True
        )
    if high_pct >= 40:
        st.markdown(
            f'<div class="good-box">✅ <b>GOOD:</b> {high_pct:.1f}% of records are in High Efficiency status.</div>',
            unsafe_allow_html=True
        )

    # Hourly trend
    st.markdown('<div class="section-header">Production Speed — Hourly Trend</div>', unsafe_allow_html=True)
    hourly = df.set_index("DateTime")["Production_Speed_units_per_hr"].resample("1H").mean().reset_index()
    hourly.columns = ["DateTime", "Speed"]
    fig_trend = px.area(
        hourly, x="DateTime", y="Speed",
        labels={"Speed": "Units/hr", "DateTime": ""},
        color_discrete_sequence=["#1976D2"]
    )
    fig_trend.update_layout(height=260, margin=dict(t=10, b=10))
    st.plotly_chart(fig_trend, use_container_width=True)


# ═══════════════════════════════════════════════════════════
#  TAB 2 – MACHINE HEALTH
# ═══════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Machine Health Index — Ranking</div>', unsafe_allow_html=True)

    mhi_df = df.groupby("Machine_ID").agg(
        MHI     = ("Machine_Health_Index", "mean"),
        Temp    = ("Temperature_C", "mean"),
        Vib     = ("Vibration_Hz", "mean"),
        Pwr     = ("Power_Consumption_kW", "mean"),
        Records = ("Machine_ID", "count")
    ).round(2).reset_index().sort_values("MHI")

    mhi_df["Color"] = mhi_df["MHI"].apply(
        lambda v: "#EF5350" if v < 35 else ("#FFA726" if v < 50 else "#43A047")
    )
    mhi_df["Status"] = mhi_df["MHI"].apply(
        lambda v: "🔴 Critical" if v < 35 else ("🟡 Warning" if v < 50 else "🟢 Healthy")
    )

    fig_mhi = go.Figure(go.Bar(
        x=mhi_df["MHI"], y=mhi_df["Machine_ID"].astype(str),
        orientation="h",
        marker_color=mhi_df["Color"],
        text=mhi_df["MHI"].round(1),
        textposition="outside"
    ))
    fig_mhi.add_vline(x=mhi_df["MHI"].mean(), line_dash="dash",
                      line_color="navy", annotation_text="Fleet Avg")
    fig_mhi.update_layout(
        height=max(400, len(mhi_df) * 14),
        xaxis_title="Machine Health Index (0–100)",
        yaxis_title="Machine ID",
        margin=dict(l=60, r=40, t=20, b=40),
        showlegend=False
    )
    st.plotly_chart(fig_mhi, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Sensor Boxplots by Operation Mode</div>', unsafe_allow_html=True)
        sensor_choice = st.selectbox(
            "Select sensor",
            ["Temperature_C", "Vibration_Hz", "Power_Consumption_kW"],
            key="sensor_box"
        )
        fig_box = px.box(
            df, x="Operation_Mode", y=sensor_choice,
            color="Operation_Mode", color_discrete_map=MODE_COLORS,
            points=False
        )
        fig_box.update_layout(height=380, showlegend=False,
                               margin=dict(t=20, b=20))
        st.plotly_chart(fig_box, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Temperature Heatmap — Machine × Mode</div>', unsafe_allow_html=True)
        pv = df.pivot_table(
            values="Temperature_C", index="Machine_ID",
            columns="Operation_Mode", aggfunc="mean"
        ).round(1)
        fig_heat = px.imshow(
            pv, color_continuous_scale="YlOrRd",
            labels={"color": "°C"},
            aspect="auto"
        )
        fig_heat.update_layout(height=380, margin=dict(t=20, b=20))
        st.plotly_chart(fig_heat, use_container_width=True)

    # Scorecard table
    st.markdown('<div class="section-header">Machine Health Scorecard</div>', unsafe_allow_html=True)
    scorecard = mhi_df[["Machine_ID", "Status", "MHI", "Temp", "Vib", "Pwr", "Records"]].copy()
    scorecard.columns = ["Machine ID", "Health Status", "Health Index", "Avg Temp (°C)",
                          "Avg Vibration (Hz)", "Avg Power (kW)", "Records"]
    st.dataframe(
        scorecard.sort_values("Health Index"),
        use_container_width=True,
        hide_index=True
    )

    # Vibration stability
    st.markdown('<div class="section-header">Vibration Stability — Rolling Std Dev by Machine (Top 10 Unstable)</div>',
                unsafe_allow_html=True)
    vib_std = df.groupby("Machine_ID")["Vibration_Hz"].std().nlargest(10).reset_index()
    vib_std.columns = ["Machine_ID", "Vib_Std"]
    fig_vstd = px.bar(
        vib_std, x="Machine_ID", y="Vib_Std", color="Vib_Std",
        color_continuous_scale="Reds",
        labels={"Vib_Std": "Vibration Std Dev (Hz)", "Machine_ID": "Machine ID"}
    )
    fig_vstd.update_layout(height=320, margin=dict(t=20, b=20), showlegend=False)
    st.plotly_chart(fig_vstd, use_container_width=True)


# ═══════════════════════════════════════════════════════════
#  TAB 3 – PRODUCTION & QUALITY
# ═══════════════════════════════════════════════════════════
with tab3:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Avg Production Speed by Machine</div>', unsafe_allow_html=True)
        spd = df.groupby("Machine_ID")["Production_Speed_units_per_hr"].agg(
            Mean="mean", Std="std"
        ).reset_index().sort_values("Mean")
        fleet_avg = spd["Mean"].mean()
        fig_spd = go.Figure(go.Bar(
            x=spd["Mean"], y=spd["Machine_ID"].astype(str),
            orientation="h",
            error_x=dict(array=spd["Std"].fillna(0), visible=True),
            marker_color=[
                "#EF5350" if v < fleet_avg * 0.8 else "#43A047"
                for v in spd["Mean"]
            ]
        ))
        fig_spd.add_vline(x=fleet_avg, line_dash="dash", line_color="navy",
                          annotation_text=f"Fleet Avg {fleet_avg:.0f}")
        fig_spd.update_layout(
            height=420, xaxis_title="Units / hr", yaxis_title="Machine ID",
            margin=dict(l=60, r=20, t=10, b=30), showlegend=False
        )
        st.plotly_chart(fig_spd, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Defect Rate by Machine (Top 20)</div>', unsafe_allow_html=True)
        def_m = df.groupby("Machine_ID")["Quality_Control_Defect_Rate_%"].mean().nlargest(20).reset_index()
        def_m.columns = ["Machine_ID", "Defect_Rate"]
        fleet_defect = df["Quality_Control_Defect_Rate_%"].mean()
        fig_def = px.bar(
            def_m, x="Machine_ID", y="Defect_Rate",
            color="Defect_Rate", color_continuous_scale="Reds",
            labels={"Defect_Rate": "Avg Defect Rate (%)", "Machine_ID": "Machine ID"}
        )
        fig_def.add_hline(y=fleet_defect, line_dash="dash", line_color="navy",
                          annotation_text=f"Fleet Avg {fleet_defect:.2f}%")
        fig_def.update_layout(height=420, margin=dict(t=10, b=30), showlegend=False)
        st.plotly_chart(fig_def, use_container_width=True)

    # Defect Heatmap
    st.markdown('<div class="section-header">Defect Rate Heatmap — Machine × Hour of Day</div>', unsafe_allow_html=True)
    top_machines_def = (df.groupby("Machine_ID")["Quality_Control_Defect_Rate_%"]
                         .mean().nlargest(20).index)
    sub_def = df[df["Machine_ID"].isin(top_machines_def)]
    pv_def  = sub_def.pivot_table(
        values="Quality_Control_Defect_Rate_%",
        index="Machine_ID", columns="Hour", aggfunc="mean"
    ).round(2)
    fig_dheat = px.imshow(
        pv_def, color_continuous_scale="Reds",
        labels={"color": "Defect %", "x": "Hour of Day", "y": "Machine ID"},
        aspect="auto"
    )
    fig_dheat.update_layout(height=400, margin=dict(t=20, b=20))
    st.plotly_chart(fig_dheat, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<div class="section-header">Error Rate Trend (24-hr Rolling Avg)</div>',
                    unsafe_allow_html=True)
        err_h = df.set_index("DateTime")["Error_Rate_%"].resample("1H").mean()
        err_roll = err_h.rolling(24, min_periods=1).mean().reset_index()
        err_roll.columns = ["DateTime", "Error_Rate"]
        fig_err = px.area(
            err_roll, x="DateTime", y="Error_Rate",
            color_discrete_sequence=["#EF5350"],
            labels={"Error_Rate": "Error Rate (%)", "DateTime": ""}
        )
        fig_err.update_layout(height=300, margin=dict(t=10, b=10))
        st.plotly_chart(fig_err, use_container_width=True)

    with col4:
        st.markdown('<div class="section-header">Error Rate by Shift & Mode</div>',
                    unsafe_allow_html=True)
        err_sm = df.groupby(["Shift", "Operation_Mode"])["Error_Rate_%"].mean().reset_index()
        fig_esm = px.bar(
            err_sm, x="Shift", y="Error_Rate_%",
            color="Operation_Mode", color_discrete_map=MODE_COLORS,
            barmode="group",
            labels={"Error_Rate_%": "Avg Error Rate (%)", "Shift": ""}
        )
        fig_esm.update_layout(height=300, margin=dict(t=10, b=10),
                               legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig_esm, use_container_width=True)

    # Speed vs Defect scatter
    st.markdown('<div class="section-header">Speed vs Defect Rate — by Efficiency Status</div>',
                unsafe_allow_html=True)
    sample_sd = df.sample(min(8000, len(df)), random_state=42)
    fig_sd = px.scatter(
        sample_sd,
        x="Production_Speed_units_per_hr",
        y="Quality_Control_Defect_Rate_%",
        color="Efficiency_Status",
        color_discrete_map=EFF_COLORS,
        opacity=0.45, trendline="ols",
        labels={
            "Production_Speed_units_per_hr": "Speed (units/hr)",
            "Quality_Control_Defect_Rate_%": "Defect Rate (%)"
        }
    )
    fig_sd.update_layout(height=400, margin=dict(t=20, b=20))
    st.plotly_chart(fig_sd, use_container_width=True)


# ═══════════════════════════════════════════════════════════
#  TAB 4 – EFFICIENCY DIAGNOSTICS
# ═══════════════════════════════════════════════════════════
with tab4:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Efficiency % by Machine</div>', unsafe_allow_html=True)
        eff_m = (df.groupby(["Machine_ID", "Efficiency_Status"])
                   .size().reset_index(name="Count"))
        tot_m = eff_m.groupby("Machine_ID")["Count"].transform("sum")
        eff_m["Pct"] = (eff_m["Count"] / tot_m * 100).round(1)
        fig_em = px.bar(
            eff_m, x="Machine_ID", y="Pct",
            color="Efficiency_Status", color_discrete_map=EFF_COLORS,
            barmode="stack",
            labels={"Pct": "%", "Machine_ID": "Machine ID"}
        )
        fig_em.update_layout(height=400, margin=dict(t=10, b=30),
                              legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig_em, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Efficiency by Shift</div>', unsafe_allow_html=True)
        eff_sh = (df.groupby(["Shift", "Efficiency_Status"])
                    .size().reset_index(name="Count"))
        tot_sh = eff_sh.groupby("Shift")["Count"].transform("sum")
        eff_sh["Pct"] = (eff_sh["Count"] / tot_sh * 100).round(1)
        fig_esh = px.bar(
            eff_sh, x="Shift", y="Pct",
            color="Efficiency_Status", color_discrete_map=EFF_COLORS,
            barmode="stack",
            labels={"Pct": "%", "Shift": ""}
        )
        fig_esh.update_layout(height=400, margin=dict(t=10, b=30),
                               legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig_esh, use_container_width=True)

    # KPI by Efficiency Class
    st.markdown('<div class="section-header">KPI Benchmarks by Efficiency Class</div>',
                unsafe_allow_html=True)
    kpi_cols_eff = [
        "Temperature_C", "Vibration_Hz", "Power_Consumption_kW",
        "Production_Speed_units_per_hr", "Quality_Control_Defect_Rate_%",
        "Error_Rate_%", "Machine_Health_Index",
        "Network_Latency_ms", "Packet_Loss_%"
    ]
    kpi_class = df.groupby("Efficiency_Status")[kpi_cols_eff].mean().round(3)
    kpi_norm  = kpi_class.div(kpi_class.max())

    fig_kpi = go.Figure()
    eff_order = ["Low", "Medium", "High"]
    bar_colors = ["#EF5350", "#FFA726", "#43A047"]
    for eff, col in zip(eff_order, bar_colors):
        if eff in kpi_norm.index:
            fig_kpi.add_trace(go.Bar(
                name=eff,
                x=[c.replace("_", " ") for c in kpi_cols_eff],
                y=kpi_norm.loc[eff],
                marker_color=col
            ))
    fig_kpi.update_layout(
        barmode="group", height=380,
        yaxis_title="Normalised Value",
        margin=dict(t=10, b=80),
        legend=dict(orientation="h", y=-0.3),
        xaxis_tickangle=-30
    )
    st.plotly_chart(fig_kpi, use_container_width=True)

    # Summary table
    st.markdown('<div class="section-header">Efficiency Class — KPI Summary Table</div>',
                unsafe_allow_html=True)
    st.dataframe(
        kpi_class.style.background_gradient(cmap="RdYlGn", axis=0),
        use_container_width=True
    )

    # Predictive Maintenance Score distribution
    st.markdown('<div class="section-header">Predictive Maintenance Score — Distribution by Efficiency</div>',
                unsafe_allow_html=True)
    fig_pdm = px.violin(
        df, x="Efficiency_Status", y="Predictive_Maintenance_Score",
        color="Efficiency_Status", color_discrete_map=EFF_COLORS,
        box=True, points=False,
        category_orders={"Efficiency_Status": ["Low", "Medium", "High"]},
        labels={"Predictive_Maintenance_Score": "PdM Score", "Efficiency_Status": ""}
    )
    fig_pdm.update_layout(height=360, showlegend=False, margin=dict(t=10, b=10))
    st.plotly_chart(fig_pdm, use_container_width=True)


# ═══════════════════════════════════════════════════════════
#  TAB 5 – CROSS-METRIC ANALYSIS
# ═══════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header">Correlation Matrix</div>', unsafe_allow_html=True)
    numeric_cols = [
        "Temperature_C", "Vibration_Hz", "Power_Consumption_kW",
        "Network_Latency_ms", "Packet_Loss_%",
        "Quality_Control_Defect_Rate_%", "Production_Speed_units_per_hr",
        "Predictive_Maintenance_Score", "Error_Rate_%",
        "Machine_Health_Index", "Defect_Density", "Eff_Num"
    ]
    corr = df[numeric_cols].corr().round(2)
    fig_corr = px.imshow(
        corr,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        text_auto=True,
        aspect="auto",
        labels={"color": "Pearson r"}
    )
    fig_corr.update_traces(textfont_size=9)
    fig_corr.update_layout(height=520, margin=dict(t=20, b=20))
    st.plotly_chart(fig_corr, use_container_width=True)

    # User-defined scatter
    st.markdown('<div class="section-header">Custom Metric Comparison (Sidebar Controls)</div>',
                unsafe_allow_html=True)
    color_by = st.selectbox(
        "Colour by", ["Efficiency_Status", "Operation_Mode", "Shift"], key="scatter_color"
    )
    color_map = EFF_COLORS if color_by == "Efficiency_Status" else (
        MODE_COLORS if color_by == "Operation_Mode" else None
    )
    sample_cm = df.sample(min(8000, len(df)), random_state=7)
    fig_cm = px.scatter(
        sample_cm, x=x_metric, y=y_metric,
        color=color_by,
        color_discrete_map=color_map,
        opacity=0.45, trendline="ols",
        labels={x_metric: x_metric.replace("_", " "),
                y_metric: y_metric.replace("_", " ")}
    )
    fig_cm.update_layout(height=440, margin=dict(t=10, b=20))
    st.plotly_chart(fig_cm, use_container_width=True)

    # Network health
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">Network Latency vs Error Rate</div>',
                    unsafe_allow_html=True)
        sample_net = df.sample(min(5000, len(df)), random_state=5)
        fig_net = px.scatter(
            sample_net, x="Network_Latency_ms", y="Error_Rate_%",
            color="Operation_Mode", color_discrete_map=MODE_COLORS,
            opacity=0.45, trendline="ols",
            labels={"Network_Latency_ms": "Latency (ms)",
                    "Error_Rate_%": "Error Rate (%)"}
        )
        fig_net.update_layout(height=360, margin=dict(t=10, b=10),
                               legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig_net, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Packet Loss vs Defect Rate</div>',
                    unsafe_allow_html=True)
        fig_pkt = px.scatter(
            sample_net, x="Packet_Loss_%", y="Quality_Control_Defect_Rate_%",
            color="Efficiency_Status", color_discrete_map=EFF_COLORS,
            opacity=0.45, trendline="ols",
            labels={"Packet_Loss_%": "Packet Loss (%)",
                    "Quality_Control_Defect_Rate_%": "Defect Rate (%)"}
        )
        fig_pkt.update_layout(height=360, margin=dict(t=10, b=10),
                               legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig_pkt, use_container_width=True)

    # Pair summary
    st.markdown('<div class="section-header">Sensor Pair Averages by Efficiency Class</div>',
                unsafe_allow_html=True)
    pair_cols = [
        "Temperature_C", "Vibration_Hz", "Power_Consumption_kW",
        "Network_Latency_ms", "Packet_Loss_%", "Error_Rate_%"
    ]
    pair_eff = df.groupby("Efficiency_Status")[pair_cols].mean().round(2)
    fig_radar_data = []
    for eff, clr in zip(["Low", "Medium", "High"], ["#EF5350", "#FFA726", "#43A047"]):
        if eff in pair_eff.index:
            vals = pair_eff.loc[eff].tolist()
            norm_vals = [v / pair_eff[c].max() if pair_eff[c].max() > 0 else 0
                         for v, c in zip(vals, pair_cols)]
            fig_radar_data.append(go.Scatterpolar(
                r=norm_vals + [norm_vals[0]],
                theta=[c.replace("_", " ") for c in pair_cols] + [pair_cols[0].replace("_", " ")],
                fill="toself", name=eff,
                line_color=clr, fillcolor=clr,
                opacity=0.3
            ))
    if fig_radar_data:
        fig_radar = go.Figure(data=fig_radar_data)
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True, height=420,
            legend=dict(orientation="h", y=-0.1),
            margin=dict(t=30, b=50)
        )
        st.plotly_chart(fig_radar, use_container_width=True)

# ── FOOTER ─────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "🏭 Manufacturing Process Health & Operational Efficiency Analysis | "
    "6G-Enabled Smart Factories | Unified Mentor × Thales Group | April 2026"
)
