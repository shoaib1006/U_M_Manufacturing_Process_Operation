# 🏭 Manufacturing Process Health & Operational Efficiency Dashboard
### 6G-Enabled Smart Factory Analytics | Unified Mentor × Thales Group

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Place the dataset
Put `smart_factory.csv` in the same folder as `app.py`.
Or let the app auto-download it on first run (requires internet + gdown).

### 3. Run the dashboard
```bash
streamlit run app.py
```

Open in browser: `http://localhost:8501`

---

## Dashboard Tabs

| Tab | Content |
|-----|---------|
| 🏠 **Factory Overview** | KPI cards, efficiency distribution, automated alerts, hourly trend |
| 🔧 **Machine Health** | MHI ranking, sensor boxplots, temperature heatmap, scorecard table |
| ⚡ **Production & Quality** | Speed ranking, defect heatmap, error trend, speed vs defect scatter |
| 📊 **Efficiency Diagnostics** | Efficiency by machine/shift/mode, KPI benchmarks, PdM score violin |
| 🔗 **Cross-Metric Analysis** | Correlation matrix, custom scatter, network vs error, radar chart |

## Sidebar Filters
- Date range picker
- Machine ID multi-select
- Operation Mode filter
- Efficiency Status filter
- Shift filter
- Custom X / Y axis metric for scatter plot

## Dataset Columns
`Machine_ID, Operation_Mode, Temperature_C, Vibration_Hz, Power_Consumption_kW,
Network_Latency_ms, Packet_Loss_%, Quality_Control_Defect_Rate_%,
Production_Speed_units_per_hr, Predictive_Maintenance_Score, Error_Rate_%, Efficiency_Status`

## Engineered Features (auto-computed)
- `Machine_Health_Index` — composite score (vibration 40%, temperature 35%, power 25%)
- `Defect_Density` — defect rate relative to production speed
- `Shift` — Night / Morning / Evening labels from timestamp
- `Temp_dev / Vib_dev / Pwr_dev` — deviation from machine-specific baselines
