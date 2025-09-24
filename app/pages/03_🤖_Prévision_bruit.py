
import pandas as pd
import plotly.express as px
import streamlit as st
from utils import load_data, train_or_load_model, predict_interval

st.set_page_config(page_title="Prévision", page_icon="🤖", layout="wide")
st.title("🤖 Prévision du niveau de bruit")

data_path = st.session_state.get("data_path", "data/sample_noise.csv")
df = load_data(data_path)
model, metrics = train_or_load_model(df)
mae = metrics.get("mae", 2.5)

# Controls
with st.sidebar:
    st.header("Paramètres de prévision")
    zone = st.selectbox("Zone", sorted(df['zone_id'].unique().tolist()))
    min_date = df['timestamp'].min().date()
    max_date = df['timestamp'].max().date()
    date = st.date_input("Date", value=min_date, min_value=min_date, max_value=max_date)
    hour_start = st.slider("Heure de début", 0, 23, 8)
    horizon_h = st.slider("Durée (h)", 1, 24, 6)

    st.markdown("---")
    st.caption("Scénario de trafic (si aucune donnée future)")
    flights_scn = st.slider("Vols par heure (scénario)", 0, 20, 6)
    heavy_ratio_scn = st.slider("Ratio gros porteurs", 0.0, 1.0, 0.35, 0.05)
    altitude_scn = st.slider("Altitude moyenne (m)", 300, 4000, 1600, 50)

start_ts = pd.Timestamp(date) + pd.Timedelta(hours=hour_start)
times = pd.date_range(start=start_ts, periods=horizon_h, freq='H')

# Build future feature grid (fallback to scenario)
fut = pd.DataFrame({
    "timestamp": times,
    "zone_id": zone,
    "flights_count": flights_scn,
    "heavy_aircraft_ratio": heavy_ratio_scn,
    "altitude_mean_m": altitude_scn,
})
fut["hour"] = fut["timestamp"].dt.hour
fut["dow"] = fut["timestamp"].dt.dayofweek
fut["is_weekend"] = (fut["dow"] >= 5).astype(int)

preds, lo, hi = predict_interval(model, fut[["zone_id","flights_count","heavy_aircraft_ratio","altitude_mean_m","hour","dow","is_weekend"]], mae)

out = fut.copy()
out["laeq_pred"] = preds
out["lower"] = lo
out["upper"] = hi

st.subheader("Prévision horaire (avec incertitude ±MAE)")
fig = px.line(out, x="timestamp", y=["laeq_pred","lower","upper"], labels={"value":"LAeq (dB)", "timestamp":"Heure"}, height=430)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Synthèse")
col1, col2, col3 = st.columns(3)
col1.metric("LAeq moyen prévu", f"{out['laeq_pred'].mean():.1f} dB")
col2.metric("Min/Max prévu", f"{out['laeq_pred'].min():.1f} / {out['laeq_pred'].max():.1f} dB")
col3.metric("Incertitude (±MAE)", f"{mae:.1f} dB")
