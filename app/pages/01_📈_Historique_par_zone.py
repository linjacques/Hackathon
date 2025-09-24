
import pandas as pd
import plotly.express as px
import streamlit as st
from utils import load_data

st.set_page_config(page_title="Historique par zone", page_icon="📈", layout="wide")

st.title("📈 Historique par zone")

data_path = st.session_state.get("data_path", "data/sample_noise.csv")
df = load_data(data_path)

# Controls
with st.sidebar:
    st.header("Filtres")
    zones = st.multiselect("Zones", sorted(df['zone_id'].unique().tolist()), default=sorted(df['zone_id'].unique().tolist())[:3])
    min_date = df['timestamp'].min().date()
    max_date = df['timestamp'].max().date()
    start, end = st.date_input("Période", (min_date, max_date), min_value=min_date, max_value=max_date)
    agg = st.selectbox("Agrégation", ["Heure", "Jour"], index=0)
    if isinstance(start, tuple) or isinstance(start, list):
        start, end = start
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

m = df['zone_id'].isin(zones) & (df['timestamp'].between(start_ts, end_ts))
dff = df.loc[m].copy()

if dff.empty:
    st.warning("Aucune donnée pour les filtres sélectionnés.")
    st.stop()

# Resample
if agg == "Jour":
    dff = dff.set_index('timestamp').groupby('zone_id').resample('D')['laeq_db'].mean().reset_index()
    x = "timestamp"
else:
    dff = dff.set_index('timestamp').groupby('zone_id').resample('H')['laeq_db'].mean().reset_index()
    x = "timestamp"

st.subheader("Évolution temporelle")
fig = px.line(dff, x=x, y="laeq_db", color="zone_id", labels={"laeq_db": "LAeq (dB)", x: "Date/Heure"}, height=450)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Résumé statistique")
stats = df.loc[m].groupby('zone_id')['laeq_db'].agg(['mean','max','min','count']).round(2).reset_index()
st.dataframe(stats, use_container_width=True)
