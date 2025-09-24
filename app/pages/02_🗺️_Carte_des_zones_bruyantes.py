
import os
import pandas as pd
import pydeck as pdk
import streamlit as st
from utils import load_data, aggregate_by_zone, color_for_db

st.set_page_config(page_title="Carte des zones", page_icon="🗺️", layout="wide")
st.title("🗺️ Carte des zones les plus bruyantes")

data_path = st.session_state.get("data_path", "data/sample_noise.csv")
df = load_data(data_path)

with st.sidebar:
    st.header("Période")
    min_date = df['timestamp'].min().date()
    max_date = df['timestamp'].max().date()
    start, end = st.date_input("Plage de dates", (min_date, max_date), min_value=min_date, max_value=max_date)
    if isinstance(start, tuple) or isinstance(start, list):
        start, end = start
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

agg = aggregate_by_zone(df, start_ts, end_ts)
agg['color'] = agg['laeq_mean'].apply(color_for_db)

st.subheader("Top zones (moyenne LAeq)")
st.dataframe(agg.sort_values('laeq_mean', ascending=False).head(10)[['zone_id','zone_name','laeq_mean','laeq_max','count']].round(2), use_container_width=True)

# Map advanced if MAPBOX_API_KEY is set
mapbox = os.environ.get("MAPBOX_API_KEY")
if mapbox:
    st.caption("Affichage avancé avec pydeck + Mapbox (clé détectée).")
    view_state = pdk.ViewState(
        latitude=agg['lat'].mean(),
        longitude=agg['lon'].mean(),
        zoom=11.2,
        pitch=0
    )
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=agg,
        get_position='[lon, lat]',
        get_radius=150 + (agg['laeq_mean'] - agg['laeq_mean'].min()) * 20,
        get_fill_color='color',
        pickable=True
    )
    r = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": "{zone_name}\nLAeq moy.: {laeq_mean} dB"})
    st.pydeck_chart(r, use_container_width=True)
else:
    st.caption("Affichage basique (st.map). Définissez MAPBOX_API_KEY pour un rendu avancé.")
    st.map(agg.rename(columns={"lat":"latitude","lon":"longitude"})[['latitude','longitude']], zoom=11)
