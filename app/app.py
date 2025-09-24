
import os
import streamlit as st
from utils import load_data, train_or_load_model

st.set_page_config(page_title="Ciel Tranquille", page_icon="✈️", layout="wide")

st.title("✈️ Ciel Tranquille — Bruit aérien urbain")
st.markdown(
    """
    Bienvenue dans le **dashboard** *Ciel Tranquille*.
    - Visualisez l'historique des niveaux sonores par zone
    - Repérez sur une carte les **zones les plus bruyantes**
    - **Prévisualisez** le niveau probable pour un lieu et une période (avec incertitude)
    - Importez vos données et ré-entraînez le modèle de prédiction

    *Astuce* : utilisez le menu **Pages** à gauche pour naviguer.
    """
)

with st.sidebar:
    st.header("Données chargées")
    data_path = st.session_state.get("data_path", os.path.join("data", "sample_noise.csv"))
    st.caption(f"Fichier : `{os.path.basename(data_path)}`")
    df = load_data(data_path)
    st.metric("Observations", len(df))
    st.metric("Zones", df['zone_id'].nunique())

    st.caption("Le modèle s'entraîne automatiquement au besoin.")
    model, metrics = train_or_load_model(df)
    st.metric("R² (test)", f"{metrics.get('r2', 0):.2f}")
    st.metric("MAE (dB)", f"{metrics.get('mae', 0):.2f}")

st.subheader("Pour commencer")
st.write("➡️ Ouvrez la page **📈 Historique par zone** ou **🗺️ Carte des zones bruyantes** dans le menu à gauche.")
st.info("Vous pouvez téléverser votre propre CSV via **⚙️ Données & paramètres**.")
