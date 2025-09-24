
import os
import io
import pandas as pd
import streamlit as st
from utils import load_data, train_or_load_model

st.set_page_config(page_title="Données & paramètres", page_icon="⚙️", layout="wide")
st.title("⚙️ Données & paramètres")

st.subheader("Téléverser un CSV")
up = st.file_uploader("Choisissez un fichier CSV conforme au schéma attendu", type=["csv"])
if up is not None:
    # Persist file
    target_path = os.path.join("data", "user_upload.csv")
    with open(target_path, "wb") as f:
        f.write(up.getvalue())
    st.session_state["data_path"] = target_path
    st.success(f"Fichier chargé : {target_path}. Les caches de données seront actualisés après un rechargement de page.")
    st.caption("Astuce: revenez sur l'accueil pour vérifier le nombre d'observations et la qualité du modèle.")

data_path = st.session_state.get("data_path", os.path.join("data", "sample_noise.csv"))
df = load_data(data_path)
st.subheader("Aperçu des données")
st.dataframe(df.head(200), use_container_width=True)

st.subheader("Schéma attendu")
st.markdown("""
- `zone_id` *(str)* — identifiant de la zone/capteur
- `zone_name` *(str)* — libellé lisible
- `lat`, `lon` *(float)* — coordonnées WGS84
- `timestamp` *(ISO 8601)* — horodatage local/UTC (géré automatiquement)
- `laeq_db` *(float)* — niveau sonore LAeq
- `flights_count` *(int)* — vols/heure (ou proxy)
- `heavy_aircraft_ratio` *(float 0–1)* — part de gros porteurs
- `altitude_mean_m` *(float)* — altitude moyenne (m)
""")

st.subheader("Entraînement du modèle")
if st.button("Ré-entraîner maintenant"):
    model, metrics = train_or_load_model.clear()  # clear cache
    model, metrics = train_or_load_model(df)      # retrain
    st.success(f"Modèle ré-entraîné — R²={metrics.get('r2',0):.2f}, MAE={metrics.get('mae',0):.2f} dB")
