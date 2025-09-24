import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.set_page_config(page_title="Modélisation du bruit des avions", layout="wide")
st.title("🧠 Prédiction du Bruit Aérien via Machine Learning")

# Vérification de l'existence du script
target_script = "func/modelisation_bruit.py"

if not os.path.exists(target_script):
    st.error(f"Le fichier {target_script} est introuvable.")
    st.stop()

with st.spinner("⏳ Entrainement des modèles en cours..."):
    try:
        with open(target_script, "r", encoding="utf-8") as f:
            code = f.read()
            exec(code, globals())
        st.success(" Modèles entraînés et évalués avec succès !")
    except Exception as e:
        st.exception(e)
        st.stop()

# Affichage des résultats finaux
if 'metadata' in globals():
    st.markdown("###  Résumé du Meilleur Modèle")
    st.json(metadata)

    st.info(f"Modèle retenu : **{metadata['best_model']}** avec un RMSE de **{metadata['rmse']:.2f} dB**")

    if os.path.exists("models/model_metadata.json"):
        with open("models/model_metadata.json") as f:
            st.download_button(
                label=" Télécharger les métadonnées JSON",
                data=f,
                file_name="model_metadata.json",
                mime="application/json"
            )
    else:
        st.warning("Fichier de métadonnées introuvable pour téléchargement.")
else:
    st.warning("Les métadonnées du modèle ne sont pas disponibles.")
