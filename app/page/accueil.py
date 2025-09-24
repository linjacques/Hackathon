import streamlit as st

st.set_page_config(page_title="Exploration complète", layout="wide")
st.title("Résultats du script Python basé sur le notebook")

with st.spinner("Exécution du script..."):
    with open("func/exploration_script.py", "r", encoding="utf-8") as f:
        code = f.read()
        exec(code, globals())

st.success("Script exécuté avec succès.")
