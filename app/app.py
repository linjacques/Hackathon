import streamlit as st

Page_accueil = st.Page(
    page="page/accueil.py",
    title="accueil",
    default=True
)

Page_Bruit = st.Page("page/exploration_bruit.py", "Analyse Bruit")
Page_Vols = st.Page("page/exploration_vols.py", "Analyse Vols")
Page_Carte = st.Page("page/carte_bruit.py", "Carte Folium")
Page_Top = st.Page("page/top_origins.py", "Top Pays Origine")


NavBarr = st.navigation(pages=[
    Page_accueil, Page_Bruit, Page_Vols, Page_Carte, Page_Top
])

NavBarr.run()
