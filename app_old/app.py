import streamlit as st

Page_accueil = st.Page(
    page="page/accueil.py",
    title="accueil",
    default=True
)


Page_ml = st.Page(
    page="page/ml.py",
    title="modélisation du bruit"
)

NavBarr = st.navigation(pages=[
    Page_accueil, Page_ml
])

NavBarr.run()
