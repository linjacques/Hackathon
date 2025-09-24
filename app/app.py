import streamlit as st

Page_accueil = st.Page(
    page="page/accueil.py",
    title="accueil",
    default=True
)


NavBarr = st.navigation(pages=[
    Page_accueil
])

NavBarr.run()
