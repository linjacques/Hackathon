import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from branca.colormap import linear
from streamlit_folium import folium_static

st.set_page_config(page_title="Analyse Bruit & Vols", layout="wide")
st.title(" Analyse exploratoire - Bruit et Trafic Aérien")

# mise en cache des données pour les perf
@st.cache_data
def load_data():
    bruit = pd.read_csv("../data/bruit_survol.csv")
    opensky = pd.read_csv("../data/opensky_snapshot.csv")
    flights = pd.read_csv("../data/flights_history.csv")

    bruit["timestamp_iso"] = pd.to_datetime(bruit["timestamp_iso"])
    opensky["time_position_unix"] = pd.to_datetime(opensky["time_position_unix"], unit="s")
    flights["first_seen_iso"] = pd.to_datetime(flights["first_seen_iso"])
    flights["last_seen_iso"] = pd.to_datetime(flights["last_seen_iso"])

    return bruit.drop_duplicates(), opensky.drop_duplicates(), flights.drop_duplicates()

bruit, opensky, flights = load_data()

# Stat descriptives
st.subheader(" Statistiques descriptives")

with st.expander("Statistiques Bruit"):
    st.dataframe(bruit.describe())

with st.expander("Statistiques Vols (OpenSky)"):
    st.dataframe(opensky.describe())

#Histogramme sur bruit
st.subheader(" Distribution des niveaux de bruit")
num_cols_bruit = [col for col in bruit.columns if bruit[col].dtype in ['float64', 'int64']]
col_bruit = st.selectbox("Choisir une colonne numérique à afficher :", num_cols_bruit, index=0)

fig, ax = plt.subplots(figsize=(8, 5))
n, bins, _ = ax.hist(bruit[col_bruit], bins=20, color="#2980b9", alpha=0.85, edgecolor="white")
ax.set_xlabel(col_bruit, fontsize=13)
ax.set_ylabel("Fréquence", fontsize=13)
ax.set_title(f"Distribution de {col_bruit}", fontsize=15)
ax.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
st.pyplot(fig)

# Série temporelle
st.subheader(" Évolution temporelle du bruit")
df_plot = bruit.dropna(subset=["timestamp_iso", col_bruit]).sort_values("timestamp_iso")
if not df_plot.empty:
    fig2 = plt.figure(figsize=(12, 5))
    sns.lineplot(data=df_plot, x="timestamp_iso", y=col_bruit, marker="o", color="#2980b9")
    plt.title(f"Évolution de {col_bruit} dans le temps")
    plt.xlabel("Date")
    plt.ylabel(col_bruit)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    st.pyplot(fig2)
else:
    st.warning("Aucune donnée valide pour la série temporelle.")

# Carte Folium
st.subheader(" Carte des mesures de bruit (Folium)")
lat_col = next((col for col in bruit.columns if 'lat' in col.lower()), None)
lon_col = next((col for col in bruit.columns if 'lon' in col.lower()), None)
val_col = col_bruit

if lat_col and lon_col and val_col:
    m = folium.Map(location=[bruit[lat_col].mean(), bruit[lon_col].mean()], zoom_start=10)
    colormap = linear.YlOrRd_09.scale(bruit[val_col].min(), bruit[val_col].max())
    colormap.caption = f"Niveau de bruit ({val_col})"

    for _, row in bruit.dropna(subset=[lat_col, lon_col, val_col]).iterrows():
        folium.CircleMarker(
            location=[row[lat_col], row[lon_col]],
            radius=7,
            color=colormap(row[val_col]),
            fill=True,
            fill_color=colormap(row[val_col]),
            fill_opacity=0.7,
            popup=folium.Popup(
                f"<b>Station:</b> {row.get('station_name', '')}<br>"
                f"<b>{val_col}:</b> {row[val_col]:.2f} dB<br>"
                f"<b>Date:</b> {row.get('timestamp_iso', '')}",
                max_width=250
            ),
        ).add_to(m)

    colormap.add_to(m)
    folium_static(m)
else:
    st.error("Colonnes latitude/longitude ou bruit manquantes pour la carte.")

# Top 5 pays origine (opensky)
st.subheader(" Top 5 pays d'origine des vols")
if "origin_country" in opensky.columns:
    top_origins = opensky["origin_country"].value_counts().head(5)
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    bars = ax3.bar(top_origins.index, top_origins.values, color=sns.color_palette("Blues", 5))
    ax3.set_title("Top 5 pays d'origine des vols")
    ax3.set_ylabel("Nombre de vols")
    ax3.set_xlabel("Pays")
    for bar in bars:
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(), int(bar.get_height()), 
                 ha='center', va='bottom', fontsize=10)
    st.pyplot(fig3)
else:
    st.error("Colonne 'origin_country' non trouvée dans opensky_snapshot.")
