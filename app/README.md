# Ciel Tranquille — Dashboard Streamlit

Tableau de bord pour **surveiller, analyser et prévoir** le bruit aérien urbain.

## Prérequis

- Python 3.10+
- `pip`

## Installation

```bash
cd ciel_tranquille_streamlit
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Lancer l'application

```bash
streamlit run app.py
```

### Données

Un jeu de données *exemple* est fourni : `data/sample_noise.csv`.  
Vous pouvez téléverser votre propre CSV depuis la page **⚙️ Données & paramètres**.  
Format attendu (colonnes) :

- `zone_id` (str)
- `zone_name` (str)
- `lat`, `lon` (float)
- `timestamp` (ISO 8601)
- `laeq_db` (float)
- `flights_count` (int)
- `heavy_aircraft_ratio` (float 0–1)
- `altitude_mean_m` (float)

## Structure

- `app.py` — page d'accueil et navigation
- `pages/01_📈_Historique_par_zone.py`
- `pages/02_🗺️_Carte_des_zones_bruyantes.py`
- `pages/03_🤖_Prévision_bruit.py`
- `pages/04_⚙️_Données_et_paramètres.py`
- `data/` — fichiers CSV
- `models/` — modèle de base (RandomForest) entraîné sur l'exemple
- `.streamlit/config.toml` — thème

## Notes

- Si vous avez une clé Mapbox (`MAPBOX_API_KEY`), la carte utilisera pydeck en mode avancé (Heatmap).  
  Sinon, un affichage de base avec `st.map` sera utilisé.
- Le modèle se réentraine automatiquement lorsque vous chargez un nouveau CSV.
- Les prévisions affichent un **intervalle d'incertitude** basé sur l'erreur absolue moyenne (MAE).
