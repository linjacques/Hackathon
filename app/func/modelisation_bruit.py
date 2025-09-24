import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from geopy.distance import geodesic
import joblib
import json

# Chargement des données
bruit = pd.read_csv("../Data/bruit_survol.csv")
opensky = pd.read_csv("../Data/opensky_snapshot.csv")
flights = pd.read_csv("../Data/flights_history.csv")

bruit["timestamp_iso"] = pd.to_datetime(bruit["timestamp_iso"])
opensky["time_position_unix"] = pd.to_datetime(opensky["time_position_unix"], unit="s")
flights["first_seen_iso"] = pd.to_datetime(flights["first_seen_iso"])
flights["last_seen_iso"] = pd.to_datetime(flights["last_seen_iso"])

bruit.drop_duplicates(inplace=True)
opensky.drop_duplicates(inplace=True)
flights.drop_duplicates(inplace=True)

# Feature engineering : temps
bruit['hour'] = bruit['timestamp_iso'].dt.hour
bruit['day_of_week'] = bruit['timestamp_iso'].dt.dayofweek
bruit['is_night'] = bruit['hour'].between(22, 5)
bruit['is_rush_hour'] = bruit['hour'].isin([7,8,9,17,18,19])
bruit['is_weekend'] = bruit['day_of_week'].isin([5,6])

# Fonction de calcul de distance
def calculate_distance(row, station_lat, station_lon):
    return geodesic((station_lat, station_lon), (row['latitude'], row['longitude'])).kilometers

def find_nearby_aircraft(noise_row, opensky_data, max_time_diff_minutes=5, max_distance_km=20):
    time_window = pd.Timedelta(minutes=max_time_diff_minutes)
    time_mask = (opensky_data['time_position_unix'] >= noise_row['timestamp_iso'] - time_window) & \
                (opensky_data['time_position_unix'] <= noise_row['timestamp_iso'] + time_window)
    nearby = opensky_data[time_mask].copy()
    if not nearby.empty:
        nearby['distance_to_station'] = nearby.apply(
            lambda x: calculate_distance(x, noise_row['latitude'], noise_row['longitude']), axis=1)
        nearby = nearby[nearby['distance_to_station'] <= max_distance_km]
        if not nearby.empty:
            return {
                'num_aircraft': len(nearby),
                'avg_altitude': nearby['baro_altitude_m'].mean(),
                'max_altitude': nearby['baro_altitude_m'].max(),
                'min_altitude': nearby['baro_altitude_m'].min(),
                'avg_velocity': nearby['velocity_m_s'].mean(),
                'max_velocity': nearby['velocity_m_s'].max(),
                'min_distance': nearby['distance_to_station'].min(),
                'avg_distance': nearby['distance_to_station'].mean(),
                'num_close_aircraft': (nearby['distance_to_station'] <= 5).sum()
            }
    return dict.fromkeys([
        'num_aircraft', 'avg_altitude', 'max_altitude', 'min_altitude',
        'avg_velocity', 'max_velocity', 'min_distance', 'avg_distance', 'num_close_aircraft'
    ], 0)

# Enrichissement des données
features_list = [find_nearby_aircraft(row, opensky) for _, row in bruit.iterrows()]
aircraft_df = pd.DataFrame(features_list)
bruit_enriched = pd.concat([bruit.reset_index(drop=True), aircraft_df], axis=1)

# Encodage et interaction
le = LabelEncoder()
bruit_enriched['station_id_encoded'] = le.fit_transform(bruit_enriched['station_id'])
bruit_enriched['airport_encoded'] = le.fit_transform(bruit_enriched['airport'])
bruit_enriched['altitude_distance_ratio'] = bruit_enriched['avg_altitude'] / (bruit_enriched['avg_distance'] + 1)
bruit_enriched['velocity_distance_ratio'] = bruit_enriched['avg_velocity'] / (bruit_enriched['avg_distance'] + 1)

# Features
feature_columns = [
    'hour', 'day_of_week', 'is_night', 'is_rush_hour', 'is_weekend',
    'num_aircraft', 'num_close_aircraft', 'avg_altitude', 'max_altitude',
    'min_altitude', 'avg_velocity', 'max_velocity', 'min_distance',
    'avg_distance', 'altitude_distance_ratio', 'velocity_distance_ratio',
    'station_id_encoded', 'airport_encoded', 'latitude', 'longitude'
]
target_column = 'LAeq_dB'

X = bruit_enriched[feature_columns]
y = bruit_enriched[target_column]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Grille d'hyperparamètres
rf_params = {
    'n_estimators': [100],
    'max_depth': [10],
    'min_samples_split': [10],
    'min_samples_leaf': [4]
}
gb_params = {
    'n_estimators': [100],
    'learning_rate': [0.01],
    'max_depth': [3],
    'min_samples_split': [10]
}

rf_grid = GridSearchCV(RandomForestRegressor(random_state=42), rf_params, cv=5, scoring='neg_root_mean_squared_error')
gb_grid = GridSearchCV(GradientBoostingRegressor(random_state=42), gb_params, cv=5, scoring='neg_root_mean_squared_error')

rf_grid.fit(X_train_scaled, y_train)
gb_grid.fit(X_train_scaled, y_train)

rf_model = rf_grid.best_estimator_
gb_model = gb_grid.best_estimator_
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

rf_pred = rf_model.predict(X_test_scaled)
gb_pred = gb_model.predict(X_test_scaled)
lr_pred = lr_model.predict(X_test_scaled)

# Évaluation
models = {
    "Random Forest": rf_pred,
    "Gradient Boosting": gb_pred,
    "Linear Regression": lr_pred
}

for name, preds in models.items():
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = np.mean(np.abs(y_test - preds))
    r2 = r2_score(y_test, preds)
    print(f"\n{name}")
    print(f"RMSE: {rmse:.2f} dB")
    print(f"MAE: {mae:.2f} dB")
    print(f"R2: {r2:.3f}")
    plt.figure(figsize=(10,6))
    plt.scatter(y_test, preds, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.title(f"{name} - Prédictions vs Réalité")
    plt.xlabel("Réel")
    plt.ylabel("Prédit")
    plt.tight_layout()
    plt.show()

    errors = preds - y_test
    plt.figure(figsize=(10,6))
    sns.histplot(errors, bins=30)
    plt.title(f"Distribution des erreurs - {name}")
    plt.axvline(x=0, color='r', linestyle='--')
    plt.tight_layout()
    plt.show()

# Importance des features
feat_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance_rf': rf_model.feature_importances_,
    'importance_gb': gb_model.feature_importances_
}).sort_values('importance_rf', ascending=False)

plt.figure(figsize=(12,6))
x = np.arange(len(feature_columns))
width = 0.35
plt.bar(x - width/2, feat_importance['importance_rf'], width, label='RF')
plt.bar(x + width/2, feat_importance['importance_gb'], width, label='GB')
plt.xticks(x, feat_importance['feature'], rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.title("Importance des caractéristiques")
plt.show()

# Sauvegarde
best_model_name = min(models, key=lambda m: mean_squared_error(y_test, models[m]))
best_model = {
    "Random Forest": rf_model,
    "Gradient Boosting": gb_model,
    "Linear Regression": lr_model
}[best_model_name]

joblib.dump(best_model, f"models/best_noise_prediction_model_{best_model_name.replace(' ', '_').lower()}.joblib")
joblib.dump(scaler, "models/feature_scaler.joblib")

metadata = {
    'feature_columns': feature_columns,
    'target_column': target_column,
    'best_model': best_model_name,
    'rmse': float(np.sqrt(mean_squared_error(y_test, models[best_model_name]))),
    'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
}


print(f"\nModèle sauvegardé : {best_model_name} avec RMSE = {metadata['rmse']:.2f} dB")
