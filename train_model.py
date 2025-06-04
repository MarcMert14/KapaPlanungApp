import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import joblib

# Excel-Datei einlesen
excel_path = 'KI_Zeitprognose_Vorlage_Projekt-W.xlsx'
df = pd.read_excel(excel_path)

# --- HIER ggf. die Spaltennamen anpassen! ---
# Beispielhafte Spaltennamen (bitte ggf. anpassen):
# 'Produkttyp', 'Fläche_m2', 'Seitenverkleidung', 'Dachtyp', 'Anzahl_Gewerke', 'Zeichnungszeit', 'Stücklistenzeit'

# Spaltennamen aus der Excel-Datei
produkttyp_col = 'Produkttyp'
flaeche_col = 'Fläche_m2'
seitenverkleidung_col = 'Seitenverkleidung'
dachtyp_col = 'Dachtyp'
anzahl_gewerke_col = 'Anzahl_Gewerke'
zeichnungszeit_col = 'Zeichnungszeit'
stuecklistenzeit_col = 'Stücklistenzeit'

X = df[[produkttyp_col, flaeche_col, seitenverkleidung_col, dachtyp_col, anzahl_gewerke_col]]
y = df[[zeichnungszeit_col, stuecklistenzeit_col]]

# Kategorische Features definieren
categorical_features = [produkttyp_col, seitenverkleidung_col, dachtyp_col]
numerical_features = [flaeche_col, anzahl_gewerke_col]

# Preprocessing-Pipeline erstellen
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', 'passthrough', numerical_features)
    ])

# Modell-Pipeline erstellen
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Modell trainieren
model.fit(X, y)

# Modell speichern
joblib.dump(model, 'ki_zeitprognose_model.joblib')

print("Modell wurde erfolgreich mit Excel-Daten trainiert und gespeichert!") 