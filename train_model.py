import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Pfad zur Excel-Datei
excel_path = 'KI_Zeitprognose_Vorlage_Projekt-W.xlsx'

# Spaltennamen in der Excel-Datei (Groß- und Kleinschreibung sowie Leerzeichen beachten)
projekt_id_col = 'Projekt-ID'
zeichnungszeit_col = 'Zeichnungszeit'
stuecklistenzeit_col = 'Stücklistenzeit'
systemanzahl_col = 'Systemanzahl'

# Präfixe für System-Spalten in der Excel-Datei
produkttyp_prefix_excel = 'Produkttyp '
anzahl_prefix_excel = 'Anzahl '
dachtyp_prefix_excel = 'Dachtyp '
seitenverkleidung_prefix_excel = 'Seitenverkleidung '
groesse_prefix_excel = 'Größe '
gesamtwert_prefix_excel = 'Gesamtwert '
pv_integration_prefix_excel = 'Photovoltaikintegration '
besonderheit_prefix_excel = 'Besonderheit '

# Namen der Features, die das Modell erwartet
produkttyp_feature = 'Produkttyp'
anahl_gewerke_feature = 'Anzahl_Gewerke' # Modell erwartet diesen Namen
dachtyp_feature = 'Dachtyp'
seitenverkleidung_feature = 'Seitenverkleidung'
flaeche_feature = 'Fläche_m2' # Modell erwartet diesen Namen

# Maximale Anzahl von Systemen, die wir erwarten (basierend auf Ihrer Struktur)
max_systems = 4 # Annahme basierend auf Ihrer Tabellenstruktur, anpassen falls nötig

# Leere Listen für das flache Dataset
flat_data = []

# Excel-Datei laden
df = pd.read_excel(excel_path, header=1) # Angabe, dass Kopfzeile in Zeile 2 (Index 1) ist

# Sicherstellen, dass alle erwarteten Basis-Spalten im flachen Dataset vorhanden sein werden
# Initialisiere ein leeres DataFrame mit den erwarteten Spalten, falls keine Daten extrahiert werden
column_names_flat = [
    produkttyp_feature,
    anahl_gewerke_feature,
    dachtyp_feature,
    seitenverkleidung_feature,
    flaeche_feature,
    'Zeichnungszeit',
    'Stücklistenzeit'
]
df_flat = pd.DataFrame(columns=column_names_flat)


# Daten zeilenweise verarbeiten und flaches Dataset erstellen
for index, row in df.iterrows():
    try:
        projekt_id = row[projekt_id_col]
        gesamt_zeichnungszeit = row[zeichnungszeit_col]
        gesamt_stuecklistenzeit = row[stuecklistenzeit_col]
        systemanzahl = int(row[systemanzahl_col]) # Systemanzahl als Integer
    except KeyError as e:
        print(f"Warnung: Erforderliche Spalte fehlt in Zeile {index}: {e}. Zeile übersprungen.")
        continue
    except ValueError:
        print(f"Warnung: Systemanzahl in Zeile {index} ist keine gültige Zahl. Zeile übersprungen.")
        continue

    # Annahme: Gesamtzeit wird gleichmäßig auf Systeme aufgeteilt
    # Dies ist eine Vereinfachung! Eine komplexere Aufteilung wäre realistischer, aber aufwendiger.
    zeit_pro_system_zeichnung = gesamt_zeichnungszeit / systemanzahl if systemanzahl > 0 else 0
    zeit_pro_system_stueckliste = gesamt_stuecklistenzeit / systemanzahl if systemanzahl > 0 else 0

    for i in range(1, systemanzahl + 1):
        # System-spezifische Spaltennamen in der Excel-Datei erstellen
        produkttyp_col_excel = f'{produkttyp_prefix_excel}{i}'
        anzahl_col_excel = f'{anzahl_prefix_excel}{i}'
        dachtyp_col_excel = f'{dachtyp_prefix_excel}{i}'
        seitenverkleidung_col_excel = f'{seitenverkleidung_prefix_excel}{i}'
        groesse_col_excel = f'{groesse_prefix_excel}{i}'

        # Daten für das aktuelle System extrahieren
        # Prüfen, ob die Spalten in der aktuellen Zeile existieren und nicht leer sind
        if (produkttyp_col_excel in row and pd.notna(row[produkttyp_col_excel]) and
                anzahl_col_excel in row and pd.notna(row[anzahl_col_excel]) and
                dachtyp_col_excel in row and pd.notna(row[dachtyp_col_excel]) and
                seitenverkleidung_col_excel in row and pd.notna(row[seitenverkleidung_col_excel]) and
                groesse_col_excel in row and pd.notna(row[groesse_col_excel])):

            system_data = {
                produkttyp_feature: row[produkttyp_col_excel],
                anahl_gewerke_feature: row[anzahl_col_excel], # Daten aus 'Anzahl X' Excel-Spalte, gespeichert als 'Anzahl_Gewerke'
                dachtyp_feature: row[dachtyp_col_excel],
                seitenverkleidung_feature: row[seitenverkleidung_col_excel],
                flaeche_feature: row[groesse_col_excel],
                'Zeichnungszeit': zeit_pro_system_zeichnung,
                'Stücklistenzeit': zeit_pro_system_stueckliste
            }
            flat_data.append(system_data)
        else:
             # Optional: Warnung ausgeben, wenn Systemdaten unvollständig sind
             # print(f"Warnung: Daten für System {i} in Projekt {projekt_id} unvollständig oder fehlen. System wird für Training übersprungen.")
             pass # Kein Fehler, einfach überspringen, wenn Daten unvollständig sind

# Flaches Dataset in DataFrame umwandeln und an das initiale df_flat anhängen
df_flat = pd.concat([df_flat, pd.DataFrame(flat_data)], ignore_index=True)

# Sicherstellen, dass numerische Spalten korrekt typisiert sind (könnten durch pd.concat strings werden)
df_flat[flaeche_feature] = pd.to_numeric(df_flat[flaeche_feature], errors='coerce')
df_flat[anahl_gewerke_feature] = pd.to_numeric(df_flat[anahl_gewerke_feature], errors='coerce')
df_flat['Zeichnungszeit'] = pd.to_numeric(df_flat['Zeichnungszeit'], errors='coerce')
df_flat['Stücklistenzeit'] = pd.to_numeric(df_flat['Stücklistenzeit'], errors='coerce')

# Zeilen mit fehlenden Werten in den Features oder Zielvariablen entfernen
df_flat.dropna(subset=column_names_flat, inplace=True)

# Überprüfen, ob nach der Bereinigung noch Daten übrig sind
if df_flat.empty:
    print("Fehler: Keine gültigen Daten zum Trainieren des Modells gefunden. Bitte überprüfen Sie die Excel-Datei und die Spaltennamen.")
else:
    # Features (X) und Zielvariablen (y) definieren
    X = df_flat[[produkttyp_feature, flaeche_feature, seitenverkleidung_feature, dachtyp_feature, anahl_gewerke_feature]]
    y = df_flat[['Zeichnungszeit', 'Stücklistenzeit']]

    # Kategorische Features in numerische umwandeln (One-Hot Encoding)
    X = pd.get_dummies(X, columns=[produkttyp_feature, seitenverkleidung_feature, dachtyp_feature], drop_first=True)

    # Daten in Trainings- und Testsets aufteilen (optional, aber empfohlen)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Trainiere das Modell
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Optional: Modell auf Testdaten evaluieren
    # score = model.score(X_test, y_test)
    # print(f"Modell-Score auf Testdaten: {score}")

    # Speichere das trainierte Modell
    joblib.dump(model, "ki_zeitprognose_model.joblib")

    print("Modell erfolgreich trainiert und gespeichert.") 