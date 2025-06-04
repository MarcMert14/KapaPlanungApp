import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import joblib

# Lade die Excel-Datei
print("Lade Excel-Datei...")
df = pd.read_excel('KI_Zeitprognose_Vorlage_Projekt-W.xlsx', header=1)

# Debug: Zeige Spaltennamen
print("\nSpaltennamen in der Excel-Datei:")
print(df.columns.tolist())

# Debug: Zeige erste Zeile
print("\nErste Zeile der Excel-Datei:")
print(df.iloc[0])

# Extrahiere die Gesamtzeiten pro Projekt
print("\nExtrahiere Gesamtzeiten pro Projekt...")
X = []
y = []

for index, row in df.iterrows():
    try:
        # Lese Basisinformationen der Projektzeile
        systemanzahl = int(row.get('Systemanzahl', 0))
        # Korrigierte Spaltennamen
        gesamt_zeichnungszeit = row.get('Zeichnungszeit')
        gesamt_stuecklistenzeit = row.get('Stücklistenzeit')

        print(f"\nVerarbeite Zeile {index}:")
        print(f"Systemanzahl: {systemanzahl}")
        print(f"Zeichnungszeit: {gesamt_zeichnungszeit}")
        print(f"Stücklistenzeit: {gesamt_stuecklistenzeit}")

        # Ignoriere Zeilen mit unzureichenden Daten
        if systemanzahl <= 0 or pd.isna(gesamt_zeichnungszeit) or pd.isna(gesamt_stuecklistenzeit):
            print("Zeile übersprungen: Unzureichende Daten")
            continue

        # Sammle alle Systeme aus der Excel-Zeile
        systems = []
        for sys_idx in range(1, systemanzahl + 1):
            produkttyp = row.get(f'Produkttyp {sys_idx}')
            anzahl = row.get(f'Anzahl {sys_idx}')
            dachtyp = row.get(f'Dachtyp {sys_idx}')
            seitenverkleidung = row.get(f'Seitenverkleidung {sys_idx}')
            groesse = row.get(f'Größe {sys_idx}')

            print(f"\nSystem {sys_idx}:")
            print(f"Produkttyp: {produkttyp}")
            print(f"Anzahl: {anzahl}")
            print(f"Dachtyp: {dachtyp}")
            print(f"Seitenverkleidung: {seitenverkleidung}")
            print(f"Größe: {groesse}")

            if (pd.notna(produkttyp) and pd.notna(anzahl) and 
                pd.notna(dachtyp) and pd.notna(seitenverkleidung) and 
                pd.notna(groesse)):
                try:
                    systems.append({
                        'Produkttyp': produkttyp,
                        'Größe': float(groesse),
                        'Seitenverkleidung': seitenverkleidung,
                        'Dachtyp': dachtyp,
                        'Anzahl': int(anzahl)
                    })
                except (ValueError, TypeError) as e:
                    print(f"Fehler bei Konvertierung: {e}")
                    continue

        if len(systems) == systemanzahl:
            # Erstelle Features für das gesamte Projekt
            features = {
                "Anzahl_Systeme": len(systems),
                "Gesamtflaeche": sum(sys['Größe'] for sys in systems),
                "Durchschnittliche_Systemgroesse": sum(sys['Größe'] for sys in systems) / len(systems),
                "Gesamt_Anzahl_Gewerke": sum(sys['Anzahl'] for sys in systems),
                "Produkttyp_Carport": sum(1 for sys in systems if sys['Produkttyp'] == 'Carport'),
                "Produkttyp_Fahrradüberdachung": sum(1 for sys in systems if sys['Produkttyp'] == 'Fahrradüberdachung'),
                "Produkttyp_Mülleinhausung": sum(1 for sys in systems if sys['Produkttyp'] == 'Mülleinhausung'),
                "Produkttyp_Pergola": sum(1 for sys in systems if sys['Produkttyp'] == 'Pergola'),
                "Produkttyp_Mülltonnenbox": sum(1 for sys in systems if sys['Produkttyp'] == 'Mülltonnenbox'),
                "Seitenverkleidung_Gittermatte": sum(1 for sys in systems if sys['Seitenverkleidung'] == 'Gittermatte'),
                "Seitenverkleidung_Ohne": sum(1 for sys in systems if sys['Seitenverkleidung'] == 'Ohne'),
                "Seitenverkleidung_Stahl-Lochblech": sum(1 for sys in systems if sys['Seitenverkleidung'] == 'Stahl-Lochblech'),
                "Seitenverkleidung_Stahl-Vollblech": sum(1 for sys in systems if sys['Seitenverkleidung'] == 'Stahl-Vollblech'),
                "Seitenverkleidung_Trespa": sum(1 for sys in systems if sys['Seitenverkleidung'] == 'Trespa'),
                "Seitenverkleidung_WL": sum(1 for sys in systems if sys['Seitenverkleidung'] == 'WL'),
                "Seitenverkleidung_WL+LBK": sum(1 for sys in systems if sys['Seitenverkleidung'] == 'WL+LBK'),
                "Dachtyp_Gründach": sum(1 for sys in systems if sys['Dachtyp'] == 'Gründach'),
                "Dachtyp_Gründach-Light": sum(1 for sys in systems if sys['Dachtyp'] == 'Gründach-Light'),
                "Dachtyp_Ohne": sum(1 for sys in systems if sys['Dachtyp'] == 'Ohne'),
                "Dachtyp_Polycarbonat": sum(1 for sys in systems if sys['Dachtyp'] == 'Polycarbonat'),
                "Dachtyp_Trapezblech": sum(1 for sys in systems if sys['Dachtyp'] == 'Trapezblech')
            }
            X.append(features)
            y.append([float(gesamt_zeichnungszeit), float(gesamt_stuecklistenzeit)])
            print("Projekt erfolgreich extrahiert")
        else:
            print(f"Zeile übersprungen: {len(systems)} Systeme gefunden, {systemanzahl} erwartet")

    except Exception as e:
        print(f"Fehler bei Zeile {index}: {e}")
        continue

# Konvertiere zu DataFrames
X_df = pd.DataFrame(X)
y_df = pd.DataFrame(y, columns=['Zeichnungszeit', 'Stuecklistenzeit'])

print(f"\nAnzahl der Projekte zum Training: {len(X_df)}")

if len(X_df) > 0:
    # Teile die Daten in Trainings- und Testsets
    X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.2, random_state=42)

    # Trainiere das Modell
    print("Trainiere Modell...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Speichere das Modell
    print("Speichere Modell...")
    joblib.dump(model, 'ki_zeitprognose_model.joblib')

    # Evaluierung
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"Trainings-Score: {train_score:.3f}")
    print(f"Test-Score: {test_score:.3f}")

    print("Fertig!")
else:
    print("Keine gültigen Projekte zum Training gefunden!") 