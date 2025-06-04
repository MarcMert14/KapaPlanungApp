import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Excel-Tabelle laden (für Modell-Lookups und Filterung)
excel_path = 'KI_Zeitprognose_Vorlage_Projekt-W.xlsx'
df_excel = pd.read_excel(excel_path, header=1) # Angabe, dass Kopfzeile in Zeile 2 (Index 1) ist

# Temporärer Print zur Fehlersuche: Zeige die gelesenen Spaltennamen
# print("Gelesene Spaltennamen aus Excel:", df_excel.columns.tolist())

# Modell laden (trainiert auf allen Daten)
model = joblib.load("ki_zeitprognose_model.joblib")

# Spaltennamen in der Excel-Datei für den Lookup (müssen exakt übereinstimmen)
EXCEL_PROJEKT_ID = 'Projekt-ID'
EXCEL_ZEICHNUNGSZEIT = 'Zeichnungszeit'
EXCEL_STUECKLISTENZEIT = 'Stücklistenzeit'
EXCEL_SYSTEMANZAHL = 'Systemanzahl'

# Präfixe für System-Spalten in der Excel-Datei (mit führendem Leerzeichen, wie im Screenshot)
EXCEL_PRODUKTTYP_PREFIX = 'Produkttyp '
EXCEL_ANZAHL_PREFIX = 'Anzahl '
EXCEL_DACHTYP_PREFIX = 'Dachtyp '
EXCEL_SEITENVERKLEIDUNG_PREFIX = 'Seitenverkleidung '
EXCEL_GROESSE_PREFIX = 'Größe '
# EXCEL_GESAMTWERT_PREFIX = 'Gesamtwert '
# EXCEL_PV_INTEGRATION_PREFIX = 'Photovoltaikintegration '
# EXCEL_BESONDERHEIT_PREFIX = 'Besonderheit '

# Namen der Features, die das Modell erwartet (aus train_model.py)
# Dies ist eine manuelle Liste basierend auf dem Training mit drop_first=True
# Eine robustere Lösung würde diese Liste aus dem gespeicherten Modell/Encoder laden.
MODEL_FEATURE_NAMES = np.array([
    'Fläche_m2',
    'Anzahl_Gewerke',
    'Produkttyp_Carport', 
    'Produkttyp_Fahrradüberdachung',
    'Produkttyp_Mülleinhausung',
    'Produkttyp_Pergola',
    'Produkttyp_Mülltonnenbox', 
    'Seitenverkleidung_Gittermatte',
    'Seitenverkleidung_Ohne',
    'Seitenverkleidung_Stahl-Lochblech',
    'Seitenverkleidung_Stahl-Vollblech',
    'Seitenverkleidung_Trespa',
    'Seitenverkleidung_WL',
    'Seitenverkleidung_WL+LBK',
    'Dachtyp_Gründach',
    'Dachtyp_Gründach-Light',
    'Dachtyp_Ohne',
    'Dachtyp_Polycarbonat',
    'Dachtyp_Trapezblech'
])

# --- Funktion zur Schätzung der Zeiten (wird von beiden Optionen genutzt) ---
def estimate_times(produkttyp_list, größe_list, seitenverkleidung_list, dachtyp_list, anzahl_gewerke_list, tortyp_list, pv_integration_list, gesamtwert_gesamt, besonderheit_gesamt, mitarbeiter_filter, df_excel, model):
    
    gesamt_zeichnungszeit_h = 0
    gesamt_stuecklistenzeit_h = 0
    # Quelle kann gemischt sein, präferieren 'Excel' wenn mindestens ein System gefunden wurde
    quelle = "KI-Modell"
    excel_match_found = False

    # Temporärer Print zur Fehlersuche (kann später entfernt werden)
    # print("estimate_times aufgerufen mit:", produkttyp_list, größe_list, seitenverkleidung_list, dachtyp_list, anzahl_gewerke_list)

    if not produkttyp_list:
        return (0, 0), "Kein System zur Schätzung gefunden"

    # Für jedes System im Projekt (aus der App-Eingabe) die Zeiten schätzen
    for i in range(len(produkttyp_list)):
        current_produkttyp = produkttyp_list[i]
        current_größe = größe_list[i]
        current_seitenverkleidung = seitenverkleidung_list[i]
        current_dachtyp = dachtyp_list[i]
        current_anzahl_gewerke = anzahl_gewerke_list[i]

        # print(f"\nVerarbeite System {i+1}: Produkttyp={current_produkttyp}, Größe={current_größe}, Seitenverkleidung={current_seitenverkleidung}, Dachtyp={current_dachtyp}, Anzahl={current_anzahl_gewerke}")

        system_zeichnungszeit_h = 0
        system_stuecklistenzeit_h = 0
        system_quelle = "KI-Modell"

        # 1. Versuch: Excel-Lookup für das aktuelle System
        found_in_excel = False
        # Iteriere durch jede Zeile in der Excel-Datei
        for index, row in df_excel.iterrows():
            try:
                # Lese Basisinformationen der Projektzeile
                excel_systemanzahl = int(row.get(EXCEL_SYSTEMANZAHL, 0)) # Nutze .get() für robustheit
                excel_gesamt_zeichnungszeit = row.get(EXCEL_ZEICHNUNGSZEIT)
                excel_gesamt_stuecklistenzeit = row.get(EXCEL_STUECKLISTENZEIT)

                # Ignoriere Zeilen mit unzureichenden Daten
                if excel_systemanzahl <= 0 or pd.isna(excel_gesamt_zeichnungszeit) or pd.isna(excel_gesamt_stuecklistenzeit):
                    continue

            except (KeyError, ValueError, TypeError) as e:
                # Handle Fehler beim Lesen der Basisspalten in dieser Zeile, überspringe sie
                # print(f"Warnung: Problem beim Lesen der Basisspalten in Excel-Zeile {index}: {e}. Zeile übersprungen.")
                continue
            except Exception as e:
                 #print(f"Unerwarteter Fehler beim Lesen der Excel-Zeile {index}: {e}. Zeile übersprungen.")
                 continue # Alle anderen Fehler beim Lesen überspringen


            # Iteriere durch die Systeme in der aktuellen Excel-Zeile (basierend auf excel_systemanzahl)
            for sys_idx in range(1, excel_systemanzahl + 1):
                # System-spezifische Spaltennamen in der Excel-Datei erstellen
                excel_produkttyp_col = f'{EXCEL_PRODUKTTYP_PREFIX}{sys_idx}'
                excel_anzahl_col = f'{EXCEL_ANZAHL_PREFIX}{sys_idx}'
                excel_dachtyp_col = f'{EXCEL_DACHTYP_PREFIX}{sys_idx}'
                excel_seitenverkleidung_col = f'{EXCEL_SEITENVERKLEIDUNG_PREFIX}{sys_idx}'
                excel_groesse_col = f'{EXCEL_GROESSE_PREFIX}{sys_idx}'

                # Daten für das aktuelle System aus der Excel-Zeile extrahieren (nutze .get() und prüfe auf NaN)
                excel_produkttyp = row.get(excel_produkttyp_col)
                excel_anzahl_raw = row.get(excel_anzahl_col)
                excel_dachtyp = row.get(excel_dachtyp_col)
                excel_seitenverkleidung = row.get(excel_seitenverkleidung_col)
                excel_groesse_raw = row.get(excel_groesse_col)

                # Prüfen, ob die Systemdaten in der Excel-Zeile vollständig und nicht leer sind
                if (pd.notna(excel_produkttyp) and pd.notna(excel_anzahl_raw) and pd.notna(excel_dachtyp) and
                        pd.notna(excel_seitenverkleidung) and pd.notna(excel_groesse_raw)):

                    # Versuche, die Anzahl und Größe in numerische Typen umzuwandeln, um Vergleiche zu ermöglichen
                    try:
                        excel_anzahl = int(excel_anzahl_raw)
                        excel_groesse = float(excel_groesse_raw)
                    except (ValueError, TypeError):
                        # Wenn Umwandlung fehlschlägt, kann es keine exakte Übereinstimmung geben, überspringe dieses System in dieser Zeile
                        continue

                    # Exakte Übereinstimmung prüfen (mit den umgewandelten Werten)
                    # Stelle sicher, dass current_größe und current_anzahl_gewerke ebenfalls numerisch sind
                    try:
                        current_größe_float = float(current_größe)
                        current_anzahl_gewerke_int = int(current_anzahl_gewerke)
                    except (ValueError, TypeError):
                         # Wenn die Eingabewerte nicht numerisch sind, kann keine Übereinstimmung gefunden werden
                         # print("Warnung: Eingabewerte für Größe oder Anzahl nicht numerisch.")
                         continue # Gehe zum nächsten System in dieser Excel-Zeile


                    if (excel_produkttyp == current_produkttyp and
                            excel_anzahl == current_anzahl_gewerke_int and
                            excel_dachtyp == current_dachtyp and
                            excel_seitenverkleidung == current_seitenverkleidung and
                            excel_groesse == current_größe_float):

                        # print(f"  Exakter Match gefunden in Excel-Zeile {index} für System {sys_idx}!")
                        # print(f"  Excel Gesamtzeiten: Zeichnung={excel_gesamt_zeichnungszeit}, Stückliste={excel_gesamt_stuecklistenzeit}, Systemanzahl={excel_systemanzahl}")

                        # Exakte Übereinstimmung gefunden! Zeit pro System aus Excel berechnen
                        # Sicherstellen, dass die Gesamtzeiten numerisch sind, bevor geteilt wird
                        try:
                            system_zeichnungszeit_h = float(excel_gesamt_zeichnungszeit) / excel_systemanzahl
                            system_stuecklistenzeit_h = float(excel_gesamt_stuecklistenzeit) / excel_systemanzahl
                            system_quelle = 'Excel-Tabelle (Lookup)'
                            excel_match_found = True # Markiere, dass mindestens ein Excel-Match gefunden wurde
                            found_in_excel = True # Markiere, dass dieses spezielle System in Excel gefunden wurde
                            # print(f"  Geschätzte Zeiten aus Excel für dieses System: Zeichnung={system_zeichnungszeit_h:.2f}, Stückliste={system_stuecklistenzeit_h:.2f}")
                            break # Beende die Suche nach diesem System in der aktuellen Excel-Zeile und allen weiteren Excel-Zeilen
                        except (ValueError, TypeError):
                            # Wenn Gesamtzeiten nicht numerisch sind, kann diese Zeile nicht für die Berechnung verwendet werden
                            # print(f"Warnung: Gesamtzeiten in Excel-Zeile {index} nicht numerisch. Kann nicht für Lookup verwendet werden. Versuche KI.")
                            # Hier brechen wir nicht ab, sondern lassen den Fallback zum KI-Modell zu
                            pass # Gehe zum nächsten System in dieser Excel-Zeile oder nächsten Excel-Zeile

            # Wenn für das aktuelle System aus der App-Eingabe ein Match in der Excel-Datei gefunden wurde, stoppen wir die Suche in Excel
            if found_in_excel:
                break # Springe zur Verarbeitung des nächsten Systems aus der App-Eingabe

        # 2. Fallback zum KI-Modell, wenn keine exakte Übereinstimmung in Excel gefunden wurde (oder Excel-Daten ungültig waren)
        # Bevor wir die KI nutzen, stellen wir sicher, dass die notwendigen Spalten im Eingabe-DataFrame vorhanden sind.
        # model_feature_names sollten jetzt aus dem geladenen Modell kommen

        if not found_in_excel or (found_in_excel and system_zeichnungszeit_h == 0 and system_stuecklistenzeit_h == 0 and system_quelle == 'Excel-Tabelle (Lookup)') : # Füge Bedingung hinzu, falls Excel-Zeiten 0 waren
            if not found_in_excel:
                # print("  Kein exakter Match in Excel gefunden. Verwende KI-Modell.")
                pass # Debug print entfernt
            elif system_quelle == 'Excel-Tabelle (Lookup)' and system_zeichnungszeit_h == 0 and system_stuecklistenzeit_h == 0:
                 # print("  Exakter Match in Excel gefunden, aber Zeiten sind 0. Verwende KI-Modell.")
                 system_quelle = 'KI-Modell (Excel-Match mit 0 Zeiten)' # Aktualisiere Quelle, falls von Excel mit 0 kam
                 
            try:
                # Erstelle DataFrame für die Vorhersage des einzelnen Systems
                X_input = pd.DataFrame([{
                    "Produkttyp": current_produkttyp,
                    "Fläche_m2": current_größe,
                    "Seitenverkleidung": current_seitenverkleidung,
                    "Dachtyp": current_dachtyp,
                    "Anzahl_Gewerke": current_anzahl_gewerke
                }])

                # --- WICHTIG: Sicherstellen, dass die Eingabe-Features mit den Trainings-Features übereinstimmen! ---
                # Lade die Feature-Namen direkt aus dem trainierten Modell, wenn möglich
                try:
                    model_feature_names = model.feature_names_in_
                    #print(f"  Modell erwartete Feature-Namen (aus Modell): {model_feature_names.tolist()}")
                except AttributeError:
                    # print("Warnung: Feature-Namen können nicht aus dem Modell extrahiert werden (model.feature_names_in_ fehlt). Verwende manuelle Liste.")
                    # Fallback: Manuell definierte erwartete Spaltennamen nach One-Hot Encoding (basierend auf Training)
                    # Diese Liste MUSS mit der genauen Ausgabe von pd.get_dummies im train_model.py Skript übereinstimmen!
                    # Die Reihenfolge ist auch wichtig.
                    # Passen Sie diese Liste basierend auf Ihrem tatsächlichen Training an!
                    model_feature_names = np.array([
                        'Fläche_m2',
                        'Anzahl_Gewerke',
                        'Produkttyp_Carport', 
                        'Produkttyp_Fahrradüberdachung',
                        'Produkttyp_Mülleinhausung',
                        'Produkttyp_Pergola',
                        'Produkttyp_Mülltonnenbox', 
                        'Seitenverkleidung_Gittermatte',
                        'Seitenverkleidung_Ohne',
                        'Seitenverkleidung_Stahl-Lochblech',
                        'Seitenverkleidung_Stahl-Vollblech',
                        'Seitenverkleidung_Trespa',
                        'Seitenverkleidung_WL',
                        'Seitenverkleidung_WL+LBK',
                        'Dachtyp_Gründach',
                        'Dachtyp_Gründach-Light',
                        'Dachtyp_Ohne',
                        'Dachtyp_Polycarbonat',
                        'Dachtyp_Trapezblech'
                        # Fügen Sie hier weitere Dummy-Spalten basierend auf Ihren tatsächlichen Trainingsdaten hinzu
                    ])

                # One-Hot-Encode die Eingabe
                X_input_processed = pd.get_dummies(X_input, columns=['Produkttyp', 'Seitenverkleidung', 'Dachtyp'], drop_first=True)

                # Füge fehlende Spalten hinzu (mit Wert 0) und ordne die Spalten neu an
                X_input_reindexed = X_input_processed.reindex(columns=model_feature_names, fill_value=0)

                # Überprüfen, ob alle Spalten des Modells vorhanden sind (optional, da reindex fehlende hinzufügt)
                # if not all(col in X_input_reindexed.columns for col in model_feature_names):
                #     missing_cols = [col for col in model_feature_names if col not in X_input_reindexed.columns]
                #     print(f"Fehler: Fehlende Spalten im KI-Input nach Reindexierung: {missing_cols}")
                #     raise ValueError("Fehlende Spalten für KI-Vorhersage nach Reindexierung.")

                 # Überprüfen, ob zusätzliche Spalten im KI-Input sind (optional, da reindex keine hinzufügt)
                # if not all(col in model_feature_names for col in X_input_reindexed.columns):
                #     extra_cols = [col for col in X_input_reindexed.columns if col not in model_feature_names]
                #     print(f"Fehler: Zusätzliche Spalten im KI-Input nach Reindexierung: {extra_cols}")
                #     raise ValueError("Zusätzliche Spalten für KI-Vorhersage nach Reindexierung.")


                # print(f"  KI-Input nach Verarbeitung (erste Zeile):\n{X_input_reindexed.head(1)}")
                # print(f"  KI-Input Spalten: {X_input_reindexed.columns.tolist()}")

                prediction = model.predict(X_input_reindexed)[0]
                system_zeichnungszeit_h = prediction[0]
                system_stuecklistenzeit_h = prediction[1]
                system_quelle = 'KI-Modell'
                # print(f"  KI-Vorhersage für dieses System: Zeichnung={system_zeichnungszeit_h:.2f}, Stückliste={system_stuecklistenzeit_h:.2f}")

            except Exception as e:
                print(f"  Fehler bei KI-Vorhersage für ein System: {e}")
                system_zeichnungszeit_h = 0
                system_stuecklistenzeit_h = 0
                system_quelle = f'Fehler bei Schätzung: {e}'

        # Zeiten des aktuellen Systems zu den Gesamtergebnissen addieren
        gesamt_zeichnungszeit_h += system_zeichnungszeit_h
        gesamt_stuecklistenzeit_h += system_stuecklistenzeit_h

    # Bestimme die endgültige Quelle basierend darauf, ob mindestens ein Excel-Match gefunden wurde
    if excel_match_found:
        quelle = 'Excel-Tabelle (mind. 1 System gefunden)'
    else:
        quelle = 'KI-Modell (kein Excel-Match)'

    # print(f"\nGesamte geschätzte Zeiten für das Projekt: Zeichnung={gesamt_zeichnungszeit_h:.2f}, Stückliste={gesamt_stuecklistenzeit_h:.2f}")
    # print(f"Endgültige Quelle: {quelle}")


    return (gesamt_zeichnungszeit_h, gesamt_stuecklistenzeit_h), quelle


# Simulierte ap+ Projektdaten (OHNE Zeiten, mit detaillierten Feldern)
# Später durch tatsächlichen ap+ Zugriff ersetzen!
SIMULATED_AP_PLUS_PROJECTS = [
    {
        'Interne_Auftragsnummer': 'AUFTRAG-001',
        'Systeme': [
            {'Produkttyp': 'Carport', 'Größe': 180, 'Seitenverkleidung': 'WL+LBK', 'Dachtyp': 'Gründach', 'Anzahl_Gewerke': 4, 'Tortyp': 'Ohne', 'Photovoltaikintegration': 'ja', 'Gesamtwert': 25000, 'Besonderheit': 'Schräges Dach'}
        ],
        'Zugewiesener_Mitarbeiter': 'Mitarbeiter 1',
    },
    {
        'Interne_Auftragsnummer': 'AUFTRAG-002',
        'Systeme': [
             {'Produkttyp': 'Fahrradüberdachung', 'Größe': 40, 'Seitenverkleidung': 'Stahl-Lochblech', 'Dachtyp': 'Trapezblech', 'Anzahl_Gewerke': 2, 'Tortyp': 'N1', 'Photovoltaikintegration': 'nein', 'Gesamtwert': 9000, 'Besonderheit': 'Keine'}
        ],
        'Zugewiesener_Mitarbeiter': 'Mitarbeiter 2',
    },
    {
        'Interne_Auftragsnummer': 'AUFTRAG-003',
        'Systeme': [
            {'Produkttyp': 'Mülleinhausung', 'Größe': 90, 'Seitenverkleidung': 'Ohne', 'Dachtyp': 'Gründach-Light', 'Anzahl_Gewerke': 3, 'Tortyp': 'Ohne', 'Photovoltaikintegration': 'ja', 'Gesamtwert': 18000, 'Besonderheit': 'Fallrohr innen'},
            {'Produkttyp': 'Mülleinhausung', 'Größe': 30, 'Seitenverkleidung': 'Trespa', 'Dachtyp': 'Ohne', 'Anzahl_Gewerke': 1, 'Tortyp': 'HST1', 'Photovoltaikintegration': 'nein', 'Gesamtwert': 7000, 'Besonderheit': 'Keine'}
        ],
        'Zugewiesener_Mitarbeiter': 'Mitarbeiter 1',
    },
    {
        'Interne_Auftragsnummer': 'AUFTRAG-004',
        'Systeme': [
            {'Produkttyp': 'Pergola', 'Größe': 50, 'Seitenverkleidung': 'Ohne', 'Dachtyp': 'Ohne', 'Anzahl_Gewerke': 1, 'Tortyp': 'Ohne', 'Photovoltaikintegration': 'nein', 'Gesamtwert': 6500, 'Besonderheit': 'Fußleiste'}
        ],
        'Zugewiesener_Mitarbeiter': 'Mitarbeiter 3',
    },
    {
        'Interne_Auftragsnummer': 'AUFTRAG-005',
        'Systeme': [
            {'Produkttyp': 'Mülltonnenbox', 'Größe': 10, 'Seitenverkleidung': 'Vorhanden', 'Dachtyp': 'Trapezblech', 'Anzahl_Gewerke': 1, 'Tortyp': 'N2', 'Photovoltaikintegration': 'nein', 'Gesamtwert': 3200, 'Besonderheit': 'Keine'}
        ],
        'Zugewiesener_Mitarbeiter': 'Mitarbeiter 4',
    },
     {
        'Interne_Auftragsnummer': 'AUFTRAG-006',
        'Systeme': [
            {'Produkttyp': 'Carport', 'Größe': 100, 'Seitenverkleidung': 'Ohne', 'Dachtyp': 'Trapezblech', 'Anzahl_Gewerke': 2, 'Tortyp': 'Ohne', 'Photovoltaikintegration': 'nein', 'Gesamtwert': 20000, 'Besonderheit': 'Keine'},
            {'Produkttyp': 'Fahrradüberdachung', 'Größe': 30, 'Seitenverkleidung': 'Gittermatte', 'Dachtyp': 'Polycarbonat', 'Anzahl_Gewerke': 1, 'Tortyp': 'Ohne', 'Photovoltaikintegration': 'nein', 'Gesamtwert': 5000, 'Besonderheit': 'Keine'}
        ],
        'Zugewiesener_Mitarbeiter': 'Mitarbeiter 2',
    }

]

df_simulated_projects = pd.DataFrame(SIMULATED_AP_PLUS_PROJECTS)

# Listen für Dropdown-Optionen
PRODUKTTYPEN = ['Carport', 'Mülleinhausung', 'Fahrradüberdachung', 'Pergola', 'Mülltonnenbox']
DACHTYPEN = ['Gründach', 'Gründach-Light', 'Trapezblech', 'Polycarbonat', 'Ohne']
SEITENVERKLEIDUNGEN = ['Gittermatte', 'WL+LBK', 'WL', 'Trespa', 'Aluline', 'Stahl-Lochblech', 'Stahl-Vollblech', 'Ohne']
TORTYPEN = ['N1', 'N2', 'HST1', 'HSTD', 'Ohne']
PV_OPTIONEN = ['ja', 'nein']
BESONDERHEITEN = ['', 'Keine', 'Schräges Dach', 'Fallrohr innen', 'Fußleiste', 'Sonderwunsch', 'Sonderfarbe', 'Sonderhöhe'] # 'Keine' hinzugefügt
MITARBEITER = ['Alle', 'Mitarbeiter 1', 'Mitarbeiter 2', 'Mitarbeiter 3', 'Mitarbeiter 4', 'Mitarbeiter 5']


# --- Option 1: Abruf über Auftragsnummer ---
st.header("Projektzeiten per Auftragsnummer abrufen")

auftragsnummer = st.text_input("Interne Auftragsnummer eingeben", key="auftragsnummer_input")

if st.button("Zeiten für Auftragsnummer abrufen", key="btn_abrufen"):
    # Projekt in simulierten Daten suchen
    project_data = df_simulated_projects[df_simulated_projects['Interne_Auftragsnummer'] == auftragsnummer]

    if project_data.empty:
        st.warning(f"Kein Projekt mit Auftragsnummer {auftragsnummer} in den simulierten Daten gefunden.")
    else:
        # Projektdetails extrahieren
        projekt_details = project_data.iloc[0]
        systeme = projekt_details['Systeme']

        st.subheader(f"Details für Projekt {auftragsnummer}")
        # Anzeige der Systemdetails
        st.write("**Systeme im Projekt:**")
        for i, system in enumerate(systeme):
            st.write(f"System {i+1}:")
            st.write(f"- Produkttyp: {system.get('Produkttyp', 'N/A')}")
            st.write(f"- Größe: {system.get('Größe', 'N/A')} m²")
            st.write(f"- Seitenverkleidung: {system.get('Seitenverkleidung', 'N/A')}")
            st.write(f"- Dachtyp: {system.get('Dachtyp', 'N/A')}")
            st.write(f"- Anzahl: {system.get('Anzahl_Gewerke', 'N/A')}") # Ändere Label zu "Anzahl"
            st.write(f"- Tortyp: {system.get('Tortyp', 'N/A')}")
            st.write(f"- Photovoltaikintegration: {system.get('Photovoltaikintegration', 'N/A')}")
            st.write(f"- Gesamtwert System: {system.get('Gesamtwert', 'N/A')} Euro") # Anzeige pro System
            st.write(f"- Besonderheit System: {system.get('Besonderheit', 'N/A')}") # Anzeige pro System
        
        # st.write(f"Gesamtwert: {projekt_details.get('Gesamtwert', 'N/A')} Euro") # Gesamtwert pro Projekt entfernt
        # st.write(f"Besonderheit: {projekt_details.get('Besonderheit', 'N/A')}") # Besonderheit pro Projekt entfernt
        
        # Anzeige des in ap+ zugewiesenen Mitarbeiters (aus simulierten Daten)
        if 'Zugewiesener_Mitarbeiter' in projekt_details:
             st.write(f"Zugewiesener Mitarbeiter (simuliert in ap+): {projekt_details['Zugewiesener_Mitarbeiter']}")

        # --- Zeiten schätzen (verwendet die Logik von unten) ---
        # Sammle die Listen für die Schätzfunktion
        produkttyp_list = [s.get('Produkttyp') for s in systeme]
        größe_list = [s.get('Größe') for s in systeme]
        seitenverkleidung_list = [s.get('Seitenverkleidung') for s in systeme]
        dachtyp_list = [s.get('Dachtyp') for s in systeme]
        anzahl_gewerke_list = [s.get('Anzahl_Gewerke') for s in systeme]
        tortyp_list = [s.get('Tortyp') for s in systeme]
        pv_integration_list = [s.get('Photovoltaikintegration') for s in systeme]

        # Gesamtwert und Besonderheit hier nicht pro System übergeben, da estimate_times diese für den Lookup nicht nutzt
        estimated_times, quelle = estimate_times(
            produkttyp_list,
            größe_list,
            seitenverkleidung_list,
            dachtyp_list,
            anzahl_gewerke_list,
            tortyp_list, # Wird in estimate_times aktuell nicht verwendet
            pv_integration_list, # Wird in estimate_times aktuell nicht verwendet
            None, # Gesamtwert pro Projekt entfernt
            None, # Besonderheit pro Projekt entfernt
            'Alle',  # Mitarbeiterfilter entfernt
            df_excel,
            model
        )

        st.subheader("Geschätzte Bearbeitungszeiten")
        st.write(f"Quelle der Werte: **{quelle}**")
        st.metric("Zeichnung", f"{estimated_times[0]:.1f} Stunden")
        st.metric("Stückliste", f"{estimated_times[1]:.1f} Stunden")
        st.write("_Hinweis: Die Zeiten sind die Summe aller Systeme im Projekt._")

# --- Option 2: Manuelle Eingabe Projektdetails ---
st.header("Manuelle Eingabe Projektdetails")
st.write("Schätzen Sie die Zeiten für ein Projekt ohne Auftragsnummer.")

# Initialisiere die Anzahl der Systeme im Session State, falls noch nicht vorhanden
if 'num_systems' not in st.session_state:
    st.session_state.num_systems = 1

# Buttons zum Hinzufügen/Entfernen von Systemen (außerhalb des Formulars)
col1, col2 = st.columns(2)
with col1:
    if st.button("+ System hinzufügen", key="add_system"):
        st.session_state.num_systems += 1
        st.rerun()
with col2:
    if st.button("- System entfernen", key="remove_system", disabled=st.session_state.num_systems <= 1):
        st.session_state.num_systems -= 1
        st.rerun()

with st.form("manual_input_form"):
    st.subheader("Details zu den Systemen")
    
    manual_systeme_inputs = []
    for i in range(st.session_state.num_systems):
        st.write(f"**System {i+1}:**")
        col1, col2 = st.columns(2)
        with col1:
            manual_produkttyp = st.selectbox(f"Produkttyp System {i+1}", PRODUKTTYPEN, key=f"manual_produkttyp_{i}")
            manual_größe = st.number_input(f"Größe System {i+1} in m²", min_value=1, value=100, key=f"manual_größe_{i}")
            manual_seitenverkleidung = st.selectbox(f"Seitenverkleidung System {i+1}", SEITENVERKLEIDUNGEN, key=f"manual_seitenverkleidung_{i}")
            manual_dachtyp = st.selectbox(f"Dachtyp System {i+1}", DACHTYPEN, key=f"manual_dachtyp_{i}")
        with col2:
            manual_anzahl_gewerke = st.slider(f"Anzahl System {i+1}", min_value=1, max_value=5, value=2, key=f"manual_anzahl_gewerke_{i}") # Ändere Label zu "Anzahl"
            manual_tortyp = st.selectbox(f"Tortyp System {i+1}", TORTYPEN, key=f"manual_tortyp_{i}")
            manual_pv_integration = st.selectbox(f"Photovoltaikintegration System {i+1}", PV_OPTIONEN, key=f"manual_pv_integration_{i}")
            manual_gesamtwert = st.number_input(f"Gesamtwert System {i+1} (Euro Brutto)", min_value=0, value=10000, key=f"manual_gesamtwert_{i}")
            manual_besonderheit = st.selectbox(f"Besonderheit System {i+1}", BESONDERHEITEN, index=1, key=f"manual_besonderheit_{i}") # Standard auf 'Keine' (Index 1)

        manual_systeme_inputs.append({
            'Produkttyp': manual_produkttyp,
            'Größe': manual_größe,
            'Seitenverkleidung': manual_seitenverkleidung,
            'Dachtyp': manual_dachtyp,
            'Anzahl_Gewerke': manual_anzahl_gewerke, # Behalte 'Anzahl_Gewerke' für Konsistenz mit Excel/Modell
            'Tortyp': manual_tortyp,
            'Photovoltaikintegration': manual_pv_integration,
            'Gesamtwert': manual_gesamtwert,
            'Besonderheit': manual_besonderheit
        })

    st.subheader("Projektdetails")
    manual_konstrukteur = st.selectbox("Zuständiger Konstrukteur", MITARBEITER, index=0, key="manual_konstrukteur")

    submit_manual_button = st.form_submit_button("Zeiten schätzen (Manuell)")

    if submit_manual_button:
        # --- Zeiten schätzen (verwendet die gleiche Logik) ---
        # Sammle die Listen für die Schätzfunktion aus den manuellen Inputs
        manual_produkttyp_list = [s.get('Produkttyp') for s in manual_systeme_inputs]
        manual_größe_list = [s.get('Größe') for s in manual_systeme_inputs]
        manual_seitenverkleidung_list = [s.get('Seitenverkleidung') for s in manual_systeme_inputs]
        manual_dachtyp_list = [s.get('Dachtyp') for s in manual_systeme_inputs]
        manual_anzahl_gewerke_list = [s.get('Anzahl_Gewerke') for s in manual_systeme_inputs]
        manual_tortyp_list = [s.get('Tortyp') for s in manual_systeme_inputs]
        manual_pv_integration_list = [s.get('Photovoltaikintegration') for s in manual_systeme_inputs]
        manual_gesamtwert_list = [s.get('Gesamtwert') for s in manual_systeme_inputs]
        manual_besonderheit_list = [s.get('Besonderheit') for s in manual_systeme_inputs]

        estimated_times_manual, quelle_manual = estimate_times(
            manual_produkttyp_list,
            manual_größe_list,
            manual_seitenverkleidung_list,
            manual_dachtyp_list,
            manual_anzahl_gewerke_list,
            manual_tortyp_list, # Wird in estimate_times aktuell nicht verwendet
            manual_pv_integration_list, # Wird in estimate_times aktuell nicht verwendet
            sum(manual_gesamtwert_list),  # Wird in estimate_times aktuell nicht verwendet
            ", ".join(filter(None, manual_besonderheit_list)),  # Wird in estimate_times aktuell nicht verwendet
            'Alle',  # Mitarbeiterfilter entfernt
            df_excel,
            model
        )

        st.subheader("Geschätzte Bearbeitungszeiten (Manuell)")
        st.write(f"Quelle der Werte: **{quelle_manual}**")
        st.metric("Zeichnung", f"{estimated_times_manual[0]:.1f} Stunden")
        st.metric("Stückliste", f"{estimated_times_manual[1]:.1f} Stunden")
        st.write("_Hinweis: Die Zeiten sind die Summe aller Systeme im Projekt._")

# Optional: Anzeige der simulierten Projektdaten (hilfreich für die Demo)
st.sidebar.subheader("Simulierte ap+ Projekte Struktur")
st.sidebar.write("\n".join([f"- {p['Interne_Auftragsnummer']}" for p in SIMULATED_AP_PLUS_PROJECTS]))