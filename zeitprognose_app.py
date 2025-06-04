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
    
    # Quelle ist immer das KI-Modell nach Entfernung des Excel-Lookups
    quelle = "KI-Modell"
    gesamt_zeichnungszeit_h = 0
    gesamt_stuecklistenzeit_h = 0

    if not produkttyp_list:
        return (0, 0), "Kein System zur Schätzung gefunden"

    # Erstelle eine Liste aller Systeme im aktuellen Projekt
    current_systems = []
    for i in range(len(produkttyp_list)):
        try:
            current_größe_float = float(größe_list[i])
            current_anzahl_gewerke_int = int(anzahl_gewerke_list[i])
        except (ValueError, TypeError):
            continue

        current_systems.append({
            'Produkttyp': produkttyp_list[i],
            'Größe': current_größe_float,
            'Seitenverkleidung': seitenverkleidung_list[i],
            'Dachtyp': dachtyp_list[i],
            'Anzahl': current_anzahl_gewerke_int
        })

    # --- KI-Modell Schätzung für das gesamte Projekt ---
    try:
        # Erstelle DataFrame für die Vorhersage des gesamten Projekts
        X_input = pd.DataFrame([{
            "Anzahl_Systeme": len(current_systems),
            "Gesamtflaeche": sum(sys['Größe'] for sys in current_systems),
            "Durchschnittliche_Systemgroesse": sum(sys['Größe'] for sys in current_systems) / len(current_systems) if len(current_systems) > 0 else 0,
            "Gesamt_Anzahl_Gewerke": sum(sys['Anzahl'] for sys in current_systems),
            "Produkttyp_Carport": sum(1 for sys in current_systems if sys['Produkttyp'] == 'Carport'),
            "Produkttyp_Fahrradüberdachung": sum(1 for sys in current_systems if sys['Produkttyp'] == 'Fahrradüberdachung'),
            "Produkttyp_Mülleinhausung": sum(1 for sys in current_systems if sys['Produkttyp'] == 'Mülleinhausung'),
            "Produkttyp_Pergola": sum(1 for sys in current_systems if sys['Produkttyp'] == 'Pergola'),
            "Produkttyp_Mülltonnenbox": sum(1 for sys in current_systems if sys['Produkttyp'] == 'Mülltonnenbox'),
            "Seitenverkleidung_Gittermatte": sum(1 for sys in current_systems if sys['Seitenverkleidung'] == 'Gittermatte'),
            "Seitenverkleidung_Ohne": sum(1 for sys in current_systems if sys['Seitenverkleidung'] == 'Ohne'),
            "Seitenverkleidung_Stahl-Lochblech": sum(1 for sys in current_systems if sys['Seitenverkleidung'] == 'Stahl-Lochblech'),
            "Seitenverkleidung_Stahl-Vollblech": sum(1 for sys in current_systems if sys['Seitenverkleidung'] == 'Stahl-Vollblech'),
            "Seitenverkleidung_Trespa": sum(1 for sys in current_systems if sys['Seitenverkleidung'] == 'Trespa'),
            "Seitenverkleidung_WL": sum(1 for sys in current_systems if sys['Seitenverkleidung'] == 'WL'),
            "Seitenverkleidung_WL+LBK": sum(1 for sys in current_systems if sys['Seitenverkleidung'] == 'WL+LBK'),
            "Dachtyp_Gründach": sum(1 for sys in current_systems if sys['Dachtyp'] == 'Gründach'),
            "Dachtyp_Gründach-Light": sum(1 for sys in current_systems if sys['Dachtyp'] == 'Gründach-Light'),
            "Dachtyp_Ohne": sum(1 for sys in current_systems if sys['Dachtyp'] == 'Ohne'),
            "Dachtyp_Polycarbonat": sum(1 for sys in current_systems if sys['Dachtyp'] == 'Polycarbonat'),
            "Dachtyp_Trapezblech": sum(1 for sys in current_systems if sys['Dachtyp'] == 'Trapezblech')
        }])

        # One-Hot-Encode die Eingabe
        X_input_processed = pd.get_dummies(X_input, drop_first=True)

        # Füge fehlende Spalten hinzu (mit Wert 0) und ordne die Spalten neu an
        # Stelle sicher, dass model.feature_names_in_ existiert
        if hasattr(model, 'feature_names_in_'):
             X_input_reindexed = X_input_processed.reindex(columns=model.feature_names_in_, fill_value=0)
        else:
             # Fallback, falls feature_names_in_ nicht verfügbar ist (z.B. bei älteren scikit-learn Versionen)
             print("Warnung: model.feature_names_in_ nicht verfügbar. Die Feature-Reihenfolge wird möglicherweise nicht korrekt gehandhabt.")
             X_input_reindexed = X_input_processed

        prediction = model.predict(X_input_reindexed)[0]
        gesamt_zeichnungszeit_h = prediction[0]
        gesamt_stuecklistenzeit_h = prediction[1]
        quelle = 'KI-Modell' # Quelle ist immer KI nach Entfernung des Lookups

    except Exception as e:
        print(f"Fehler bei KI-Vorhersage: {e}")
        gesamt_zeichnungszeit_h = 0
        gesamt_stuecklistenzeit_h = 0
        quelle = f'Fehler bei Schätzung: {e}'

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