import streamlit as st
import joblib
import pandas as pd

# Excel-Tabelle laden (für Modell-Lookups und Filterung)
excel_path = 'KI_Zeitprognose_Vorlage_Projekt-W.xlsx'
df_excel = pd.read_excel(excel_path)

# Modell laden (trainiert auf allen Daten)
model = joblib.load("ki_zeitprognose_model.joblib")

# --- Funktion zur Schätzung der Zeiten (wird von beiden Optionen genutzt) ---
def estimate_times(produkttyp, fläche, seitenverkleidung, dachtyp, anzahl_gewerke, mitarbeiter_filter, df_excel, model):
    
    zeichnungszeit_h = 0
    stuecklistenzeit_h = 0
    quelle = "Unbekannt"

    # 1. Excel-Lookup mit optionalem Mitarbeiterfilter
    df_excel_filtered = df_excel.copy()
    if mitarbeiter_filter != 'Alle':
         # Filtere Excel-Daten nach ausgewähltem Mitarbeiter (Spalte muss existieren!)
         if 'Zugewiesener_Mitarbeiter' in df_excel_filtered.columns:
             df_excel_filtered = df_excel_filtered[df_excel_filtered['Zugewiesener_Mitarbeiter'] == int(mitarbeiter_filter)]
             mitarbeiter_info = f', Mitarbeiter {mitarbeiter_filter}'
         else:
             st.warning("Spalte 'Zugewiesener_Mitarbeiter' nicht in Excel gefunden. Mitarbeiterfilter kann nicht angewendet werden.")
             df_excel_filtered = df_excel.copy() # Setze Filter zurück
             mitarbeiter_info = ''
    else:
        mitarbeiter_info = ''

    match = df_excel_filtered[
        (df_excel_filtered['Produkttyp'] == produkttyp) &
        (df_excel_filtered['Fläche_m2'] == fläche) &
        (df_excel_filtered['Seitenverkleidung'] == seitenverkleidung) &
        (df_excel_filtered['Dachtyp'] == dachtyp) &
        (df_excel_filtered['Anzahl_Gewerke'] == anzahl_gewerke)
    ]

    if len(match) == 1:
        zeichnungszeit_h = match.iloc[0]['Zeichnungszeit']
        stuecklistenzeit_h = match.iloc[0]['Stücklistenzeit']
        quelle = f'Excel-Tabelle (exakt 1 Treffer{mitarbeiter_info})'
    else:
        # 2. Fallback zum Excel-Lookup in allen Daten (falls Mitarbeiterfilter aktiv war und keinen exakten Treffer lieferte)
        if mitarbeiter_filter != 'Alle': # Nur prüfen, wenn vorher gefiltert wurde
            match_all = df_excel[
                (df_excel['Produkttyp'] == produkttyp) &
                (df_excel['Fläche_m2'] == fläche) &
                (df_excel['Seitenverkleidung'] == seitenverkleidung) &
                (df_excel['Dachtyp'] == dachtyp) &
                (df_excel['Anzahl_Gewerke'] == anzahl_gewerke)
            ]
            if len(match_all) == 1:
                zeichnungszeit_h = match_all.iloc[0]['Zeichnungszeit']
                stuecklistenzeit_h = match_all.iloc[0]['Stücklistenzeit']
                quelle = 'Excel-Tabelle (exakt 1 Treffer in allen Daten)'
            else:
                 # 3. Fallback zum KI-Modell
                 X_input = pd.DataFrame([{
                     "Produkttyp": produkttyp,
                     "Fläche_m2": fläche,
                     "Seitenverkleidung": seitenverkleidung,
                     "Dachtyp": dachtyp,
                     "Anzahl_Gewerke": anzahl_gewerke
                 }])
                 prediction = model.predict(X_input)[0]
                 zeichnungszeit_h = prediction[0]
                 stuecklistenzeit_h = prediction[1]
                 quelle = 'KI-Modell'
        else: # Kein Mitarbeiterfilter aktiv, direkt zum KI-Modell
             X_input = pd.DataFrame([{
                 "Produkttyp": produkttyp,
                 "Fläche_m2": fläche,
                 "Seitenverkleidung": seitenverkleidung,
                 "Dachtyp": dachtyp,
                 "Anzahl_Gewerke": anzahl_gewerke
             }])
             prediction = model.predict(X_input)[0]
             zeichnungszeit_h = prediction[0]
             stuecklistenzeit_h = prediction[1]
             quelle = 'KI-Modell'

    return (zeichnungszeit_h, stuecklistenzeit_h), quelle


st.title("KI-Zeitprognose für technische Bearbeitung")

# Simulierte ap+ Projektdaten (OHNE Zeiten)
# Später durch tatsächlichen ap+ Zugriff ersetzen!
SIMULATED_AP_PLUS_PROJECTS = [
    {
        'Interne_Auftragsnummer': 'AUFTRAG-001',
        'Produkttyp': 'Carport',
        'Fläche_m2': 180,
        'Seitenverkleidung': 'Vorhanden',
        'Dachtyp': 'Gründach',
        'Anzahl_Gewerke': 4,
        'Zugewiesener_Mitarbeiter': 1,
    },
    {
        'Interne_Auftragsnummer': 'AUFTRAG-002',
        'Produkttyp': 'Fahrradeinhausung',
        'Fläche_m2': 40,
        'Seitenverkleidung': 'Vorhanden',
        'Dachtyp': 'Trapezblech',
        'Anzahl_Gewerke': 2,
        'Zugewiesener_Mitarbeiter': 2,
    },
    {
        'Interne_Auftragsnummer': 'AUFTRAG-003',
        'Produkttyp': 'Mülleinhausung',
        'Fläche_m2': 90,
        'Seitenverkleidung': 'Ohne',
        'Dachtyp': 'Gründach-Light',
        'Anzahl_Gewerke': 3,
        'Zugewiesener_Mitarbeiter': 1,
    },
    {
        'Interne_Auftragsnummer': 'AUFTRAG-004',
        'Produkttyp': 'Carport',
        'Fläche_m2': 300,
        'Seitenverkleidung': 'Ohne',
        'Dachtyp': 'Ohne',
        'Anzahl_Gewerke': 2,
        'Zugewiesener_Mitarbeiter': 2,
    },
    {
        'Interne_Auftragsnummer': 'AUFTRAG-005',
        'Produkttyp': 'Fahrradeinhausung',
        'Fläche_m2': 55,
        'Seitenverkleidung': 'Vorhanden',
        'Dachtyp': 'Gründach',
        'Anzahl_Gewerke': 1,
        'Zugewiesener_Mitarbeiter': 1,
    }
]

df_simulated_projects = pd.DataFrame(SIMULATED_AP_PLUS_PROJECTS)

# --- Option 1: Abruf über Auftragsnummer ---
st.header("Projektzeiten per Auftragsnummer abrufen")

auftragsnummer = st.text_input("Interne Auftragsnummer eingeben", key="auftragsnummer_input")

# Optionale Mitarbeiterfilterung (basiert auf allen bekannten Mitarbeitern in Excel und simulierten Daten)
all_employees = pd.concat([
    df_excel['Zugewiesener_Mitarbeiter'].dropna() if 'Zugewiesener_Mitarbeiter' in df_excel.columns else pd.Series([], dtype=int),
    df_simulated_projects['Zugewiesener_Mitarbeiter'].dropna() if 'Zugewiesener_Mitarbeiter' in df_simulated_projects.columns else pd.Series([], dtype=int)
]).unique()

mitarbeiter_filter_optionen = ['Alle'] + sorted(all_employees.astype(str).tolist()) # Mitarbeiter IDs als String für Selectbox
selected_mitarbeiter_auftragsnr = st.selectbox("Optional: Zeiten basierend auf Projekten von Mitarbeiter...", mitarbeiter_filter_optionen, key="mitarbeiter_filter_auftragsnr")

if st.button("Zeiten für Auftragsnummer abrufen", key="btn_abrufen"):
    # Projekt in simulierten Daten suchen
    project_data = df_simulated_projects[df_simulated_projects['Interne_Auftragsnummer'] == auftragsnummer]

    if project_data.empty:
        st.warning(f"Kein Projekt mit Auftragsnummer {auftragsnummer} in den simulierten Daten gefunden.")
    else:
        # Projektdetails extrahieren
        projekt_details = project_data.iloc[0]
        # Mitarbeiter aus den simulierten Daten übernehmen, falls im Excel-Filter 'Alle' gewählt ist
        # Ansonsten wird der Filter angewendet.
        # mitarbeiter_fuer_schaetzung = projekt_details['Zugewiesener_Mitarbeiter'] if selected_mitarbeiter_auftragsnr == 'Alle' else int(selected_mitarbeiter_auftragsnr)

        st.subheader(f"Details für Projekt {auftragsnummer}")
        st.write(f"Produkttyp: {projekt_details['Produkttyp']}")
        st.write(f"Fläche: {projekt_details['Fläche_m2']} m²")
        st.write(f"Seitenverkleidung: {projekt_details['Seitenverkleidung']}")
        st.write(f"Dachtyp: {projekt_details['Dachtyp']}")
        st.write(f"Anzahl Gewerke: {projekt_details['Anzahl_Gewerke']}")
        # Anzeige des in ap+ zugewiesenen Mitarbeiters (aus simulierten Daten)
        if 'Zugewiesener_Mitarbeiter' in projekt_details:
             st.write(f"Zugewiesener Mitarbeiter (simuliert in ap+): {projekt_details['Zugewiesener_Mitarbeiter']}")

        # --- Zeiten schätzen (verwendet die Logik von unten) ---
        estimated_times, quelle = estimate_times(
            projekt_details['Produkttyp'],
            projekt_details['Fläche_m2'],
            projekt_details['Seitenverkleidung'],
            projekt_details['Dachtyp'],
            projekt_details['Anzahl_Gewerke'],
            selected_mitarbeiter_auftragsnr, # Übergabe des Filters
            df_excel, # Übergabe der Excel-Daten
            model # Übergabe des Modells
        )

        st.subheader("Geschätzte Bearbeitungszeiten")
        st.write(f"Quelle der Werte: **{quelle}**")
        st.metric("Zeichnung", f"{estimated_times[0]:.1f} Stunden")
        st.metric("Stückliste", f"{estimated_times[1]:.1f} Stunden")

# --- Option 2: Manuelle Eingabe ---
st.header("Manuelle Eingabe Projektdetails")
st.write("Schätzen Sie die Zeiten für ein Projekt ohne Auftragsnummer.")

with st.form("manual_input_form"):
    manual_produkttyp = st.selectbox("Produkttyp", ["Carport", "Fahrradeinhausung", "Mülleinhausung"], key="manual_produkttyp")
    manual_fläche = st.number_input("Fläche in m²", min_value=1, value=100, key="manual_fläche")
    manual_seitenverkleidung = st.selectbox("Seitenverkleidung", ["Vorhanden", "Ohne"], key="manual_seitenverkleidung")
    manual_dachtyp = st.selectbox("Dachtyp", ["Gründach", "Gründach-Light", "Trapezblech", "Ohne"], key="manual_dachtyp")
    manual_anzahl_gewerke = st.slider("Anzahl Gewerke", min_value=1, max_value=5, value=2, key="manual_anzahl_gewerke")

    # Optional: Mitarbeiter für manuelle Eingabe filtern
    selected_mitarbeiter_manual = st.selectbox("Optional: Zeiten basierend auf Projekten von Mitarbeiter...", mitarbeiter_filter_optionen, key="mitarbeiter_filter_manual")

    submit_manual_button = st.form_submit_button("Zeiten schätzen (Manuell)")

    if submit_manual_button:
        # --- Zeiten schätzen (verwendet die gleiche Logik) ---
        estimated_times_manual, quelle_manual = estimate_times(
            manual_produkttyp,
            manual_fläche,
            manual_seitenverkleidung,
            manual_dachtyp,
            manual_anzahl_gewerke,
            selected_mitarbeiter_manual, # Übergabe des Filters
            df_excel, # Übergabe der Excel-Daten
            model # Übergabe des Modells
        )

        st.subheader("Geschätzte Bearbeitungszeiten (Manuell)")
        st.write(f"Quelle der Werte: **{quelle_manual}**")
        st.metric("Zeichnung", f"{estimated_times_manual[0]:.1f} Stunden")
        st.metric("Stückliste", f"{estimated_times_manual[1]:.1f} Stunden")

# Optional: Anzeige der simulierten Projektdaten (hilfreich für die Demo)
st.sidebar.subheader("Simulierte ap+ Projekte")
st.sidebar.dataframe(df_simulated_projects)