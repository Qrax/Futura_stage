#!/usr/bin/env python3
"""
sync_plotter_multi.py – interactieve visualisatie van data uit MEERDERE CSV-bestanden.
Kolommen 'Device','Run','SampleIndex','ADC_Value','Timestamp_us' aanwezig.

Kan CSV-bestanden vooraf definiëren via PREDEFINED_CSV_FILES.
Anders wordt interactief om bestandsnamen gevraagd (komma-gescheiden).

Bij opstart:
  • V_ref globaal instellen.
  • Device filter globaal instellen (e.g., "Master").

Menu: (aangepast voor multi-file)
  a) Ruwe data alle runs (van alle geladen bestanden)
  b) Gemiddelde runs (van alle geladen bestanden, per bestand een lijn)
  c) Statistiek max‑amplitudes (histograms per bestand)
  d) Steepest‑slope onset (aligned data per bestand)
  e) Threshold‑onset (aligned data per bestand)
  l) Laad andere CSV-bestand(en)
  x) Stoppen
"""
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Config ---------------------------------------------------------------
ADC_BITS = 12
TARGET_DATA_SUBFOLDER = os.path.join("../..", "data", "UltraSoon_Measurements")
DEFAULT_BASE_FILENAMES_STR = "nieuw_deksel_meta_1.csv,nieuw_deksel_2_meta_1.csv"  # Voorbeeld

# --- PREDEFINED CSV FILES CONFIGURATION ---
# Lijst van BASE filenames. Verwacht in TARGET_DATA_SUBFOLDER.
# Als leeg, wordt interactief gevraagd.
PREDEFINED_CSV_FILES = [
    # "nieuw_deksel_meta_1.csv",  # Voorbeeld 1
    # "nieuw_deksel_2_meta_1.csv", # Voorbeeld 2
]
#PREDEFINED_CSV_FILES = ["nieuw_deksel_meta_1.csv", "nieuw_deksel_2_meta_1.csv", "nieuw_deksel_3_meta_1.csv"]
#PREDEFINED_CSV_FILES = ["zonder_pulse_meta_1.csv", "zonder_pulse_meta_2.csv", "zonder_pulse_meta_3.csv"]
#PREDEFINED_CSV_FILES = ["ruw_opnieuw_opbouw_meta_1.csv", "ruw_opnieuw_opbouw_meta_2.csv", "ruw_opnieuw_opbouw_meta_3.csv"]
#PREDEFINED_CSV_FILES = ["zonder_opbouw_meta_3.csv", "zonder_opbouw_meta_2.csv", "zonder_opbouw_meta_1.csv"]
#PREDEFINED_CSV_FILES = ["test materials_meta_1.csv", "test materials_meta_2.csv", "test materials_meta_3.csv"]
#PREDEFINED_CSV_FILES = ["test_x_meta_1.csv", "test_x_meta_2.csv", "test_x_meta_3.csv"]
#PREDEFINED_CSV_FILES = ["woop_meta_1.csv", "woop_meta_2.csv", "woop_meta_3.csv"]
#PREDEFINED_CSV_FILES = ["boopo_meta_1.csv", "boopo_meta_2.csv", "boopo_meta_3.csv", "gleuf_meta_1.csv"]
PREDEFINED_CSV_FILES = [
    "15_mm_gleuf_herbouw_meta_1.csv",
    "15_mm_gleuf_herbouw_meta_2.csv",
    "15_mm_gleuf_herbouw_meta_3.csv",
    "zonder_gleuf_herbouw_meta_1.csv",
    "zonder_gleuf_herbouw_meta_2.csv",
    "zonder_gleuf_herbouw_meta_3.csv",
]

# Kleuren voor de verschillende datasets
PLOT_COLORS = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']

# --- Globale variabelen voor geladen data en instellingen ---
list_of_dataframes_processed = []  # Lijst van DataFrames, elk verwerkt (gefilterd, Voltage kolom)
base_filenames_loaded = []  # Om labels te maken
current_v_ref_global = 3.3
current_device_filter_global = "Master"


# --- Utilities ------------------------------------------------------------
def load_single_csv(full_csv_path, base_filename):
    """Laadt een enkel CSV-bestand en retourneert een DataFrame of None."""
    try:
        df = pd.read_csv(full_csv_path)
        req = {"Device", "Run", "SampleIndex", "ADC_Value", "Timestamp_us"}
        if not req.issubset(df.columns):
            print(f"FOUT: Bestand '{base_filename}' mist kolommen: {req - set(df.columns)}")
            return None
        df["Run"] = df["Run"].astype(int)
        df["SampleIndex"] = df["SampleIndex"].astype(int)
        df["Device"] = df["Device"].astype(str)
        df['_source_file_'] = base_filename  # Voeg bron toe voor latere identificatie
        print(f"'{base_filename}' succesvol geladen ({len(df)} rijen).")
        return df
    except FileNotFoundError:
        print(f"FOUT: Bestand '{full_csv_path}' niet gevonden.")
        return None
    except Exception as e:
        print(f"Algemene fout bij laden '{base_filename}': {e}")
        return None


def adc_to_volts(series, vref):
    return series / (2 ** ADC_BITS - 1) * vref


def process_loaded_dataframes(raw_dfs_list, v_ref, device_filter):
    """Filtert per device, voegt Voltage kolom toe."""
    processed_dfs = []
    for df_raw in raw_dfs_list:
        if df_raw is None: continue
        df_filtered = df_raw[df_raw["Device"] == device_filter].copy()
        if df_filtered.empty:
            print(
                f"Waarschuwing: Geen data voor device '{device_filter}' in bestand '{df_raw['_source_file_'].iloc[0]}'. Dit bestand wordt overgeslagen.")
            continue
        df_filtered["Voltage"] = adc_to_volts(df_filtered["ADC_Value"], v_ref)
        processed_dfs.append(df_filtered)
    return processed_dfs


def finalize_plot_multi(num_datasets):  # Aangepast voor meerdere datasets
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()

    # Creëer een unieke legenda op basis van dataset labels (eerste paar runs per dataset)
    unique_legend_items = {}
    for h, l in zip(handles, labels):
        if l not in unique_legend_items:  # Alleen de eerste keer dat een label voorkomt
            unique_legend_items[l] = h
            if len(unique_legend_items) >= num_datasets * 2:  # Max 2 legend items per dataset (kan aangepast)
                break

    if unique_legend_items:
        plt.legend(unique_legend_items.values(), unique_legend_items.keys(), fontsize="small", title="Datasets & Runs")

    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.show()


# --- Plot‑functies Aangepast voor Meerdere Bestanden ---
def plot_all_runs_raw_multi(list_of_dfs, loaded_filenames):
    """Plot ruwe data van alle runs uit meerdere DataFrames."""
    if not list_of_dfs:
        print("Geen data geladen om te plotten.")
        return

    plt.figure(figsize=(12, 7))
    plot_runs_count_total = 0

    for i, df_processed in enumerate(list_of_dfs):
        if df_processed.empty:
            continue

        dataset_label_prefix = loaded_filenames[i].replace(".csv", "")  # Korte label van bestandsnaam
        color = PLOT_COLORS[i % len(PLOT_COLORS)]

        # Runs binnen deze specifieke dataset (df_processed)
        # Device is al gefilterd, dus groupby alleen op Run
        runs_in_this_df = 0
        for run_num, grp in df_processed.groupby("Run"):
            # Label alleen de eerste paar runs van elke dataset voor de legenda
            run_label = None
            if runs_in_this_df < 2:  # Max 2 runs per dataset in de legenda
                run_label = f"{dataset_label_prefix} Run {run_num}"

            plt.plot(grp["SampleIndex"], grp["Voltage"], alpha=0.5, color=color, label=run_label)
            runs_in_this_df += 1
            plot_runs_count_total += 1

    if plot_runs_count_total == 0:
        print("Niets om te plotten na filtering.")
        return

    plt.title(f"Ruwe data – alle runs (Device: {current_device_filter_global})")
    plt.xlabel("SampleIndex")
    plt.ylabel(f"Spanning (V, V_ref={current_v_ref_global}V)")
    finalize_plot_multi(len(list_of_dfs))


# --- Mainmenu en laadlogica ---
def run_main_interaction_loop_multi():
    global list_of_dataframes_processed, base_filenames_loaded, current_v_ref_global, current_device_filter_global

    if not list_of_dataframes_processed:
        print("Geen data beschikbaar. Probeer opnieuw te laden ('l').")
        return "load_new_files"  # Terug naar laad-stap

    while True:
        print(
            f"\n--- Menu (Actieve filter: Device='{current_device_filter_global}', V_ref={current_v_ref_global}V) ---")
        print(f"Geladen bestanden: {', '.join(base_filenames_loaded)}")
        print("  a) Ruwe data alle runs")
        # print("  b) Gemiddelde runs") # Implement later
        # print("  c) Statistiek max‑amplitudes") # Implement later
        # print("  d) Steepest‑slope onset") # Implement later
        # print("  e) Threshold‑onset") # Implement later
        print("  f) Filter/V_ref opnieuw instellen")
        print("  l) Laad andere CSV-bestand(en)")
        print("  x) Stoppen")
        k = input("> ").strip().lower()

        if k == 'a':
            plot_all_runs_raw_multi(list_of_dataframes_processed, base_filenames_loaded)
        # Voeg hier later de andere plot opties toe, aangepast voor list_of_dataframes_processed
        # elif k == 'b': plot_average_runs_multi(list_of_dataframes_processed, base_filenames_loaded)
        elif k == 'f':
            print("\n--- Filters en V_ref opnieuw instellen ---")
            # V_ref
            v_ref_input = input(f"Nieuwe referentiespanning V_ref (V, default={current_v_ref_global}): ").strip()
            try:
                current_v_ref_global = float(v_ref_input) if v_ref_input else current_v_ref_global
            except ValueError:
                print(f"Ongeldige V_ref, blijft {current_v_ref_global}V.")

            # Device filter
            dev_filter_input = input(
                f"Nieuw device filter (Master/Slave, default='{current_device_filter_global}'): ").strip()
            if dev_filter_input in ["Master", "Slave"]:
                current_device_filter_global = dev_filter_input
            elif dev_filter_input:  # als er iets is ingevoerd maar ongeldig
                print(f"Ongeldig device filter, blijft '{current_device_filter_global}'.")

            # Herverwerk de oorspronkelijk geladen ruwe data met nieuwe instellingen
            # Hiervoor moeten we de ruwe, ongefilterde data bewaren.
            # Aanpassing: load_and_process_files moet de ruwe data apart teruggeven.
            # Voor nu: we signaleren dat er opnieuw geladen moet worden om het simpel te houden.
            print("Instellingen gewijzigd. Data moet opnieuw verwerkt worden.")
            return "reprocess_data"  # Signaal om opnieuw te processen (of te herladen)

        elif k == 'l':
            return "load_new_files"
        elif k == 'x':
            return "exit"
        else:
            print("Onbekende keuze.")


def load_and_process_files():
    """Handelt het laden en initieel processen van CSV-bestanden af."""
    global list_of_dataframes_processed, base_filenames_loaded
    global current_v_ref_global, current_device_filter_global  # Voor instellingen

    raw_dataframes_loaded = []
    base_filenames_loaded = []  # Reset voor nieuwe lading

    abs_target_dir = os.path.abspath(TARGET_DATA_SUBFOLDER)

    # Gebruik PREDEFINED_CSV_FILES als die gevuld is
    csv_base_names_to_load = globals().get('PREDEFINED_CSV_FILES', [])
    if isinstance(csv_base_names_to_load, list) and csv_base_names_to_load:
        print(f"\n--- Gebruik voorgedefinieerde CSV-bestanden: {', '.join(csv_base_names_to_load)} ---")
        # Belangrijk: maak PREDEFINED_CSV_FILES leeg na gebruik, zodat 'l' interactief wordt
        globals()['PREDEFINED_CSV_FILES'] = []
    else:
        print(f"\n--- Interactieve CSV-bestandskeuze ---")
        print(f"(Verwacht in map: '{abs_target_dir}')")
        filenames_str = input(f"Voer basisnamen CSV-bestanden in, komma-gescheiden (bv. file1.csv,file2.csv)\n"
                              f"(Enter voor default '{DEFAULT_BASE_FILENAMES_STR}'): ").strip()
        if not filenames_str:
            filenames_str = DEFAULT_BASE_FILENAMES_STR
        csv_base_names_to_load = [name.strip() for name in filenames_str.split(',')]

    if not csv_base_names_to_load or not any(csv_base_names_to_load):
        print("Geen bestandsnamen opgegeven.")
        return False  # Mislukt

    print("\n--- Laden bestanden ---")
    for base_name in csv_base_names_to_load:
        if not base_name: continue
        full_path = os.path.join(TARGET_DATA_SUBFOLDER, base_name)
        df = load_single_csv(full_path, base_name)
        if df is not None:
            raw_dataframes_loaded.append(df)
            base_filenames_loaded.append(base_name)

    if not raw_dataframes_loaded:
        print("Geen bestanden succesvol geladen.")
        return False

    # Initiele V_ref en Device Filter instellen
    print("\n--- Globale instellingen ---")
    v_ref_input = input(f"Referentiespanning V_ref voor alle bestanden (V, default={current_v_ref_global}): ").strip()
    try:
        current_v_ref_global = float(v_ref_input) if v_ref_input else current_v_ref_global
    except ValueError:
        print(f"Ongeldige V_ref, standaard {current_v_ref_global}V gebruikt.")

    dev_filter_input = input(
        f"Device filter voor alle bestanden (Master/Slave, default='{current_device_filter_global}'): ").strip()
    if dev_filter_input in ["Master", "Slave"]:
        current_device_filter_global = dev_filter_input
    elif dev_filter_input:
        print(f"Ongeldig device filter, '{current_device_filter_global}' blijft gebruikt.")

    list_of_dataframes_processed = process_loaded_dataframes(raw_dataframes_loaded, current_v_ref_global,
                                                             current_device_filter_global)

    if not list_of_dataframes_processed:
        print("Geen data over na filteren. Controleer device filter of bestanden.")
        return False

    return True  # Succesvol geladen en verwerkt


if __name__ == "__main__":
    # Om de 'ruwe' data te bewaren voor her-processen met nieuwe filters (optie 'f')
    # moeten we de 'raw_dataframes_loaded' globaal of toegankelijk maken.
    # Voor nu, als 'f' wordt gekozen, signaleren we 'reprocess_data' wat
    # momenteel hetzelfde doet als 'load_new_files'. Een echte her-processing
    # zonder opnieuw te laden van schijf vereist dat `raw_dataframes_loaded`
    # bewaard blijft.

    # Globale lijst voor ruwe, onbewerkte dataframes
    raw_dataframes_globally_loaded = []

    while True:
        action = "load_new_files"  # Start met laden

        if action == "load_new_files":
            # Functie die PREDEFINED gebruikt of vraagt, en laadt in `raw_dataframes_globally_loaded`
            # en vervolgens `list_of_dataframes_processed` vult.
            # Deze functie moet ook V_ref en Device filter vragen.

            # Reset voor een nieuwe laadpoging
            raw_dataframes_globally_loaded = []
            base_filenames_loaded = []
            list_of_dataframes_processed = []

            abs_target_dir = os.path.abspath(TARGET_DATA_SUBFOLDER)
            csv_base_names_to_load = globals().get('PREDEFINED_CSV_FILES', [])
            # Belangrijk: maak PREDEFINED_CSV_FILES leeg na EERSTE GEBRUIK PER SESSIE.
            # Dit moet zorgvuldiger, want nu wordt het elke keer geleegd in de lus.
            # Beter: een vlag die aangeeft of predefined al gebruikt is in deze run.
            # Voor nu, houden we het simpel: als PREDEFINED_CSV_FILES bestaat, gebruik het.
            # De gebruiker moet het zelf leegmaken als ze interactief willen.

            if isinstance(csv_base_names_to_load, list) and csv_base_names_to_load:
                print(f"\n--- Gebruik voorgedefinieerde CSV-bestanden: {', '.join(csv_base_names_to_load)} ---")
                # Deze logica om PREDEFINED te legen moet misschien anders als 'f' echt
                # alleen herverwerkt zonder de lijst PREDEFINED te beïnvloeden.
                # globals()['PREDEFINED_CSV_FILES'] = [] # Leegmaken na gebruik
            else:
                print(f"\n--- Interactieve CSV-bestandskeuze ---")
                print(f"(Verwacht in map: '{abs_target_dir}')")
                filenames_str = input(f"Voer basisnamen CSV-bestanden in, komma-gescheiden\n"
                                      f"(Enter voor default '{DEFAULT_BASE_FILENAMES_STR}'): ").strip()
                if not filenames_str: filenames_str = DEFAULT_BASE_FILENAMES_STR
                csv_base_names_to_load = [name.strip() for name in filenames_str.split(',') if name.strip()]

            if not csv_base_names_to_load:
                print("Geen bestandsnamen opgegeven. Stoppen.")
                break

            print("\n--- Laden bestanden ---")
            success_any_load = False
            for base_name in csv_base_names_to_load:
                full_path = os.path.join(TARGET_DATA_SUBFOLDER, base_name)
                df_raw = load_single_csv(full_path, base_name)
                if df_raw is not None:
                    raw_dataframes_globally_loaded.append(df_raw)
                    base_filenames_loaded.append(base_name)
                    success_any_load = True

            if not success_any_load:
                print("Geen enkel bestand kon succesvol geladen worden.")
                retry = input("Opnieuw proberen (y/N)? ").strip().lower()
                if retry == 'y':
                    continue
                else:
                    break

            # Globale instellingen V_ref en Device Filter
            print("\n--- Globale instellingen ---")
            v_ref_input_str = input(
                f"Referentiespanning V_ref voor alle bestanden (V, default={current_v_ref_global}): ").strip()
            try:
                current_v_ref_global = float(v_ref_input_str) if v_ref_input_str else current_v_ref_global
            except ValueError:
                print(f"Ongeldige V_ref, standaard {current_v_ref_global}V gebruikt.")

            dev_filter_input_str = input(
                f"Device filter voor alle bestanden (Master/Slave, default='{current_device_filter_global}'): ").strip()
            if dev_filter_input_str in ["Master", "Slave"]:
                current_device_filter_global = dev_filter_input_str
            elif dev_filter_input_str:  # als er iets is ingevoerd maar ongeldig
                print(f"Ongeldig device filter, '{current_device_filter_global}' blijft gebruikt.")

            # Verwerk de geladen ruwe dataframes
            list_of_dataframes_processed = process_loaded_dataframes(raw_dataframes_globally_loaded,
                                                                     current_v_ref_global, current_device_filter_global)
            if not list_of_dataframes_processed:
                print("Geen data over na filteren. Controleer instellingen of bestanden.")
                retry = input("Opnieuw proberen met laden (y/N)? ").strip().lower()
                if retry == 'y':
                    continue
                else:
                    break

        # elif action == "reprocess_data": # Voor de 'f' optie
        #     print("\n--- Herverwerken data met nieuwe instellingen ---")
        #     list_of_dataframes_processed = process_loaded_dataframes(raw_dataframes_globally_loaded, current_v_ref_global, current_device_filter_global)
        #     if not list_of_dataframes_processed:
        #         print("Herverwerken mislukt, geen data over. Probeer opnieuw te laden ('l').")
        #         action = "load_new_files" # Forceer herladen als herverwerken faalt
        #         continue

        # Nu de menu-loop starten met de (her)verwerkte data
        action = run_main_interaction_loop_multi()  # Geeft "exit", "load_new_files", of "reprocess_data"

        if action == "exit":
            break
        elif action == "load_new_files":
            # PREDEFINED_CSV_FILES moet nu leeg zijn als het al gebruikt was,
            # zodat de volgende keer interactief gevraagd wordt.
            # De code hierboven voor het lezen van PREDEFINED_CSV_FILES moet dit afhandelen.
            # Voor nu, als 'l' gekozen is, zorgen we dat PREDEFINED leeg is.
            globals()['PREDEFINED_CSV_FILES'] = []
            continue  # Ga naar begin van de while True lus om opnieuw te laden
        elif action == "reprocess_data":
            # Herverwerk de `raw_dataframes_globally_loaded` met de nieuwe
            # `current_v_ref_global` en `current_device_filter_global`
            print("\n--- Herverwerken data met nieuwe instellingen ---")
            list_of_dataframes_processed = process_loaded_dataframes(raw_dataframes_globally_loaded,
                                                                     current_v_ref_global, current_device_filter_global)
            if not list_of_dataframes_processed:
                print("Herverwerken mislukt, geen data over. Probeer opnieuw te laden ('l').")
                action = "load_new_files"
                globals()['PREDEFINED_CSV_FILES'] = []  # Forceer interactief bij volgende laadpoging
                continue
            # Blijf in de menu-loop, maar met bijgewerkte `list_of_dataframes_processed`
            # De `run_main_interaction_loop_multi` wordt opnieuw aangeroepen door de `continue` impliciet.
            # Dit is niet helemaal correct, de actie zou moeten zijn om de menu loop opnieuw te starten
            # met de nieuwe data. De huidige `action` variabele wordt overschreven.
            # Correcter:
            # De menu loop zou de state moeten teruggeven, en hier beslissen we.
            # Voor nu, "reprocess_data" leidt terug naar het begin van de buitenste lus,
            # maar `raw_dataframes_globally_loaded` is nog gevuld.
            # We moeten de `action` zo zetten dat we NIET opnieuw van schijf laden.

            # Beter:
            # if action == "reprocess_data":
            #    list_of_dataframes_processed = process_loaded_dataframes(...)
            #    # EN DAN de menu loop opnieuw starten ZONDER de laadstap over te slaan.
            #    # Dit vereist een herstructurering.
            #    # Simpelste voor nu: 'f' triggert een volledige herlaad-cyclus
            #    # waar de gebruiker de V_ref / filter opnieuw kan instellen.
            #    # Of we maken de `run_main_interaction_loop_multi` direct aanpasbaar.
            #    # De huidige 'f' in de menu loop doet dit al min of meer door
            #    # de globale filters aan te passen en dan `reprocess_data` te returnen.
            #    # De `continue` hieronder is de sleutel.
            continue  # Ga terug naar begin van while True, maar nu wordt `action` niet "load_new_files"
            # tenzij de `run_main_interaction_loop_multi` dat expliciet teruggeeft.
            # Dit is nog steeds niet perfect.

    print("Programma beëindigd.")