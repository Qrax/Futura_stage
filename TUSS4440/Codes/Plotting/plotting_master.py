# Sla dit op als: plotting_master.py (VERVANG JE HUIDIGE BESTAND)

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import detrend
from scipy.stats import t
import importlib
import tkinter as tk  # *** NIEUW: Nodig om schermgrootte op te vragen

"""
plotting_master.py – Compares data from one to six CSV files using various plot modes.
NIEUW: Implementeert analyseprofielen om verschillende instellingen per materiaaltype te gebruiken.
"""

# ---------------- PLOTMODUS – Select modes to run -----------------
PLOT_MODES_TO_RUN = [
    # "RAW",
    # "THRESHOLD",
    #"SUMMARY",
    # "DIFFERENCE", # Nog niet geïmplementeerd als module
    # "DEVIATION_FROM_LINEAR", # Nog niet geïmplementeerd als module
    # "FFT_OF_DEVIATION", # Nog niet geïmplementeerd als module
    "POST_PEAK_PERIODOGRAM",
    # "INDIVIDUAL_RUN_PERIODOGRAM",
    # "PERIODOGRAM_ANALYSIS_STEPS",
    # "EXPLAIN_EXP_TREND"
    # "VARIABILITY_TABLE",          # Voeg deze toe voor de tabel
    # "PEAK_PROMINENCE",            # Voeg deze toe voor de bar chart
    # "RUN_CONSISTENCY_HEATMAP",    # Voeg deze toe voor de heatmaps

    #"#appendix_figures",
    # "PEAK_AMPLITUDE_BOXPLOT",
    # "APPENDIX_PEAK_FINDING_EXPLAINED",
#     "15MM_SCATTER",
#     "SESSION_PERIODOGRAM",
#     "DIAGNOSTIC_PEAK_FINDING"
]

# ... (de rest van je profielen en bestandslijsten blijft hetzelfde) ...
# ----------------------------------------------------------------
# --- NIEUW: DEFINIEER HIER JE ANALYSEPROFIELEN ---
# ----------------------------------------------------------------
ANALYSIS_PROFILES = {
    'aluminium': {
        'DESCRIPTION': 'Settings for Aluminium samples',
        'THRESH': 1500,
        'MAX_WINDOW_AFTER': 2000,
        'DETREND_TYPE': 'exponential',
        'MAX_PERIOD_PLOT_US': 1800,  # Aangepaste maximale periode in µs
    },
    'g10': {
        'DESCRIPTION': 'Settings for G10 composite samples',
        'THRESH': 1000,
        'MAX_WINDOW_AFTER': 450,
        'DETREND_TYPE': 'exponential',
        'MAX_PERIOD_PLOT_US': 600,  # Aangepaste maximale periode in µs
    },
    'default': {  # Een fallback voor als bestandsnaam niet matcht
        'DESCRIPTION': 'Default settings',
        'THRESH': 1500,
        'MAX_WINDOW_AFTER': 1000,
        'DETREND_TYPE': 'linear',
        'MAX_PERIOD_PLOT_US': 500,
    }
}
# ----------------------------------------------------------------

# --------- General Settings -------------------------------------
TARGET_DATA_SUBFOLDER = os.path.join("..", "..", "data", "UltraSoon_Measurements")
SAMPLE_TIME_DELTA_US = 4.63

_CSV_BASE_FILES = [
    # Aluminium

    # "al_0mm_please_meta_3.csv", "al_0mm_please_meta_2.csv", "al_0mm_laatste_metingen_meta_1.csv", "al_0mm_laatste_metingen_meta_3.csv", "al_0mm_laatste_metingen_meta_4.csv",

   # "al_5mm_5_metingen_meta_1.csv", "al_5mm_5_metingen_meta_2.csv", "al_5mm_5_metingen_meta_3.csv", "al_5mm_5_metingen_meta_4.csv", "al_5mm_5_metingen_meta_5.csv",

    #Meting 1
    # "al_15mm_please_meta_1.csv", "al_15mm_please_meta_2.csv", "al_15mm_please_meta_3.csv", "al_15mm_please_meta_4.csv","al_15mm_please_meta_5.csv",

    #Meting 2
    # "al_15mm_gleuf_teuf_meta_1.csv", "al_15mm_gleuf_teuf_meta_2.csv", "al_15mm_gleuf_teuf_meta_3.csv", "al_15mm_gleuf_teuf_meta_4.csv",

    #Meting 3
    # "al_15mm_ff_opnieuw_meta_1.csv", "al_15mm_ff_opnieuw_meta_2.csv", "al_15mm_ff_opnieuw_meta_3.csv", "al_15mm_ff_opnieuw_meta_4.csv", "al_15mm_ff_opnieuw_meta_5.csv",


    # G10
    #     "g10_0mm_3e_test_meta_1.csv", "g10_0mm_3e_test_meta_2.csv", "g10_0mm_3e_test_meta_3.csv", "g10_0mm_3e_test_meta_4.csv", "g10_0mm_3e_test_meta_5.csv",
        "g10_5mm_final_meta_1.csv", "g10_5mm_5_metingen_meta_2.csv", "g10_5mm_5_metingen_meta_3.csv", "g10_5mm_5_metingen_meta_4.csv", "g10_5mm_5_metingen_meta_5.csv",
        #"g10_15mm_5_metingen_meta_1.csv", "g10_15mm_5_metingen_meta_2.csv", "g10_15mm_5_metingen_meta_3.csv", "g10_15mm_5_metingen_meta_4.csv", "g10_15mm_5_metingen_meta_5.csv"
]

LABELS = [
    # Aluminium

   # "AL Meting 1 (Defectloos)", "AL Meting 2 (Defectloos)", "AL Meting 3 (Defectloos)", "AL Meting 4 (Defectloos)","AL Meting 5 (Defectloos)"

   #  "AL Meting 1 (5mm)", "AL Meting 2 (5mm)", "AL Meting 3 (5mm)", "AL Meting 4 (5mm)", "AL Meting 5 (5mm)",

    # Sessie 1
    # "AL 15mm (S1-M1)", "AL 15mm (S1-M2)", "AL 15mm (S1-M3)", "AL 15mm (S1-M4)", "AL 15mm (S1-M5)",
    # Sessie 2
    # "AL 15mm (S2-M1)", "AL 15mm (S2-M2)", "AL 15mm (S2-M3)", "AL 15mm (S2-M4)",
    # Sessie 3
   # "AL 15mm (S3-M1)", "AL 15mm (S3-M2)", "AL 15mm (S3-M3)", "AL 15mm (S3-M4)", "AL 15mm (S3-M5)",


    # G10
   # "G10 Meting 1 (Defectloos)", "G10 Meting 2 (Defectloos)", "G10 Meting 3 (Defectloos)", "G10 Meting 4 (Defectloos)", "G10 Meting 5 (Defectloos)",
    "G10 Meting 1 (5mm)", "G10 Meting 2 (5mm)", "G10 Meting 3 (5mm)", "G10 Meting 4 (5mm)", "G10 Meting 5 (5mm)",
   # "G10 Meting 1 (15mm)", "G10 Meting 2 (15mm)", "G10 Meting 3 (15mm)", "G10 Meting 4 (15mm)", "G10 Meting 5 (15mm)",
]

CSV_FILES = [os.path.join(TARGET_DATA_SUBFOLDER, fname.strip()) for fname in _CSV_BASE_FILES]
if len(CSV_FILES) != len(LABELS): raise ValueError("Mismatch CSV_FILES/LABELS.")

DEVICE_FILTER = "Master"
MAX_RUNS = 100
MAX_POINTS = None
WINDOW_BEFORE = 50
POST_PEAK_OFFSET_SAMPLES = 20
FIT_WINDOW_POST_PEAK = 1400
DETREND_PERIODOGRAM = True
APPLY_FFT_WINDOW = False
TIME_UNIT = "us"


def adc_to_v(adc_val):
    return adc_val


THESIS_MODE = True
if THESIS_MODE:
    plt.rcParams.update({
        'font.size': 18,
        'axes.titlesize': 22,
        'axes.labelsize': 20,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 16,
        'figure.titlesize': 24
    })

PLOT_COLORS = plt.get_cmap('Set1').colors
PLOT_LINESTYLES = ['-']


def get_profile_key_for_file(filename):
    fn_lower = filename.lower()
    if 'g10' in fn_lower: return 'g10'
    if 'al_' in fn_lower or 'alu' in fn_lower: return 'aluminium'
    return 'default'


# --- Core Helper Functions (blijven ongewijzigd) ---
def load_and_prepare_data_main(csv_f, dev_filter, settings):
    try:
        df = pd.read_csv(csv_f)
    except FileNotFoundError:
        return None
    df = df[df["Device"] == dev_filter].copy()
    if df.empty: return None
    df["Voltage"] = settings.adc_to_v(df["ADC_Value"])
    time_per_sample = settings.SAMPLE_TIME_DELTA_US
    time_unit_factor = settings.ts_factor_for_raw_timeplot
    df['TimePlot'] = df.groupby('Run').cumcount() * time_per_sample * time_unit_factor
    df["ADC_Value_V"] = df["Voltage"]
    df['DataSource'] = os.path.basename(csv_f)
    return df


def get_summary_data_main(df_single_file, max_runs_to_process, window_b, profile_settings):
    unique_runs = sorted(df_single_file.Run.unique())
    runs_to_consider = unique_runs[:max_runs_to_process] if max_runs_to_process is not None else unique_runs
    threshold_val = profile_settings['THRESH']
    max_window_a = profile_settings['MAX_WINDOW_AFTER']
    aligned_traces = []
    for run_id in runs_to_consider:
        run_data_group = df_single_file[df_single_file.Run == run_id].reset_index(drop=True)
        voltages_this_run = run_data_group.Voltage.values
        indices_above_thresh = np.where(voltages_this_run > threshold_val)[0]
        if not len(indices_above_thresh): continue
        onset_index_in_run = indices_above_thresh[0]
        segment_start_abs = onset_index_in_run - window_b
        segment_stop_abs = onset_index_in_run + max_window_a
        if segment_start_abs >= 0 and segment_stop_abs <= len(voltages_this_run):
            segment = voltages_this_run[segment_start_abs:segment_stop_abs]
            if len(segment) > 0:
                aligned_traces.append(segment)
    return aligned_traces, len(aligned_traces)


def calculate_trace_stats_main(aligned_traces):
    if not aligned_traces: return np.array([]), 0, np.array([]), np.array([])
    N = len(aligned_traces)
    max_len = max(len(trace) for trace in aligned_traces)
    padded_matrix = np.full((N, max_len), np.nan)
    for i, trace in enumerate(aligned_traces):
        padded_matrix[i, :len(trace)] = trace
    mean_trace = np.nanmean(padded_matrix, axis=0)
    std_trace = np.zeros_like(mean_trace)
    if N > 1:
        std_trace = np.nanstd(padded_matrix, axis=0, ddof=1)
    return mean_trace, N, None, std_trace


class PlotSettingsContainer:
    def __init__(self):
        # *** GEWIJZIGD: FIGUURGROOTTE NAAR VIERKANT (1:1) ***
        # Dit is ideaal voor een 2-koloms scriptie-layout.
        self.FIG_SIZE = (8, 8)

        self.TARGET_DATA_SUBFOLDER = TARGET_DATA_SUBFOLDER
        self.SAMPLE_TIME_DELTA_US = SAMPLE_TIME_DELTA_US
        # ... (rest van de settings blijft hetzelfde) ...
        self.CSV_FILES = CSV_FILES
        self.LABELS = LABELS
        self.DEVICE_FILTER = DEVICE_FILTER
        self.MAX_RUNS = MAX_RUNS
        self.MAX_POINTS = MAX_POINTS
        self.WINDOW_BEFORE = WINDOW_BEFORE
        self.POST_PEAK_OFFSET_SAMPLES = POST_PEAK_OFFSET_SAMPLES
        self.FIT_WINDOW_POST_PEAK = FIT_WINDOW_POST_PEAK
        self.DETREND_PERIODOGRAM = DETREND_PERIODOGRAM
        self.APPLY_FFT_WINDOW = APPLY_FFT_WINDOW
        self.TIME_UNIT = TIME_UNIT
        self.PLOT_COLORS = PLOT_COLORS
        self.PLOT_LINESTYLES = PLOT_LINESTYLES
        self.ANALYSIS_PROFILES = ANALYSIS_PROFILES
        self.adc_to_v = adc_to_v

        if self.TIME_UNIT.lower() == "s":
            self.ts_factor_for_raw_timeplot, self.tu_raw_lbl = 1e-6, "s"
        elif self.TIME_UNIT.lower() == "ms":
            self.ts_factor_for_raw_timeplot, self.tu_raw_lbl = 1e-3, "ms"
        else:
            self.ts_factor_for_raw_timeplot, self.tu_raw_lbl = 1.0, "µs"


# --- Main script execution ---
if __name__ == "__main__":
    settings = PlotSettingsContainer()

    # *** NIEUW: Schermgrootte dynamisch opvragen ***
    root = tk.Tk()
    root.withdraw()  # Voorkom dat er een leeg tk venster verschijnt
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()
    print(f"I: Detected screen resolution: {screen_width}x{screen_height}")

    print("Loading data...")
    dfs_loaded, actual_labels_used = [], []
    for i, filepath in enumerate(settings.CSV_FILES):
        # ... (data laad-loop blijft hetzelfde) ...
        if not os.path.exists(filepath): continue
        df_loaded_single = load_and_prepare_data_main(filepath, settings.DEVICE_FILTER, settings)
        if df_loaded_single is not None:
            dfs_loaded.append(df_loaded_single)
            actual_labels_used.append(settings.LABELS[i])

    if not dfs_loaded: exit("E: No data loaded.")
    print(f"Data loaded for: {', '.join(actual_labels_used)}")

    summary_cache = {}
    print("Pre-calculating summary data (aligned run traces)...")
    for i, df_current_file in enumerate(dfs_loaded):
        # ... (pre-calculatie loop blijft hetzelfde) ...
        label_current = actual_labels_used[i]
        filename_current = df_current_file['DataSource'].iloc[0]
        profile_key = get_profile_key_for_file(filename_current)
        current_profile_settings = settings.ANALYSIS_PROFILES[profile_key]
        print(f"  '{label_current}' uses '{profile_key}' profile.")
        matrix_list_current, runs_count_current = get_summary_data_main(
            df_current_file, settings.MAX_RUNS,
            settings.WINDOW_BEFORE, current_profile_settings
        )
        mean_trace, N_for_mean, _, std_trace = calculate_trace_stats_main(matrix_list_current)
        summary_cache[label_current] = {
            "matrix_list": matrix_list_current, "runs_count": runs_count_current,
            "mean_trace": mean_trace, "N_for_mean": N_for_mean,
            "std_trace": std_trace, "profile_key": profile_key,
            "filename": filename_current
        }
        if N_for_mean > 0:
            print(f"    -> {N_for_mean} valid runs found and cached.")
        else:
            print(f"    -> WARNING: 0 valid runs found. Check THRESH and MAX_WINDOW_AFTER for this profile.")
    print("Pre-calculation done.")

    for plot_mode_name_iter in PLOT_MODES_TO_RUN:
        # ... (plot generatie loop blijft hetzelfde) ...
        print(f"\n--- Generating plot for mode: {plot_mode_name_iter.upper()} ---")
        module_name = f"Plot_Modes.plot_{plot_mode_name_iter.lower()}"
        function_name = f"generate_plot_{plot_mode_name_iter.lower()}"
        try:
            plot_module = importlib.import_module(module_name)
            plot_function = getattr(plot_module, function_name)
            result_fig = plot_function(dfs_loaded, actual_labels_used, settings, summary_cache, plt)
            if result_fig is None:
                print(f"I: Plot mode '{plot_mode_name_iter.upper()}' did not return a figure (this may be normal).")
        except Exception as e:
            print(f"E: An error occurred while running plot mode {plot_mode_name_iter.upper()}: {e}")
            import traceback;

            traceback.print_exc()

    if plt.get_fignums():
        print(f"\nDisplaying {len(plt.get_fignums())} plot(s).")

        # *** GEWIJZIGD: ROBUUSTE MANIER OM VENSTER TE MAXIMALISEREN ***
        try:
            fig_manager = plt.get_current_fig_manager()
            # Gebruik setGeometry om het venster EXACt op de schermgrootte in te stellen
            fig_manager.window.setGeometry(0, 0, screen_width, screen_height)
            print("I: Forcing plot window to full screen.")
        except Exception:
            print("I: Could not force full screen (backend might not support it). Falling back to normal view.")

        plt.show()
    else:
        print("\nNo plots were generated.")

    print("\nScript finished.")