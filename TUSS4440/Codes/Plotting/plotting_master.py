#--- START OF FILE plotting_master.py ---

# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import detrend  # Alleen nodig voor calculate_trace_stats (indien nog in gebruik)
from scipy.stats import t  # Alleen nodig voor calculate_trace_stats (indien nog in gebruik)
import importlib  # Voor dynamisch importeren van plotmodules

"""
plotting_master.py – Compares data from one to six CSV files using various plot modes.
Modules for each plot mode are in the 'Plot_Modes' subfolder.
"""

# ---------------- PLOTMODUS – Select modes to run -----------------
PLOT_MODES_TO_RUN = [
    # "RAW",
    # "THRESHOLD",
    "SUMMARY",
    # "DIFFERENCE", # Nog niet geïmplementeerd als module
    # "DEVIATION_FROM_LINEAR", # Nog niet geïmplementeerd als module
    # "FFT_OF_DEVIATION", # Nog niet geïmplementeerd als module
    "POST_PEAK_PERIODOGRAM",
    # "INDIVIDUAL_RUN_PERIOD
    # OGRAM",
    "PERIODOGRAM_ANALYSIS_STEPS",
    # "EXPLAIN_EXP_TREND"
]

# ----------------------------------------------------------------
# --------- General Settings -------------------------------------
TARGET_DATA_SUBFOLDER = os.path.join("..", "..", "data", "UltraSoon_Measurements")  # Pas aan indien nodig
SAMPLE_TIME_DELTA_US = 4.63

# --- MODIFIED: Lijst voor 8 bestanden ---
_CSV_BASE_FILES = [
    "al_0mm_3ms-final_meta_1.csv", "al_0mm_3ms_final_meta_2.csv",
    "AL_5mm_meta_1.csv", "AL_5mm_meta_2.csv",
    #"al_10mm_final_meta_1.csv", "al_10mm_final_meta_2.csv",
    "al_10mm_retry_meta_1.csv", "al_10mm_retry_meta_2.csv",
    "al_15mm_last_meta_1.csv", "al_15mm_last_meta_2.csv",
]
# --- MODIFIED: Labels voor 8 metingen (duplicatie gecorrigeerd) ---
LABELS = [
    "Meting 1 (Defectloos)", "Meting 2 (Defectloos)",
    "Meting 1 (5mm)", "Meting 2 (5mm)",
    "Meting 1 (10mm)", "Meting 2 (10mm)",
    "Meting 1 (15mm)", "Meting 2 (15mm)",
]

CSV_FILES = [os.path.join(TARGET_DATA_SUBFOLDER, fname.strip()) for fname in
             _CSV_BASE_FILES]
if len(CSV_FILES) != len(LABELS): raise ValueError("Mismatch CSV_FILES/LABELS.")

DEVICE_FILTER = "Master"
ADC_BITS = 12
V_REF = 3.3
MAX_RUNS = 100  # Max runs to *process*
THRESH = 1500
WINDOW_BEFORE = 0

# --- NEW: Dynamic Window Settings ---
USE_DYNAMIC_WINDOW_END = False
END_THRESH = 900
MAX_WINDOW_AFTER = 2000

POST_PEAK_OFFSET_SAMPLES = 20
FIT_WINDOW_POST_PEAK = 1400

DETREND_PERIODOGRAM = True
DETREND_TYPE = "linear"  # "linear", "exponential", "none"

MIN_PERIOD_PLOT = 2
MAX_PERIOD_PLOT_ABS = 400
APPLY_FFT_WINDOW = False
TIME_UNIT = "us"
MAX_POINTS = None
# ----------------------------------------------------------------

# --- MODIFIED: Use a colormap for more colors and solid lines for clarity ---
PLOT_COLORS = plt.get_cmap('tab20').colors
PLOT_LINESTYLES = ['-'] # Force solid lines for plots that use this setting

# --- NEW: Thesis-quality Plot Settings ---
# Set to True to enable larger fonts for all plots
THESIS_MODE = True
if THESIS_MODE:
    plt.rcParams.update({
        'font.size': 14,          # Main font size
        'axes.titlesize': 18,     # Title font size
        'axes.labelsize': 16,     # X and Y labels font size
        'xtick.labelsize': 12,    # X tick labels
        'ytick.labelsize': 12,    # Y tick labels
        'legend.fontsize': 12,    # Legend font size
        'figure.titlesize': 20    # Figure title (suptitle)
    })

# --- Core Helper Functions ---
def adc_to_v_main(adc_val):
    return adc_val


def load_and_prepare_data_main(csv_f, dev_filter, ts_factor_for_timeplot):
    try:
        df = pd.read_csv(csv_f)
    except FileNotFoundError:
        print(f"E: File '{csv_f}' not found.");
        return None
    df = df[df["Device"] == dev_filter].copy()
    if df.empty:
        print(f"E: No data for filter '{dev_filter}' in '{csv_f}'.");
        return None
    required_cols = ["Run", "Timestamp_us", "ADC_Value", "Device"]
    if not all(c in df.columns for c in required_cols):
        print(f"E: CSV '{csv_f}' is missing required columns.");
        return None
    df["Voltage"] = adc_to_v_main(df["ADC_Value"])
    df["TimePlot"] = df["Timestamp_us"] * ts_factor_for_timeplot
    df['DataSource'] = os.path.basename(csv_f)
    return df


def get_summary_data_main(df_single_file, max_runs_to_process,
                          window_b, max_window_a, threshold_val,
                          use_dynamic_end, end_threshold_val):
    """
    Processes a single DataFrame to extract aligned runs.
    If use_dynamic_end is True, each run can have a different length.
    Returns a list of 1D numpy arrays (the traces).
    """
    if "Run" not in df_single_file.columns or "Voltage" not in df_single_file.columns:
        return [], 0  # Return empty list and zero count

    unique_runs = sorted(df_single_file.Run.unique())
    runs_to_consider = unique_runs[:max_runs_to_process] if max_runs_to_process is not None else unique_runs

    aligned_traces = []  # Will hold numpy arrays of potentially different lengths

    for run_id in runs_to_consider:
        run_data_group = df_single_file[df_single_file.Run == run_id].reset_index(drop=True)
        voltages_this_run = run_data_group.Voltage.values
        indices_above_thresh = np.where(voltages_this_run > threshold_val)[0]

        if not len(indices_above_thresh):
            continue

        onset_index_in_run = indices_above_thresh[0]
        segment_start_abs = onset_index_in_run - window_b

        if use_dynamic_end:
            search_area = voltages_this_run[onset_index_in_run:]
            indices_below_end_thresh = np.where(search_area < end_threshold_val)[0]
            if len(indices_below_end_thresh) > 0:
                end_offset = indices_below_end_thresh[0]
                segment_stop_abs = onset_index_in_run + end_offset + 1
            else:
                segment_stop_abs = min(len(voltages_this_run), onset_index_in_run + max_window_a + 1)
        else:
            segment_stop_abs = onset_index_in_run + max_window_a + 1

        if segment_start_abs >= 0 and segment_stop_abs <= len(voltages_this_run):
            segment = voltages_this_run[segment_start_abs:segment_stop_abs]
            if len(segment) > 0:
                aligned_traces.append(segment)

    return aligned_traces, len(aligned_traces)


def calculate_trace_stats_main(aligned_traces):
    """
    Calculates mean, std, sem from a list of potentially variable-length traces.
    """
    if not aligned_traces:
        return np.array([]), 0, np.array([]), np.array([])

    N = len(aligned_traces)
    if N == 0:
        return np.array([]), 0, np.array([]), np.array([])

    max_len = max(len(trace) for trace in aligned_traces)
    padded_matrix = np.full((N, max_len), np.nan)

    for i, trace in enumerate(aligned_traces):
        padded_matrix[i, :len(trace)] = trace

    mean_trace = np.nanmean(padded_matrix, axis=0)
    std_trace = np.zeros_like(mean_trace)
    sem_trace = np.zeros_like(mean_trace)

    if N > 1:
        std_trace = np.nanstd(padded_matrix, axis=0, ddof=1)
        n_per_point = np.sum(~np.isnan(padded_matrix), axis=0)
        valid_n = np.where(n_per_point > 1, n_per_point, np.nan)
        sem_trace = std_trace / np.sqrt(valid_n)
        sem_trace = np.nan_to_num(sem_trace, nan=0.0)

    return mean_trace, N, sem_trace, std_trace


# --- PlotSettings Class (to pass settings to modules) ---
class PlotSettingsContainer:
    def __init__(self):
        # Populate with all necessary global settings
        self.TARGET_DATA_SUBFOLDER = TARGET_DATA_SUBFOLDER
        self.SAMPLE_TIME_DELTA_US = SAMPLE_TIME_DELTA_US
        self.CSV_FILES = CSV_FILES
        self.LABELS = LABELS
        self.DEVICE_FILTER = DEVICE_FILTER
        self.ADC_BITS = ADC_BITS
        self.V_REF = V_REF
        self.MAX_RUNS = MAX_RUNS
        self.THRESH = THRESH
        self.WINDOW_BEFORE = WINDOW_BEFORE
        self.USE_DYNAMIC_WINDOW_END = USE_DYNAMIC_WINDOW_END
        self.END_THRESH = END_THRESH
        self.WINDOW_AFTER = MAX_WINDOW_AFTER

        self.POST_PEAK_OFFSET_SAMPLES = POST_PEAK_OFFSET_SAMPLES
        self.FIT_WINDOW_POST_PEAK = FIT_WINDOW_POST_PEAK
        self.DETREND_PERIODOGRAM = DETREND_PERIODOGRAM
        self.DETREND_TYPE = DETREND_TYPE
        self.MIN_PERIOD_PLOT = MIN_PERIOD_PLOT
        self.MAX_PERIOD_PLOT_ABS = MAX_PERIOD_PLOT_ABS
        self.APPLY_FFT_WINDOW = APPLY_FFT_WINDOW
        self.TIME_UNIT = TIME_UNIT
        self.MAX_POINTS = MAX_POINTS
        self.PLOT_COLORS = PLOT_COLORS
        self.PLOT_LINESTYLES = PLOT_LINESTYLES
        self.adc_to_v = adc_to_v_main

        if TIME_UNIT.lower() == "s":
            self.ts_factor_for_raw_timeplot, self.tu_raw_lbl = 1e-6, "s"
        elif TIME_UNIT.lower() == "ms":
            self.ts_factor_for_raw_timeplot, self.tu_raw_lbl = 1e-3, "ms"
        else:
            self.ts_factor_for_raw_timeplot, self.tu_raw_lbl = 1.0, "µs"


# --- Main script execution ---
if __name__ == "__main__":
    settings = PlotSettingsContainer()

    print("Loading data...")
    dfs_loaded, actual_labels_used = [], []
    for i, filepath in enumerate(settings.CSV_FILES):
        if not os.path.exists(filepath):
            print(f"W: File '{filepath}' not found. Skipping.")
            continue
        df_loaded_single = load_and_prepare_data_main(filepath, settings.DEVICE_FILTER,
                                                      settings.ts_factor_for_raw_timeplot)
        if df_loaded_single is not None and not df_loaded_single.empty:
            dfs_loaded.append(df_loaded_single)
            actual_labels_used.append(settings.LABELS[i])
        else:
            print(f"W: Skipped '{filepath}' (load error or no data for filter).")

    if not dfs_loaded:
        print("E: No data loaded. Exiting.")
        exit()
    print(f"Data loaded for: {', '.join(actual_labels_used)}")

    summary_cache = {}
    needs_summary_calculation = any(mode in PLOT_MODES_TO_RUN for mode in
                                    ["SUMMARY", "POST_PEAK_PERIODOGRAM", "INDIVIDUAL_RUN_PERIODOGRAM",
                                     "PERIODOGRAM_ANALYSIS_STEPS", "EXPLAIN_EXP_TREND", "THRESHOLD"])

    if needs_summary_calculation:
        print("Pre-calculating summary data (aligned run traces)...")
        for i, df_current_file in enumerate(dfs_loaded):
            label_current = actual_labels_used[i]

            matrix_list_current, runs_count_current = get_summary_data_main(
                df_current_file, settings.MAX_RUNS,
                settings.WINDOW_BEFORE, settings.WINDOW_AFTER, settings.THRESH,
                settings.USE_DYNAMIC_WINDOW_END, settings.END_THRESH
            )
            mean_trace, N_for_mean, sem_trace, std_trace = calculate_trace_stats_main(
                matrix_list_current
            )

            summary_cache[label_current] = {
                "matrix_list": matrix_list_current, "runs_count": runs_count_current,
                "mean_trace": mean_trace, "N_for_mean": N_for_mean,
                "sem_trace": sem_trace, "std_trace": std_trace
            }
            if N_for_mean > 0:
                print(f"  '{label_current}': {N_for_mean} valid aligned runs stored in cache.")
            else:
                print(f"  W: '{label_current}': No valid aligned runs stored in cache.")
        print("Summary/Matrix pre-calculation done.")

    for plot_mode_name_iter in PLOT_MODES_TO_RUN:
        current_plot_mode_upper = plot_mode_name_iter.upper()
        print(f"\n--- Generating plot for mode: {current_plot_mode_upper} ---")

        module_name_to_import = f"Plot_Modes.plot_{plot_mode_name_iter.lower()}"
        plot_function_name_to_call = f"generate_plot_{plot_mode_name_iter.lower()}"

        try:
            plot_module = importlib.import_module(module_name_to_import)
            plot_function = getattr(plot_module, plot_function_name_to_call)
            result_fig = plot_function(dfs_loaded, actual_labels_used, settings, summary_cache, plt)

        except ImportError:
            print(f"E: Could not import module '{module_name_to_import}'.")
        except AttributeError:
            print(f"E: Could not find function '{plot_function_name_to_call}' in module '{module_name_to_import}'.")
        except Exception as e:
            print(f"E: An error occurred while running plot mode {current_plot_mode_upper}: {e}")
            import traceback

            traceback.print_exc()

    if plt.get_fignums():
        print(f"\nDisplaying {len(plt.get_fignums())} plot(s). Close plot windows to exit.")
        plt.show()
    else:
        print("\nNo plots were generated or all were closed due to no data.")

    print("\nScript finished.")
#--- END OF FILE plotting_master.py ---