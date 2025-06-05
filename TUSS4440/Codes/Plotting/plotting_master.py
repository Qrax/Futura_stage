# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import detrend # Alleen nodig voor calculate_trace_stats (indien nog in gebruik)
from scipy.stats import t # Alleen nodig voor calculate_trace_stats (indien nog in gebruik)
# numpy.fft is nu in common_plot_utils
import importlib # Voor dynamisch importeren van plotmodules

"""
plotting_master.py – Compares data from one to six CSV files using various plot modes.
Modules for each plot mode are in the 'Plot_Modes' subfolder.
"""

# ---------------- PLOTMODUS – Select modes to run -----------------
PLOT_MODES_TO_RUN = [
    "RAW",
    "THRESHOLD",
    # "SUMMARY",
    # "DIFFERENCE", # Nog niet geïmplementeerd als module
    # "DEVIATION_FROM_LINEAR", # Nog niet geïmplementeerd als module
    # "FFT_OF_DEVIATION", # Nog niet geïmplementeerd als module
    # "POST_PEAK_PERIODOGRAM",
    # "INDIVIDUAL_RUN_PERIODOGRAM",
    #"PERIODOGRAM_ANALYSIS_STEPS",
    # "EXPLAIN_EXP_TREND"
]

# ----------------------------------------------------------------
# --------- General Settings -------------------------------------
TARGET_DATA_SUBFOLDER = os.path.join("..", "..", "data", "UltraSoon_Measurements") # Pas aan indien nodig
SAMPLE_TIME_DELTA_US = 4.63

_CSV_BASE_FILES = [
    "al_0mm_meta_1.csv",
    "al_0mm_meta_2.csv",
    "al_15mm_meta_1.csv",
    "al_15mm_2_meta_1.csv",
]
LABELS = [
    "Meting 1 (Defectloos)",
    "Meting 2 (Defectloos)",
    "Meting 1 (15mm)",
    "Meting 2 (15mm)",
]

CSV_FILES = [os.path.join(TARGET_DATA_SUBFOLDER, fname) for fname in _CSV_BASE_FILES]
if len(CSV_FILES) != len(LABELS): raise ValueError("Mismatch CSV_FILES/LABELS.")
if not 1 <= len(CSV_FILES) <= 6: raise ValueError(f"Need 1-6 CSV files, got {len(CSV_FILES)}.")

DEVICE_FILTER = "Master"
ADC_BITS = 12
V_REF = 3.3
MAX_RUNS = 100 # Max runs to *process* for THRESHOLD, and to *plot* for RAW, and *use* from matrix_list
THRESH = 1500
WINDOW_BEFORE = 0
WINDOW_AFTER = 1200

POST_PEAK_OFFSET_SAMPLES = 200
FIT_WINDOW_POST_PEAK = 1000 # Length of segment for FFT (Method A, B, PAS figs), and for Method C N_SEGMENT_LENGTH

DETREND_PERIODOGRAM = True
DETREND_TYPE = "exponential"  # "linear", "exponential", "none"

MIN_PERIOD_PLOT = 2
MAX_PERIOD_PLOT_ABS = 250
APPLY_FFT_WINDOW = False
TIME_UNIT = "us"
MAX_POINTS = None # Max points to plot for RAW mode (per run)
# ----------------------------------------------------------------

PLOT_COLORS = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']
PLOT_LINESTYLES = ['-', '--', ':', '-.', (0, (3, 1, 1, 1)), (0, (5, 10))]

# --- Core Helper Functions (kept in main or could be in a general utils.py) ---
def adc_to_v_main(adc_val): # Renamed to avoid conflict if imported directly elsewhere
    # Replace with your actual ADC to Voltage conversion if needed
    # For now, assuming ADC values are used directly as "Voltage"
    return adc_val

def load_and_prepare_data_main(csv_f, dev_filter, ts_factor_for_timeplot):
    try:
        df = pd.read_csv(csv_f)
    except FileNotFoundError:
        print(f"E: File '{csv_f}' not found."); return None
    df = df[df["Device"] == dev_filter].copy() # Ensure it's a copy
    if df.empty:
        print(f"E: No data for filter '{dev_filter}' in '{csv_f}'."); return None
    required_cols = ["Run", "Timestamp_us", "ADC_Value", "Device"]
    if not all(c in df.columns for c in required_cols):
        print(f"E: CSV '{csv_f}' is missing one or more required columns: {required_cols}. "
              f"Available columns: {df.columns.tolist()}"); return None
    df["Voltage"] = adc_to_v_main(df["ADC_Value"])
    df["TimePlot"] = df["Timestamp_us"] * ts_factor_for_timeplot # For RAW plot time axis
    df['DataSource'] = os.path.basename(csv_f) # Store base filename for reference
    return df

def get_summary_data_main(df_single_file, max_runs_to_process,
                           window_b, window_a, threshold_val):
    """
    Processes a single DataFrame to extract aligned runs for the summary cache.
    Returns: (empty_list_placeholder, aligned_run_matrix, valid_run_count)
    """
    # Ensure 'Run' and 'Voltage' columns exist
    if "Run" not in df_single_file.columns or "Voltage" not in df_single_file.columns:
        print(f"W: DataFrame for {df_single_file['DataSource'].iloc[0] if not df_single_file.empty else 'Unknown'} is missing 'Run' or 'Voltage' columns.")
        return [], [], 0

    unique_runs = sorted(df_single_file.Run.unique())
    runs_to_consider = unique_runs[:max_runs_to_process] if max_runs_to_process is not None else unique_runs

    aligned_matrix = []
    expected_segment_length = window_b + window_a + 1
    valid_runs_count = 0

    for run_id in runs_to_consider:
        run_data_group = df_single_file[df_single_file.Run == run_id].reset_index(drop=True)
        voltages_this_run = run_data_group.Voltage.values
        indices_above_thresh = np.where(voltages_this_run > threshold_val)[0]

        if not len(indices_above_thresh):
            continue # No trigger point in this run

        onset_index_in_run = indices_above_thresh[0]
        segment_start_abs = onset_index_in_run - window_b
        segment_stop_abs = onset_index_in_run + window_a + 1 # Slice stop is exclusive

        # Check if the full segment is within the run's data boundaries
        if segment_start_abs >= 0 and segment_stop_abs <= len(voltages_this_run):
            segment = voltages_this_run[segment_start_abs:segment_stop_abs]
            if len(segment) == expected_segment_length: # Double check length
                aligned_matrix.append(segment)
                valid_runs_count += 1
        # else: Segment would be truncated, so skip it for sum_cache consistency

    return [], aligned_matrix, valid_runs_count # Placeholder for first item for compatibility

def calculate_trace_stats_main(aligned_run_matrix, window_b, window_a):
    """ Calculates mean, std, sem from the matrix of aligned runs. """
    expected_len = window_b + window_a + 1
    if not aligned_run_matrix: # Empty list
        return np.zeros(expected_len), 0, np.zeros(expected_len), np.zeros(expected_len)

    aligned_data_np = np.array(aligned_run_matrix)
    N = aligned_data_np.shape[0] # Number of runs in the matrix

    if N == 0:
        return np.zeros(expected_len), 0, np.zeros(expected_len), np.zeros(expected_len)

    mean_trace = aligned_data_np.mean(axis=0)
    std_trace = np.zeros_like(mean_trace)
    sem_trace = np.zeros_like(mean_trace)

    if N > 1:
        std_trace = aligned_data_np.std(axis=0, ddof=1) # ddof=1 for sample standard deviation
        sem_trace = std_trace / np.sqrt(N)

    return mean_trace, N, sem_trace, std_trace

# --- PlotSettings Class (to pass settings to modules) ---
class PlotSettingsContainer:
    def __init__(self):
        # Populate with all necessary global settings
        self.TARGET_DATA_SUBFOLDER = TARGET_DATA_SUBFOLDER
        self.SAMPLE_TIME_DELTA_US = SAMPLE_TIME_DELTA_US
        self.CSV_FILES = CSV_FILES # List of full paths
        self.LABELS = LABELS
        self.DEVICE_FILTER = DEVICE_FILTER
        self.ADC_BITS = ADC_BITS
        self.V_REF = V_REF
        self.MAX_RUNS = MAX_RUNS
        self.THRESH = THRESH
        self.WINDOW_BEFORE = WINDOW_BEFORE
        self.WINDOW_AFTER = WINDOW_AFTER
        self.POST_PEAK_OFFSET_SAMPLES = POST_PEAK_OFFSET_SAMPLES
        self.FIT_WINDOW_POST_PEAK = FIT_WINDOW_POST_PEAK # Also N_SEGMENT_LENGTH for PAS
        self.DETREND_PERIODOGRAM = DETREND_PERIODOGRAM
        self.DETREND_TYPE = DETREND_TYPE
        self.MIN_PERIOD_PLOT = MIN_PERIOD_PLOT
        self.MAX_PERIOD_PLOT_ABS = MAX_PERIOD_PLOT_ABS
        self.APPLY_FFT_WINDOW = APPLY_FFT_WINDOW
        self.TIME_UNIT = TIME_UNIT
        self.MAX_POINTS = MAX_POINTS
        self.PLOT_COLORS = PLOT_COLORS
        self.PLOT_LINESTYLES = PLOT_LINESTYLES
        self.adc_to_v = adc_to_v_main # Pass the actual function

        # Derived settings
        if TIME_UNIT.lower() == "s":
            self.ts_factor_for_raw_timeplot, self.tu_raw_lbl = 1e-6, "s"
        elif TIME_UNIT.lower() == "ms":
            self.ts_factor_for_raw_timeplot, self.tu_raw_lbl = 1e-3, "ms"
        else: # Default to µs
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
        df_loaded_single = load_and_prepare_data_main(filepath, settings.DEVICE_FILTER, settings.ts_factor_for_raw_timeplot)
        if df_loaded_single is not None and not df_loaded_single.empty:
            dfs_loaded.append(df_loaded_single)
            actual_labels_used.append(settings.LABELS[i])
        else:
            print(f"W: Skipped '{filepath}' (load error or no data for filter).")

    if not dfs_loaded:
        print("E: No data loaded. Exiting.")
        exit()
    print(f"Data loaded for: {', '.join(actual_labels_used)}")

    # Pre-calculate summary data (aligned run matrices) for relevant plot modes
    summary_cache = {}
    needs_summary_calculation = any(
        mode in ["SUMMARY", "POST_PEAK_PERIODOGRAM", "INDIVIDUAL_RUN_PERIODOGRAM", "PERIODOGRAM_ANALYSIS_STEPS","EXPLAIN_EXP_TREND"]
        for mode in PLOT_MODES_TO_RUN
    ) or "THRESHOLD" in PLOT_MODES_TO_RUN # THRESHOLD title uses runs_count from sum_cache

    if needs_summary_calculation:
        print("Pre-calculating summary data (and aligned run matrices)...")
        for i, df_current_file in enumerate(dfs_loaded):
            label_current = actual_labels_used[i]
            # get_summary_data_main uses MAX_RUNS from settings for processing
            _, matrix_list_current, runs_count_current = get_summary_data_main(
                df_current_file, settings.MAX_RUNS,
                settings.WINDOW_BEFORE, settings.WINDOW_AFTER, settings.THRESH
            )
            mean_trace, N_for_mean, sem_trace, std_trace = calculate_trace_stats_main(
                matrix_list_current, settings.WINDOW_BEFORE, settings.WINDOW_AFTER
            )
            summary_cache[label_current] = {
                "matrix_list": matrix_list_current, # This respects MAX_RUNS from get_summary_data_main
                "runs_count": runs_count_current,   # Actual number of runs in matrix_list
                "mean_trace": mean_trace,
                "N_for_mean": N_for_mean,           # Should be same as runs_count_current
                "sem_trace": sem_trace,
                "std_trace": std_trace
            }
            if N_for_mean > 0:
                print(f"  '{label_current}': {N_for_mean} valid aligned runs stored in cache.")
            else:
                print(f"  W: '{label_current}': No valid aligned runs stored in cache.")
        print("Summary/Matrix pre-calculation done.")

    # --- Loop through selected plot modes and call their modules ---
    for plot_mode_name_iter in PLOT_MODES_TO_RUN:
        current_plot_mode_upper = plot_mode_name_iter.upper()
        print(f"\n--- Generating plot for mode: {current_plot_mode_upper} ---")

        module_name_to_import = f"Plot_Modes.plot_{plot_mode_name_iter.lower()}"
        plot_function_name_to_call = f"generate_plot_{plot_mode_name_iter.lower()}"

        try:
            plot_module = importlib.import_module(module_name_to_import)
            plot_function = getattr(plot_module, plot_function_name_to_call)

            # Call the specific plot function from the module
            # Pass dfs_loaded, actual_labels_used, settings object, summary_cache, and plt module
            # The plot function is responsible for creating and managing its own figure(s)
            # For PERIODOGRAM_ANALYSIS_STEPS, it might not return a fig but manage multiple.
            result_fig = plot_function(dfs_loaded, actual_labels_used, settings, summary_cache, plt)

            # if result_fig is None and current_plot_mode_upper != "PERIODOGRAM_ANALYSIS_STEPS":
            #     print(f"W({current_plot_mode_upper}): Plot function returned None, indicating no figure was generated or an issue occurred.")
            # elif result_fig is not None and result_fig != "periodogram_analysis_steps_completed":
            #     pass # Figure was returned, main loop will call plt.show()

        except ImportError:
            print(f"E: Could not import module '{module_name_to_import}'. "
                  f"Ensure Plot_Modes/plot_{plot_mode_name_iter.lower()}.py exists.")
        except AttributeError:
            print(f"E: Could not find function '{plot_function_name_to_call}' in module '{module_name_to_import}'.")
        except Exception as e:
            print(f"E: An error occurred while running plot mode {current_plot_mode_upper}: {e}")
            import traceback
            traceback.print_exc()

    # --- Show all generated plots ---
    if plt.get_fignums(): # Check if any figures were created and not closed
        print(f"\nDisplaying {len(plt.get_fignums())} plot(s). Close plot windows to exit.")
        plt.show()
    else:
        print("\nNo plots were generated or all were closed due to no data.")

    print("\nScript finished.")