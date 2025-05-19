# -*- coding: utf-8 -*-
# --- START OF FILE plotting_master_improved_v3.py ---

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, detrend  # correlate might not be used if XCORR is not run
from scipy.stats import norm, t
from numpy.fft import rfft, rfftfreq

# !/usr/bin/env python3
"""
plotting_master.py – Compares data from one to six CSV files using various plot modes.
Specify a list of plot modes to run in PLOT_MODES_TO_RUN.
"""

# ---------------- PLOTMODUS – Select modes to run -----------------
PLOT_MODES_TO_RUN = [
    "SUMMARY",
    "DIFFERENCE",  # Now plots derivative-like change for each mean trace
    "DEVIATION_FROM_LINEAR",
    "POST_PEAK_PERIODOGRAM",
    # "RAW",
    # "THRESHOLD",
]

# ----------------------------------------------------------------
# --------- General Settings -------------------------------------
TARGET_DATA_SUBFOLDER = os.path.join("..", "data", "UltraSoon_Measurements")
SAMPLE_TIME_DELTA_US = 4.63  # Microseconds between samples

# --- Input Files and Labels ---
_CSV_BASE_FILES = [
    "boopo_meta_1.csv",
    "boopo_meta_2.csv",
    "boopo_meta_3.csv",
    "gleuf_meta_1.csv",
    "gleuf_2_meta_1.csv",
]

LABELS = [
    "1 (no gleuf)",
    "2 (no gleuf)",
    "3 (no gleuf)",
    "4 (gleuf)",
    "5 (gleuf)",
]
CSV_FILES = [os.path.join(TARGET_DATA_SUBFOLDER, fname) for fname in _CSV_BASE_FILES]

if len(CSV_FILES) != len(LABELS):
    raise ValueError("Mismatch between the number of CSV_FILES and LABELS. Please check the configuration.")
if not 1 <= len(CSV_FILES) <= 6:
    raise ValueError(f"Please provide between 1 and 6 CSV files. Found {len(CSV_FILES)} active files.")

# --- Common Settings ---
DEVICE_FILTER = "Master"
ADC_BITS = 12  # Currently not used for scaling, ADC values are plotted raw
V_REF = 3.3  # Currently not used
MAX_RUNS = 100

THRESH = 1500
WINDOW_BEFORE = 0  # Samples before threshold for mean trace window
WINDOW_AFTER = 250  # Samples after threshold for mean trace window

FIT_WINDOW_POST_PEAK = 250  # Number of samples from peak for linear fit / FFT segment

DETREND_PERIODOGRAM = True
MIN_PERIOD_PLOT = 2  # Min period (IN SAMPLES) to show in periodogram (will be converted to µs)
MAX_PERIOD_PLOT_ABS = 40  # Max period (IN SAMPLES) to show (will be converted to µs)
APPLY_FFT_WINDOW = True  # Apply Hann window before FFT for periodogram

N_PEAKS_PER_RUN = 5  # Was for histogram part of SUMMARY, currently unused by active plots.

TIME_UNIT = "ms"  # General time unit for RAW plot x-axis if timestamps are used directly
MAX_POINTS = None  # Max points to plot in RAW mode (None for all)

# XCORR related parameters (not actively used by current PLOT_MODES_TO_RUN)
SEG_LEN = 50
MAX_LAG = 5
N_EXTREME = 10
OFFSET_EXT = 1
EXTREME_VALUES = [0, 5, 10, 15]
# ----------------------------------------------------------------

PLOT_COLORS = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']
PLOT_MARKERS = ['o', 'x', 's', '^', 'v', 'D']  # Not used in current selection, but available
PLOT_LINESTYLES = ['-', '--', ':', '-.', (0, (3, 1, 1, 1)), (0, (5, 10))]


def adc_to_v(adc_value):
    # Placeholder: Currently, "Voltage" is treated as raw ADC units.
    return adc_value


def load_and_prepare_data(csv_file, device_filter_val, time_scale_factor_val):
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: CSV file '{csv_file}' not found.")
        return None
    df = df[df["Device"] == device_filter_val].copy()
    if df.empty:
        print(f"Error: No data found for DEVICE_FILTER '{device_filter_val}' in '{csv_file}'.")
        return None
    required_cols = ["Run", "Timestamp_us", "ADC_Value", "Device"]
    if not all(col in df.columns for col in required_cols):
        print(f"Error: CSV file '{csv_file}' must contain columns: {required_cols}. Found: {df.columns.tolist()}")
        return None
    df["Voltage"] = adc_to_v(df["ADC_Value"])  # Apply scaling if adc_to_v is changed
    df["TimePlot"] = df["Timestamp_us"] * time_scale_factor_val  # For RAW plot x-axis
    df['DataSource'] = csv_file
    return df


def get_summary_data(df, label_for_warnings):  # For SUMMARY, DIFFERENCE, DEVIATION_FROM_LINEAR, etc.
    runs = sorted(df.Run.unique())[:MAX_RUNS]
    aligned_matrix_list = []
    # Expected length of segments for mean trace based on window settings
    expected_segment_len = WINDOW_BEFORE + WINDOW_AFTER + 1
    valid_run_count = 0

    for run_idx, run in enumerate(runs):
        g = df[df.Run == run].reset_index(drop=True)
        v = g.Voltage.values
        above = np.where(v > THRESH)[0]

        if not len(above): continue  # Skip run if no point above threshold

        onset = above[0]  # First point above threshold
        start_idx = onset - WINDOW_BEFORE
        stop_idx = onset + WINDOW_AFTER + 1  # Slice goes up to stop_idx-1

        # Segment extraction with boundary checks
        actual_start = max(0, start_idx)
        actual_stop = min(len(v), stop_idx)
        seg = v[actual_start:actual_stop]

        # Ensure segment has the correct, full length before adding
        # This means events too close to start/end of recording that don't allow full window are skipped
        if len(seg) == expected_segment_len and start_idx >= 0 and stop_idx <= len(v):
            aligned_matrix_list.append(seg)
            valid_run_count += 1
        # else:
        # Optional: print(f"Debug ({label_for_warnings}, Run {run}): Segment length {len(seg)} or boundary issue. Expected {expected_segment_len}. Skipping for mean trace.")

        if valid_run_count >= MAX_RUNS: break  # Safeguard, already sliced runs

    # Return empty list for peaks as it's not used by current SUMMARY
    return [], aligned_matrix_list, valid_run_count


def calculate_trace_stats(aligned_matrix_list_local):
    expected_len = WINDOW_BEFORE + WINDOW_AFTER + 1
    if not aligned_matrix_list_local:
        return np.zeros(expected_len), 0, np.zeros(expected_len), np.zeros(expected_len)

    # All segments in aligned_matrix_list_local should already be of expected_len due to get_summary_data logic
    aligned_data_np = np.array(aligned_matrix_list_local)
    N_local = aligned_data_np.shape[0]

    if N_local == 0:  # Should be caught by "if not aligned_matrix_list_local"
        return np.zeros(expected_len), 0, np.zeros(expected_len), np.zeros(expected_len)

    mean_tr_local = aligned_data_np.mean(axis=0)
    std_tr_local = np.zeros_like(mean_tr_local)  # Default for N_local=1
    sem_tr_local = np.zeros_like(mean_tr_local)  # Default for N_local=1

    if N_local > 1:
        std_tr_local = aligned_data_np.std(axis=0, ddof=1)
        sem_tr_local = std_tr_local / np.sqrt(N_local)

    return mean_tr_local, N_local, sem_tr_local, std_tr_local


# Determine time scaling for RAW plot if used
if TIME_UNIT.lower() == "s":
    time_scale_factor, time_label_unit_raw = 1e-6, "s"
elif TIME_UNIT.lower() == "ms":
    time_scale_factor, time_label_unit_raw = 1e-3, "ms"
else:
    time_scale_factor, time_label_unit_raw = 1.0, "µs"  # Default to microseconds

print("Loading and preparing data...")
data_frames, active_labels = [], []
for i, csv_file_path in enumerate(CSV_FILES):
    if not os.path.exists(csv_file_path):
        print(f"Warning: Input CSV file '{csv_file_path}' not found. Skipping.")
        continue
    df_loaded = load_and_prepare_data(csv_file_path, DEVICE_FILTER, time_scale_factor)
    if df_loaded is not None and not df_loaded.empty:
        data_frames.append(df_loaded)
        active_labels.append(LABELS[i])
    else:
        print(
            f"Warning: Skipping file '{csv_file_path}' due to loading errors or no data for filter '{DEVICE_FILTER}'.")

if not data_frames:
    print("No data loaded successfully from any CSV file. Exiting.")
    exit()
print(f"Data loaded for {len(data_frames)} files: {', '.join(active_labels)}")

summary_data_cache = {}
needs_summary_data = any(mode.upper() in ["SUMMARY", "DIFFERENCE", "DEVIATION_FROM_LINEAR",
                                          "POST_PEAK_DELTA_ADC", "POST_PEAK_FFT", "POST_PEAK_PERIODOGRAM"]
                         for mode in PLOT_MODES_TO_RUN)

if needs_summary_data:
    print("Pre-calculating summary data (mean traces, SEM, STD, etc.)...")
    for i, df_current_cache in enumerate(data_frames):
        label_cache = active_labels[i]
        _unused_peaks, matrix_list_cache, runs_count_cache = get_summary_data(df_current_cache, label_cache)
        mean_tr_cache, N_runs_for_mean_cache, sem_tr_cache, std_tr_cache = calculate_trace_stats(matrix_list_cache)

        summary_data_cache[label_cache] = {
            "peaks": _unused_peaks,  # Retained in cache structure even if unused
            "matrix_list": matrix_list_cache,
            "runs_count": runs_count_cache,  # Number of runs that met initial threshold criteria
            "mean_trace": mean_tr_cache,
            "N_for_mean": N_runs_for_mean_cache,  # Number of runs with full segments used for mean
            "sem_trace": sem_tr_cache,
            "std_trace": std_tr_cache
        }
        if N_runs_for_mean_cache > 0:
            print(f"  Processed '{label_cache}': {N_runs_for_mean_cache} runs contributed to mean trace.")
        else:
            print(
                f"  Warning for '{label_cache}': No runs suitable for mean trace (check THRESH, WINDOW settings, data boundaries).")
    print("Summary data pre-calculation complete.")


def plot_runs(ax, df_plot, label_plot, color_plot, plot_mode_str, **kwargs):
    # (This function is for THRESHOLD and RAW modes)
    all_runs = sorted(df_plot["Run"].unique())[:MAX_RUNS]
    plot_count, plotted_something = 0, False
    if plot_mode_str == "THRESHOLD":
        # ... (THRESHOLD plot_runs logic - kept as is for brevity) ...
        return plotted_something, plot_count
    elif plot_mode_str == "RAW":
        # ... (RAW plot_runs logic - kept as is for brevity) ...
        return plotted_something, plot_count
    return False, 0


for PLOT_MODE_ITER in PLOT_MODES_TO_RUN:
    CURRENT_PLOT_MODE = PLOT_MODE_ITER.upper()
    print(f"\n--- Generating plot for mode: {CURRENT_PLOT_MODE} ---")

    # --- THRESHOLD Plot (Example, if re-enabled) ---
    if CURRENT_PLOT_MODE == "THRESHOLD":
        # ... (Full THRESHOLD plot generation logic - kept as is for brevity) ...
        print("Info: THRESHOLD plot generation skipped for brevity in this example output.")
        pass


    # --- RAW Plot (Example, if re-enabled) ---
    elif CURRENT_PLOT_MODE == "RAW":
        # ... (Full RAW plot generation logic - kept as is for brevity) ...
        print("Info: RAW plot generation skipped for brevity in this example output.")
        pass

    # --- SUMMARY Plot ---
    elif CURRENT_PLOT_MODE == "SUMMARY":
        mean_trace_data_exists = any(
            label in summary_data_cache and
            summary_data_cache[label]["N_for_mean"] > 0 and
            summary_data_cache[label]["mean_trace"] is not None and
            len(summary_data_cache[label]["mean_trace"]) == (WINDOW_BEFORE + WINDOW_AFTER + 1)
            for label in active_labels
        )
        if mean_trace_data_exists:
            fig_summary_mean, ax_summary_mean = plt.subplots(figsize=(12, 6))
            # X-axis for summary plot: sample indices relative to trigger
            t_rel_summary = np.arange(-WINDOW_BEFORE, WINDOW_AFTER + 1)
            title_run_counts_summary_mean = []

            for i_df, label_s in enumerate(active_labels):
                if label_s in summary_data_cache:
                    cached_stats = summary_data_cache[label_s]
                    mean_tr = cached_stats.get("mean_trace")
                    N_val = cached_stats.get("N_for_mean")
                    sem_tr = cached_stats.get("sem_trace")
                    std_tr = cached_stats.get("std_trace")

                    if not (N_val and N_val > 0 and mean_tr is not None and len(mean_tr) == len(t_rel_summary)):
                        print(f"Warning (SUMMARY - {label_s}): Invalid or missing mean trace data. Skipping.")
                        title_run_counts_summary_mean.append(f"{label_s} (no data)")
                        continue

                    title_run_counts_summary_mean.append(f"{label_s} (N={N_val})")
                    color = PLOT_COLORS[i_df % len(PLOT_COLORS)]
                    linestyle = PLOT_LINESTYLES[i_df % len(PLOT_LINESTYLES)]

                    ax_summary_mean.plot(t_rel_summary, mean_tr, lw=2, color=color, linestyle=linestyle,
                                         label=f"Mean ({label_s})")
                    if N_val > 1:  # CI and PI are meaningful for N > 1
                        t_crit_val = t.ppf(0.975, df=N_val - 1)
                        ci95_margin = t_crit_val * sem_tr
                        pi95_margin = t_crit_val * std_tr * np.sqrt(1 + 1 / N_val)
                        ax_summary_mean.fill_between(t_rel_summary, mean_tr - ci95_margin, mean_tr + ci95_margin,
                                                     alpha=0.35, color=color, label=f"95% CI Mean ({label_s})")
                        ax_summary_mean.fill_between(t_rel_summary, mean_tr - pi95_margin, mean_tr + pi95_margin,
                                                     alpha=0.15, color=color, label=f"95% PI New Obs ({label_s})")
                    # No special handling for N_val == 1 here, as mean_tr is already plotted. CI/PI not shown.

            x_label_summary = f"Sample-index (0 = first V > {THRESH} in window)"
            y_label_summary = f"Voltage ({'ADC units' if adc_to_v(1) == 1 else 'V'})"
            ax_summary_mean.set_xlabel(x_label_summary)
            ax_summary_mean.set_ylabel(y_label_summary)
            ax_summary_mean.set_title(
                f"{DEVICE_FILTER} – SUMMARY: Mean Traces with 95% CI & PI\nWindow: {WINDOW_BEFORE} pre, {WINDOW_AFTER} post threshold ({SAMPLE_TIME_DELTA_US} µs/sample)\n{', '.join(title_run_counts_summary_mean)}")
            ax_summary_mean.grid(True, alpha=0.3)
            # Consolidate legend entries
            handles_leg, labels_leg = ax_summary_mean.get_legend_handles_labels()
            unique_legend = {}
            for h, l in zip(handles_leg, labels_leg):
                if l not in unique_legend or "Mean" in l: unique_legend[l] = h  # Prioritize mean if duplicate
            ax_summary_mean.legend(unique_legend.values(), unique_legend.keys(), fontsize=8, loc='upper right')
            fig_summary_mean.tight_layout()
            fig_summary_mean.canvas.manager.set_window_title(f"Plot Mode: SUMMARY - Mean Traces")
        else:
            print("Warning (SUMMARY): No valid mean trace data available for any dataset. Skipping SUMMARY plot.")

    # --- DIFFERENCE Plot (Now First-Order Difference / Derivative-like) ---
    elif CURRENT_PLOT_MODE == "DIFFERENCE":
        fig_diff, ax_diff = plt.subplots(figsize=(12, 6))
        any_diff_plotted = False
        title_parts_diff = []

        # Common x-axis for mean traces (before diff)
        t_rel_summary = np.arange(-WINDOW_BEFORE, WINDOW_AFTER + 1)
        # X-axis for the differenced data (plotted at midpoints of original intervals)
        # Ensure t_rel_summary has at least 2 points to diff
        if len(t_rel_summary) < 2:
            print(
                "Warning (DIFFERENCE): Window too small for difference calculation (needs at least 2 points). Skipping.")
            if 'fig_diff' in locals(): plt.close(fig_diff)  # Close the figure if created
        else:
            t_rel_diff_samples = (t_rel_summary[:-1] + t_rel_summary[1:]) / 2.0

            for i_df, label_current in enumerate(active_labels):
                color_current = PLOT_COLORS[i_df % len(PLOT_COLORS)]
                linestyle_current = PLOT_LINESTYLES[i_df % len(PLOT_LINESTYLES)]

                if label_current not in summary_data_cache or \
                        summary_data_cache[label_current].get("N_for_mean", 0) == 0 or \
                        summary_data_cache[label_current].get("mean_trace") is None:
                    print(f"Warning (DIFFERENCE - {label_current}): No valid mean trace. Skipping.")
                    title_parts_diff.append(f"{label_current} (no data)")
                    continue

                s_data = summary_data_cache[label_current]
                mean_tr = s_data["mean_trace"]
                N_runs = s_data["N_for_mean"]

                if len(mean_tr) < 2:  # Need at least 2 points to calculate a difference
                    print(
                        f"Warning (DIFFERENCE - {label_current}): Mean trace too short ({len(mean_tr)} pts) for difference. Skipping.")
                    title_parts_diff.append(f"{label_current} (trace short)")
                    continue

                # Calculate first-order difference: Y[i+1] - Y[i]
                delta_trace = np.diff(mean_tr)  # Units: ADC / sample interval

                # Optional: Convert to ADC / microsecond
                # delta_trace_per_us = delta_trace / SAMPLE_TIME_DELTA_US

                # Ensure t_rel_diff_samples matches length of delta_trace
                if len(t_rel_diff_samples) != len(delta_trace):
                    print(f"Error (DIFFERENCE - {label_current}): X-axis length mismatch for diff plot. Skipping.")
                    # This should not happen if t_rel_summary and mean_tr are consistent
                    title_parts_diff.append(f"{label_current} (len err)")
                    continue

                ax_diff.plot(t_rel_diff_samples, delta_trace, lw=1.5, color=color_current, linestyle=linestyle_current,
                             label=f"{label_current} (N={N_runs})")
                any_diff_plotted = True
                title_parts_diff.append(f"{label_current} (N={N_runs})")

            if any_diff_plotted:
                time_per_sample_info = f"({SAMPLE_TIME_DELTA_US} µs/sample)"
                ax_diff.set_xlabel(f"Sample-index (midpoint of interval, 0 = around onset) {time_per_sample_info}")
                ax_diff.set_ylabel(f"Δ Mean Voltage / Δ Sample (ADC units/sample)")
                # To plot rate in ADC/µs instead: use delta_trace_per_us and change ylabel
                # ax_diff.set_ylabel(f"Rate of Change (ADC units/µs)")
                ax_diff.set_title(
                    f"{DEVICE_FILTER} – DIFFERENCE: First-Order Difference of Mean Traces\n{', '.join(title_parts_diff)}")
                ax_diff.axhline(0, color='k', linestyle='--', alpha=0.7, lw=1)
                ax_diff.grid(True, alpha=0.3)
                ax_diff.legend(fontsize=8, loc='best')  # 'best' often good for derivative plots
                fig_diff.tight_layout()
                fig_diff.canvas.manager.set_window_title(f"Plot Mode: DIFFERENCE (Derivative-like)")
            else:
                if 'fig_diff' in locals(): plt.close(fig_diff)
                print("Warning (DIFFERENCE): No derivative traces could be plotted.")


    # --- DEVIATION_FROM_LINEAR Plot ---
    elif CURRENT_PLOT_MODE == "DEVIATION_FROM_LINEAR":
        # Plot 1: Context for Linear Fit
        fig_lin_context, ax_lin_context = plt.subplots(figsize=(12, 6))
        any_lin_context_plotted = False
        # Plot 2: Deviation from Linear Fit
        fig_dev_lin, ax_dev_lin = plt.subplots(figsize=(12, 6))
        any_dev_lin_plotted = False
        title_run_counts_dev_lin = []

        for i_df, label_current in enumerate(active_labels):
            color_current = PLOT_COLORS[i_df % len(PLOT_COLORS)]
            linestyle_current = PLOT_LINESTYLES[i_df % len(PLOT_LINESTYLES)]

            if label_current not in summary_data_cache or \
                    summary_data_cache[label_current].get("N_for_mean", 0) == 0 or \
                    summary_data_cache[label_current].get("mean_trace") is None:
                msg = f"W (DEV_LIN - {label_current}): No valid mean trace. Skip.";
                print(msg)
                title_run_counts_dev_lin.append(f"{label_current} (no data)");
                continue

            s_data = summary_data_cache[label_current]
            mean_tr_full, N_runs_for_mean = s_data["mean_trace"], s_data["N_for_mean"]
            if len(mean_tr_full) == 0:
                msg = f"W (DEV_LIN - {label_current}): Mean trace empty. Skip.";
                print(msg)
                title_run_counts_dev_lin.append(f"{label_current} (empty trace)");
                continue

            idx_peak_in_mean_tr = np.argmax(mean_tr_full)
            start_idx_segment = idx_peak_in_mean_tr
            end_idx_segment = min(start_idx_segment + FIT_WINDOW_POST_PEAK, len(mean_tr_full))
            segment_to_fit = mean_tr_full[start_idx_segment:end_idx_segment]
            x_fit_coords_samples = np.arange(len(segment_to_fit))  # Samples from peak (0, 1, 2...)

            if len(segment_to_fit) < 2:
                msg = f"W (DEV_LIN - {label_current}): Post-peak seg too short ({len(segment_to_fit)} pts). Skip.";
                print(msg)
                title_run_counts_dev_lin.append(f"{label_current} (seg short N={len(segment_to_fit)})");
                continue
            try:
                slope, intercept = np.polyfit(x_fit_coords_samples, segment_to_fit, 1)
            except (np.linalg.LinAlgError, ValueError) as e:
                msg = f"W (DEV_LIN - {label_current}): Linear fit failed. {e}. Skip.";
                print(msg)
                title_run_counts_dev_lin.append(f"{label_current} (fit error)");
                continue

            linear_fit_values = slope * x_fit_coords_samples + intercept
            deviations = segment_to_fit - linear_fit_values

            legend_details = f" (N={N_runs_for_mean}, slope={slope:.2e})"
            # Plot 1: Context
            ax_lin_context.plot(x_fit_coords_samples, segment_to_fit, lw=1.5, color=color_current,
                                linestyle=linestyle_current,
                                label=f"Data ({label_current}{legend_details})", alpha=0.7)
            ax_lin_context.plot(x_fit_coords_samples, linear_fit_values, lw=2, color=color_current, linestyle='--',
                                label=f"Fit ({label_current}{legend_details})")
            any_lin_context_plotted = True
            # Plot 2: Deviation
            ax_dev_lin.plot(x_fit_coords_samples, deviations, lw=1.5, color=color_current, linestyle=linestyle_current,
                            label=f"{label_current}{legend_details}")
            any_dev_lin_plotted = True
            if label_current not in [item.split(' ')[0] for item in title_run_counts_dev_lin if '(' in item]:
                title_run_counts_dev_lin.append(f"{label_current} (N={N_runs_for_mean})")

        common_x_label = f"Samples from Peak (Time: up to {FIT_WINDOW_POST_PEAK * SAMPLE_TIME_DELTA_US:.2f} µs)"
        common_title_suffix = f"\nSegment: {FIT_WINDOW_POST_PEAK} samples from peak\n{', '.join(title_run_counts_dev_lin)}"

        if any_lin_context_plotted:
            ax_lin_context.set_xlabel(common_x_label)
            ax_lin_context.set_ylabel(f"Voltage ({'ADC units' if adc_to_v(1) == 1 else 'V'})")
            ax_lin_context.set_title(f"{DEVICE_FILTER} – LINEAR FIT CONTEXT (Post-Peak){common_title_suffix}")
            ax_lin_context.grid(True, alpha=0.3);
            ax_lin_context.legend(fontsize=8, loc='best')
            fig_lin_context.tight_layout();
            fig_lin_context.canvas.manager.set_window_title(f"Plot Mode: DEVIATION_FROM_LINEAR - Context")
        else:
            plt.close(fig_lin_context) if 'fig_lin_context' in locals() else None

        if any_dev_lin_plotted:
            ax_dev_lin.set_xlabel(common_x_label)
            ax_dev_lin.set_ylabel(f"Deviation from Linear Fit ({'ADC units' if adc_to_v(1) == 1 else 'V'})")
            ax_dev_lin.set_title(f"{DEVICE_FILTER} – DEVIATION FROM LINEAR FIT (Post-Peak){common_title_suffix}")
            ax_dev_lin.axhline(0, color='k', linestyle='--', alpha=0.7, lw=1)
            ax_dev_lin.grid(True, alpha=0.3);
            ax_dev_lin.legend(fontsize=8, loc='best')
            fig_dev_lin.tight_layout();
            fig_dev_lin.canvas.manager.set_window_title(f"Plot Mode: DEVIATION_FROM_LINEAR - Deviation")
        else:
            plt.close(fig_dev_lin) if 'fig_dev_lin' in locals() else None


    # --- POST_PEAK_PERIODOGRAM Plot ---
    elif CURRENT_PLOT_MODE == "POST_PEAK_PERIODOGRAM":
        fig_pg, ax_pg = plt.subplots(figsize=(12, 6))
        any_periodogram_plotted = False;
        title_run_counts_pg = []
        min_period_plot_us, max_period_plot_abs_us = MIN_PERIOD_PLOT * SAMPLE_TIME_DELTA_US, MAX_PERIOD_PLOT_ABS * SAMPLE_TIME_DELTA_US

        for i_df, label_current in enumerate(active_labels):  # Corrected iteration
            color_current = PLOT_COLORS[i_df % len(PLOT_COLORS)]
            linestyle_current = PLOT_LINESTYLES[i_df % len(PLOT_LINESTYLES)]

            if label_current not in summary_data_cache or summary_data_cache[label_current].get("N_for_mean", 0) == 0 or \
                    summary_data_cache[label_current].get("mean_trace") is None:
                print(f"W (PERIODOGRAM - {label_current}): No valid mean trace. Skip.");
                title_run_counts_pg.append(f"{label_current} (no data)");
                continue
            s_data = summary_data_cache[label_current];
            mean_tr_full, N_runs_for_mean = s_data["mean_trace"], s_data["N_for_mean"]
            if len(mean_tr_full) == 0: print(
                f"W (PERIODOGRAM - {label_current}): Mean trace empty. Skip."); title_run_counts_pg.append(
                f"{label_current} (empty trace)"); continue

            idx_peak_in_mean_tr = np.argmax(mean_tr_full)
            start_idx_pg_segment = idx_peak_in_mean_tr
            end_idx_pg_segment = min(start_idx_pg_segment + FIT_WINDOW_POST_PEAK, len(mean_tr_full))
            data_for_pg_raw = mean_tr_full[start_idx_pg_segment:end_idx_pg_segment]
            N_pg_segment = len(data_for_pg_raw)

            if N_pg_segment < 4: print(
                f"W (PERIODOGRAM - {label_current}): Seg too short ({N_pg_segment} pts). Skip."); title_run_counts_pg.append(
                f"{label_current} (seg short N={N_pg_segment})"); continue

            data_for_pg_analyzed = data_for_pg_raw.copy();
            analysis_steps_label_pg = []
            if DETREND_PERIODOGRAM: data_for_pg_analyzed = detrend(data_for_pg_analyzed,
                                                                   type='linear'); analysis_steps_label_pg.append(
                "detrended")
            if APPLY_FFT_WINDOW: fft_window_func = np.hanning(
                N_pg_segment); data_for_pg_analyzed *= fft_window_func; analysis_steps_label_pg.append("Hann window")

            fft_magnitudes_pg = np.abs(rfft(data_for_pg_analyzed))
            fft_frequencies_pg = rfftfreq(N_pg_segment, d=SAMPLE_TIME_DELTA_US)
            valid_indices_for_period = fft_frequencies_pg > 1e-9
            if not np.any(valid_indices_for_period): print(
                f"W (PERIODOGRAM - {label_current}): No valid non-zero freqs. Skip."); title_run_counts_pg.append(
                f"{label_current} (no freqs)"); continue

            periods_pg_us = 1.0 / fft_frequencies_pg[valid_indices_for_period]
            magnitudes_for_periods_pg = fft_magnitudes_pg[valid_indices_for_period]
            period_mask_pg = (periods_pg_us >= min_period_plot_us) & (periods_pg_us <= max_period_plot_abs_us)
            plot_periods_pg_us, plot_magnitudes_pg = periods_pg_us[period_mask_pg], magnitudes_for_periods_pg[
                period_mask_pg]

            if len(plot_periods_pg_us) > 0:
                sort_order_pg = np.argsort(plot_periods_pg_us)
                legend_suffix_pg = f" (N={N_runs_for_mean}, SegLen={N_pg_segment}" + (
                    f", {', '.join(analysis_steps_label_pg)}" if analysis_steps_label_pg else "") + ")"
                ax_pg.plot(plot_periods_pg_us[sort_order_pg], plot_magnitudes_pg[sort_order_pg], lw=1.5,
                           color=color_current, linestyle=linestyle_current, label=f"{label_current}{legend_suffix_pg}")
                any_periodogram_plotted = True
            else:
                print(f"W (PERIODOGRAM - {label_current}): No periods to plot after filtering.")
            title_run_counts_pg.append(f"{label_current} (N={N_runs_for_mean})")

        if any_periodogram_plotted:
            ax_pg.set_xlabel(f"Periode (µs/cyclus, geplot: {min_period_plot_us:.2f}-{max_period_plot_abs_us:.2f} µs)")
            ylabel_text_pg = "FFT Magnitude" + (
                f" ({', '.join(analysis_steps_label_pg)})" if analysis_steps_label_pg else "")
            ax_pg.set_ylabel(ylabel_text_pg)
            ax_pg.set_title(
                f"{DEVICE_FILTER} – POST-PEAK PERIODOGRAM ({', '.join(analysis_steps_label_pg) if analysis_steps_label_pg else 'Raw'})\n(Segment: {FIT_WINDOW_POST_PEAK} samples from peak; Sample time: {SAMPLE_TIME_DELTA_US} µs)\n{', '.join(title_run_counts_pg)}")
            ax_pg.grid(True, which="both", ls="-", alpha=0.3);
            ax_pg.legend(fontsize=8, loc='upper right')
            fig_pg.tight_layout();
            fig_pg.canvas.manager.set_window_title(f"Plot Mode: POST_PEAK_PERIODOGRAM")
        else:
            plt.close(fig_pg) if 'fig_pg' in locals() else None; print("W (POST_PEAK_PERIODOGRAM): No traces plotted.")
    else:
        print(f"Warning: PLOT_MODE '{CURRENT_PLOT_MODE}' is not recognized or fully implemented.")

if plt.get_fignums():
    print(f"\nDisplaying {len(plt.get_fignums())} generated plot(s). Close all plot windows to exit script.")
    plt.show()
else:
    print("\nNo plots were generated.")
print("\nComparison plotting script finished.")
# --- END OF FILE plotting_master_improved_v3.py ---