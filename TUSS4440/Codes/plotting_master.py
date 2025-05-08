# -*- coding: utf-8 -*-
# --- START OF FILE plotting_master_compare.py ---

# !/usr/bin/env python3
"""
plotting_master_compare.py – Compares data from one to six CSV files using various plot modes:
  1) THRESHOLD-plot: Voltage around first V>THRESH (relative sample index)
  2) RAW-plot     : Full raw voltage vs. time
  3) XCORR        : Align signals using cross-correlation (independent alignment per file)
  4) SHIFT_EXTREMES: XCORR + extra shift for runs with extreme shifts (independent per file)
  5) SHIFT_EXTREMES_MULTI: 2x2 grid with SHIFT_EXTREMES for N_EXTREME=[0,5,10,15] (independent per file)
  6) SUMMARY      : Histogram of peak amplitudes + average trace with CI/PI (overlaid comparison)
  7) MAX_VALUE    : Scatter plot of (peak_time, peak_V) per run (overlaid comparison)

Choose the mode by uncommenting exactly one of the seven PLOT_MODE lines below.
"""

# ---------------- PLOTMODUS – comment/uncomment -----------------
# PLOT_MODE = "THRESHOLD"      # voltage around threshold
# PLOT_MODE = "RAW"           # raw data vs time
# PLOT_MODE = "XCORR"
# PLOT_MODE = "SHIFT_EXTREMES"
# PLOT_MODE = "SHIFT_EXTREMES_MULTI"
PLOT_MODE = "SUMMARY"  # Histogram + Mean trace
# PLOT_MODE = "MAX_VALUE"     # peak amplitude vs time

# ----------------------------------------------------------------

# --------- General Settings -------------------------------------
# --- Input Files and Labels ---
# Define up to 6 CSV files and their corresponding labels.
# Comment out lines for files you don't want to include (minimum 1).
# Ensure CSV_FILES and LABELS have the same number of active (uncommented) entries.
CSV_FILES = [
    "50x1_al_gel_bouw_1_1_meta_1.csv",  # First dataset
    "50x1_al_gel_bouw_1_1_meta_2_meta_1.csv",  # Second dataset (CHANGE THIS or comment out)
    "50x1_al_gel_gleuf_meta_1.csv",      # Third dataset
    "50x1_al_gel_gleuf_bouw_2_meta_1.csv",     # Fourth dataset
    # "50x1_al_gel_meta_los_meta_dag_later_meta_1.csv",      # Fifth dataset
    # "50x1_al_gel_meta_los_meta_1.csv",      # Sixth dataset
]
LABELS = [
    "Meting 1",  # Label for first dataset
    "Meting 2",  # Label for second dataset (CHANGE THIS or comment out)
    "Meting met Gleuf", #Label for third dataset
    "Meting 2 Met Gleuf",                 # Label for fourth dataset
    #"Dag Later",                  # Label for fifth dataset
    #"Losse Meting",                  # Label for sixth dataset
]

if len(CSV_FILES) != len(LABELS):
    raise ValueError("Mismatch between the number of CSV_FILES and LABELS. Please check the configuration.")
if not 1 <= len(CSV_FILES) <= 6:
    raise ValueError(f"Please provide between 1 and 6 CSV files. Found {len(CSV_FILES)} active files.")

# --- Common Settings ---
DEVICE_FILTER = "Master"  # Filter for this board (applied to all files)
ADC_BITS = 12
V_REF = 3.3
MAX_RUNS = 100  # Max runs *per file* to plot/analyze

# THRESHOLD & SUMMARY & MAX_VALUE mode - Threshold and window
# THRESH         = 0.94          # Threshold in V (if Voltage is V)
THRESH = 1500  # Threshold in ADC Units (if Voltage is ADC_Value)
WINDOW_BEFORE = 0  # Samples before crossing (THRESHOLD/SUMMARY/MAX_VALUE)
WINDOW_AFTER = 1500  # Samples after crossing (THRESHOLD/SUMMARY/MAX_VALUE)

# SUMMARY mode - How many highest peaks per run for histogram
N_PEAKS_PER_RUN = 5  # Number of highest values per run in histogram

# RAW & MAX_VALUE mode - Time unit
TIME_UNIT = "ms"  # "us", "ms" or "s"
MAX_POINTS = None  # Max points to plot in RAW mode (None = all)

# XCORR, SHIFT_EXTREMES, SHIFT_EXTREMES_MULTI - Alignment parameters
SEG_LEN = 50  # Samples after threshold for alignment segment
MAX_LAG = 5  # Max integer shift (samples) for correlation

# SHIFT_EXTREMES mode - parameters
N_EXTREME = 10  # Top/bottom runs (based on shift) to shift extra
OFFSET_EXT = 1  # Extra samples to shift extremes

# SHIFT_EXTREMES_MULTI mode - parameter
EXTREME_VALUES = [0, 5, 10, 15]  # N_EXTREME values to try in the grid
# ----------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
from scipy.stats import norm, t

# Colors and markers for plotting multiple datasets
PLOT_COLORS = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']
PLOT_MARKERS = ['o', 'x', 's', '^', 'v', 'D']
PLOT_LINESTYLES = ['-', '--', ':', '-.', (0, (3, 1, 1, 1)), (0, (5, 10))]


def adc_to_v(adc):
    # NOTE: Currently no conversion! Voltage = ADC_Value
    # If you want real voltage:
    # return adc / (2 ** ADC_BITS - 1) * V_REF
    return adc


# --- Data Loading and Preparation ---
def load_and_prepare_data(csv_file, device_filter, time_scale_factor):
    """Loads CSV, filters by device, adds Voltage and TimePlot columns."""
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: CSV file '{csv_file}' not found.")
        return None

    df = df[df["Device"] == device_filter].copy()
    if df.empty:
        print(f"Error: No data found for DEVICE_FILTER '{device_filter}' in '{csv_file}'.")
        return None

    required_cols = ["Run", "Timestamp_us", "ADC_Value", "Device"]
    if not all(col in df.columns for col in required_cols):
        print(f"Error: CSV file '{csv_file}' must contain columns: {required_cols}")
        return None

    df["Voltage"] = adc_to_v(df["ADC_Value"])
    df["TimePlot"] = df["Timestamp_us"] * time_scale_factor
    df['DataSource'] = csv_file  # Add identifier
    return df


# --- Helper function for XCORR-based alignment data ---
def get_shifts_and_onsets(df, label_for_warnings, thresh_val, seg_len_val, max_lag_val):
    runs = sorted(df.Run.unique())[:MAX_RUNS]
    onsets = {}
    valid_runs = []
    for run in runs:
        v = df[df.Run == run].Voltage.values
        on_idx = np.where(v > thresh_val)[0]
        if len(on_idx) > 0:
            onsets[run] = on_idx[0]
            valid_runs.append(run)
    runs = valid_runs  # Use only runs with a threshold crossing
    if not runs: return {}, {}, None, []

    ref_run = runs[0]
    if ref_run not in onsets:  # Should not happen if runs is populated
        print(f"Warning ({label_for_warnings}): Reference run {ref_run} unexpectedly missing from onsets dict.")
        return {}, {}, None, []

    ref_on = onsets[ref_run]
    v_ref_series = df[df.Run == ref_run].Voltage
    if v_ref_series.empty:
        print(f"Warning ({label_for_warnings}): Reference run {ref_run} has no voltage data.")
        return {}, {}, None, []
    v_ref = v_ref_series.values

    if ref_on + seg_len_val > len(v_ref):
        print(
            f"Warning ({label_for_warnings}): Reference run {ref_run} (len {len(v_ref)}, onset {ref_on}) too short for SEG_LEN ({seg_len_val}).")
        return {}, {}, None, []
    ref_seg = v_ref[ref_on: ref_on + seg_len_val]
    ref_seg_mean = ref_seg.mean()

    shifts = {}
    calculated_runs = []
    for run in runs:  # Iterate through 'valid_runs'
        v_series = df[df.Run == run].Voltage
        if v_series.empty: continue
        v = v_series.values

        on = onsets[run]
        if on + seg_len_val > len(v):
            # print(f"Warning ({label_for_warnings}): Run {run} (len {len(v)}, onset {on}) too short for SEG_LEN ({seg_len_val}). Skipping run.")
            continue

        seg = v[on: on + seg_len_val]
        # Check for constant segments which cause issues with correlation if std is zero
        if np.std(seg) == 0 or np.std(ref_seg) == 0:
            # print(f"Warning ({label_for_warnings}): Segment for run {run} or reference segment is constant. Using shift=0.")
            shifts[run] = 0  # Assign a default shift (e.g. 0) or skip
            calculated_runs.append(run)
            continue

        seg_zero = seg - seg.mean()
        ref_zero = ref_seg - ref_seg_mean

        try:
            full = correlate(seg_zero, ref_zero, mode="full")
        except ValueError as e:
            print(f"Warning ({label_for_warnings}): ValueError during correlation for run {run}: {e}. Skipping run.")
            continue

        if len(seg_zero) <= max_lag_val:
            # print(f"Warning ({label_for_warnings}): Segment for run {run} (len {len(seg_zero)}) too short for MAX_LAG ({max_lag_val}). Skipping run.")
            continue

        centre = len(seg_zero) - 1
        lag_start = centre - max_lag_val
        lag_end = centre + max_lag_val + 1

        if lag_start < 0 or lag_end > len(full):
            print(f"Warning ({label_for_warnings}): Lag window out of bounds for run {run}. Skipping run.")
            continue

        window = full[lag_start: lag_end]
        if len(window) == 0:
            print(f"Warning ({label_for_warnings}): Correlation window is empty for run {run}. Skipping run.")
            continue

        shifts[run] = np.argmax(window) - max_lag_val
        calculated_runs.append(run)
    return shifts, onsets, ref_on, calculated_runs


# Determine time scale and label
if TIME_UNIT == "s":
    time_scale_factor = 1e-6
    time_label_unit = "s"
elif TIME_UNIT == "ms":
    time_scale_factor = 1e-3
    time_label_unit = "ms"
else:  # "us" or default
    time_scale_factor = 1.0
    time_label_unit = "µs"

# Load data for all specified files
data_frames = []
active_labels = []
for i, csv_file_path in enumerate(CSV_FILES):
    df = load_and_prepare_data(csv_file_path, DEVICE_FILTER, time_scale_factor)
    if df is not None and not df.empty:
        data_frames.append(df)
        active_labels.append(LABELS[i])
    else:
        print(f"Warning: Skipping file '{csv_file_path}' due to loading errors or no data for filter.")

if not data_frames:
    print("No data loaded successfully from any CSV file. Exiting.")
    exit()


# --- Plotting Functions (Helper for Repetitive Tasks) ---
def plot_runs(ax, df, label, color, plot_mode_str, **kwargs):  # Renamed plot_mode to plot_mode_str
    """Helper to plot runs for a single DataFrame."""
    all_runs = sorted(df["Run"].unique())[:MAX_RUNS]
    plot_count = 0
    plotted_something = False

    if plot_mode_str == "THRESHOLD":
        window_before = kwargs.get("window_before", WINDOW_BEFORE)
        window_after = kwargs.get("window_after", WINDOW_AFTER)
        thresh = kwargs.get("thresh", THRESH)
        for run in all_runs:
            g = df[df["Run"] == run].reset_index(drop=True)
            above = np.where(g["Voltage"] > thresh)[0]
            if not len(above): continue
            onset = above[0]
            g["idx_rel"] = g.index - onset
            mask = (g["idx_rel"] >= -window_before) & (g["idx_rel"] <= window_after)
            if mask.any():
                ax.scatter(g.loc[mask, "idx_rel"], g.loc[mask, "Voltage"],
                           s=16, alpha=0.5, c=color,
                           label=f"{label} Run {run}" if plot_count < 2 else None)
                plot_count += 1
                plotted_something = True
        return plotted_something, plot_count

    elif plot_mode_str == "RAW":
        max_points = kwargs.get("max_points", MAX_POINTS)
        # Apply max_points per run for RAW mode
        for run_id in all_runs:  # Iterate using MAX_RUNS limited list
            run_df = df[df["Run"] == run_id]
            if run_df.empty: continue

            plot_df_run = run_df
            if max_points and len(run_df) > max_points:
                plot_df_run = run_df.iloc[:max_points]

            if not plot_df_run.empty:
                ax.plot(plot_df_run["TimePlot"], plot_df_run["Voltage"], "-", lw=1, alpha=0.6, c=color,
                        label=f"{label} Run {run_id}" if plot_count < 2 else None)
                plot_count += 1
                plotted_something = True
        return plotted_something, plot_count
    return False, 0


# --------- MODE 1: THRESHOLD ----------------------------------
if PLOT_MODE.upper() == "THRESHOLD":
    fig, ax = plt.subplots(figsize=(10, 5))
    total_plot_counts = []
    any_plotted = False

    for i, df_current in enumerate(data_frames):
        label_current = active_labels[i]
        color_current = PLOT_COLORS[i % len(PLOT_COLORS)]

        plotted_df, plot_count_df = plot_runs(ax, df_current, label_current, color_current, "THRESHOLD",
                                              window_before=WINDOW_BEFORE, window_after=WINDOW_AFTER, thresh=THRESH)
        if plotted_df:
            any_plotted = True
        total_plot_counts.append(plot_count_df)

    ax.set_xlabel(f"Sample-index (0 = 1e V > {THRESH})")
    ax.set_ylabel("Spanning (ADC Units or V)")

    title_label_parts = [f"{active_labels[j]} ({total_plot_counts[j]} runs)" for j in range(len(active_labels))]
    title_str = " vs ".join(title_label_parts)
    ax.set_title(f"{DEVICE_FILTER} – Voltage around threshold V > {THRESH}\n"
                 f"Comparing {title_str}")

    if any_plotted:
        handles, legend_labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, legend_labels, title="First 2 runs shown per dataset", fontsize=8)


# --------- MODE XCORR: integer shift per run (independent alignment) ---
elif PLOT_MODE.upper() == "XCORR":
    fig, ax = plt.subplots(figsize=(10, 5))
    all_run_counts = []
    # all_dataset_shifts = {} # Not strictly needed for plotting but could be useful
    any_plotted = False


    def align_and_plot_xcorr(ax_plot, df, label, color_plot):  # Renamed ax, color
        runs = sorted(df.Run.unique())[:MAX_RUNS]
        if not runs: return 0, []

        ref_run = runs[0]
        ref_df = df[df.Run == ref_run]
        if ref_df.empty:
            print(f"Warning ({label}): Reference run {ref_run} has no data.")
            return 0, []
        ref_v = ref_df.Voltage.values
        ref_on_indices = np.where(ref_v > THRESH)[0]
        if not len(ref_on_indices):
            print(f"Warning ({label}): No threshold crossing in reference run {ref_run}.")
            return 0, []
        ref_on = ref_on_indices[0]
        if ref_on + SEG_LEN > len(ref_v):
            print(f"Warning ({label}): Reference run {ref_run} too short for SEG_LEN.")
            return 0, []
        ref_seg = ref_v[ref_on: ref_on + SEG_LEN]
        ref_seg_mean = ref_seg.mean()

        plot_count = 0
        shifts_list = []
        for run in runs:
            g = df[df.Run == run].reset_index(drop=True)
            v = g.Voltage.values
            on_indices = np.where(v > THRESH)[0]
            if not len(on_indices): continue
            on = on_indices[0]
            if on + SEG_LEN > len(v): continue

            seg = v[on: on + SEG_LEN]
            if np.std(seg) == 0 or np.std(ref_seg) == 0:  # Handle constant segments
                shift = 0  # Default shift
            else:
                seg_zero = seg - seg.mean()
                ref_zero = ref_seg - ref_seg_mean
                try:
                    full = correlate(seg_zero, ref_zero, mode="full")
                except ValueError:
                    continue

                if len(seg_zero) <= MAX_LAG: continue  # seg_zero too short
                centre = len(seg_zero) - 1
                lag_start = centre - MAX_LAG
                lag_end = centre + MAX_LAG + 1
                if lag_start < 0 or lag_end > len(full): continue
                window = full[lag_start: lag_end]
                if len(window) == 0: continue
                shift = np.argmax(window) - MAX_LAG

            shifts_list.append(shift)
            idx = np.arange(len(v))
            idx_corr = idx - (on - ref_on + shift)
            ax_plot.scatter(idx_corr, v, s=16, alpha=0.5, c=color_plot,
                            label=f"{label} Run {run} (Δ={shift})" if plot_count < 2 else None)
            plot_count += 1
        return plot_count, shifts_list


    for i, df_current in enumerate(data_frames):
        label_current = active_labels[i]
        color_current = PLOT_COLORS[i % len(PLOT_COLORS)]

        plot_count_df, _ = align_and_plot_xcorr(ax, df_current, label_current, color_current)
        if plot_count_df > 0:
            any_plotted = True
        all_run_counts.append(plot_count_df)
        # all_dataset_shifts[label_current] = shifts_list_df

    ax.set_xlabel("Sample-index (Aligned Independently per Dataset)")
    ax.set_ylabel("Spanning (ADC Units or V)")
    title_label_parts = [f"{active_labels[j]} ({all_run_counts[j]} runs)" for j in range(len(active_labels))]
    title_str = " vs ".join(title_label_parts)
    ax.set_title(f"{DEVICE_FILTER} – XCORR Alignment Comparison\n"
                 f"{title_str}")
    if any_plotted:
        handles, legend_labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, legend_labels, title="First 2 runs shown per dataset", fontsize=8)


# --------- MODE SHIFT_EXTREMES_MULTI -----------------------------
elif PLOT_MODE.upper() == "SHIFT_EXTREMES_MULTI":
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    all_datasets_shifts_info = []
    for i, df_current in enumerate(data_frames):
        label_current = active_labels[i]
        shifts, onsets, ref_on, calc_runs = get_shifts_and_onsets(df_current, label_current, THRESH, SEG_LEN, MAX_LAG)
        df_s = pd.DataFrame.from_dict(shifts, orient="index", columns=["shift"]).reset_index().rename(
            columns={"index": "run"}).sort_values("shift") if shifts else pd.DataFrame(columns=['run', 'shift'])
        all_datasets_shifts_info.append({
            "df_orig": df_current,
            "label": label_current, "shifts": shifts, "onsets": onsets,
            "ref_on": ref_on, "calc_runs": calc_runs, "df_s": df_s
        })

    plotted_runs_counts_per_dataset = [0] * len(data_frames)

    for subplot_idx, N_EXTREME_val in enumerate(EXTREME_VALUES):
        ax_sub = axes_flat[subplot_idx]  # Renamed ax to ax_sub

        for dataset_idx, data_info in enumerate(all_datasets_shifts_info):
            df_plot = data_info["df_orig"]
            label_plot = data_info["label"]
            color_plot = PLOT_COLORS[dataset_idx % len(PLOT_COLORS)]
            marker_plot = PLOT_MARKERS[dataset_idx % len(PLOT_MARKERS)]

            shifts_dict, onsets_dict = data_info["shifts"], data_info["onsets"]
            ref_on_val, calc_runs_list, df_s_sorted = data_info["ref_on"], data_info["calc_runs"], data_info["df_s"]

            if not calc_runs_list or ref_on_val is None: continue

            low_runs = set(
                df_s_sorted.head(N_EXTREME_val)["run"]) if N_EXTREME_val > 0 and not df_s_sorted.empty else set()
            high_runs = set(
                df_s_sorted.tail(N_EXTREME_val)["run"]) if N_EXTREME_val > 0 and not df_s_sorted.empty else set()

            for run_idx, run in enumerate(calc_runs_list):
                if run not in shifts_dict or run not in onsets_dict: continue
                v_series = df_plot[df_plot.Run == run].Voltage
                if v_series.empty: continue
                v = v_series.values
                on, shift = onsets_dict[run], shifts_dict[run]

                idx_raw = np.arange(len(v))
                idx_corr = idx_raw - (on - ref_on_val + shift)
                extra = -OFFSET_EXT if run in low_runs else (+OFFSET_EXT if run in high_runs else 0)

                ax_sub.plot(idx_corr + extra, v, marker=marker_plot, ms=3, alpha=0.4, c=color_plot, ls='-')
                # Legend will be handled globally later
                plotted_runs_counts_per_dataset[dataset_idx] += 1

        ax_sub.set_title(f"N_extreme = {N_EXTREME_val}")
        ax_sub.grid(alpha=0.3)
        if subplot_idx >= 2: ax_sub.set_xlabel("Sample-index (Aligned + Shifted Extremes)")
        if subplot_idx % 2 == 0: ax_sub.set_ylabel("Spanning (ADC Units or V)")

    # Custom legend for the whole figure
    legend_handles_fig = []
    legend_labels_fig = []
    for dataset_idx, data_info in enumerate(all_datasets_shifts_info):
        if plotted_runs_counts_per_dataset[dataset_idx] > 0:  # Only add if dataset was plotted
            color_current = PLOT_COLORS[dataset_idx % len(PLOT_COLORS)]
            marker_current = PLOT_MARKERS[dataset_idx % len(PLOT_MARKERS)]
            dummy_line = plt.Line2D([0], [0], color=color_current, marker=marker_current, linestyle='-', markersize=4,
                                    label=data_info["label"])
            legend_handles_fig.append(dummy_line)
            legend_labels_fig.append(data_info["label"])

    if legend_handles_fig:
        fig.legend(legend_handles_fig, legend_labels_fig, title="Datasets", loc="upper right", fontsize=8)

    avg_runs_text_parts = [
        f"{active_labels[j]} ({plotted_runs_counts_per_dataset[j] // len(EXTREME_VALUES) if EXTREME_VALUES else 0} runs)"
        for j in range(len(active_labels))]
    title_runs_str = " vs ".join(avg_runs_text_parts)
    fig.suptitle(f"{DEVICE_FILTER} – SHIFT_EXTREMES_MULTI Comparison (Offset={OFFSET_EXT})\n"
                 f"{title_runs_str}", y=1.02)
    fig.tight_layout(rect=[0, 0, 0.9, 0.96])


# ---------- MODE SHIFT_EXTREMES : Shift highest and lowest runs extra (independent) ----------
elif PLOT_MODE.upper() == "SHIFT_EXTREMES":
    fig, ax = plt.subplots(figsize=(10, 5))
    all_run_counts = []
    any_plotted = False


    def align_and_plot_shift_extreme(ax_plot, df, label, color_plot, marker_plot):  # Renamed args
        shifts, onsets, ref_on, calc_runs = get_shifts_and_onsets(df, label, THRESH, SEG_LEN, MAX_LAG)
        if not calc_runs or ref_on is None: return 0

        df_s = pd.DataFrame.from_dict(shifts, orient="index", columns=["shift"]).reset_index().rename(
            columns={"index": "run"}).sort_values("shift")
        low_runs = set(df_s.head(N_EXTREME)["run"]) if N_EXTREME > 0 and not df_s.empty else set()
        high_runs = set(df_s.tail(N_EXTREME)["run"]) if N_EXTREME > 0 and not df_s.empty else set()

        plot_count = 0
        for run in calc_runs:
            if run not in shifts or run not in onsets: continue  # Should be present
            v_series = df[df.Run == run].Voltage
            if v_series.empty: continue
            v = v_series.values
            on, shift = onsets[run], shifts[run]

            idx = np.arange(len(v))
            idx_corr = idx - (on - ref_on + shift)

            extra, label_suffix = 0, ""
            if run in low_runs:
                extra = -OFFSET_EXT; label_suffix = f" (extr -, Δ={shift:+d})"
            elif run in high_runs:
                extra = +OFFSET_EXT; label_suffix = f" (extr +, Δ={shift:+d})"
            else:
                label_suffix = f" (Δ={shift:+d})"

            ax_plot.plot(idx_corr + extra, v, marker=marker_plot, ms=4, alpha=0.6, c=color_plot, ls='-',
                         label=f"{label} Run {run}{label_suffix}" if plot_count < 2 else None)
            plot_count += 1
        return plot_count


    for i, df_current in enumerate(data_frames):
        label_current = active_labels[i]
        color_current = PLOT_COLORS[i % len(PLOT_COLORS)]
        marker_current = PLOT_MARKERS[i % len(PLOT_MARKERS)]

        plot_count_df = align_and_plot_shift_extreme(ax, df_current, label_current, color_current, marker_current)
        if plot_count_df > 0:
            any_plotted = True
        all_run_counts.append(plot_count_df)

    ax.set_xlabel(f"Sample-index (Aligned + Shifted Extremes by ±{OFFSET_EXT})")
    ax.set_ylabel("Spanning (ADC Units or V)")
    title_label_parts = [f"{active_labels[j]} ({all_run_counts[j]} runs)" for j in range(len(active_labels))]
    title_str = " vs ".join(title_label_parts)
    ax.set_title(f"{DEVICE_FILTER} – SHIFT_EXTREMES Comparison (N={N_EXTREME})\n"
                 f"{title_str}")
    if any_plotted:
        handles, legend_labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, legend_labels, title=f"First 2 runs shown per dataset", fontsize=8)


# --------- MODE 2: RAW data vs time -------------------------
elif PLOT_MODE.upper() == "RAW":
    fig, ax = plt.subplots(figsize=(10, 5))
    xlabel = f"Tijd ({time_label_unit})"
    total_plot_counts = []
    any_plotted = False

    for i, df_current in enumerate(data_frames):
        label_current = active_labels[i]
        color_current = PLOT_COLORS[i % len(PLOT_COLORS)]

        plotted_df, plot_count_df = plot_runs(ax, df_current, label_current, color_current, "RAW",
                                              max_points=MAX_POINTS)
        if plotted_df:
            any_plotted = True
        total_plot_counts.append(plot_count_df)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Spanning (ADC Units or V)")
    title_label_parts = [f"{active_labels[j]} ({total_plot_counts[j]} runs)" for j in range(len(active_labels))]
    title_str = " vs ".join(title_label_parts)
    ax.set_title(f"{DEVICE_FILTER} – Ruwe spanning versus tijd Comparison\n"
                 f"{title_str}")
    if any_plotted:
        handles, legend_labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, legend_labels, title=f"First 2 runs shown per dataset", fontsize=8)


# --------- MODE SUMMARY: histogram + mean/CI trace ---------------
elif PLOT_MODE.upper() == "SUMMARY":

    def get_summary_data(df, label_for_warnings):  # Added label for warnings
        runs = sorted(df.Run.unique())[:MAX_RUNS]
        top_n_peak_vals = []
        aligned_matrix_list = []  # Changed to list for append
        hist_seg_len = WINDOW_BEFORE + WINDOW_AFTER + 1
        valid_run_count = 0

        for run in runs:
            g = df[df.Run == run].reset_index(drop=True)
            v = g.Voltage.values
            above = np.where(v > THRESH)[0]
            if not len(above): continue
            onset = above[0]
            start = onset - WINDOW_BEFORE
            stop = onset + WINDOW_AFTER + 1  # Slice goes up to, but not including, stop

            if start < 0 or stop > len(v):
                # print(f"Warning ({label_for_warnings}): Run {run} segment out of bounds. Skipping.")
                continue
            seg = v[start:stop]
            if len(seg) != hist_seg_len:
                # print(f"Warning ({label_for_warnings}): Run {run} segment length {len(seg)} != expected {hist_seg_len}. Skipping.")
                continue

            num_peaks_to_get = min(N_PEAKS_PER_RUN, len(seg))  # Ensure N_PEAKS_PER_RUN is not > len(seg)
            if num_peaks_to_get > 0:
                # Ensure seg is not empty before argpartition
                indices_of_top_n = np.argpartition(seg, -num_peaks_to_get)[-num_peaks_to_get:]
                current_top_n_values = seg[indices_of_top_n]
                top_n_peak_vals.extend(current_top_n_values)

            aligned_matrix_list.append(seg)
            valid_run_count += 1
        return top_n_peak_vals, aligned_matrix_list, valid_run_count


    all_peaks_data_list = []
    all_matrix_data_list = []
    all_valid_run_counts_list = []

    for i, df_current in enumerate(data_frames):
        label_current = active_labels[i]
        peaks, matrix_list, runs_count = get_summary_data(df_current, label_current)
        all_peaks_data_list.append(peaks)
        all_matrix_data_list.append(matrix_list)
        all_valid_run_counts_list.append(runs_count)

    combined_all_peaks_flat = [p for sublist in all_peaks_data_list for p in sublist]

    if not combined_all_peaks_flat:
        print("Warning: No peak values collected for histogram across all datasets.")
    else:
        fig_hist, ax1 = plt.subplots(figsize=(7, 4))
        min_bin_val = np.min(combined_all_peaks_flat)
        max_bin_val = np.max(combined_all_peaks_flat)
        n_bins_total = max(10, min(50, int(np.sqrt(len(combined_all_peaks_flat))))) if len(
            combined_all_peaks_flat) > 1 else 10

        # Ensure min_bin_val and max_bin_val are different for linspace
        if min_bin_val == max_bin_val:
            bins_hist = np.linspace(min_bin_val - 0.5, max_bin_val + 0.5, n_bins_total + 1)  # Handle single value case
        else:
            bins_hist = np.linspace(min_bin_val, max_bin_val, n_bins_total + 1)

        bin_width_val = bins_hist[1] - bins_hist[0] if len(bins_hist) > 1 and n_bins_total > 0 else 1.0

        for i, peaks_df_list in enumerate(all_peaks_data_list):
            if not peaks_df_list: continue

            label_current = active_labels[i]
            color_current = PLOT_COLORS[i % len(PLOT_COLORS)]
            runs_df_count = all_valid_run_counts_list[i]

            mu1, sigma1 = np.mean(peaks_df_list), np.std(peaks_df_list, ddof=1) if len(peaks_df_list) > 1 else 0
            ax1.hist(peaks_df_list, bins=bins_hist, density=False, alpha=0.6, color=color_current,
                     label=f"{label_current}: {len(peaks_df_list)} top-{N_PEAKS_PER_RUN} vals ({runs_df_count} runs)")

            if sigma1 > 0 and bin_width_val > 0:
                x_norm = np.linspace(bins_hist[0], bins_hist[-1], 200)
                pdf1 = norm.pdf(x_norm, mu1, sigma1)
                scale1 = len(peaks_df_list) * bin_width_val
                ax1.plot(x_norm, pdf1 * scale1, linestyle=PLOT_LINESTYLES[i % len(PLOT_LINESTYLES)], lw=2,
                         color=color_current,
                         label=f"Norm Fit ({label_current}) μ={mu1:.2f}, σ={sigma1:.2f}")
            elif len(peaks_df_list) > 0:  # If sigma is 0 or not enough data for std, plot mean
                ax1.axvline(mu1, color=color_current, linestyle=PLOT_LINESTYLES[i % len(PLOT_LINESTYLES)], lw=1.5,
                            label=f"Mean ({label_current}) μ={mu1:.2f}" + (f" (σ={sigma1:.2f})" if sigma1 > 0 else ""))

        ax1.set_xlabel(f"Top-{N_PEAKS_PER_RUN} Amplitudes per Run (ADC Units or V)")
        ax1.set_ylabel("Aantal Waarnemingen")
        ax1.set_title(f"{DEVICE_FILTER} – Verdeling Top-{N_PEAKS_PER_RUN} Amplitudes Comparison")
        ax1.grid(alpha=0.3)
        ax1.legend(fontsize=8)
        fig_hist.tight_layout()

    if not any(m for m in all_matrix_data_list if m):  # Check if any dataset has non-empty matrix data
        print("Warning: Cannot create mean trace plot - no aligned data found in any dataset.")
    else:
        fig_mean, ax2 = plt.subplots(figsize=(10, 5))
        t_rel = np.arange(-WINDOW_BEFORE, WINDOW_AFTER + 1)


        def plot_mean_trace(ax_plot, aligned_matrix_list_local, label, color_plot, linestyle_plot):  # Renamed args
            if not aligned_matrix_list_local: return  # Empty list of segments
            aligned = np.array(aligned_matrix_list_local)
            N = aligned.shape[0]
            if N == 0: return

            mean_tr = aligned.mean(axis=0)
            std_tr = aligned.std(axis=0, ddof=1) if N > 1 else np.zeros_like(mean_tr)
            sem_tr = std_tr / np.sqrt(N) if N > 1 else np.zeros_like(mean_tr)

            ci95_hi, ci95_lo = mean_tr, mean_tr  # Default for N=1
            pred_hi, pred_lo = mean_tr, mean_tr  # Default for N=1

            if N > 1:
                t_crit = t.ppf(0.975, df=N - 1)
                ci95_hi = mean_tr + t_crit * sem_tr
                ci95_lo = mean_tr - t_crit * sem_tr
                pred_hi = mean_tr + t_crit * std_tr * np.sqrt(1 + 1 / N)
                pred_lo = mean_tr - t_crit * std_tr * np.sqrt(1 + 1 / N)

            ax_plot.fill_between(t_rel, pred_lo, pred_hi, alpha=0.15, color=color_plot, label=f"95% PI ({label})")
            ax_plot.fill_between(t_rel, ci95_lo, ci95_hi, alpha=0.35, color=color_plot, label=f"95% CI ({label})")
            ax_plot.plot(t_rel, mean_tr, lw=1.5, color=color_plot, linestyle=linestyle_plot,
                         label=f"Gemiddelde ({label}, N={N})")


        title_run_counts_parts = []
        for i, matrix_df_list in enumerate(all_matrix_data_list):
            label_current = active_labels[i]
            color_current = PLOT_COLORS[i % len(PLOT_COLORS)]
            linestyle_current = PLOT_LINESTYLES[i % len(PLOT_LINESTYLES)]

            # Number of runs for this dataset is len(matrix_df_list)
            num_runs_this_dataset = len(matrix_df_list) if matrix_df_list else 0
            title_run_counts_parts.append(f"{label_current} ({num_runs_this_dataset} runs)")

            if num_runs_this_dataset > 0:  # Only plot if there's data
                plot_mean_trace(ax2, matrix_df_list, label_current, color_current, linestyle_current)

        ax2.set_xlabel(f"Sample-index (0 = 1e V > {THRESH})")
        ax2.set_ylabel("Spanning (ADC Units or V)")
        title_runs_str = " vs ".join(title_run_counts_parts)
        ax2.set_title(f"{DEVICE_FILTER} – Gemiddelde trace Comparison ({title_runs_str})\n"
                      f"±95% CI and ±95% PI shown per dataset")
        ax2.grid(alpha=0.3)
        if any(m for m in all_matrix_data_list if m):  # Only add legend if something was plotted
            ax2.legend(fontsize=9)
        fig_mean.tight_layout()


# --------- MODE MAX_VALUE: Scatter plot van piek tijd vs piek waarde --------
elif PLOT_MODE.upper() == "MAX_VALUE":
    fig, ax = plt.subplots(figsize=(10, 5))
    xlabel_plot = f"Tijd van Piek ({time_label_unit})"  # Renamed
    ylabel_plot = "Piek Spanning (ADC Units or V)"  # Renamed
    any_plotted = False


    def get_peak_data(df, label_for_warnings):  # Added label for warnings
        runs = sorted(df.Run.unique())[:MAX_RUNS]
        peak_times_plot_list = []  # Renamed
        peak_values_list = []  # Renamed
        valid_run_count = 0
        for run in runs:
            g = df[df.Run == run].reset_index(drop=True)
            v = g.Voltage.values
            t_us = g.Timestamp_us.values  # Use original timestamp for indexing
            above = np.where(v > THRESH)[0]
            if not len(above): continue
            onset = above[0]
            start = max(0, onset - WINDOW_BEFORE)
            stop = min(len(v), onset + WINDOW_AFTER + 1)
            if start >= stop: continue

            seg_v = v[start:stop]
            if len(seg_v) == 0: continue

            idx_in_seg = np.argmax(seg_v)
            peak_v_val = seg_v[idx_in_seg]  # Renamed
            abs_idx = start + idx_in_seg

            # Ensure abs_idx is within bounds of t_us
            if abs_idx >= len(t_us):
                # print(f"Warning ({label_for_warnings}): Calculated peak index {abs_idx} out of bounds for timestamps (len {len(t_us)}) in run {run}. Skipping.")
                continue

            peak_t_us_val = t_us[abs_idx]  # Renamed
            peak_values_list.append(peak_v_val)
            peak_times_plot_list.append(peak_t_us_val * time_scale_factor)  # Apply scale factor here
            valid_run_count += 1
        return peak_times_plot_list, peak_values_list, valid_run_count


    title_run_counts_parts = []
    for i, df_current in enumerate(data_frames):
        label_current = active_labels[i]
        color_current = PLOT_COLORS[i % len(PLOT_COLORS)]
        marker_current = PLOT_MARKERS[i % len(PLOT_MARKERS)]
        linestyle_current = PLOT_LINESTYLES[i % len(PLOT_LINESTYLES)]

        times_df_list, vals_df_list, count_df_val = get_peak_data(df_current, label_current)
        title_run_counts_parts.append(f"{label_current} ({count_df_val} runs)")

        if count_df_val > 0:
            ax.plot(times_df_list, vals_df_list, marker=marker_current, ms=5, alpha=0.7, c=color_current,
                    linestyle='None',
                    label=f"{label_current} ({count_df_val} runs)")
            if vals_df_list:  # Ensure list is not empty before mean
                avg_peak_df = np.mean(vals_df_list)
                ax.axhline(avg_peak_df, color=color_current, linestyle=linestyle_current, lw=1.5,
                           label=f'Gem. Piek {label_current} = {avg_peak_df:.2f}')
            any_plotted = True

    ax.set_xlabel(xlabel_plot)
    ax.set_ylabel(ylabel_plot)
    title_runs_str = " vs ".join(title_run_counts_parts)
    ax.set_title(f"{DEVICE_FILTER} – Piek Amplitude vs. Tijd van Piek Comparison ({title_runs_str})")
    if any_plotted:
        ax.legend(fontsize=9)

else:
    valid_modes = ["THRESHOLD", "RAW", "XCORR", "SHIFT_EXTREMES", "SHIFT_EXTREMES_MULTI", "SUMMARY", "MAX_VALUE"]
    raise ValueError(f"PLOT_MODE '{PLOT_MODE}' ongeldig. Kies uit: {', '.join(valid_modes)}")

# --------- General Finishing (only if a plot was created) --------
if plt.get_fignums():  # Check if any figures were created
    plt.show()
else:
    print(
        "Geen plots gegenereerd (mogelijk geen geldige data gevonden voor de gekozen modus, of modus niet geimplementeerd).")

print("\nComparison plotting script finished.")

# --- END OF FILE plotting_master_compare.py ---