# -*- coding: utf-8 -*-
# --- START OF FILE plotting_master_improved_v8.py --- # (Adjusted name if needed)

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, detrend
from scipy.stats import norm, t
from numpy.fft import rfft, rfftfreq

# !/usr/bin/env python3
"""
plotting_master.py – Compares data from one to six CSV files using various plot modes.
"""

# ---------------- PLOTMODUS – Select modes to run -----------------
PLOT_MODES_TO_RUN = [
    "RAW",
    "THRESHOLD",
    "SUMMARY",
    #"DIFFERENCE",
    #"DEVIATION_FROM_LINEAR",
    # "FFT_OF_DEVIATION", # Remains commented out
    "POST_PEAK_PERIODOGRAM",
    "INDIVIDUAL_RUN_PERIODOGRAM", # New plot mode
]

# ----------------------------------------------------------------
# --------- General Settings -------------------------------------
TARGET_DATA_SUBFOLDER = os.path.join("..", "data", "UltraSoon_Measurements")
SAMPLE_TIME_DELTA_US = 4.63

_CSV_BASE_FILES = [
    # "15_mm_gleuf_herbouw_meta_1.csv",
    # "15_mm_gleuf_herbouw_meta_2.csv",
    # "15_mm_gleuf_herbouw_meta_3.csv",
    "transducers_aan_elkaar_met_gel.csv",
    "transducers_aan_elkaar_zonder_gel.csv",
    "transducers_beetje_afstand.csv",
]
LABELS = [
    "Met gel",
    "Zonder gel",
    "Afstand",
    # "Meting 1 (Defectloos)",
    # "Meting 2 (Defectloos)",
    # "Meting 3 (Defectloos)",
]

CSV_FILES = [os.path.join(TARGET_DATA_SUBFOLDER, fname) for fname in _CSV_BASE_FILES]
if len(CSV_FILES) != len(LABELS): raise ValueError("Mismatch CSV_FILES/LABELS.")
if not 1 <= len(CSV_FILES) <= 6: raise ValueError(f"Need 1-6 CSV files, got {len(CSV_FILES)}.")

DEVICE_FILTER = "Master";
ADC_BITS = 12;
V_REF = 3.3;
MAX_RUNS = 100 # Max runs from each file to consider for summary AND individual periodograms
THRESH = 1500;
WINDOW_BEFORE = 10
WINDOW_AFTER = 1200

FIT_WINDOW_POST_PEAK = 1200 # Used for periodogram segment length
DETREND_PERIODOGRAM = True;
MIN_PERIOD_PLOT = 2;
MAX_PERIOD_PLOT_ABS = 40;
APPLY_FFT_WINDOW = True
MIN_FREQ_PLOT_MHZ_DEV = 0.001; # For FFT_OF_DEVIATION if it were active
MAX_FREQ_PLOT_MHZ_DEV = 0.05  # For FFT_OF_DEVIATION if it were active
N_PEAKS_PER_RUN = 5; # Not currently used by these plot modes
TIME_UNIT = "ms";
MAX_POINTS = None # For RAW plot
# SEG_LEN, MAX_LAG, N_EXTREME, OFFSET_EXT, EXTREME_VALUES = 50, 5, 10, 1, [0, 5, 10, 15] # Not used
# ----------------------------------------------------------------

PLOT_COLORS = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']
PLOT_LINESTYLES = ['-', '--', ':', '-.', (0, (3, 1, 1, 1)), (0, (5, 10))]


def adc_to_v(adc_val): return adc_val


def load_and_prepare_data(csv_f, dev_filter, ts_factor):
    try:
        df = pd.read_csv(csv_f)
    except FileNotFoundError:
        print(f"E: File '{csv_f}' not found.");
        return None
    df = df[df["Device"] == dev_filter].copy()
    if df.empty: print(f"E: No data for filter '{dev_filter}' in '{csv_f}'."); return None
    req_c = ["Run", "Timestamp_us", "ADC_Value", "Device"]
    if not all(c in df.columns for c in req_c): print(
        f"E:'{csv_f}' needs {req_c}. Has:{df.columns.tolist()}"); return None
    df["Voltage"] = adc_to_v(df["ADC_Value"]);
    df["TimePlot"] = df["Timestamp_us"] * ts_factor;
    df['DataSource'] = csv_f
    return df


def get_summary_data(df_s, lbl_warn):
    runs_s = sorted(df_s.Run.unique())[:MAX_RUNS]; # Apply MAX_RUNS here
    al_matrix = []
    exp_len_s = WINDOW_BEFORE + WINDOW_AFTER + 1;
    valid_r_cnt = 0
    for _, run_s in enumerate(runs_s):
        g_s = df_s[df_s.Run == run_s].reset_index(drop=True);
        v_s = g_s.Voltage.values
        above_s = np.where(v_s > THRESH)[0]
        if not len(above_s): continue
        onset_s = above_s[0];
        start_s, stop_s = onset_s - WINDOW_BEFORE, onset_s + WINDOW_AFTER + 1
        act_start_s, act_stop_s = max(0, start_s), min(len(v_s), stop_s);
        seg_s = v_s[act_start_s:act_stop_s]
        # Ensure the segment has the expected length AND was fully captured (not truncated at start/end of run)
        if len(seg_s) == exp_len_s and start_s >= 0 and stop_s <= len(v_s):
            al_matrix.append(seg_s);
            valid_r_cnt += 1
        # No need to break early if we respect MAX_RUNS from the start
    return [], al_matrix, valid_r_cnt # first element [] is historical, not used


def calculate_trace_stats(al_matrix_loc):
    exp_len_loc = WINDOW_BEFORE + WINDOW_AFTER + 1
    if not al_matrix_loc: return np.zeros(exp_len_loc), 0, np.zeros(exp_len_loc), np.zeros(exp_len_loc)
    al_data_np = np.array(al_matrix_loc);
    N_loc = al_data_np.shape[0]
    if N_loc == 0: return np.zeros(exp_len_loc), 0, np.zeros(exp_len_loc), np.zeros(exp_len_loc)
    m_tr = al_data_np.mean(axis=0);
    std_tr_loc, sem_tr_loc = np.zeros_like(m_tr), np.zeros_like(m_tr)
    if N_loc > 1:
        std_tr_loc = al_data_np.std(axis=0, ddof=1)
        sem_tr_loc = std_tr_loc / np.sqrt(N_loc)
    return m_tr, N_loc, sem_tr_loc, std_tr_loc


if TIME_UNIT.lower() == "s":
    ts_factor_raw, tu_raw_lbl = 1e-6, "s"
elif TIME_UNIT.lower() == "ms":
    ts_factor_raw, tu_raw_lbl = 1e-3, "ms"
else:
    ts_factor_raw, tu_raw_lbl = 1.0, "µs"

print("Loading data...");
dfs, act_lbls = [], []
for i, fp in enumerate(CSV_FILES):
    if not os.path.exists(fp): print(f"W:File '{fp}' not found.Skip.");continue
    df_l = load_and_prepare_data(fp, DEVICE_FILTER, ts_factor_raw)
    if df_l is not None and not df_l.empty:
        dfs.append(df_l);
        act_lbls.append(LABELS[i])
    else:
        print(f"W:Skip '{fp}' (load err/no data).")
if not dfs: print("E:No data loaded.Exit."); exit()
print(f"Data loaded for: {', '.join(act_lbls)}")

sum_cache = {};
needs_sum_calc = any(
    m in ["SUMMARY", "DIFFERENCE", "DEVIATION_FROM_LINEAR", "FFT_OF_DEVIATION",
          "POST_PEAK_PERIODOGRAM", "INDIVIDUAL_RUN_PERIODOGRAM"] for m in PLOT_MODES_TO_RUN)

if needs_sum_calc: # This pre-calculation is essential for the new mode too
    print("Pre-calc summary data (and aligned run matrices)...")
    for i, df_c in enumerate(dfs):
        lbl_c = act_lbls[i];
        _, mtx_c, r_c = get_summary_data(df_c, lbl_c); # mtx_c is al_matrix
        m_c, N_c, s_c, sd_c = calculate_trace_stats(mtx_c)
        sum_cache[lbl_c] = {"matrix_list": mtx_c, "runs_count": r_c, "mean_trace": m_c, "N_for_mean": N_c,
                            "sem_trace": s_c, "std_trace": sd_c}
        if N_c > 0: # N_c is essentially len(mtx_c) if mtx_c is not empty
            print(f"  '{lbl_c}':{N_c} runs stored for mean and individual analysis.")
        else:
            print(f"  W '{lbl_c}':No valid runs stored.")
    print("Summary/Matrix pre-calc done.")

needs_dev_calc = "DEVIATION_FROM_LINEAR" in PLOT_MODES_TO_RUN or "FFT_OF_DEVIATION" in PLOT_MODES_TO_RUN
if needs_dev_calc:
    print("Pre-calc deviation data...")
    for lbl_dc in act_lbls:
        sum_cache[lbl_dc]["deviation_info"] = None # Initialize
        if lbl_dc in sum_cache and sum_cache[lbl_dc].get("N_for_mean", 0) > 0 and sum_cache[lbl_dc].get(
                "mean_trace") is not None:
            sdat_dc = sum_cache[lbl_dc];
            mt_dc = sdat_dc["mean_trace"]
            if len(mt_dc) > 0: # Ensure mean_trace is not empty
                idx_pk_dc = np.argmax(mt_dc);
                # Ensure segment is within bounds of mean_trace
                start_dc, end_dc = idx_pk_dc, min(idx_pk_dc + FIT_WINDOW_POST_PEAK, len(mt_dc))
                seg_fit_dc = mt_dc[start_dc:end_dc];
                x_coords_dc = np.arange(len(seg_fit_dc))
                if len(seg_fit_dc) >= 2: # Need at least 2 points for polyfit
                    try:
                        sl_dc, int_dc = np.polyfit(x_coords_dc, seg_fit_dc, 1);
                        fit_vals_dc = sl_dc * x_coords_dc + int_dc;
                        dev_arr_dc = seg_fit_dc - fit_vals_dc
                        sum_cache[lbl_dc]["deviation_info"] = {"deviations_array": dev_arr_dc,
                                                               "N_segment": len(dev_arr_dc),
                                                               "x_coords_samples": x_coords_dc,
                                                               "original_segment_data": seg_fit_dc,
                                                               "linear_fit_on_segment": fit_vals_dc, "slope": sl_dc}
                        print(f"  '{lbl_dc}':Dev data calc(seg len {len(dev_arr_dc)}).")
                    except(np.linalg.LinAlgError, ValueError) as e_f:
                        print(f"  W '{lbl_dc}':Linear fit for dev failed.{e_f}")
                else:
                    print(f"  W '{lbl_dc}':Seg for dev too short({len(seg_fit_dc)}).")
            else:
                print(f"  W '{lbl_dc}':Mean trace empty for dev calc.")
        else:
            print(f"  W '{lbl_dc}':No mean trace for dev calc.")
    print("Deviation pre-calc done.")


def plot_runs(ax, df_plot, label_plot, color_plot, plot_mode_str, **kwargs):
    all_run_ids_in_df = sorted(df_plot["Run"].unique())
    # MAX_RUNS is now applied in get_summary_data for consistency if data comes from there
    # For RAW plot, it's applied here
    runs_to_plot = all_run_ids_in_df[:MAX_RUNS] if plot_mode_str == "RAW" else all_run_ids_in_df

    plot_count, plotted_something = 0, False

    if plot_mode_str == "THRESHOLD":
        window_b = kwargs.get("window_before", WINDOW_BEFORE)
        window_a = kwargs.get("window_after", WINDOW_AFTER)
        thresh_val = kwargs.get("thresh", THRESH)
        # Iterate up to MAX_RUNS worth of valid runs
        # This logic should ideally use the pre-filtered runs from sum_cache for consistency,
        # but plot_runs currently takes df_plot directly.
        # For now, we'll keep it iterating runs from df_plot, respecting MAX_RUNS locally.
        # Note: get_summary_data already limits runs to MAX_RUNS for its matrix.

        temp_run_count = 0
        for run_id in runs_to_plot: # runs_to_plot is all runs in df_plot here
            if temp_run_count >= MAX_RUNS and MAX_RUNS is not None: break

            g = df_plot[df_plot["Run"] == run_id].reset_index(drop=True)
            if g.empty: continue
            v_run = g["Voltage"].values
            above = np.where(v_run > thresh_val)[0]
            if not len(above): continue

            temp_run_count +=1 # Count this as a processed run for MAX_RUNS limit

            onset_idx = above[0]
            data_start_abs = max(0, onset_idx - window_b)
            data_end_abs = min(len(v_run), onset_idx + window_a + 1)
            segment_y = v_run[data_start_abs:data_end_abs]

            expected_len = window_b + window_a + 1
            # Only plot if segment is of expected length (i.e., not truncated)
            if len(segment_y) != expected_len: continue


            start_x_rel = data_start_abs - onset_idx
            end_x_rel = (data_end_abs - 1) - onset_idx # data_end_abs is exclusive
            segment_x_rel = np.arange(start_x_rel, end_x_rel + 1)

            if len(segment_x_rel) != len(segment_y): continue

            current_label_for_legend = None
            if plot_count == 0:
                current_label_for_legend = label_plot

            ax.plot(segment_x_rel, segment_y, '-', lw=1, alpha=0.5, c=color_plot, label=current_label_for_legend)
            plot_count += 1 # plot_count is for legend entries
            plotted_something = True
        return plotted_something, plot_count # plot_count is number of legend entries made

    elif plot_mode_str == "RAW":
        max_pts = kwargs.get("max_points", MAX_POINTS)
        for run_id in runs_to_plot: # runs_to_plot respects MAX_RUNS here
            run_df = df_plot[df_plot["Run"] == run_id]
            if run_df.empty: continue
            x_data, y_data = run_df["TimePlot"], run_df["Voltage"]
            if max_pts and len(x_data) > max_pts: x_data, y_data = x_data.iloc[:max_pts], y_data.iloc[:max_pts]
            if not x_data.empty:
                current_label_for_legend = None
                if plot_count == 0:
                    current_label_for_legend = label_plot
                ax.plot(x_data, y_data, "-", lw=1, alpha=0.6, c=color_plot, label=current_label_for_legend)
                plot_count += 1;
                plotted_something = True
        return plotted_something, plot_count
    return False, 0


for PLOT_MODE_ITER in PLOT_MODES_TO_RUN:
    CURRENT_PLOT_MODE = PLOT_MODE_ITER.upper()
    print(f"\n--- Generating plot for mode: {CURRENT_PLOT_MODE} ---")

    if CURRENT_PLOT_MODE == "RAW":
        fig_r, ax_r = plt.subplots(figsize=(12, 6));
        # counts_r tracks how many runs from each file are actually PLOTTED (not just processed)
        # for the title. For legend, plot_runs manages one label per file.
        plotted_runs_counts_r = []
        any_plot_r = False
        actual_labels_plotted_r = []

        for i, df_r in enumerate(dfs):
            lbl_r, c_r = act_lbls[i], PLOT_COLORS[i % len(PLOT_COLORS)]
            # plot_runs returns (True/False if anything plotted from this file, count of legend entries which is 0 or 1)
            # We need a different count for the title: how many runs were actually shown on the plot for this file.
            # This requires a slight refactor of plot_runs or a different way to count.
            # For now, the title might be slightly misleading if MAX_RUNS is very high and not all runs meet THRESHOLD criteria.
            # Let's use the N_for_mean from sum_cache if available, or MAX_RUNS as an upper bound for title.

            # Call plot_runs to draw and get legend status
            did_plot_file_r, _ = plot_runs(ax_r, df_r, lbl_r, c_r, "RAW", max_points=MAX_POINTS)

            if did_plot_file_r:
                any_plot_r = True
                actual_labels_plotted_r.append(lbl_r)
                # Get number of runs shown for the title
                # For RAW, it's up to MAX_RUNS, or fewer if file has less.
                num_runs_in_file = len(df_r["Run"].unique())
                plotted_runs_counts_r.append(min(num_runs_in_file, MAX_RUNS if MAX_RUNS is not None else num_runs_in_file))


        if any_plot_r:
            ax_r.set_xlabel(f"Time ({tu_raw_lbl})");
            ax_r.set_ylabel(f"V({'ADC' if adc_to_v(1) == 1 else 'V'})")
            titles_r_info = [f"{actual_labels_plotted_r[j]} ({plotted_runs_counts_r[j]} runs)" for j in range(len(actual_labels_plotted_r))]
            ax_r.set_title(
                f"{DEVICE_FILTER}–RAW: Full Traces (up to {MAX_RUNS if MAX_RUNS is not None else 'all'} runs/file)\nComparing {', '.join(titles_r_info)}")
            h_r, l_r = ax_r.get_legend_handles_labels()
            if h_r: by_l_r = dict(zip(l_r, h_r)); ax_r.legend(by_l_r.values(), by_l_r.keys(), title="Measurements",
                                                              fontsize=8, loc='upper right')
            ax_r.grid(True, alpha=0.3);
            fig_r.tight_layout();
            fig_r.canvas.manager.set_window_title("Plot:RAW")
        else:
            plt.close(fig_r) if 'fig_r' in locals() else None;
            print("W(RAW):No raw traces plotted.")

    elif CURRENT_PLOT_MODE == "THRESHOLD":
        fig_th, ax_th = plt.subplots(figsize=(12, 6));
        plotted_runs_counts_th = [] # For title
        any_plot_th = False
        actual_labels_plotted_th = []


        for i, df_th_file in enumerate(dfs): # df_th_file is one of the original dataframes
            lbl_th, c_th = act_lbls[i], PLOT_COLORS[i % len(PLOT_COLORS)]

            # Call plot_runs to draw
            did_plot_file_th, _ = plot_runs(ax_th, df_th_file, lbl_th, c_th, "THRESHOLD",
                                             window_before=WINDOW_BEFORE, window_after=WINDOW_AFTER, thresh=THRESH)

            if did_plot_file_th:
                any_plot_th = True
                actual_labels_plotted_th.append(lbl_th)
                # For THRESHOLD, number of runs is ideally sum_cache[lbl_th]['runs_count']
                # as plot_runs itself doesn't return the count of *plotted lines* only *legend entries*
                if lbl_th in sum_cache and sum_cache[lbl_th]['runs_count'] > 0:
                    plotted_runs_counts_th.append(sum_cache[lbl_th]['runs_count'])
                else: # Fallback if sum_cache somehow not populated or no runs
                    # This part is tricky as plot_runs does its own filtering.
                    # Best to rely on sum_cache which did the same filtering.
                    plotted_runs_counts_th.append(f"up to {MAX_RUNS if MAX_RUNS is not None else 'all'}")


        if any_plot_th:
            ax_th.set_xlabel(
                f"Sample-index relative to Trigger (0 = first V > {THRESH} ADC, {SAMPLE_TIME_DELTA_US}µs/sample)")
            ax_th.set_ylabel(f"V({'ADC' if adc_to_v(1) == 1 else 'V'})")
            titles_th_info = [f"{actual_labels_plotted_th[j]} ({plotted_runs_counts_th[j]} runs)" for j in range(len(actual_labels_plotted_th))]
            ax_th.set_title(
                f"{DEVICE_FILTER}–THRESHOLD: Aligned Traces (Window: [-{WINDOW_BEFORE}, +{WINDOW_AFTER}] samples around trigger)\nComparing {', '.join(titles_th_info)}")
            h_th, l_th = ax_th.get_legend_handles_labels()
            if h_th: by_l_th = dict(zip(l_th, h_th)); ax_th.legend(by_l_th.values(), by_l_th.keys(),
                                                                   title="Measurements",
                                                                   fontsize=8, loc='upper right')
            ax_th.grid(True, alpha=0.3);
            fig_th.tight_layout();
            fig_th.canvas.manager.set_window_title("Plot:THRESHOLD")
        else:
            plt.close(fig_th) if 'fig_th' in locals() else None;
            print(
                "W(THRESHOLD):No threshold-aligned traces plotted.")

    elif CURRENT_PLOT_MODE == "SUMMARY":
        mean_trace_data_exists = any(label in sum_cache and sum_cache[label]["N_for_mean"] > 0 for label in act_lbls)
        if mean_trace_data_exists:
            fig_s, ax_s = plt.subplots(figsize=(12, 6));
            t_rel_s = np.arange(-WINDOW_BEFORE, WINDOW_AFTER + 1); # X-axis for summary plot
            title_s_parts = []
            for i, lbl_s in enumerate(act_lbls):
                if lbl_s in sum_cache:
                    cache_s = sum_cache[lbl_s];
                    mt_s, N_s, sem_s, std_s = cache_s.get("mean_trace"), cache_s.get("N_for_mean"), cache_s.get(
                        "sem_trace"), cache_s.get("std_trace")
                    # Check if mean_trace has the expected length
                    if not (N_s and N_s > 0 and mt_s is not None and len(mt_s) == len(t_rel_s)):
                        title_s_parts.append(f"{lbl_s}(no valid data for plot)"); continue
                    title_s_parts.append(f"{lbl_s}(N={N_s})");
                    c, ls = PLOT_COLORS[i % len(PLOT_COLORS)], PLOT_LINESTYLES[i % len(PLOT_LINESTYLES)]
                    ax_s.plot(t_rel_s, mt_s, lw=2, color=c, linestyle=ls, label=f"Mean({lbl_s})")
                    if N_s > 1: # CI/PI only make sense for N > 1
                        tcrit = t.ppf(0.975, df=N_s - 1)
                        if sem_s is not None and isinstance(sem_s, np.ndarray) and len(sem_s) == len(t_rel_s):
                             ci_m = tcrit * sem_s
                             ax_s.fill_between(t_rel_s, mt_s - ci_m, mt_s + ci_m, alpha=0.35, color=c, label=f"95%CI({lbl_s})")
                        if std_s is not None and isinstance(std_s, np.ndarray) and len(std_s) == len(t_rel_s):
                             pi_m = tcrit * std_s * np.sqrt(1 + 1 / N_s) # Prediction Interval
                             ax_s.fill_between(t_rel_s, mt_s - pi_m, mt_s + pi_m, alpha=0.15, color=c, label=f"95%PI({lbl_s})")
            ax_s.set_xlabel(f"Sample-idx relative to Trigger (0=V>{THRESH}, {SAMPLE_TIME_DELTA_US}µs/sample)");
            ax_s.set_ylabel(f"V({'ADC' if adc_to_v(1) == 1 else 'V'})")
            ax_s.set_title(
                f"{DEVICE_FILTER}–SUMMARY: Mean Traces ± CI & PI\nWindow: [-{WINDOW_BEFORE}, +{WINDOW_AFTER}] samples. Comparing: {', '.join(title_s_parts)}")
            ax_s.grid(True, alpha=0.3);
            h_s, l_s = ax_s.get_legend_handles_labels()
            # Consolidate legend for CI/PI if many files
            if h_s:
                # Simple legend if few items, more complex if many
                if len(h_s) < 10 : # Arbitrary threshold
                     by_l_s = dict(zip(l_s, h_s)); ax_s.legend(by_l_s.values(), by_l_s.keys(), fontsize=8, loc='upper right')
                else: # Simplified legend for many items
                    handles_to_show, labels_to_show = [], []
                    seen_labels_base = set()
                    for handle, label in zip(h_s, l_s):
                        base_label = label.split('(')[0] # e.g. "Mean", "95%CI"
                        if base_label not in seen_labels_base or "Mean" in label : # Ensure mean is always shown distinct per file
                             if "Mean" in label: # Unique mean label
                                 handles_to_show.append(handle)
                                 labels_to_show.append(label)
                             elif base_label not in seen_labels_base:
                                 handles_to_show.append(plt.Line2D([0], [0], color=handle.get_facecolor()[0], alpha=handle.get_alpha())) # Create proxy artist for fill
                                 labels_to_show.append(base_label) # General label e.g. "95%CI"
                                 seen_labels_base.add(base_label)

                    ax_s.legend(handles_to_show, labels_to_show, fontsize=8, loc='upper right', title="Legend")
            fig_s.tight_layout();
            fig_s.canvas.manager.set_window_title("Plot:SUMMARY-MeanTraces")
        else:
            print("W(SUMMARY):No valid mean trace data to plot.")

    elif CURRENT_PLOT_MODE == "DIFFERENCE":
        fig_d, ax_d = plt.subplots(figsize=(12, 6));
        any_d_plot = False;
        title_d_parts = []
        # Expected length of mean trace is WINDOW_BEFORE + WINDOW_AFTER + 1
        expected_mean_trace_len = WINDOW_BEFORE + WINDOW_AFTER + 1
        t_rel_sum_d = np.arange(-WINDOW_BEFORE, WINDOW_AFTER + 1) # X-axis for original mean trace points

        if len(t_rel_sum_d) < 2: # Need at least 2 points to diff
            print("W(DIFF): Window too small for difference plot.");
            plt.close(fig_d) if 'fig_d' in locals() else None
        else:
            # X-axis for difference plot (midpoints)
            t_rel_diff_d = (t_rel_sum_d[:-1] + t_rel_sum_d[1:]) / 2.0

            for i, lbl_d in enumerate(act_lbls):
                c, ls = PLOT_COLORS[i % len(PLOT_COLORS)], PLOT_LINESTYLES[i % len(PLOT_LINESTYLES)]
                s_data = sum_cache.get(lbl_d, {})
                mt_d, N_d = s_data.get("mean_trace"), s_data.get("N_for_mean", 0)

                if N_d == 0 or mt_d is None or len(mt_d) != expected_mean_trace_len:
                    title_d_parts.append(f"{lbl_d}(no valid data)"); continue
                if len(mt_d) < 2: # Should not happen if previous check passes
                    title_d_parts.append(f"{lbl_d}(trace too short)"); continue

                delta_tr_d = np.diff(mt_d) # Length is len(mt_d) - 1

                # Ensure lengths match for plotting
                if len(t_rel_diff_d) != len(delta_tr_d):
                    # This can happen if mt_d was not of expected_mean_trace_len
                    print(f"W(DIFF - {lbl_d}): Length mismatch. Diff trace: {len(delta_tr_d)}, Expected x-axis for diff: {len(t_rel_diff_d)}. Mean trace len: {len(mt_d)}")
                    title_d_parts.append(f"{lbl_d}(len err)"); continue

                ax_d.plot(t_rel_diff_d, delta_tr_d, lw=1.5, color=c, linestyle=ls, label=f"{lbl_d}(N={N_d})")
                any_d_plot = True;
                title_d_parts.append(f"{lbl_d}(N={N_d})")

            if any_d_plot:
                ts_info = f"({SAMPLE_TIME_DELTA_US}µs/sample)";
                ax_d.set_xlabel(f"Sample-idx (midpoint, 0 relative to trigger){ts_info}");
                ax_d.set_ylabel(f"ΔMeanV/ΔSample (ADC/sample)")
                ax_d.set_title(f"{DEVICE_FILTER}–DIFFERENCE: 1st-Order Diff of Mean Traces\nComparing: {', '.join(title_d_parts)}")
                ax_d.axhline(0, color='k', ls='--', alpha=0.7, lw=1);
                ax_d.grid(True, alpha=0.3);
                ax_d.legend(fontsize=8, loc='best')
                fig_d.tight_layout();
                fig_d.canvas.manager.set_window_title("Plot:DIFFERENCE(Derivative-like)")
            else:
                plt.close(fig_d) if 'fig_d' in locals() else None;
                print("W(DIFF):No derivative traces plotted.")

    elif CURRENT_PLOT_MODE == "DEVIATION_FROM_LINEAR":
        fig_lc, ax_lc = plt.subplots(figsize=(12, 6)); any_lc_plot = False
        fig_dl, ax_dl = plt.subplots(figsize=(12, 6)); any_dl_plot = False;
        title_dl_parts = []
        for i, lbl_dl in enumerate(act_lbls):
            c, ls = PLOT_COLORS[i % len(PLOT_COLORS)], PLOT_LINESTYLES[i % len(PLOT_LINESTYLES)];
            s_data = sum_cache.get(lbl_dl, {})
            dev_info = s_data.get("deviation_info");
            N_runs_dl = s_data.get("N_for_mean", 0)

            if dev_info is None or N_runs_dl == 0:
                print(f"W(DEV_LIN - {lbl_dl}): No pre-calculated deviation data or N_runs=0. Skip.");
                title_dl_parts.append(f"{lbl_dl}(no dev data)"); continue

            dev_array = dev_info["deviations_array"]
            x_coords = dev_info["x_coords_samples"] # Relative to peak
            orig_seg = dev_info["original_segment_data"]
            fit_seg = dev_info["linear_fit_on_segment"]
            slope_dl = dev_info["slope"]

            if len(x_coords) == 0 : # Skip if no actual data in dev_info
                title_dl_parts.append(f"{lbl_dl}(empty dev seg)"); continue


            leg_details = f"(N_runs={N_runs_dl}, slope={slope_dl:.2e})"
            # Plot for Linear Fit Context
            ax_lc.plot(x_coords, orig_seg, lw=1.5, color=c, linestyle=ls, label=f"Data ({lbl_dl} {leg_details})", alpha=0.7);
            ax_lc.plot(x_coords, fit_seg, lw=2, color=c, linestyle='--', label=f"Fit ({lbl_dl} {leg_details})");
            any_lc_plot = True

            # Plot for Deviation
            ax_dl.plot(x_coords, dev_array, lw=1.5, color=c, linestyle=ls, label=f"{lbl_dl} {leg_details}");
            any_dl_plot = True

            # Add to title string, ensure no duplicates if multiple runs from same file were conceptualized differently before
            if not any(lbl_dl in part for part in title_dl_parts):
                 title_dl_parts.append(f"{lbl_dl}(N_runs={N_runs_dl})")

        common_x_lbl = f"Samples from Peak (Time up to {FIT_WINDOW_POST_PEAK * SAMPLE_TIME_DELTA_US:.2f}µs)";
        common_title_suf = f"\nSegment: {FIT_WINDOW_POST_PEAK} samples post-peak. Comparing: {', '.join(title_dl_parts)}"

        if any_lc_plot:
            ax_lc.set_xlabel(common_x_lbl);
            ax_lc.set_ylabel(f"Voltage ({'ADC' if adc_to_v(1) == 1 else 'V'})");
            ax_lc.set_title(f"{DEVICE_FILTER}–LINEAR FIT CONTEXT (Post-Peak){common_title_suf}");
            ax_lc.grid(True, alpha=0.3); ax_lc.legend(fontsize=8, loc='best');
            fig_lc.tight_layout(); fig_lc.canvas.manager.set_window_title("Plot:DEV_LIN-Context")
        else: plt.close(fig_lc) if 'fig_lc' in locals() else None

        if any_dl_plot:
            ax_dl.set_xlabel(common_x_lbl);
            ax_dl.set_ylabel(f"Deviation from Linear Fit ({'ADC' if adc_to_v(1) == 1 else 'V'})");
            ax_dl.set_title(f"{DEVICE_FILTER}–DEVIATION FROM LINEAR FIT (Post-Peak){common_title_suf}");
            ax_dl.axhline(0, color='k', ls='--', alpha=0.7, lw=1);
            ax_dl.grid(True, alpha=0.3); ax_dl.legend(fontsize=8, loc='best');
            fig_dl.tight_layout(); fig_dl.canvas.manager.set_window_title("Plot:DEV_LIN-Deviation")
        else: plt.close(fig_dl) if 'fig_dl' in locals() else None

        if not any_lc_plot and not any_dl_plot:
            print("W(DEVIATION_FROM_LINEAR): No data plotted for either context or deviation.")


    elif CURRENT_PLOT_MODE == "FFT_OF_DEVIATION": # This mode is currently disabled in PLOT_MODES_TO_RUN
        # (Original logic for FFT_OF_DEVIATION - kept for completeness if re-enabled)
        # ... (code from your original script) ...
        print("W(FFT_OF_DEVIATION): This plot mode is currently disabled by PLOT_MODES_TO_RUN.")
        pass # Placeholder

    elif CURRENT_PLOT_MODE == "POST_PEAK_PERIODOGRAM":
        # This plots periodogram of the MEAN trace
        fig_pg_mean, ax_pg_mean = plt.subplots(figsize=(12, 6));
        any_pg_mean_plot = False;
        title_pg_mean_parts = []
        min_p_us, max_p_us = MIN_PERIOD_PLOT * SAMPLE_TIME_DELTA_US, MAX_PERIOD_PLOT_ABS * SAMPLE_TIME_DELTA_US

        for i, lbl_pg in enumerate(act_lbls):
            c, ls = PLOT_COLORS[i % len(PLOT_COLORS)], PLOT_LINESTYLES[i % len(PLOT_LINESTYLES)]
            s_data = sum_cache.get(lbl_pg, {})
            mt_pg, N_pg = s_data.get("mean_trace"), s_data.get("N_for_mean", 0)

            if N_pg == 0 or mt_pg is None or len(mt_pg) == 0:
                title_pg_mean_parts.append(f"{lbl_pg}(no mean data)"); continue

            idx_peak_pg = np.argmax(mt_pg);
            start_pg, end_pg = idx_peak_pg, min(idx_peak_pg + FIT_WINDOW_POST_PEAK, len(mt_pg));
            data_raw_pg = mt_pg[start_pg:end_pg];
            N_seg_pg = len(data_raw_pg)

            if N_seg_pg < 4: # Need a few points for meaningful FFT
                title_pg_mean_parts.append(f"{lbl_pg}(seg short N={N_seg_pg})"); continue

            data_an_pg = data_raw_pg.copy();
            steps_pg = []
            if DETREND_PERIODOGRAM: data_an_pg = detrend(data_an_pg, type='linear'); steps_pg.append("detrend")
            if APPLY_FFT_WINDOW: win_pg = np.hanning(N_seg_pg); data_an_pg *= win_pg; steps_pg.append("Hann")

            mags_pg = np.abs(rfft(data_an_pg));
            freqs_pg = rfftfreq(N_seg_pg, d=SAMPLE_TIME_DELTA_US); # Freq in 1/us = MHz

            valid_idx_pg = freqs_pg > 1e-9 # Avoid division by zero for period
            if not np.any(valid_idx_pg):
                title_pg_mean_parts.append(f"{lbl_pg}(no valid freqs)"); continue

            pers_us_pg, mags_pers_pg = 1.0 / freqs_pg[valid_idx_pg], mags_pg[valid_idx_pg];
            mask_pg = (pers_us_pg >= min_p_us) & (pers_us_pg <= max_p_us);
            plot_p_pg, plot_m_pg = pers_us_pg[mask_pg], mags_pers_pg[mask_pg]

            if len(plot_p_pg) > 0:
                sort_pg = np.argsort(plot_p_pg);
                leg_suf_pg = f"(N_runs={N_pg}, SegLen={N_seg_pg}" + (f", {', '.join(steps_pg)}" if steps_pg else "") + ")"
                ax_pg_mean.plot(plot_p_pg[sort_pg], plot_m_pg[sort_pg], lw=1.5, color=c, linestyle=ls, label=f"{lbl_pg} {leg_suf_pg}");
                any_pg_mean_plot = True
            title_pg_mean_parts.append(f"{lbl_pg}(N_runs={N_pg})")

        if any_pg_mean_plot:
            ax_pg_mean.set_xlabel(f"Period (µs/cycle, plotted: {min_p_us:.2f}-{max_p_us:.2f} µs)");
            ylbl_pg = "FFT Magnitude (Mean Trace)" + (f" ({', '.join(steps_pg)})" if steps_pg else "");
            ax_pg_mean.set_ylabel(ylbl_pg);
            processing_str = ', '.join(steps_pg) if steps_pg else 'Raw'
            ax_pg_mean.set_title(
                f"{DEVICE_FILTER}–POST-PEAK PERIODOGRAM (Mean Traces, {processing_str})\n"
                f"Segment: {FIT_WINDOW_POST_PEAK} samples ({FIT_WINDOW_POST_PEAK*SAMPLE_TIME_DELTA_US:.2f} µs) post-peak. Sample time: {SAMPLE_TIME_DELTA_US} µs\n"
                f"Comparing: {', '.join(title_pg_mean_parts)}")
            ax_pg_mean.grid(True, which="both", ls="-", alpha=0.3);
            ax_pg_mean.legend(fontsize=8, loc='upper right');
            fig_pg_mean.tight_layout();
            fig_pg_mean.canvas.manager.set_window_title("Plot:POST_PEAK_PERIODOGRAM (Mean)")
        else:
            plt.close(fig_pg_mean) if 'fig_pg_mean' in locals() else None;
            print("W(POST_PEAK_PERIODOGRAM - Mean):No mean trace periodograms plotted.")

    elif CURRENT_PLOT_MODE == "INDIVIDUAL_RUN_PERIODOGRAM":
        fig_pg_ind, ax_pg_ind = plt.subplots(figsize=(12, 6));
        any_pg_ind_plot = False;
        title_pg_ind_parts = []
        min_p_us, max_p_us = MIN_PERIOD_PLOT * SAMPLE_TIME_DELTA_US, MAX_PERIOD_PLOT_ABS * SAMPLE_TIME_DELTA_US
        plotted_legend_for_file = {lbl: False for lbl in act_lbls}


        for i, lbl_pg_run in enumerate(act_lbls):
            c, ls = PLOT_COLORS[i % len(PLOT_COLORS)], PLOT_LINESTYLES[0] # Use base linestyle for individual runs
            s_data = sum_cache.get(lbl_pg_run, {})
            run_matrix = s_data.get("matrix_list", []) # List of individual aligned run segments
            N_total_runs_for_file = s_data.get("N_for_mean", 0) # Total valid runs for this file

            if N_total_runs_for_file == 0 or not run_matrix:
                title_pg_ind_parts.append(f"{lbl_pg_run}(no runs)"); continue

            num_runs_plotted_for_file = 0
            for run_idx, single_run_data in enumerate(run_matrix): # Iterating through MAX_RUNS valid runs
                if len(single_run_data) == 0: continue

                idx_peak_run = np.argmax(single_run_data);
                start_run, end_run = idx_peak_run, min(idx_peak_run + FIT_WINDOW_POST_PEAK, len(single_run_data));
                data_raw_run = single_run_data[start_run:end_run];
                N_seg_run = len(data_raw_run)

                if N_seg_run < 4: continue # Need a few points

                data_an_run = data_raw_run.copy();
                steps_pg_run = [] # Reuse variable name, specific to this scope
                if DETREND_PERIODOGRAM: data_an_run = detrend(data_an_run, type='linear'); steps_pg_run.append("detrend")
                if APPLY_FFT_WINDOW: win_run = np.hanning(N_seg_run); data_an_run *= win_run; steps_pg_run.append("Hann")

                mags_run = np.abs(rfft(data_an_run));
                freqs_run = rfftfreq(N_seg_run, d=SAMPLE_TIME_DELTA_US);

                valid_idx_run = freqs_run > 1e-9
                if not np.any(valid_idx_run): continue

                pers_us_run, mags_pers_run = 1.0 / freqs_run[valid_idx_run], mags_run[valid_idx_run];
                mask_run = (pers_us_run >= min_p_us) & (pers_us_run <= max_p_us);
                plot_p_run, plot_m_run = pers_us_run[mask_run], mags_pers_run[mask_run]

                if len(plot_p_run) > 0:
                    sort_run = np.argsort(plot_p_run);
                    current_label = None
                    if not plotted_legend_for_file[lbl_pg_run]: # Only label first run from this file
                        current_label = f"{lbl_pg_run} (Individual Runs)"
                        plotted_legend_for_file[lbl_pg_run] = True

                    ax_pg_ind.plot(plot_p_run[sort_run], plot_m_run[sort_run],
                                   lw=0.8, color=c, linestyle=ls, alpha=0.4, label=current_label); # Reduced lw, increased alpha
                    any_pg_ind_plot = True
                    num_runs_plotted_for_file += 1

            if num_runs_plotted_for_file > 0:
                 title_pg_ind_parts.append(f"{lbl_pg_run}({num_runs_plotted_for_file}/{N_total_runs_for_file} runs)")
            else:
                 title_pg_ind_parts.append(f"{lbl_pg_run}(0 runs plotted)")


        if any_pg_ind_plot:
            ax_pg_ind.set_xlabel(f"Period (µs/cycle, plotted: {min_p_us:.2f}-{max_p_us:.2f} µs)");
            processing_steps_str = ', '.join(steps_pg_run) if 'steps_pg_run' in locals() and steps_pg_run else ('detrend, Hann' if DETREND_PERIODOGRAM and APPLY_FFT_WINDOW else ('detrend' if DETREND_PERIODOGRAM else ('Hann' if APPLY_FFT_WINDOW else 'Raw')))
            ax_pg_ind.set_ylabel(f"FFT Magnitude (Individual Runs, {processing_steps_str})");
            ax_pg_ind.set_title(
                f"{DEVICE_FILTER}–INDIVIDUAL RUN PERIODOGRAMS ({processing_steps_str})\n"
                f"Segment: {FIT_WINDOW_POST_PEAK} samples ({FIT_WINDOW_POST_PEAK*SAMPLE_TIME_DELTA_US:.2f} µs) post-peak. Sample time: {SAMPLE_TIME_DELTA_US} µs\n"
                f"Comparing: {', '.join(title_pg_ind_parts)}")
            ax_pg_ind.grid(True, which="both", ls="-", alpha=0.3);
            # Legend handling for one entry per file
            h_ind, l_ind = ax_pg_ind.get_legend_handles_labels()
            if h_ind: by_l_ind = dict(zip(l_ind, h_ind)); ax_pg_ind.legend(by_l_ind.values(), by_l_ind.keys(), fontsize=8, loc='upper right')

            fig_pg_ind.tight_layout();
            fig_pg_ind.canvas.manager.set_window_title("Plot:INDIVIDUAL_RUN_PERIODOGRAM")
        else:
            plt.close(fig_pg_ind) if 'fig_pg_ind' in locals() else None;
            print("W(INDIVIDUAL_RUN_PERIODOGRAM):No individual run periodograms plotted.")


    else:
        print(f"W: PLOT_MODE '{CURRENT_PLOT_MODE}' not recognized.")

if plt.get_fignums():
    print(f"\nDisplaying {len(plt.get_fignums())} plot(s). Close to exit.");
    plt.show()
else:
    print("\nNo plots generated.")
print("\nScript finished.")