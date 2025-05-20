# -*- coding: utf-8 -*-
# --- START OF FILE plotting_master_improved_v8.py ---

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
    "THRESHOLD",  # Zou nu correct moeten uitlijnen
    "SUMMARY",
    "DIFFERENCE",
    "DEVIATION_FROM_LINEAR",
    "FFT_OF_DEVIATION",
    "POST_PEAK_PERIODOGRAM",
]

# ----------------------------------------------------------------
# --------- General Settings -------------------------------------
TARGET_DATA_SUBFOLDER = os.path.join("..", "data", "UltraSoon_Measurements")
SAMPLE_TIME_DELTA_US = 4.63

_CSV_BASE_FILES = [
    #"boopo_meta_1.csv",
    # "boopo_meta_2.csv",
    #"boopo_meta_3.csv",
    "gleuf_meta_1.csv",
    # "gleuf_2_meta_1.csv",
]
LABELS = [
    #"1 (no gleuf)",
    # "2 (no gleuf)",
    #"3 (no gleuf)",
    "4 (gleuf)",
    # "5 (gleuf)",
]
CSV_FILES = [os.path.join(TARGET_DATA_SUBFOLDER, fname) for fname in _CSV_BASE_FILES]
if len(CSV_FILES) != len(LABELS): raise ValueError("Mismatch CSV_FILES/LABELS.")
if not 1 <= len(CSV_FILES) <= 6: raise ValueError(f"Need 1-6 CSV files, got {len(CSV_FILES)}.")

DEVICE_FILTER = "Master";
ADC_BITS = 12;
V_REF = 3.3;
MAX_RUNS = 100
THRESH = 1400;
# WINDOW_BEFORE en WINDOW_AFTER worden nu consistent gebruikt door THRESHOLD en SUMMARY
WINDOW_BEFORE = 50  # Samples voor de trigger (voor THRESHOLD en SUMMARY context)
WINDOW_AFTER = 500  # Samples na de trigger (voor THRESHOLD en SUMMARY context)

FIT_WINDOW_POST_PEAK = 500
DETREND_PERIODOGRAM = True;
MIN_PERIOD_PLOT = 2;
MAX_PERIOD_PLOT_ABS = 40;
APPLY_FFT_WINDOW = True
MIN_FREQ_PLOT_MHZ_DEV = 0.001;
MAX_FREQ_PLOT_MHZ_DEV = 0.05
N_PEAKS_PER_RUN = 5;
TIME_UNIT = "ms";
MAX_POINTS = None
SEG_LEN, MAX_LAG, N_EXTREME, OFFSET_EXT, EXTREME_VALUES = 50, 5, 10, 1, [0, 5, 10, 15]
# ----------------------------------------------------------------

PLOT_COLORS = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']
PLOT_LINESTYLES = ['-', '--', ':', '-.', (0, (3, 1, 1, 1)), (0, (5, 10))]


def adc_to_v(adc_val): return adc_val


def load_and_prepare_data(csv_f, dev_filter, ts_factor):
    try:
        df = pd.read_csv(csv_f)
    except FileNotFoundError:
        print(f"E: File '{csv_f}' not found."); return None
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
    runs_s = sorted(df_s.Run.unique())[:MAX_RUNS];
    al_matrix = []
    # Gebruik hier dezelfde WINDOW_BEFORE en WINDOW_AFTER als THRESHOLD plot voor consistentie
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
        if len(seg_s) == exp_len_s and start_s >= 0 and stop_s <= len(v_s):
            al_matrix.append(seg_s);
            valid_r_cnt += 1
        if valid_r_cnt >= MAX_RUNS: break
    return [], al_matrix, valid_r_cnt


def calculate_trace_stats(al_matrix_loc):
    exp_len_loc = WINDOW_BEFORE + WINDOW_AFTER + 1  # Consistent met get_summary_data
    if not al_matrix_loc: return np.zeros(exp_len_loc), 0, np.zeros(exp_len_loc), np.zeros(exp_len_loc)
    al_data_np = np.array(al_matrix_loc);
    N_loc = al_data_np.shape[0]
    if N_loc == 0: return np.zeros(exp_len_loc), 0, np.zeros(exp_len_loc), np.zeros(exp_len_loc)
    m_tr = al_data_np.mean(axis=0);
    std_tr_loc, sem_tr_loc = np.zeros_like(m_tr), np.zeros_like(m_tr)
    if N_loc > 1:
        std_tr_loc = al_data_np.std(axis=0, ddof=1)
        sem_tr_loc = std_tr_loc / np.sqrt(N_loc)  # Gecorrigeerd
    return m_tr, N_loc, sem_tr_loc, std_tr_loc


if TIME_UNIT.lower() == "s":
    ts_factor_raw, tu_raw_lbl = 1e-6, "s"
elif TIME_UNIT.lower() == "ms":
    ts_factor_raw, tu_raw_lbl = 1e-3, "ms"
else:
    ts_factor_raw, tu_raw_lbl = 1.0, "µs"

print("Loading data...");
dfs, act_lbls = [], []  # dfs en act_lbls zijn globaal hier
for i, fp in enumerate(CSV_FILES):
    if not os.path.exists(fp): print(f"W:File '{fp}' not found.Skip.");continue
    df_l = load_and_prepare_data(fp, DEVICE_FILTER, ts_factor_raw)
    if df_l is not None and not df_l.empty:
        dfs.append(df_l);act_lbls.append(LABELS[i])
    else:
        print(f"W:Skip '{fp}' (load err/no data).")
if not dfs: print("E:No data loaded.Exit."); exit()
print(f"Data loaded for: {', '.join(act_lbls)}")

sum_cache = {};
needs_sum_calc = any(
    m in ["SUMMARY", "DIFFERENCE", "DEVIATION_FROM_LINEAR", "FFT_OF_DEVIATION", "POST_PEAK_PERIODOGRAM"] for m in
    PLOT_MODES_TO_RUN)
if needs_sum_calc:
    print("Pre-calc summary data...")
    for i, df_c in enumerate(dfs):  # Gebruik dfs
        lbl_c = act_lbls[i];
        _, mtx_c, r_c = get_summary_data(df_c, lbl_c);
        m_c, N_c, s_c, sd_c = calculate_trace_stats(mtx_c)
        sum_cache[lbl_c] = {"matrix_list": mtx_c, "runs_count": r_c, "mean_trace": m_c, "N_for_mean": N_c,
                            "sem_trace": s_c, "std_trace": sd_c}
        if N_c > 0:
            print(f"  '{lbl_c}':{N_c} runs for mean.")
        else:
            print(f"  W '{lbl_c}':No runs for mean.")
    print("Summary pre-calc done.")

needs_dev_calc = "DEVIATION_FROM_LINEAR" in PLOT_MODES_TO_RUN or "FFT_OF_DEVIATION" in PLOT_MODES_TO_RUN
if needs_dev_calc:
    print("Pre-calc deviation data...")
    for lbl_dc in act_lbls:  # Gebruik act_lbls
        sum_cache[lbl_dc]["deviation_info"] = None
        if lbl_dc in sum_cache and sum_cache[lbl_dc].get("N_for_mean", 0) > 0 and sum_cache[lbl_dc].get(
                "mean_trace") is not None:
            sdat_dc = sum_cache[lbl_dc];
            mt_dc = sdat_dc["mean_trace"]
            if len(mt_dc) > 0:
                idx_pk_dc = np.argmax(mt_dc);
                start_dc, end_dc = idx_pk_dc, min(idx_pk_dc + FIT_WINDOW_POST_PEAK, len(mt_dc))
                seg_fit_dc = mt_dc[start_dc:end_dc];
                x_coords_dc = np.arange(len(seg_fit_dc))
                if len(seg_fit_dc) >= 2:
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
    runs_to_plot = all_run_ids_in_df[:MAX_RUNS]
    plot_count, plotted_something = 0, False

    if plot_mode_str == "THRESHOLD":
        # Haal window_before en window_after uit kwargs, met fallback naar globale instellingen
        window_b = kwargs.get("window_before", WINDOW_BEFORE)
        window_a = kwargs.get("window_after", WINDOW_AFTER)
        thresh_val = kwargs.get("thresh", THRESH)

        for run_id in runs_to_plot:
            g = df_plot[df_plot["Run"] == run_id].reset_index(drop=True)
            if g.empty: continue
            v_run = g["Voltage"].values
            above = np.where(v_run > thresh_val)[0]
            if not len(above): continue
            onset_idx = above[0]  # Absolute index van onset in deze run

            # Bepaal het segment van Y-waarden rond de onset
            data_start_abs = max(0, onset_idx - window_b)
            data_end_abs = min(len(v_run), onset_idx + window_a + 1)
            segment_y = v_run[data_start_abs:data_end_abs]

            if len(segment_y) == 0: continue

            # Maak de X-as relatief aan de onset_idx
            # De eerste sample in segment_y komt van v_run[data_start_abs]
            # De relatieve index daarvan is data_start_abs - onset_idx
            # De laatste sample in segment_y komt van v_run[data_end_abs - 1]
            # De relatieve index daarvan is (data_end_abs - 1) - onset_idx
            start_x_rel = data_start_abs - onset_idx
            end_x_rel = (data_end_abs - 1) - onset_idx
            segment_x_rel = np.arange(start_x_rel, end_x_rel + 1)

            # Zorg ervoor dat segment_x_rel en segment_y even lang zijn (zou moeten)
            if len(segment_x_rel) != len(segment_y):
                # Dit zou niet moeten gebeuren als de logica correct is.
                # print(f"Debug: Mismatch len x ({len(segment_x_rel)}) and y ({len(segment_y)}) for run {run_id}")
                continue

            run_label = None
            if plot_count < 1 and len(runs_to_plot) == 1:
                run_label = f"{label_plot} Run {run_id}"
            elif plot_count < 2 and len(runs_to_plot) > 1:
                run_label = f"{label_plot} Run {run_id}"
            elif plot_count < 1 and len(act_lbls) == 1 and len(runs_to_plot) > 0:
                run_label = f"{label_plot} Run {run_id}"

            ax.plot(segment_x_rel, segment_y, '-', lw=1, alpha=0.5, c=color_plot, label=run_label)
            plot_count += 1
            plotted_something = True
        return plotted_something, plot_count

    elif plot_mode_str == "RAW":
        # (RAW logica ongewijzigd)
        max_pts = kwargs.get("max_points", MAX_POINTS)
        for run_id in runs_to_plot:
            run_df = df_plot[df_plot["Run"] == run_id]
            if run_df.empty: continue
            x_data, y_data = run_df["TimePlot"], run_df["Voltage"]
            if max_pts and len(x_data) > max_pts: x_data, y_data = x_data.iloc[:max_pts], y_data.iloc[:max_pts]
            if not x_data.empty:
                run_label = None
                if plot_count < 1 and len(runs_to_plot) == 1:
                    run_label = f"{label_plot} Run {run_id}"
                elif plot_count < 2 and len(runs_to_plot) > 1:
                    run_label = f"{label_plot} Run {run_id}"
                elif plot_count < 1 and len(act_lbls) == 1 and len(runs_to_plot) > 0:
                    run_label = f"{label_plot} Run {run_id}"
                ax.plot(x_data, y_data, "-", lw=1, alpha=0.6, c=color_plot, label=run_label)
                plot_count += 1;
                plotted_something = True
        return plotted_something, plot_count
    return False, 0


for PLOT_MODE_ITER in PLOT_MODES_TO_RUN:
    CURRENT_PLOT_MODE = PLOT_MODE_ITER.upper()
    print(f"\n--- Generating plot for mode: {CURRENT_PLOT_MODE} ---")

    if CURRENT_PLOT_MODE == "RAW":
        # (RAW plot hoofd logica ongewijzigd)
        fig_r, ax_r = plt.subplots(figsize=(12, 6));
        counts_r = [];
        any_plot_r = False
        for i, df_r in enumerate(dfs):
            lbl_r, c_r = act_lbls[i], PLOT_COLORS[i % len(PLOT_COLORS)]
            plot_df_r, count_r = plot_runs(ax_r, df_r, lbl_r, c_r, "RAW", max_points=MAX_POINTS)
            if plot_df_r: any_plot_r = True
            counts_r.append(count_r)
        if any_plot_r:
            ax_r.set_xlabel(f"Time ({tu_raw_lbl})");
            ax_r.set_ylabel(f"V({'ADC' if adc_to_v(1) == 1 else 'V'})")
            titles_r = [f"{act_lbls[j]}({counts_r[j]} runs)" for j in range(len(act_lbls)) if counts_r[j] > 0]
            ax_r.set_title(
                f"{DEVICE_FILTER}–RAW:Full Traces(up to {MAX_RUNS} runs/file)\nComparing {', '.join(titles_r)}")
            h_r, l_r = ax_r.get_legend_handles_labels()
            if h_r: by_l_r = dict(zip(l_r, h_r)); ax_r.legend(by_l_r.values(), by_l_r.keys(), title="Runs Shown",
                                                              fontsize=8, loc='upper right')
            ax_r.grid(True, alpha=0.3);
            fig_r.tight_layout();
            fig_r.canvas.manager.set_window_title("Plot:RAW")
        else:
            plt.close(fig_r) if 'fig_r' in locals() else None; print("W(RAW):No raw traces plotted.")

    elif CURRENT_PLOT_MODE == "THRESHOLD":
        fig_th, ax_th = plt.subplots(figsize=(12, 6));
        counts_th = [];
        any_plot_th = False
        for i, df_th in enumerate(dfs):  # dfs is globaal
            lbl_th, c_th = act_lbls[i], PLOT_COLORS[i % len(PLOT_COLORS)]  # act_lbls is globaal
            # Gebruik WINDOW_BEFORE en WINDOW_AFTER uit de globale scope voor deze plot
            # Deze worden doorgegeven via kwargs aan plot_runs
            plot_df_th, count_th = plot_runs(ax_th, df_th, lbl_th, c_th, "THRESHOLD",
                                             window_before=WINDOW_BEFORE, window_after=WINDOW_AFTER, thresh=THRESH)
            if plot_df_th: any_plot_th = True
            counts_th.append(count_th)
        if any_plot_th:
            # De x-as is nu relatief, 0 is het triggerpunt.
            ax_th.set_xlabel(
                f"Sample-index relative to Trigger (0 = first V > {THRESH} ADC, {SAMPLE_TIME_DELTA_US}µs/sample)")
            ax_th.set_ylabel(f"V({'ADC' if adc_to_v(1) == 1 else 'V'})")
            titles_th = [f"{act_lbls[j]}({counts_th[j]} runs)" for j in range(len(act_lbls)) if counts_th[j] > 0]
            ax_th.set_title(
                f"{DEVICE_FILTER}–THRESHOLD: Aligned Traces (Window: [-{WINDOW_BEFORE}, +{WINDOW_AFTER}] samples around trigger)\nComparing {', '.join(titles_th)}")
            h_th, l_th = ax_th.get_legend_handles_labels()
            if h_th: by_l_th = dict(zip(l_th, h_th)); ax_th.legend(by_l_th.values(), by_l_th.keys(), title="Runs Shown",
                                                                   fontsize=8, loc='upper right')
            ax_th.grid(True, alpha=0.3);
            fig_th.tight_layout();
            fig_th.canvas.manager.set_window_title("Plot:THRESHOLD")
        else:
            plt.close(fig_th) if 'fig_th' in locals() else None; print(
                "W(THRESHOLD):No threshold-aligned traces plotted.")

    elif CURRENT_PLOT_MODE == "SUMMARY":
        # (Logica ongewijzigd)
        mean_trace_data_exists = any(label in sum_cache and sum_cache[label]["N_for_mean"] > 0 for label in act_lbls)
        if mean_trace_data_exists:
            fig_s, ax_s = plt.subplots(figsize=(12, 6));
            t_rel_s = np.arange(-WINDOW_BEFORE, WINDOW_AFTER + 1);
            title_s = []
            for i, lbl_s in enumerate(act_lbls):
                if lbl_s in sum_cache:
                    cache_s = sum_cache[lbl_s];
                    mt_s, N_s, sem_s, std_s = cache_s.get("mean_trace"), cache_s.get("N_for_mean"), cache_s.get(
                        "sem_trace"), cache_s.get("std_trace")
                    if not (N_s and N_s > 0 and mt_s is not None and len(mt_s) == len(t_rel_s)): title_s.append(
                        f"{lbl_s}(no data)"); continue
                    title_s.append(f"{lbl_s}(N={N_s})");
                    c, ls = PLOT_COLORS[i % len(PLOT_COLORS)], PLOT_LINESTYLES[i % len(PLOT_LINESTYLES)]
                    ax_s.plot(t_rel_s, mt_s, lw=2, color=c, linestyle=ls, label=f"Mean({lbl_s})")
                    if N_s > 1:
                        tcrit = t.ppf(0.975, df=N_s - 1)
                        if sem_s is not None and isinstance(sem_s, np.ndarray): ci_m = tcrit * sem_s;ax_s.fill_between(
                            t_rel_s, mt_s - ci_m, mt_s + ci_m, alpha=0.35, color=c, label=f"95%CI({lbl_s})")
                        if std_s is not None and isinstance(std_s, np.ndarray): pi_m = tcrit * std_s * np.sqrt(
                            1 + 1 / N_s);ax_s.fill_between(t_rel_s, mt_s - pi_m, mt_s + pi_m, alpha=0.15, color=c,
                                                           label=f"95%PI({lbl_s})")
            ax_s.set_xlabel(f"Sample-idx(0=V>{THRESH},{SAMPLE_TIME_DELTA_US}µs/sample)");
            ax_s.set_ylabel(f"V({'ADC' if adc_to_v(1) == 1 else 'V'})")
            ax_s.set_title(
                f"{DEVICE_FILTER}–SUMMARY:MeanTraces±CI&PI\nWin:{WINDOW_BEFORE}pre,{WINDOW_AFTER}post\n{', '.join(title_s)}")
            ax_s.grid(True, alpha=0.3);
            h_s, l_s = ax_s.get_legend_handles_labels()
            if h_s: by_l_s = dict(zip(l_s, h_s));ax_s.legend(by_l_s.values(), by_l_s.keys(), fontsize=8,
                                                             loc='upper right')
            fig_s.tight_layout();
            fig_s.canvas.manager.set_window_title("Plot:SUMMARY-MeanTraces")
        else:
            print("W(SUMMARY):No valid mean trace data.")

    elif CURRENT_PLOT_MODE == "DIFFERENCE":
        # (Logica ongewijzigd)
        fig_d, ax_d = plt.subplots(figsize=(12, 6));
        any_d_plot = False;
        title_d = []
        t_rel_sum_d = np.arange(-WINDOW_BEFORE, WINDOW_AFTER + 1)
        if len(t_rel_sum_d) < 2:
            print("W(DIFF):Win too small.");plt.close(fig_d) if 'fig_d' in locals() else None
        else:
            t_rel_diff_d = (t_rel_sum_d[:-1] + t_rel_sum_d[1:]) / 2.0
            for i, lbl_d in enumerate(act_lbls):
                c, ls = PLOT_COLORS[i % len(PLOT_COLORS)], PLOT_LINESTYLES[i % len(PLOT_LINESTYLES)]
                if lbl_d not in sum_cache or sum_cache[lbl_d].get("N_for_mean", 0) == 0 or sum_cache[lbl_d].get(
                    "mean_trace") is None: title_d.append(f"{lbl_d}(no data)"); continue
                sdat_d = sum_cache[lbl_d];
                mt_d, N_d = sdat_d["mean_trace"], sdat_d["N_for_mean"]
                if len(mt_d) < 2: title_d.append(f"{lbl_d}(trace short)"); continue
                delta_tr_d = np.diff(mt_d)
                if len(t_rel_diff_d) != len(delta_tr_d): title_d.append(f"{lbl_d}(len err)"); continue
                ax_d.plot(t_rel_diff_d, delta_tr_d, lw=1.5, color=c, linestyle=ls, label=f"{lbl_d}(N={N_d})")
                any_d_plot = True;
                title_d.append(f"{lbl_d}(N={N_d})")
            if any_d_plot:
                ts_info = f"({SAMPLE_TIME_DELTA_US}µs/sample)";
                ax_d.set_xlabel(f"Sample-idx(midpoint,0=around onset){ts_info}");
                ax_d.set_ylabel(f"ΔMeanV/ΔSample(ADC/sample)")
                ax_d.set_title(f"{DEVICE_FILTER}–DIFFERENCE:1st-OrderDiff of MeanTraces\n{', '.join(title_d)}")
                ax_d.axhline(0, color='k', ls='--', alpha=0.7, lw=1);
                ax_d.grid(True, alpha=0.3);
                ax_d.legend(fontsize=8, loc='best')
                fig_d.tight_layout();
                fig_d.canvas.manager.set_window_title("Plot:DIFFERENCE(Derivative-like)")
            else:
                plt.close(fig_d) if 'fig_d' in locals() else None; print("W(DIFF):No derivative traces plotted.")

    elif CURRENT_PLOT_MODE == "DEVIATION_FROM_LINEAR":
        # (Logica ongewijzigd)
        fig_lc, ax_lc = plt.subplots(figsize=(12, 6));
        any_lc_plot = False
        fig_dl, ax_dl = plt.subplots(figsize=(12, 6));
        any_dl_plot = False;
        title_dl_all = []
        for i, lbl_dl in enumerate(act_lbls):
            c, ls = PLOT_COLORS[i % len(PLOT_COLORS)], PLOT_LINESTYLES[i % len(PLOT_LINESTYLES)];
            dev_info = sum_cache.get(lbl_dl, {}).get("deviation_info");
            N_runs_dl = sum_cache.get(lbl_dl, {}).get("N_for_mean", 0)
            if dev_info is None: print(f"W(DEV_LIN - {lbl_dl}): No pre-calc dev data. Skip."); title_dl_all.append(
                f"{lbl_dl}(no dev data)"); continue
            dev_array, x_coords, orig_seg, fit_seg, slope_dl = dev_info["deviations_array"], dev_info[
                "x_coords_samples"], dev_info["original_segment_data"], dev_info["linear_fit_on_segment"], dev_info[
                "slope"]
            leg_details = f"(N={N_runs_dl}, slope={slope_dl:.2e})"
            ax_lc.plot(x_coords, orig_seg, lw=1.5, color=c, linestyle=ls, label=f"Data({lbl_dl}{leg_details})",
                       alpha=0.7);
            ax_lc.plot(x_coords, fit_seg, lw=2, color=c, linestyle='--', label=f"Fit({lbl_dl}{leg_details})");
            any_lc_plot = True
            ax_dl.plot(x_coords, dev_array, lw=1.5, color=c, linestyle=ls, label=f"{lbl_dl}{leg_details}");
            any_dl_plot = True
            if lbl_dl not in [item.split(' ')[0] for item in title_dl_all if '(' in item]: title_dl_all.append(
                f"{lbl_dl}(N={N_runs_dl})")
        common_x_lbl = f"Samples from Peak(Time up to {FIT_WINDOW_POST_PEAK * SAMPLE_TIME_DELTA_US:.2f}µs)";
        common_title_suf = f"\nSeg:{FIT_WINDOW_POST_PEAK}samples\n{', '.join(title_dl_all)}"
        if any_lc_plot:
            ax_lc.set_xlabel(common_x_lbl);ax_lc.set_ylabel(f"V({'ADC' if adc_to_v(1) == 1 else 'V'})");ax_lc.set_title(
                f"{DEVICE_FILTER}–LINEAR FIT CONTEXT(Post-Peak){common_title_suf}");ax_lc.grid(True,
                                                                                               alpha=0.3);ax_lc.legend(
                fontsize=8, loc='best');fig_lc.tight_layout();fig_lc.canvas.manager.set_window_title(
                "Plot:DEV_LIN-Context")
        else:
            plt.close(fig_lc) if 'fig_lc' in locals() else None
        if any_dl_plot:
            ax_dl.set_xlabel(common_x_lbl);ax_dl.set_ylabel(
                f"Deviation from Linear Fit({'ADC' if adc_to_v(1) == 1 else 'V'})");ax_dl.set_title(
                f"{DEVICE_FILTER}–DEVIATION FROM LINEAR FIT(Post-Peak){common_title_suf}");ax_dl.axhline(0, color='k',
                                                                                                         ls='--',
                                                                                                         alpha=0.7,
                                                                                                         lw=1);ax_dl.grid(
                True, alpha=0.3);ax_dl.legend(fontsize=8,
                                              loc='best');fig_dl.tight_layout();fig_dl.canvas.manager.set_window_title(
                "Plot:DEV_LIN-Deviation")
        else:
            plt.close(fig_dl) if 'fig_dl' in locals() else None

    elif CURRENT_PLOT_MODE == "FFT_OF_DEVIATION":
        # (Logica ongewijzigd)
        fig_fft_dev, ax_fft_dev = plt.subplots(figsize=(12, 6));
        any_fft_dev_plotted = False;
        title_fft_dev_all = []
        for i_df, label_current_fft_dev in enumerate(act_lbls):
            color_current, linestyle_current = PLOT_COLORS[i_df % len(PLOT_COLORS)], PLOT_LINESTYLES[
                i_df % len(PLOT_LINESTYLES)];
            dev_info = sum_cache.get(label_current_fft_dev, {}).get("deviation_info");
            N_runs_fft_dev = sum_cache.get(label_current_fft_dev, {}).get("N_for_mean", 0)
            if dev_info is None: print(
                f"W(FFT_DEV-{label_current_fft_dev}):No pre-calc dev data.Skip.");title_fft_dev_all.append(
                f"{label_current_fft_dev}(no dev data)"); continue
            deviations_signal, N_dev_segment = dev_info["deviations_array"], dev_info["N_segment"]
            if N_dev_segment < 4: print(
                f"W(FFT_DEV-{label_current_fft_dev}):Dev seg too short({N_dev_segment}pts).Skip.");title_fft_dev_all.append(
                f"{label_current_fft_dev}(seg short N={N_dev_segment})"); continue
            processed_deviations = deviations_signal.copy();
            fft_processing_steps = []
            if APPLY_FFT_WINDOW: hann_window = np.hanning(
                N_dev_segment);processed_deviations *= hann_window;fft_processing_steps.append("Hann")
            fft_magnitudes = np.abs(rfft(processed_deviations));
            fft_frequencies_mhz = rfftfreq(N_dev_segment, d=SAMPLE_TIME_DELTA_US)
            freq_mask = (fft_frequencies_mhz >= MIN_FREQ_PLOT_MHZ_DEV) & (fft_frequencies_mhz <= MAX_FREQ_PLOT_MHZ_DEV);
            plot_frequencies, plot_magnitudes = fft_frequencies_mhz[freq_mask], fft_magnitudes[freq_mask]
            if len(plot_frequencies) > 0:
                sort_order = np.argsort(plot_frequencies);
                legend_label = f"{label_current_fft_dev}(N_runs={N_runs_fft_dev},SegLen={N_dev_segment}" + (
                    f", {', '.join(fft_processing_steps)}" if fft_processing_steps else "") + ")"
                ax_fft_dev.plot(plot_frequencies[sort_order], plot_magnitudes[sort_order], lw=1.5, color=color_current,
                                linestyle=linestyle_current, label=legend_label);
                any_fft_dev_plotted = True
            else:
                print(
                    f"W(FFT_DEV-{label_current_fft_dev}):No freqs in plot range[{MIN_FREQ_PLOT_MHZ_DEV:.3f}-{MAX_FREQ_PLOT_MHZ_DEV:.3f}MHz].")
            title_fft_dev_all.append(f"{label_current_fft_dev}(N={N_runs_fft_dev})")
        if any_fft_dev_plotted:
            ax_fft_dev.set_xlabel(
                f"Freq(MHz=1/µs).Plot range:{MIN_FREQ_PLOT_MHZ_DEV:.3f}-{MAX_FREQ_PLOT_MHZ_DEV:.3f}MHz");
            y_label_fft_dev = "FFT Mag(Deviation Signal)" + (
                f"({', '.join(fft_processing_steps)})" if fft_processing_steps else "");
            ax_fft_dev.set_ylabel(y_label_fft_dev)
            title_main_fft_dev = f"{DEVICE_FILTER}–FFT of (Mean Trace Deviation from Linear Fit)";
            title_suffix_fft_dev = f"\nSeg:{FIT_WINDOW_POST_PEAK}samples({FIT_WINDOW_POST_PEAK * SAMPLE_TIME_DELTA_US:.2f}µs)from peak\n{', '.join(title_fft_dev_all)}";
            ax_fft_dev.set_title(title_main_fft_dev + title_suffix_fft_dev)
            ax_fft_dev.grid(True, which="both", ls="-", alpha=0.3);
            ax_fft_dev.legend(fontsize=8, loc='upper right');
            fig_fft_dev.tight_layout();
            fig_fft_dev.canvas.manager.set_window_title("Plot:FFT_OF_DEVIATION")
        else:
            plt.close(fig_fft_dev) if 'fig_fft_dev' in locals() else None;print(
                "W(FFT_OF_DEVIATION):No FFT traces plotted.")

    elif CURRENT_PLOT_MODE == "POST_PEAK_PERIODOGRAM":
        # (Logica ongewijzigd)
        fig_pg, ax_pg = plt.subplots(figsize=(12, 6));
        any_pg_plot = False;
        title_pg = []
        min_p_us, max_p_us = MIN_PERIOD_PLOT * SAMPLE_TIME_DELTA_US, MAX_PERIOD_PLOT_ABS * SAMPLE_TIME_DELTA_US
        for i, lbl_pg in enumerate(act_lbls):  # dfs en act_lbls zijn globaal
            c, ls = PLOT_COLORS[i % len(PLOT_COLORS)], PLOT_LINESTYLES[i % len(PLOT_LINESTYLES)]
            if lbl_pg not in sum_cache or sum_cache[lbl_pg].get("N_for_mean", 0) == 0 or sum_cache[lbl_pg].get(
                "mean_trace") is None: title_pg.append(f"{lbl_pg}(no data)"); continue
            sdat_pg = sum_cache[lbl_pg];
            mt_pg, N_pg = sdat_pg["mean_trace"], sdat_pg["N_for_mean"]
            if len(mt_pg) == 0: title_pg.append(f"{lbl_pg}(empty trace)"); continue
            idx_peak_pg = np.argmax(mt_pg);
            start_pg, end_pg = idx_peak_pg, min(idx_peak_pg + FIT_WINDOW_POST_PEAK, len(mt_pg));
            data_raw_pg = mt_pg[start_pg:end_pg];
            N_seg_pg = len(data_raw_pg)
            if N_seg_pg < 4: title_pg.append(f"{lbl_pg}(seg short N={N_seg_pg})"); continue
            data_an_pg = data_raw_pg.copy();
            steps_pg = []
            if DETREND_PERIODOGRAM: data_an_pg = detrend(data_an_pg, type='linear');steps_pg.append("detrend")
            if APPLY_FFT_WINDOW: win_pg = np.hanning(N_seg_pg);data_an_pg *= win_pg;steps_pg.append("Hann")
            mags_pg = np.abs(rfft(data_an_pg));
            freqs_pg = rfftfreq(N_seg_pg, d=SAMPLE_TIME_DELTA_US);
            valid_idx_pg = freqs_pg > 1e-9
            if not np.any(valid_idx_pg): title_pg.append(f"{lbl_pg}(no freqs)"); continue
            pers_us_pg, mags_pers_pg = 1.0 / freqs_pg[valid_idx_pg], mags_pg[valid_idx_pg];
            mask_pg = (pers_us_pg >= min_p_us) & (pers_us_pg <= max_p_us);
            plot_p_pg, plot_m_pg = pers_us_pg[mask_pg], mags_pers_pg[mask_pg]
            if len(plot_p_pg) > 0: sort_pg = np.argsort(plot_p_pg);leg_suf_pg = f"(N={N_pg},SegL={N_seg_pg}" + (
                f", {', '.join(steps_pg)}" if steps_pg else "") + ")";ax_pg.plot(plot_p_pg[sort_pg], plot_m_pg[sort_pg],
                                                                                 lw=1.5, color=c, linestyle=ls,
                                                                                 label=f"{lbl_pg}{leg_suf_pg}");any_pg_plot = True
            title_pg.append(f"{lbl_pg}(N={N_pg})")
        if any_pg_plot:
            ax_pg.set_xlabel(f"Periode(µs/cycle,plotted:{min_p_us:.2f}-{max_p_us:.2f}µs)");
            ylbl_pg = "FFT Mag" + (f"({', '.join(steps_pg)})" if steps_pg else "");
            ax_pg.set_ylabel(ylbl_pg);
            ax_pg.set_title(
                f"{DEVICE_FILTER}–POST-PEAK PERIODOGRAM({', '.join(steps_pg) if steps_pg else 'Raw'})\n(Seg:{FIT_WINDOW_POST_PEAK}samples;Sample time:{SAMPLE_TIME_DELTA_US}µs)\n{', '.join(title_pg)}")
            ax_pg.grid(True, which="both", ls="-", alpha=0.3);
            ax_pg.legend(fontsize=8, loc='upper right');
            fig_pg.tight_layout();
            fig_pg.canvas.manager.set_window_title("Plot:POST_PEAK_PERIODOGRAM")
        else:
            plt.close(fig_pg) if 'fig_pg' in locals() else None; print("W(PERIODOGRAM):No traces plotted.")

    else:
        print(f"W: PLOT_MODE '{CURRENT_PLOT_MODE}' not recognized.")

if plt.get_fignums():
    print(f"\nDisplaying {len(plt.get_fignums())} plot(s). Close to exit."); plt.show()
else:
    print("\nNo plots generated.")
print("\nScript finished.")
# --- END OF FILE plotting_master_improved_v8.py ---