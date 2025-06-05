# Plot_Mods/common_plot_utils.py
import numpy as np
from scipy.signal import detrend
from numpy.fft import rfft, rfftfreq

# De adc_to_v functie wordt via het settings object doorgegeven in deze refactoring.

# ── bovenin het bestand (extra import) ────────────────────────────────────
from scipy.optimize import curve_fit
# -------------------------------------------------------------------------


def _detrend_exponential_data(segment_data, return_trend_component=False):
    """
    Detrends data by fitting   y = C + A * exp(B*x)   (C = DC-offset)
    Falls back to the old log-polyfit (A*exp(Bx)) if non-linear fitting fails.

    Returns:
        (residual, params_dict)               # if return_trend_component == False
        (residual, trend, params_dict)        # if return_trend_component == True

    params_dict keys: "B_slope", "A_amplitude", "C_offset", "method"
    """
    # ── Voorwaarden checken ───────────────────────────────────────────────
    params = {"B_slope": None, "A_amplitude": None,
              "C_offset": None, "method": None}

    if segment_data is None or len(segment_data) < 4:
        return (segment_data, None, params) if return_trend_component \
               else (segment_data, params)

    y = segment_data.astype(float).copy()
    x = np.arange(len(y))

    # ── 1. Offset (C) grof schatten uit de staart ─────────────────────────
    frac_tail = 0.10
    c_guess   = float(np.mean(y[int((1 - frac_tail) * len(y)):]))

    # ── 2. Niet-lineaire fit  y = C + A·exp(Bx)  ──────────────────────────
    def _model(x_, A, B, C):          # B < 0 verwacht
        return C + A * np.exp(B * x_)

    a_guess = max(y[0] - c_guess, 1e-6)
    b_guess = -1.0 / max(len(y), 1)   # milde negatieve helling

    try:
        popt, _ = curve_fit(
            _model, x, y,
            p0=[a_guess, b_guess, c_guess],
            bounds=([0, -np.inf, c_guess * 0.5],
                    [np.inf, 0,     c_guess * 1.5]),
            maxfev=10_000
        )
        A_fit, B_fit, C_fit = popt
        trend = _model(x, *popt)
        residual = y - trend

        params.update({"B_slope": B_fit,
                       "A_amplitude": A_fit,
                       "C_offset": C_fit,
                       "method": "offset_exp"})
        result_tuple = (residual, trend, params) if return_trend_component \
                       else (residual, params)
        return result_tuple

    except Exception:
        # ── 3. Fallback op oude log-polyfit zonder C ────────────────────
        try:
            # kleine offset zodat alles positief blijft
            offset = 1e-8 - np.min(y) if np.min(y) <= 0 else 0.0
            log_y  = np.log(y + offset)
            B_lin, log_A_lin = np.polyfit(x, log_y, 1)
            trend = np.exp(log_A_lin + B_lin * x) - offset
            residual = y - trend
            params.update({"B_slope": B_lin,
                           "A_amplitude": np.exp(log_A_lin),
                           "C_offset": 0.0,
                           "method": "legacy_exp"})
            result_tuple = (residual, trend, params) if return_trend_component \
                           else (residual, params)
            return result_tuple
        except Exception:
            # laatste redmiddel: geen detrend
            if return_trend_component:
                return segment_data, None, params
            else:
                return segment_data, params



def plot_runs_common(ax, df_plot, label_plot, color_plot, plot_mode_str, settings):
    """
    Plots individual runs for RAW or THRESHOLD modes.
    plot_mode_str: "RAW" or "THRESHOLD"
    settings: An object containing MAX_RUNS, MAX_POINTS, WINDOW_BEFORE, WINDOW_AFTER, THRESH.
    """
    all_run_ids_in_df = sorted(df_plot["Run"].unique())
    plot_count_legend_entries = 0
    plotted_something_from_file = False

    if plot_mode_str == "THRESHOLD":
        window_b = settings.WINDOW_BEFORE
        window_a = settings.WINDOW_AFTER
        thresh_val = settings.THRESH
        processed_run_attempts = 0
        for run_id in all_run_ids_in_df:
            if settings.MAX_RUNS is not None and processed_run_attempts >= settings.MAX_RUNS:
                break # Stop processing more runs for this file if limit reached

            g = df_plot[df_plot["Run"] == run_id].reset_index(drop=True)
            if g.empty: continue

            processed_run_attempts += 1 # Count this as an attempt

            v_run = g["Voltage"].values
            above = np.where(v_run > thresh_val)[0]
            if not len(above): continue # No trigger found in this run

            onset_idx = above[0]
            data_start_abs = max(0, onset_idx - window_b)
            data_end_abs = min(len(v_run), onset_idx + window_a + 1)
            segment_y = v_run[data_start_abs:data_end_abs]

            expected_len = window_b + window_a + 1
            # Only plot if segment is of expected length (i.e., not truncated at start/end of run)
            if len(segment_y) != expected_len:
                continue

            # X-axis is relative to onset
            start_x_rel = data_start_abs - onset_idx # Should be -window_b if not truncated
            end_x_rel = (data_end_abs - 1) - onset_idx # Should be +window_a if not truncated
            segment_x_rel = np.arange(start_x_rel, end_x_rel + 1)

            if len(segment_x_rel) != len(segment_y): # Should not happen if expected_len is met
                continue

            current_label_for_legend = None
            if plot_count_legend_entries == 0: current_label_for_legend = label_plot

            ax.plot(segment_x_rel, segment_y, '-', lw=1, alpha=0.5, c=color_plot, label=current_label_for_legend)
            if plot_count_legend_entries == 0: plot_count_legend_entries += 1
            plotted_something_from_file = True

        return plotted_something_from_file, plot_count_legend_entries

    elif plot_mode_str == "RAW":
        runs_to_plot = all_run_ids_in_df[:settings.MAX_RUNS] if settings.MAX_RUNS is not None else all_run_ids_in_df
        max_pts = settings.MAX_POINTS
        for run_id in runs_to_plot:
            run_df = df_plot[df_plot["Run"] == run_id]
            if run_df.empty: continue
            x_data, y_data = run_df["TimePlot"], run_df["Voltage"]
            if max_pts and len(x_data) > max_pts: x_data, y_data = x_data.iloc[:max_pts], y_data.iloc[:max_pts]

            if not x_data.empty:
                current_label_for_legend = None
                if plot_count_legend_entries == 0: current_label_for_legend = label_plot
                ax.plot(x_data, y_data, "-", lw=1, alpha=0.6, c=color_plot, label=current_label_for_legend)
                if plot_count_legend_entries == 0: plot_count_legend_entries += 1
                plotted_something_from_file = True
        return plotted_something_from_file, plot_count_legend_entries
    return False, 0


def _calculate_generic_periodogram_common(data_segment, sample_delta_us,
                                          do_detrend_gate, current_detrend_type,
                                          do_apply_fft_window,
                                          min_period_us_plot, max_period_us_plot):
    """
    Calculates a periodogram for a given data segment.
    Returns: (plot_periods, plot_mags, processing_steps_list) or (None, None, [])
    """
    if data_segment is None or len(data_segment) < 4: # Need at least a few points for FFT
        return None, None, []

    data_proc = data_segment.copy()
    current_segment_len = len(data_proc)
    processing_steps = []

    if do_detrend_gate and current_detrend_type.lower() != "none":
        if current_detrend_type.lower() == 'exponential':
            data_proc, _ = _detrend_exponential_data(data_proc.copy(), return_trend_component=False) # Ignore params here
            processing_steps.append("exp_detrend")
        elif current_detrend_type.lower() == 'linear':
            data_proc = detrend(data_proc.copy(), type='linear')
            processing_steps.append("lin_detrend")
        else:
            print(f"W(_cgp_common): Unrecognized DETREND_TYPE '{current_detrend_type}'. No detrending performed by this function call.")
            # No processing step added if detrend type is unknown and not "none"

    if do_apply_fft_window:
        if len(data_proc) > 0: # Ensure data_proc is not empty after potential detrending failure
            data_proc *= np.hanning(len(data_proc))
            processing_steps.append("Hann")
        # else: print(f"W(_cgp_common): data_proc empty before Hanning. Skipping.")

    if len(data_proc) < 4: # Re-check after processing, Hanning doesn't change length
        return None, None, processing_steps # Return steps taken so far

    # Perform FFT

    # ---- resolutie-instelling ------------------------------------
    ZERO_PAD_FACTOR = 16  # 1 = uit, 4–16 werkt goed
    n_fft = int(2 ** np.ceil(np.log2(len(data_proc) * ZERO_PAD_FACTOR)))

    mags_fft = np.abs(rfft(data_proc, n=n_fft))
    freqs_fft = rfftfreq(n_fft, d=sample_delta_us)

    # Filter out zero frequency and convert to periods
    valid_freq_idx = freqs_fft > 1e-9 # Avoid division by zero for period calculation
    if not np.any(valid_freq_idx):
        return None, None, processing_steps

    periods_us_all = 1.0 / freqs_fft[valid_freq_idx]
    mags_at_periods = mags_fft[valid_freq_idx]

    # Filter by plotting range for periods
    plot_range_mask = (periods_us_all >= min_period_us_plot) & (periods_us_all <= max_period_us_plot)
    plot_periods = periods_us_all[plot_range_mask]
    plot_mags = mags_at_periods[plot_range_mask]

    if len(plot_periods) == 0:
        return None, None, processing_steps # No data in the desired period range

    # Sort by period for plotting
    sort_idx = np.argsort(plot_periods)
    return plot_periods[sort_idx], plot_mags[sort_idx], processing_steps