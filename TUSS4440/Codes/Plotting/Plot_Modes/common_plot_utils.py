# --- START OF FILE Plot_Modes/common_plot_utils.py ---

import numpy as np
from scipy.signal import detrend as sp_detrend, get_window
from numpy.fft import rfft, rfftfreq
from scipy.optimize import curve_fit


def _detrend_exponential_data(segment_data, return_trend_component=False):
    """
    Detrends data with an exponential decay model y = C + A*exp(B*x).
    First tries a non-linear fit, then falls back to a log-linear fit.
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
    c_guess = float(np.mean(y[int((1 - frac_tail) * len(y)):]))

    # ── 2. Niet-lineaire fit  y = C + A·exp(Bx)  ──────────────────────────
    def _model(x_, A, B, C):  # B < 0 verwacht
        return C + A * np.exp(B * x_)

    a_guess = max(y[0] - c_guess, 1e-6)
    b_guess = -1.0 / max(len(y), 1)  # milde negatieve helling

    try:
        popt, _ = curve_fit(
            _model, x, y,
            p0=[a_guess, b_guess, c_guess],
            bounds=([0, -np.inf, c_guess * 0.5],
                    [np.inf, 0, c_guess * 1.5]),
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
            log_y = np.log(y + offset)
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


def get_periodogram_analysis_steps(data_segment, sample_delta_us, do_detrend_gate, current_detrend_type,
                                   do_apply_fft_window):
    """
    Performs all steps for periodogram analysis and returns intermediate results for plotting.
    This is the new core function for periodogram calculations.
    """
    if data_segment is None or len(data_segment) < 4:
        return None

    original_segment = data_segment.copy()
    processing_steps = []
    trend_line = None
    detrended_segment = original_segment.copy()

    # Step 1: Detrending
    if do_detrend_gate and current_detrend_type.lower() != "none":
        if current_detrend_type.lower() == 'exponential':
            detrended_segment, trend_line, _ = _detrend_exponential_data(original_segment, return_trend_component=True)
            processing_steps.append("exp_detrend")
        elif current_detrend_type.lower() == 'linear':
            detrended_segment = sp_detrend(original_segment, type='linear')
            trend_line = original_segment - detrended_segment
            processing_steps.append("lin_detrend")
    else:
        processing_steps.append("Raw (No Detrend)")

    # Step 2: Windowing (Optional)
    final_segment_for_fft = detrended_segment
    if do_apply_fft_window:
        win = np.hanning(len(detrended_segment))
        final_segment_for_fft = detrended_segment * win
        processing_steps.append("Hann")

    # Step 3: FFT (using your zero-padding logic)
    ZERO_PAD_FACTOR = 16
    n_fft = int(2 ** np.ceil(np.log2(len(final_segment_for_fft) * ZERO_PAD_FACTOR)))

    mags_fft = np.abs(rfft(final_segment_for_fft, n=n_fft))
    freqs_fft = rfftfreq(n_fft, d=sample_delta_us)

    # Convert frequency to period
    valid_freq_idx = freqs_fft > 1e-9
    if not np.any(valid_freq_idx):
        return None

    periods_us_all = 1.0 / freqs_fft[valid_freq_idx]
    mags_at_periods = mags_fft[valid_freq_idx]

    return {
        "original_segment": original_segment,
        "trend_line": trend_line,
        "detrended_segment": detrended_segment,
        "final_segment_for_fft": final_segment_for_fft,
        "periods": periods_us_all,
        "magnitudes": mags_at_periods,
        "processing_steps_str": ", ".join(processing_steps) if processing_steps else "None"
    }


def _calculate_generic_periodogram_common(data_segment, sample_delta_us,
                                          do_detrend_gate, current_detrend_type,
                                          do_apply_fft_window,
                                          min_period_us_plot, max_period_us_plot):
    """
    Refactored to use the new analysis function. This ensures all periodogram
    calculations are consistent and avoids code duplication.
    """
    analysis_data = get_periodogram_analysis_steps(
        data_segment, sample_delta_us, do_detrend_gate,
        current_detrend_type, do_apply_fft_window
    )

    if analysis_data is None:
        return None, None, []

    periods_us_all = analysis_data["periods"]
    mags_at_periods = analysis_data["magnitudes"]

    # Filter by plotting range for periods
    plot_range_mask = (periods_us_all >= min_period_us_plot) & (periods_us_all <= max_period_us_plot)
    plot_periods = periods_us_all[plot_range_mask]
    plot_mags = mags_at_periods[plot_range_mask]

    if len(plot_periods) == 0:
        return None, None, analysis_data["processing_steps_str"].split(", ")

    # Sort by period for plotting
    sort_idx = np.argsort(plot_periods)
    return plot_periods[sort_idx], plot_mags[sort_idx], analysis_data["processing_steps_str"].split(", ")


def plot_runs_common(ax, df_plot, label_plot, color_plot, plot_mode_str, settings):
    """
    Plots individual runs for RAW mode.
    """
    all_run_ids_in_df = sorted(df_plot["Run"].unique())
    plot_count_legend_entries = 0
    plotted_something_from_file = False

    if plot_mode_str == "RAW":
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

    print(f"W(plot_runs_common): Called with unsupported mode '{plot_mode_str}'. Only 'RAW' is valid.")
    return False, 0

# --- END OF FILE Plot_Modes/common_plot_utils.py ---