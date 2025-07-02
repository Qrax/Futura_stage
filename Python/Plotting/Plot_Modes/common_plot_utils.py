# Sla dit op als: Plot_Modes/common_plot_utils.py (Vervang de volledige inhoud)

import numpy as np
from scipy.signal import detrend as sp_detrend, get_window
from numpy.fft import rfft, rfftfreq
from scipy.optimize import curve_fit


def _detrend_exponential_data(segment_data, return_trend_component=False):
    """
    Detrends data with an exponential decay model y = C + A*exp(B*x).
    First tries a non-linear fit, then falls back to a log-linear fit.
    """
    params = {"B_slope": None, "A_amplitude": None, "C_offset": None, "method": None}
    if segment_data is None or len(segment_data) < 4:
        return (segment_data, None, params) if return_trend_component else (segment_data, params)
    y = segment_data.astype(float).copy()
    x = np.arange(len(y))
    frac_tail = 0.10
    c_guess = float(np.mean(y[int((1 - frac_tail) * len(y)):]))

    def _model(x_, A, B, C):
        return C + A * np.exp(B * x_)

    a_guess = max(y[0] - c_guess, 1e-6)
    b_guess = -1.0 / max(len(y), 1)

    try:
        popt, _ = curve_fit(_model, x, y, p0=[a_guess, b_guess, c_guess],
                            bounds=([0, -np.inf, c_guess * 0.5], [np.inf, 0, c_guess * 1.5]), maxfev=10_000)
        trend = _model(x, *popt)
        residual = y - trend
        params.update({"B_slope": popt[1], "A_amplitude": popt[0], "C_offset": popt[2], "method": "offset_exp"})
        return (residual, trend, params) if return_trend_component else (residual, params)
    except Exception:
        try:
            offset = 1e-8 - np.min(y) if np.min(y) <= 0 else 0.0
            log_y = np.log(y + offset)
            B_lin, log_A_lin = np.polyfit(x, log_y, 1)
            trend = np.exp(log_A_lin + B_lin * x) - offset
            residual = y - trend
            params.update({"B_slope": B_lin, "A_amplitude": np.exp(log_A_lin), "C_offset": 0.0, "method": "legacy_exp"})
            return (residual, trend, params) if return_trend_component else (residual, params)
        except Exception:
            return (segment_data, None, params) if return_trend_component else (segment_data, params)


def get_periodogram_analysis_steps(data_segment, sample_delta_us, do_detrend_gate, current_detrend_type,
                                   do_apply_fft_window):
    """
    Performs all steps for periodogram analysis and returns intermediate results.
    """
    if data_segment is None or len(data_segment) < 4: return None
    original_segment = data_segment.copy()
    trend_line, detrended_segment = None, original_segment.copy()

    if do_detrend_gate and current_detrend_type.lower() != "none":
        if current_detrend_type.lower() == 'exponential':
            detrended_segment, trend_line, _ = _detrend_exponential_data(original_segment, return_trend_component=True)
        elif current_detrend_type.lower() == 'linear':
            detrended_segment = sp_detrend(original_segment, type='linear')
            trend_line = original_segment - detrended_segment

    final_segment_for_fft = detrended_segment * np.hanning(
        len(detrended_segment)) if do_apply_fft_window else detrended_segment

    n_fft = int(2 ** np.ceil(np.log2(len(final_segment_for_fft) * 16)))
    mags_fft = np.abs(rfft(final_segment_for_fft, n=n_fft))

    # --- DE REPARATIE ---
    # `d` moet in seconden zijn. We converteren microseconden naar seconden.
    sample_delta_s = sample_delta_us * 1e-6
    freqs_fft = rfftfreq(n_fft, d=sample_delta_s)

    valid_freq_idx = freqs_fft > 1e-9
    if not np.any(valid_freq_idx): return None

    # (1.0 / Hz) geeft seconden. Vermenigvuldig met 1e6 voor microseconden (µs).
    periods_us_all = (1.0 / freqs_fft[valid_freq_idx]) * 1e6
    mags_at_periods = mags_fft[valid_freq_idx]

    return {
        "original_segment": original_segment, "trend_line": trend_line,
        "detrended_segment": detrended_segment, "final_segment_for_fft": final_segment_for_fft,
        "periods": periods_us_all, "magnitudes": mags_at_periods
    }


def _calculate_generic_periodogram_common(data_segment, sample_delta_us, do_detrend_gate, current_detrend_type,
                                          do_apply_fft_window, min_period_us_plot, max_period_us_plot):
    """
    Wrapper to ensure all periodogram calculations are consistent.
    """
    analysis_data = get_periodogram_analysis_steps(data_segment, sample_delta_us, do_detrend_gate, current_detrend_type,
                                                   do_apply_fft_window)
    if analysis_data is None: return None, None, []

    periods_us_all, mags_at_periods = analysis_data["periods"], analysis_data["magnitudes"]
    plot_range_mask = (periods_us_all >= min_period_us_plot) & (periods_us_all <= max_period_us_plot)
    plot_periods, plot_mags = periods_us_all[plot_range_mask], mags_at_periods[plot_range_mask]

    if len(plot_periods) == 0: return None, None, []

    sort_idx = np.argsort(plot_periods)
    return plot_periods[sort_idx], plot_mags[sort_idx], []

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