# Plot_Modes/plot_post_peak_periodogram.py
import matplotlib.pyplot as plt
import numpy as np
from .common_plot_utils import _calculate_generic_periodogram_common

def generate_plot_post_peak_periodogram(dfs, act_lbls, settings, sum_cache, plt_instance):
    """
    Generates the POST_PEAK_PERIODOGRAM plot.
    This calculates the periodogram on the MEAN TRACE of each file.
    This version creates two plots: one with a linear y-axis and one with a logarithmic y-axis.
    """
    # Create two figures: one for linear scale, one for log scale
    fig_lin, ax_lin = plt_instance.subplots(figsize=(12, 7))
    fig_log, ax_log = plt_instance.subplots(figsize=(12, 7))

    any_pg_mean_plot = False
    min_p_us = settings.MIN_PERIOD_PLOT * settings.SAMPLE_TIME_DELTA_US
    max_p_us = settings.MAX_PERIOD_PLOT_ABS * settings.SAMPLE_TIME_DELTA_US
    global_steps_for_title_mean = []

    for i, lbl_pg_mean in enumerate(act_lbls):
        c = settings.PLOT_COLORS[i % len(settings.PLOT_COLORS)]
        s_data = sum_cache.get(lbl_pg_mean, {})
        mean_trace = s_data.get("mean_trace")
        N = s_data.get("N_for_mean", 0)

        if mean_trace is None or N == 0 or len(mean_trace) < (settings.POST_PEAK_OFFSET_SAMPLES + 4):
            continue

        # Segment extraction on the mean trace
        idx_peak_mean = np.argmax(mean_trace)
        segment_start_idx = idx_peak_mean + settings.POST_PEAK_OFFSET_SAMPLES
        segment_end_idx = segment_start_idx + settings.FIT_WINDOW_POST_PEAK

        segment_start_idx_clipped = min(max(0, segment_start_idx), len(mean_trace) - 1)
        segment_end_idx_clipped = min(segment_end_idx, len(mean_trace))

        if segment_start_idx_clipped >= segment_end_idx_clipped or \
                (segment_end_idx_clipped - segment_start_idx_clipped) < 4:
            continue

        data_segment_for_fft = mean_trace[segment_start_idx_clipped:segment_end_idx_clipped]

        plot_p_mean, plot_m_mean, steps_pg_mean = _calculate_generic_periodogram_common(
            data_segment_for_fft, settings.SAMPLE_TIME_DELTA_US,
            settings.DETREND_PERIODOGRAM, settings.DETREND_TYPE, settings.APPLY_FFT_WINDOW,
            min_p_us, max_p_us
        )

        if not global_steps_for_title_mean and steps_pg_mean:
            global_steps_for_title_mean = steps_pg_mean

        if plot_p_mean is not None and len(plot_p_mean) > 0:
            current_label = f"{lbl_pg_mean} (N={N})"
            # MODIFIED: Thicker, solid lines for clarity
            # Plot on both linear and log axes
            ax_lin.plot(plot_p_mean, plot_m_mean, lw=2.5, color=c, linestyle='-',
                        label=current_label)
            ax_log.plot(plot_p_mean, plot_m_mean, lw=2.5, color=c, linestyle='-',
                        label=current_label)
            any_pg_mean_plot = True

    if any_pg_mean_plot:
        # MODIFIED: Cleaner title and labels for thesis
        processing_str = ', '.join(global_steps_for_title_mean) if global_steps_for_title_mean else "None"
        base_title = (
            f"Periodogram of Mean Signal Traces\n"
            f"Post-Peak Segment (Offset: {settings.POST_PEAK_OFFSET_SAMPLES} samp, "
            f"Length: {settings.FIT_WINDOW_POST_PEAK} samp) | Processing: {processing_str}"
        )

        # --- Configure Linear Plot ---
        ax_lin.set_title(base_title, pad=20)
        ax_lin.set_xlabel(f"Period ({settings.tu_raw_lbl})")
        ax_lin.set_ylabel("FFT Magnitude")
        ax_lin.grid(True, which="both", ls="--", alpha=0.5)
        ax_lin.legend(title="Source Measurement")
        ax_lin.set_xlim(left=min_p_us, right=max_p_us)
        fig_lin.tight_layout()
        fig_lin.canvas.manager.set_window_title(f"Plot: Periodogram of Mean Traces (Linear) - {settings.DEVICE_FILTER}")

        # --- Configure Log Plot ---
        ax_log.set_title(base_title, pad=20)
        ax_log.set_xlabel(f"Period ({settings.tu_raw_lbl})")
        ax_log.set_ylabel("FFT Magnitude (log scale)")
        ax_log.set_yscale('log')  # Set y-axis to log scale
        ax_log.grid(True, which="both", ls="--", alpha=0.5)
        ax_log.legend(title="Source Measurement")
        ax_log.set_xlim(left=min_p_us, right=max_p_us)
        fig_log.tight_layout()
        fig_log.canvas.manager.set_window_title(f"Plot: Periodogram of Mean Traces (Log) - {settings.DEVICE_FILTER}")

        return [fig_lin, fig_log]
    else:
        plt_instance.close(fig_lin)
        plt_instance.close(fig_log)
        print("W(POST_PEAK_PERIODOGRAM): No periodograms of mean traces were plotted.")
        return None