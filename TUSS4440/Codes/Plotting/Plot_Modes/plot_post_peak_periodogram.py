
# Plot_Modes/plot_post_peak_periodogram.py
import matplotlib.pyplot as plt
import numpy as np
from .common_plot_utils import _calculate_generic_periodogram_common

def generate_plot_post_peak_periodogram(dfs, act_lbls, settings, sum_cache, plt_instance):
    """
    Generates the POST_PEAK_PERIODOGRAM plot.
    This calculates the periodogram on the MEAN TRACE of each file.
    """
    fig_pg_mean, ax_pg_mean = plt_instance.subplots(figsize=(12, 7))
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
            ax_pg_mean.plot(plot_p_mean, plot_m_mean, lw=2.5, color=c, linestyle='-',
                            label=current_label)
            any_pg_mean_plot = True

    if any_pg_mean_plot:
        # MODIFIED: Cleaner title and labels for thesis
        processing_str = ', '.join(global_steps_for_title_mean) if global_steps_for_title_mean else "None"
        title = (
            f"Periodogram of Mean Signal Traces\n"
            f"Post-Peak Segment (Offset: {settings.POST_PEAK_OFFSET_SAMPLES} samp, "
            f"Length: {settings.FIT_WINDOW_POST_PEAK} samp) | Processing: {processing_str}"
        )
        ax_pg_mean.set_title(title, pad=20)

        ax_pg_mean.set_xlabel(f"Period ({settings.tu_raw_lbl})")
        ax_pg_mean.set_ylabel("FFT Magnitude")
        ax_pg_mean.grid(True, which="both", ls="--", alpha=0.5)
        ax_pg_mean.legend(title="Source Measurement")
        ax_pg_mean.set_xlim(left=min_p_us, right=max_p_us)

        fig_pg_mean.tight_layout()
        fig_pg_mean.canvas.manager.set_window_title(f"Plot: Periodogram of Mean Traces - {settings.DEVICE_FILTER}")
        return fig_pg_mean
    else:
        plt_instance.close(fig_pg_mean)
        print("W(POST_PEAK_PERIODOGRAM): No periodograms of mean traces were plotted.")
        return None
