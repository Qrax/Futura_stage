# Plot_Mods/plot_post_peak_periodogram.py
import matplotlib.pyplot as plt
import numpy as np
from .common_plot_utils import _calculate_generic_periodogram_common

def generate_plot_post_peak_periodogram(dfs, act_lbls, settings, sum_cache, plt_instance):
    """
    Generates the POST_PEAK_PERIODOGRAM plot (FFT of mean trace segment).
    """
    fig_pg_mean, ax_pg_mean = plt_instance.subplots(figsize=(12, 6))
    any_pg_mean_plot = False
    title_pg_mean_parts = []
    min_p_us = settings.MIN_PERIOD_PLOT * settings.SAMPLE_TIME_DELTA_US
    max_p_us = settings.MAX_PERIOD_PLOT_ABS * settings.SAMPLE_TIME_DELTA_US
    global_steps_for_title = [] # Store steps from first successful plot for consistent title

    for i, lbl_pg in enumerate(act_lbls):
        c = settings.PLOT_COLORS[i % len(settings.PLOT_COLORS)]
        ls = settings.PLOT_LINESTYLES[i % len(settings.PLOT_LINESTYLES)]
        s_data = sum_cache.get(lbl_pg, {})
        mt_pg = s_data.get("mean_trace")
        N_pg = s_data.get("N_for_mean", 0)

        title_pg_mean_parts.append(f"{lbl_pg}(N={N_pg if N_pg else 0})")

        if N_pg == 0 or mt_pg is None or len(mt_pg) < (settings.POST_PEAK_OFFSET_SAMPLES + 4):
            continue # Need at least 4 points in segment for FFT

        idx_peak_pg = np.argmax(mt_pg)
        segment_start_idx = idx_peak_pg + settings.POST_PEAK_OFFSET_SAMPLES
        # FIT_WINDOW_POST_PEAK is the desired length of the segment for FFT
        segment_end_idx = segment_start_idx + settings.FIT_WINDOW_POST_PEAK

        # Clip segment to actual data boundaries
        segment_start_idx_clipped = min(max(0, segment_start_idx), len(mt_pg) - 1)
        segment_end_idx_clipped = min(segment_end_idx, len(mt_pg))

        if segment_start_idx_clipped >= segment_end_idx_clipped or \
           (segment_end_idx_clipped - segment_start_idx_clipped) < 4:
            continue # Segment invalid or too short after clipping

        data_segment_for_fft = mt_pg[segment_start_idx_clipped:segment_end_idx_clipped]

        plot_p_pg, plot_m_pg, steps_pg = _calculate_generic_periodogram_common(
            data_segment_for_fft, settings.SAMPLE_TIME_DELTA_US,
            settings.DETREND_PERIODOGRAM, settings.DETREND_TYPE, settings.APPLY_FFT_WINDOW,
            min_p_us, max_p_us
        )

        if not global_steps_for_title and steps_pg: # Capture steps from the first successful plot
            global_steps_for_title = steps_pg

        if plot_p_pg is not None and len(plot_p_pg) > 0:
            leg_suf_pg = (f"(N={N_pg}, SegLen={len(data_segment_for_fft)}, "
                          f"Steps: {', '.join(steps_pg) or 'Raw'})")
            ax_pg_mean.plot(plot_p_pg, plot_m_pg, lw=1.5, color=c, linestyle=ls,
                            label=f"{lbl_pg} {leg_suf_pg}")
            any_pg_mean_plot = True

    if any_pg_mean_plot:
        processing_str_title = f" ({', '.join(global_steps_for_title)})" if global_steps_for_title else " (Raw)"
        ax_pg_mean.set_xlabel(
            f"Period ({settings.tu_raw_lbl}/cycle, plotted: {min_p_us:.1f}-{max_p_us:.1f} {settings.tu_raw_lbl})")
        ax_pg_mean.set_ylabel(f"FFT Magnitude (Mean Trace){processing_str_title}")
        ax_pg_mean.set_title(
            f"{settings.DEVICE_FILTER}–POST-PEAK PERIODOGRAM (Mean Traces){processing_str_title}\n"
            f"Segment from Pk+{settings.POST_PEAK_OFFSET_SAMPLES}samp, Len up to {settings.FIT_WINDOW_POST_PEAK}samp. "
            f"Sample time: {settings.SAMPLE_TIME_DELTA_US}{settings.tu_raw_lbl}\n"
            f"Comparing: {', '.join(title_pg_mean_parts)}")
        ax_pg_mean.grid(True, which="both", ls="-", alpha=0.3)
        ax_pg_mean.legend(fontsize=8, loc='upper right')
        fig_pg_mean.tight_layout()
        fig_pg_mean.canvas.manager.set_window_title(f"Plot:POST_PEAK_PERIODOGRAM (Mean) - {settings.DEVICE_FILTER}")
        return fig_pg_mean
    else:
        plt_instance.close(fig_pg_mean)
        print(f"W(POST_PEAK_PERIODOGRAM - Mean): No periodograms plotted. Title parts: {', '.join(title_pg_mean_parts)}")
        return None