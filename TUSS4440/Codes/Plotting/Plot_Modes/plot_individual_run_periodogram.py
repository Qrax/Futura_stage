
# Plot_Modes/plot_individual_run_periodogram.py
import matplotlib.pyplot as plt
import numpy as np
# This utility function is assumed to exist in a file named common_plot_utils.py
# in the same directory (Plot_Modes).
from .common_plot_utils import _calculate_generic_periodogram_common


def generate_plot_individual_run_periodogram(dfs, act_lbls, settings, sum_cache, plt_instance):
    """
    Generates the INDIVIDUAL_RUN_PERIODOGRAM plot.
    For each file, this plots periodograms for all individual valid runs (transparently)
    and overlays the average of these periodograms (as a solid, thick line).
    """
    fig, ax = plt_instance.subplots(figsize=(12, 8))
    any_plot = False
    global_steps_for_title = []
    min_p_us = settings.MIN_PERIOD_PLOT * settings.SAMPLE_TIME_DELTA_US
    max_p_us = settings.MAX_PERIOD_PLOT_ABS * settings.SAMPLE_TIME_DELTA_US

    for i, lbl in enumerate(act_lbls):
        c = settings.PLOT_COLORS[i % len(settings.PLOT_COLORS)]
        s_data = sum_cache.get(lbl, {})
        individual_traces = s_data.get("matrix_list", [])

        if not individual_traces:
            continue

        all_run_magnitudes = []
        period_axis = None  # To store the common period axis for averaging

        # Plot individual runs transparently
        for trace in individual_traces:
            if len(trace) < (settings.POST_PEAK_OFFSET_SAMPLES + 4):
                continue

            idx_peak = np.argmax(trace)
            segment_start_idx = idx_peak + settings.POST_PEAK_OFFSET_SAMPLES
            segment_end_idx = segment_start_idx + settings.FIT_WINDOW_POST_PEAK

            segment_start_idx_clipped = min(max(0, segment_start_idx), len(trace) - 1)
            segment_end_idx_clipped = min(segment_end_idx, len(trace))

            if segment_start_idx_clipped >= segment_end_idx_clipped or \
                    (segment_end_idx_clipped - segment_start_idx_clipped) < 4:
                continue

            data_segment = trace[segment_start_idx_clipped:segment_end_idx_clipped]

            periods, magnitudes, steps = _calculate_generic_periodogram_common(
                data_segment, settings.SAMPLE_TIME_DELTA_US,
                settings.DETREND_PERIODOGRAM, settings.DETREND_TYPE, settings.APPLY_FFT_WINDOW,
                min_p_us, max_p_us
            )

            if not global_steps_for_title and steps:
                global_steps_for_title = steps

            if periods is not None and len(periods) > 0:
                if period_axis is None:
                    period_axis = periods

                # Only include in average if segment length (and thus period axis) is consistent
                if np.array_equal(period_axis, periods):
                    all_run_magnitudes.append(magnitudes)
                    ax.plot(periods, magnitudes, lw=1.0, color=c, alpha=0.15, linestyle='-')
                    any_plot = True

        # Plot the average of the individual runs with a thick, solid line
        if all_run_magnitudes:
            mean_magnitudes = np.mean(all_run_magnitudes, axis=0)
            label_for_avg = f"{lbl} (Avg. of {len(all_run_magnitudes)} runs)"
            ax.plot(period_axis, mean_magnitudes, lw=3.0, color=c, linestyle='-', label=label_for_avg)

    if any_plot:
        processing_str = ', '.join(global_steps_for_title) if global_steps_for_title else "None"
        title = (
            f"Periodograms of Individual Runs and their Average\n"
            f"Post-peak Segment (Offset: {settings.POST_PEAK_OFFSET_SAMPLES} samp, "
            f"Length: {settings.FIT_WINDOW_POST_PEAK} samp) | Processing: {processing_str}"
        )
        ax.set_title(title, pad=20)
        ax.set_xlabel(f"Period ({settings.tu_raw_lbl})")
        ax.set_ylabel("FFT Magnitude")
        ax.grid(True, which="both", ls="--", alpha=0.5)
        ax.legend(title="Source Measurement (Average)")
        ax.set_xlim(left=min_p_us, right=max_p_us)

        fig.tight_layout()
        fig.canvas.manager.set_window_title(f"Plot: Individual Run Periodograms - {settings.DEVICE_FILTER}")
        return fig
    else:
        plt_instance.close(fig)
        print("W(INDIVIDUAL_RUN_PERIODOGRAM): No individual run periodograms were plotted.")
        return None

