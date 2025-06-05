# Plot_Mods/plot_individual_run_periodogram.py
import matplotlib.pyplot as plt
import numpy as np
from .common_plot_utils import _calculate_generic_periodogram_common

def generate_plot_individual_run_periodogram(dfs, act_lbls, settings, sum_cache, plt_instance):
    """
    Generates the INDIVIDUAL_RUN_PERIODOGRAM plot.
    """
    fig_pg_ind, ax_pg_ind = plt_instance.subplots(figsize=(12, 6))
    any_pg_ind_plot = False
    title_pg_ind_parts = []
    min_p_us = settings.MIN_PERIOD_PLOT * settings.SAMPLE_TIME_DELTA_US
    max_p_us = settings.MAX_PERIOD_PLOT_ABS * settings.SAMPLE_TIME_DELTA_US
    plotted_legend_for_file = {lbl: False for lbl in act_lbls}
    global_steps_for_title_ind = []

    for i, lbl_pg_run in enumerate(act_lbls):
        c = settings.PLOT_COLORS[i % len(settings.PLOT_COLORS)]
        ls = settings.PLOT_LINESTYLES[0] # Consistent linestyle for individual runs
        s_data = sum_cache.get(lbl_pg_run, {})
        run_matrix = s_data.get("matrix_list", []) # Aligned runs from sum_cache
        N_total_runs_for_file = s_data.get("N_for_mean", 0) # N_for_mean is count of valid aligned runs

        num_runs_actually_plotted_for_file = 0
        if N_total_runs_for_file == 0 or not run_matrix:
            title_pg_ind_parts.append(f"{lbl_pg_run}(0/{N_total_runs_for_file} runs)")
            continue

        for run_idx, single_run_data in enumerate(run_matrix): # run_matrix already respects MAX_RUNS
            if len(single_run_data) < (settings.POST_PEAK_OFFSET_SAMPLES + 4):
                continue

            idx_peak_run = np.argmax(single_run_data)
            segment_start_idx = idx_peak_run + settings.POST_PEAK_OFFSET_SAMPLES
            segment_end_idx = segment_start_idx + settings.FIT_WINDOW_POST_PEAK

            segment_start_idx_clipped = min(max(0, segment_start_idx), len(single_run_data) - 1)
            segment_end_idx_clipped = min(segment_end_idx, len(single_run_data))

            if segment_start_idx_clipped >= segment_end_idx_clipped or \
               (segment_end_idx_clipped - segment_start_idx_clipped) < 4:
                continue
            data_segment_for_fft = single_run_data[segment_start_idx_clipped:segment_end_idx_clipped]

            plot_p_run, plot_m_run, steps_pg_run = _calculate_generic_periodogram_common(
                data_segment_for_fft, settings.SAMPLE_TIME_DELTA_US,
                settings.DETREND_PERIODOGRAM, settings.DETREND_TYPE, settings.APPLY_FFT_WINDOW,
                min_p_us, max_p_us
            )

            if not global_steps_for_title_ind and steps_pg_run:
                global_steps_for_title_ind = steps_pg_run

            if plot_p_run is not None and len(plot_p_run) > 0:
                current_label = None
                if not plotted_legend_for_file[lbl_pg_run]:
                    current_label = (f"{lbl_pg_run} (Indiv. Runs, "
                                     f"Steps: {', '.join(steps_pg_run) or 'Raw'})")
                    plotted_legend_for_file[lbl_pg_run] = True
                ax_pg_ind.plot(plot_p_run, plot_m_run, lw=0.8, color=c, linestyle=ls, alpha=0.4,
                               label=current_label)
                any_pg_ind_plot = True
                num_runs_actually_plotted_for_file += 1

        title_pg_ind_parts.append(
            f"{lbl_pg_run}({num_runs_actually_plotted_for_file}/{N_total_runs_for_file} plotted)")

    if any_pg_ind_plot:
        processing_str_title = f" ({', '.join(global_steps_for_title_ind)})" if global_steps_for_title_ind else " (Raw)"
        ax_pg_ind.set_xlabel(
            f"Period ({settings.tu_raw_lbl}/cycle, plotted: {min_p_us:.1f}-{max_p_us:.1f} {settings.tu_raw_lbl})")
        ax_pg_ind.set_ylabel(f"FFT Magnitude (Individual Runs){processing_str_title}")
        ax_pg_ind.set_title(
            f"{settings.DEVICE_FILTER}–INDIVIDUAL RUN PERIODOGRAMS{processing_str_title}\n"
            f"Segment from Pk+{settings.POST_PEAK_OFFSET_SAMPLES}samp, Len up to {settings.FIT_WINDOW_POST_PEAK}samp. "
            f"Sample time: {settings.SAMPLE_TIME_DELTA_US}{settings.tu_raw_lbl}\n"
            f"Comparing: {', '.join(title_pg_ind_parts)}")
        ax_pg_ind.grid(True, which="both", ls="-", alpha=0.3)

        h_ind, l_ind = ax_pg_ind.get_legend_handles_labels()
        if h_ind: # Remove duplicate labels for legend
            by_l_ind = dict(zip(l_ind, h_ind))
            ax_pg_ind.legend(by_l_ind.values(), by_l_ind.keys(), fontsize=8, loc='upper right')

        fig_pg_ind.tight_layout()
        fig_pg_ind.canvas.manager.set_window_title(f"Plot:INDIVIDUAL_RUN_PERIODOGRAM - {settings.DEVICE_FILTER}")
        return fig_pg_ind
    else:
        plt_instance.close(fig_pg_ind)
        print(f"W(INDIVIDUAL_RUN_PERIODOGRAM): No periodograms plotted. Title parts: {', '.join(title_pg_ind_parts)}")
        return None