# Plot_Mods/plot_periodogram_analysis_steps.py
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import detrend  # For linear detrend visual
from numpy.fft import rfftfreq  # For Method C common axis
from .common_plot_utils import _calculate_generic_periodogram_common, _detrend_exponential_data


def generate_plot_periodogram_analysis_steps(dfs, act_lbls, settings, sum_cache, plt_instance):
    """
    Generates all figures for PERIODOGRAM_ANALYSIS_STEPS.
    This function will manage its own figures and does not return a single figure object.
    """
    min_p_us = settings.MIN_PERIOD_PLOT * settings.SAMPLE_TIME_DELTA_US
    max_p_us = settings.MAX_PERIOD_PLOT_ABS * settings.SAMPLE_TIME_DELTA_US
    # N_SEGMENT_LENGTH for PAS is FIT_WINDOW_POST_PEAK
    N_SEGMENT_LENGTH = settings.FIT_WINDOW_POST_PEAK
    time_unit_label_pas = settings.tu_raw_lbl
    created_figs = []  # To keep track of created figures

    # --- Figure 1: Original Post-Peak Segments & Fitted Trends ---
    fig_orig_td, ax_orig_td = plt_instance.subplots(figsize=(12, 7))
    ax_orig_td.set_title(
        f"Original Segments (Pk+{settings.POST_PEAK_OFFSET_SAMPLES}samp, len up to {N_SEGMENT_LENGTH}) & Trends")
    ax_orig_td.set_xlabel(
        f"Time from Segment Start ({time_unit_label_pas}) [Seg starts Pk+{settings.POST_PEAK_OFFSET_SAMPLES}samp]")
    ax_orig_td.set_ylabel(f"Voltage ({'ADC' if settings.adc_to_v(1) == 1 else 'V'})")
    ax_orig_td.grid(True, alpha=0.3)
    plotted_anything_fig1 = False

    for i_file, lbl_data in enumerate(act_lbls):
        s_data = sum_cache.get(lbl_data, {})
        mt = s_data.get("mean_trace")
        N_runs = s_data.get("N_for_mean", 0)
        if mt is None or N_runs == 0 or len(mt) < (settings.POST_PEAK_OFFSET_SAMPLES + 4):
            continue

        idx_peak = np.argmax(mt)
        segment_start_index = idx_peak + settings.POST_PEAK_OFFSET_SAMPLES
        segment_end_index = segment_start_index + N_SEGMENT_LENGTH
        segment_start_index_clipped = min(max(0, segment_start_index), len(mt) - 1)
        segment_end_index_clipped = min(segment_end_index, len(mt))

        if segment_start_index_clipped >= segment_end_index_clipped or \
                (segment_end_index_clipped - segment_start_index_clipped) < 2:  # Need at least 2 for polyfit
            continue

        seg_orig = mt[segment_start_index_clipped: segment_end_index_clipped]
        if len(seg_orig) < 2: continue  # Ensure segment is usable

        time_axis_seg_relative_to_segment_start = np.arange(len(seg_orig)) * settings.SAMPLE_TIME_DELTA_US
        file_color = settings.PLOT_COLORS[i_file % len(settings.PLOT_COLORS)]
        file_linestyle = settings.PLOT_LINESTYLES[i_file % len(settings.PLOT_LINESTYLES)]

        ax_orig_td.plot(time_axis_seg_relative_to_segment_start, seg_orig,
                        label=f"{lbl_data} - Orig (N={N_runs}, SegLen:{len(seg_orig)})",
                        color=file_color, linestyle=file_linestyle)
        plotted_anything_fig1 = True

        if settings.DETREND_PERIODOGRAM and settings.DETREND_TYPE.lower() != "none" and len(seg_orig) >= 2:
            fitted_trend_to_plot = None
            trend_label_suffix = ""
            exp_params_str = ""

            if settings.DETREND_TYPE.lower() == 'linear':
                x_coords = np.arange(len(seg_orig))
                try:
                    slope, intercept = np.polyfit(x_coords, seg_orig, 1)
                    fitted_trend_to_plot = slope * x_coords + intercept
                    trend_label_suffix = f"Lin Trend (B={slope:.2e})"
                except Exception as e_lin:
                    print(f"W(PAS_Fig1 - {lbl_data}): Linear fit vis failed: {e_lin}")
            elif settings.DETREND_TYPE.lower() == 'exponential':
                # Use the common util for consistency, requesting trend component and params
                _, fitted_trend_component, exp_params = _detrend_exponential_data(seg_orig.copy(),
                                                                                  return_trend_component=True)
                if fitted_trend_component is not None and len(fitted_trend_component) == len(seg_orig):
                    # Check if the trend component is substantially different from the original segment
                    if not np.array_equal(fitted_trend_component, seg_orig):  # Basic check
                        fitted_trend_to_plot = fitted_trend_component
                        B_slope = exp_params.get("B_slope")
                        # A_amp = exp_params.get("A_amplitude") # A is A_amplitude * exp(B * 0_offset), which is just A_amplitude
                        if B_slope is not None:
                            exp_params_str = f" (B={B_slope:.2e})"  # A={A_amp:.2e}
                        trend_label_suffix = f"Exp. Trend{exp_params_str}"

            if fitted_trend_to_plot is not None and len(fitted_trend_to_plot) == len(
                    time_axis_seg_relative_to_segment_start):
                ax_orig_td.plot(time_axis_seg_relative_to_segment_start, fitted_trend_to_plot,
                                label=f"{lbl_data} - {trend_label_suffix}",
                                color=file_color, linestyle='--', alpha=0.7)
    if plotted_anything_fig1:
        ax_orig_td.legend(fontsize=8, loc='best')
        fig_orig_td.tight_layout()
        fig_orig_td.canvas.manager.set_window_title(
            f"PAS: Fig 1 - Orig Seg (Pk+{settings.POST_PEAK_OFFSET_SAMPLES}) & Trends - {settings.DEVICE_FILTER}")
        created_figs.append(fig_orig_td)
    else:
        plt_instance.close(fig_orig_td)
        print(f"I(PAS_Fig1): No data for Original Segments (Pk+{settings.POST_PEAK_OFFSET_SAMPLES}).")

    # --- Figure 2: Detrended Post-Peak Segments (Time Domain) ---
    if settings.DETREND_PERIODOGRAM and settings.DETREND_TYPE.lower() != "none":
        fig_detrend_td, ax_detrend_td = plt_instance.subplots(figsize=(12, 7))
        ax_detrend_td.set_title(
            f"'{settings.DETREND_TYPE.capitalize()}' Detrended (Pk+{settings.POST_PEAK_OFFSET_SAMPLES}samp, len up to {N_SEGMENT_LENGTH})")
        ax_detrend_td.set_xlabel(
            f"Time from Segment Start ({time_unit_label_pas}) [Seg starts Pk+{settings.POST_PEAK_OFFSET_SAMPLES}samp]")
        ax_detrend_td.set_ylabel(f"Detrended Voltage ({'ADC' if settings.adc_to_v(1) == 1 else 'V'})")
        ax_detrend_td.grid(True, alpha=0.3)
        plotted_anything_fig2 = False

        for i_file, lbl_data in enumerate(act_lbls):
            s_data = sum_cache.get(lbl_data, {})
            mt = s_data.get("mean_trace")
            N_runs = s_data.get("N_for_mean", 0)
            if mt is None or N_runs == 0 or len(mt) < (settings.POST_PEAK_OFFSET_SAMPLES + 4):
                continue
            idx_peak = np.argmax(mt)
            segment_start_index = idx_peak + settings.POST_PEAK_OFFSET_SAMPLES
            segment_end_index = segment_start_index + N_SEGMENT_LENGTH
            segment_start_index_clipped = min(max(0, segment_start_index), len(mt) - 1)
            segment_end_index_clipped = min(segment_end_index, len(mt))
            if segment_start_index_clipped >= segment_end_index_clipped: continue

            seg_orig_for_detrend = mt[segment_start_index_clipped: segment_end_index_clipped]
            if len(seg_orig_for_detrend) < 2: continue  # Need at least 2 for detrending

            seg_detrended_display = np.array([])
            if settings.DETREND_TYPE.lower() == 'exponential':
                seg_detrended_display, _ = _detrend_exponential_data(seg_orig_for_detrend.copy(),
                                                                     return_trend_component=False)
            elif settings.DETREND_TYPE.lower() == 'linear':
                if len(seg_orig_for_detrend) >= 2:  # detrend needs at least 2 points
                    seg_detrended_display = detrend(seg_orig_for_detrend.copy(), type='linear')
                else:
                    seg_detrended_display = seg_orig_for_detrend.copy()  # or handle as error

            if seg_detrended_display.size > 0 and len(seg_detrended_display) >= 2:
                time_axis_seg_rel = np.arange(len(seg_detrended_display)) * settings.SAMPLE_TIME_DELTA_US
                ax_detrend_td.plot(time_axis_seg_rel, seg_detrended_display,
                                   label=f"{lbl_data} (N={N_runs}, SegLen:{len(seg_detrended_display)})",
                                   color=settings.PLOT_COLORS[i_file % len(settings.PLOT_COLORS)],
                                   linestyle=settings.PLOT_LINESTYLES[i_file % len(settings.PLOT_LINESTYLES)])
                plotted_anything_fig2 = True
        if plotted_anything_fig2:
            ax_detrend_td.legend(fontsize=8, loc='best')
            fig_detrend_td.tight_layout()
            fig_detrend_td.canvas.manager.set_window_title(
                f"PAS: Fig 2 - {settings.DETREND_TYPE.capitalize()} Detrended (Pk+{settings.POST_PEAK_OFFSET_SAMPLES}) - {settings.DEVICE_FILTER}")
            created_figs.append(fig_detrend_td)
        else:
            plt_instance.close(fig_detrend_td)
            print(
                f"I(PAS_Fig2): No data for {settings.DETREND_TYPE} Detrended Segments (Pk+{settings.POST_PEAK_OFFSET_SAMPLES}).")
    else:
        print(f"I(PAS_Fig2): Detrending (type: {settings.DETREND_TYPE}) not active or 'none'. Skipping detrended plot.")

    # --- Figure 3: Final Preprocessed Segments for FFT (Time Domain) ---
    fig_final_td, ax_final_td = plt_instance.subplots(figsize=(12, 7))
    processing_steps_str_fig3_list = []
    if settings.DETREND_PERIODOGRAM and settings.DETREND_TYPE.lower() != "none":
        processing_steps_str_fig3_list.append(f"{settings.DETREND_TYPE.capitalize()}_Detrend")
    if settings.APPLY_FFT_WINDOW:
        processing_steps_str_fig3_list.append("Hann")
    title_suffix_fig3 = f" ({', '.join(processing_steps_str_fig3_list)})" if processing_steps_str_fig3_list else " (Raw)"

    ax_final_td.set_title(
        f"Final Segments for FFT Input{title_suffix_fig3} (Pk+{settings.POST_PEAK_OFFSET_SAMPLES}samp, len up to {N_SEGMENT_LENGTH})")
    ax_final_td.set_xlabel(
        f"Time from Segment Start ({time_unit_label_pas}) [Seg starts Pk+{settings.POST_PEAK_OFFSET_SAMPLES}samp]")
    ax_final_td.set_ylabel("Processed Value")
    ax_final_td.grid(True, alpha=0.3)
    plotted_anything_fig3 = False

    for i_file, lbl_data in enumerate(act_lbls):
        s_data = sum_cache.get(lbl_data, {})
        mt = s_data.get("mean_trace")
        N_runs = s_data.get("N_for_mean", 0)
        if mt is None or N_runs == 0 or len(mt) < (settings.POST_PEAK_OFFSET_SAMPLES + 4):
            continue
        idx_peak = np.argmax(mt)
        segment_start_index = idx_peak + settings.POST_PEAK_OFFSET_SAMPLES
        segment_end_index = segment_start_index + N_SEGMENT_LENGTH
        segment_start_index_clipped = min(max(0, segment_start_index), len(mt) - 1)
        segment_end_index_clipped = min(segment_end_index, len(mt))
        if segment_start_index_clipped >= segment_end_index_clipped: continue

        seg_to_process = mt[segment_start_index_clipped: segment_end_index_clipped]
        if len(seg_to_process) < 2: continue  # Need at least 2 for processing

        current_seg_processed = seg_to_process.copy()
        if settings.DETREND_PERIODOGRAM and settings.DETREND_TYPE.lower() != "none":
            if settings.DETREND_TYPE.lower() == 'exponential':
                current_seg_processed, _ = _detrend_exponential_data(current_seg_processed.copy(),
                                                                     return_trend_component=False)
            elif settings.DETREND_TYPE.lower() == 'linear':
                if len(current_seg_processed) >= 2:
                    current_seg_processed = detrend(current_seg_processed.copy(), type='linear')

        if settings.APPLY_FFT_WINDOW:
            if len(current_seg_processed) > 0:
                current_seg_processed *= np.hanning(len(current_seg_processed))

        if len(current_seg_processed) > 0 and len(current_seg_processed) >= 2:
            time_axis_seg_rel = np.arange(len(current_seg_processed)) * settings.SAMPLE_TIME_DELTA_US
            ax_final_td.plot(time_axis_seg_rel, current_seg_processed,
                             label=f"{lbl_data} (N={N_runs}, SegLen:{len(current_seg_processed)})",
                             color=settings.PLOT_COLORS[i_file % len(settings.PLOT_COLORS)],
                             linestyle=settings.PLOT_LINESTYLES[i_file % len(settings.PLOT_LINESTYLES)])
            plotted_anything_fig3 = True
    if plotted_anything_fig3:
        ax_final_td.legend(fontsize=8, loc='best')
        fig_final_td.tight_layout()
        fig_final_td.canvas.manager.set_window_title(
            f"PAS: Fig 3 - Final Segments for FFT (Pk+{settings.POST_PEAK_OFFSET_SAMPLES}) - {settings.DEVICE_FILTER}")
        created_figs.append(fig_final_td)
    else:
        plt_instance.close(fig_final_td)
        print(f"I(PAS_Fig3): No data for Final Preprocessed Segments (Pk+{settings.POST_PEAK_OFFSET_SAMPLES}).")

    # --- Figure 4: Periodogram of Mean Traces (Method A) ---
    fig_method_a, ax_method_a = plt_instance.subplots(figsize=(12, 7))
    any_plot_methA = False
    title_methA_parts = []
    global_steps_methA = []

    for i_file, lbl_methA in enumerate(act_lbls):
        s_data = sum_cache.get(lbl_methA, {})
        mt = s_data.get("mean_trace")
        N_runs = s_data.get("N_for_mean", 0)
        title_methA_parts.append(f"{lbl_methA}(N={N_runs if N_runs else 0})")

        if mt is None or N_runs == 0 or len(mt) < (settings.POST_PEAK_OFFSET_SAMPLES + 4):
            continue
        idx_peak = np.argmax(mt)
        segment_start_index = idx_peak + settings.POST_PEAK_OFFSET_SAMPLES
        segment_end_index = segment_start_index + N_SEGMENT_LENGTH
        segment_start_index_clipped = min(max(0, segment_start_index), len(mt) - 1)
        segment_end_index_clipped = min(segment_end_index, len(mt))
        if segment_start_index_clipped >= segment_end_index_clipped or \
                (segment_end_index_clipped - segment_start_index_clipped) < 4:
            continue

        seg_for_pg = mt[segment_start_index_clipped: segment_end_index_clipped]
        if len(seg_for_pg) < 4: continue

        periods, mags, steps = _calculate_generic_periodogram_common(
            seg_for_pg, settings.SAMPLE_TIME_DELTA_US,
            settings.DETREND_PERIODOGRAM, settings.DETREND_TYPE, settings.APPLY_FFT_WINDOW,
            min_p_us, max_p_us)

        if not global_steps_methA and steps: global_steps_methA = steps

        if periods is not None and len(periods) > 0:
            ax_method_a.plot(periods, mags,
                             color=settings.PLOT_COLORS[i_file % len(settings.PLOT_COLORS)],
                             linestyle=settings.PLOT_LINESTYLES[i_file % len(settings.PLOT_LINESTYLES)],
                             label=f"{lbl_methA} (N={N_runs}, SegLen={len(seg_for_pg)}, Steps: {', '.join(steps) or 'Raw'})")
            any_plot_methA = True

    processing_title_methA = f" ({', '.join(global_steps_methA)})" if global_steps_methA else " (Raw)"
    ax_method_a.set_title(
        f"Periodogram of Mean (Method A: Pk+{settings.POST_PEAK_OFFSET_SAMPLES}samp -> Avg -> FFT){processing_title_methA}\n"
        f"Comparing: {', '.join(title_methA_parts)}")
    ax_method_a.set_xlabel(
        f"Period ({time_unit_label_pas}/cycle, plotted: {min_p_us:.1f}-{max_p_us:.1f} {time_unit_label_pas})")
    ax_method_a.set_ylabel("FFT Mag (of Mean Trace)")
    ax_method_a.grid(True, alpha=0.3, which='both')
    if any_plot_methA:
        ax_method_a.legend(fontsize=8, loc='upper right')
        fig_method_a.tight_layout()
        fig_method_a.canvas.manager.set_window_title(
            f"PAS: Fig 4 - Periodogram of Mean (Method A, Pk+{settings.POST_PEAK_OFFSET_SAMPLES}) - {settings.DEVICE_FILTER}")
        created_figs.append(fig_method_a)
    else:
        plt_instance.close(fig_method_a)
        print(
            f"I(PAS_Fig4_MethA): No mean trace periodograms (Pk+{settings.POST_PEAK_OFFSET_SAMPLES}). Title parts: {', '.join(title_methA_parts)}")

    # --- Figure 5: Overlay of Individual Run Periodograms (Method B) ---
    fig_method_b, ax_method_b = plt_instance.subplots(figsize=(12, 7))
    any_plot_methB = False
    plotted_legend_methB = {lbl: False for lbl in act_lbls}
    title_methB_parts = []
    global_steps_methB = []

    for i_file, lbl_methB in enumerate(act_lbls):
        s_data = sum_cache.get(lbl_methB, {})
        run_matrix = s_data.get("matrix_list", [])
        N_total = s_data.get("N_for_mean", 0)  # Total valid runs for this file
        num_runs_actually_plotted_methB = 0

        if not run_matrix or N_total == 0:
            title_methB_parts.append(f"{lbl_methB}(0/{N_total} runs)")
            continue

        for run_data in run_matrix:  # run_matrix already limited by MAX_RUNS
            if len(run_data) < (settings.POST_PEAK_OFFSET_SAMPLES + 4):
                continue
            idx_peak_run = np.argmax(run_data)
            segment_start_index_run = idx_peak_run + settings.POST_PEAK_OFFSET_SAMPLES
            segment_end_index_run = segment_start_index_run + N_SEGMENT_LENGTH  # Max length
            segment_start_index_run_clipped = min(max(0, segment_start_index_run), len(run_data) - 1)
            segment_end_index_run_clipped = min(segment_end_index_run, len(run_data))

            if segment_start_index_run_clipped >= segment_end_index_run_clipped or \
                    (segment_end_index_run_clipped - segment_start_index_run_clipped) < 4:
                continue
            seg_run_for_pg = run_data[segment_start_index_run_clipped: segment_end_index_run_clipped]
            if len(seg_run_for_pg) < 4: continue

            periods, mags, steps = _calculate_generic_periodogram_common(
                seg_run_for_pg, settings.SAMPLE_TIME_DELTA_US,
                settings.DETREND_PERIODOGRAM, settings.DETREND_TYPE, settings.APPLY_FFT_WINDOW,
                min_p_us, max_p_us)

            if not global_steps_methB and steps: global_steps_methB = steps

            if periods is not None and len(periods) > 0:
                label_str = None
                if not plotted_legend_methB[lbl_methB]:
                    label_str = f"{lbl_methB} (Indiv. Runs, Steps: {', '.join(steps) or 'Raw'})"
                    plotted_legend_methB[lbl_methB] = True
                ax_method_b.plot(periods, mags,
                                 color=settings.PLOT_COLORS[i_file % len(settings.PLOT_COLORS)],
                                 lw=0.7, alpha=0.35, label=label_str)
                any_plot_methB = True
                num_runs_actually_plotted_methB += 1
        title_methB_parts.append(f"{lbl_methB}({num_runs_actually_plotted_methB}/{N_total} plotted)")

    processing_title_methB = f" ({', '.join(global_steps_methB)})" if global_steps_methB else " (Raw)"
    ax_method_b.set_title(
        f"Overlay Indiv. Run PGs (Method B: Pk+{settings.POST_PEAK_OFFSET_SAMPLES}samp -> FFT -> Plot){processing_title_methB}\n"
        f"Comparing: {', '.join(title_methB_parts)}")
    ax_method_b.set_xlabel(
        f"Period ({time_unit_label_pas}/cycle, plotted: {min_p_us:.1f}-{max_p_us:.1f} {time_unit_label_pas})")
    ax_method_b.set_ylabel("FFT Mag (Individual Runs)")
    ax_method_b.grid(True, alpha=0.3, which='both')
    if any_plot_methB:
        h_b, l_b = ax_method_b.get_legend_handles_labels()
        if h_b: by_l_b = dict(zip(l_b, h_b)); ax_method_b.legend(by_l_b.values(), by_l_b.keys(), fontsize=8,
                                                                 loc='upper right')
        fig_method_b.tight_layout()
        fig_method_b.canvas.manager.set_window_title(
            f"PAS: Fig 5 - Individual PGs (Method B, Pk+{settings.POST_PEAK_OFFSET_SAMPLES}) - {settings.DEVICE_FILTER}")
        created_figs.append(fig_method_b)
    else:
        plt_instance.close(fig_method_b)
        print(
            f"I(PAS_Fig5_MethB): No individual run PGs (Pk+{settings.POST_PEAK_OFFSET_SAMPLES}). Title parts: {', '.join(title_methB_parts)}")

    # --- Figure 6: Mean of Individual Periodograms (Method C) ---
    max_possible_segment_after_offset = settings.WINDOW_AFTER - settings.POST_PEAK_OFFSET_SAMPLES
    if N_SEGMENT_LENGTH > max_possible_segment_after_offset and max_possible_segment_after_offset > 0:
        print(f"W(PAS_Fig6_MethC): N_SEGMENT_LENGTH ({N_SEGMENT_LENGTH}) for Method C > "
              f"max possible after offset ({max_possible_segment_after_offset}). "
              f"Method C may have 0 runs. Reduce FIT_WINDOW_POST_PEAK or POST_PEAK_OFFSET_SAMPLES, or increase WINDOW_AFTER.")
    elif max_possible_segment_after_offset <= 0:
        print(
            f"W(PAS_Fig6_MethC): POST_PEAK_OFFSET_SAMPLES ({settings.POST_PEAK_OFFSET_SAMPLES}) >= WINDOW_AFTER ({settings.WINDOW_AFTER}). "
            f"No data available after offset for Method C.")

    fig_method_c, ax_method_c = plt_instance.subplots(figsize=(12, 7))
    any_plot_methC = False
    title_methC_parts = []
    _plot_periods_axis_for_method_c = None
    _valid_common_freqs_idx = None
    _plot_mask_c_for_common_axis = None
    _sort_idx_c_methC_for_common_axis = None
    global_steps_methC = []  # For title, from first successful FFT processing

    # Determine common frequency/period axis for N_SEGMENT_LENGTH
    if N_SEGMENT_LENGTH >= 4:  # Need enough points for rfftfreq
        _common_freqs_for_method_c = rfftfreq(N_SEGMENT_LENGTH, d=settings.SAMPLE_TIME_DELTA_US)
        _valid_common_freqs_idx = _common_freqs_for_method_c > 1e-9
        if np.any(_valid_common_freqs_idx):
            _common_periods_full_range = 1.0 / _common_freqs_for_method_c[_valid_common_freqs_idx]
            _plot_mask_c_for_common_axis = (_common_periods_full_range >= min_p_us) & \
                                           (_common_periods_full_range <= max_p_us)
            temp_plot_periods = _common_periods_full_range[_plot_mask_c_for_common_axis]
            if len(temp_plot_periods) > 0:
                _sort_idx_c_methC_for_common_axis = np.argsort(temp_plot_periods)
                _plot_periods_axis_for_method_c = temp_plot_periods[_sort_idx_c_methC_for_common_axis]
    if _plot_periods_axis_for_method_c is None:
        print(
            f"W(PAS_Fig6_MethC): No plottable periods on common axis for SegLen {N_SEGMENT_LENGTH}. Method C plot will be empty.")

    for i_file, lbl_methC in enumerate(act_lbls):
        s_data = sum_cache.get(lbl_methC, {})
        run_matrix = s_data.get("matrix_list", [])
        N_total_runs = s_data.get("N_for_mean", 0)  # Total valid runs for this file
        collected_mags_for_avg = []
        num_runs_for_avg = 0  # Runs that meet the fixed length criteria

        if _plot_periods_axis_for_method_c is None or not run_matrix or N_total_runs == 0:
            title_methC_parts.append(f"{lbl_methC}(0/{N_total_runs if N_total_runs else 'N/A'} runs)")
            continue

        for run_idx, run_data in enumerate(run_matrix):
            if len(run_data) == 0: continue
            idx_peak_run = np.argmax(run_data)
            segment_start_index_run_c = idx_peak_run + settings.POST_PEAK_OFFSET_SAMPLES
            # For Method C, segment MUST be exactly N_SEGMENT_LENGTH
            if segment_start_index_run_c >= 0 and \
                    segment_start_index_run_c + N_SEGMENT_LENGTH <= len(run_data):
                seg_run_fixed = run_data[segment_start_index_run_c: segment_start_index_run_c + N_SEGMENT_LENGTH]

                # Process this fixed-length segment
                data_proc = seg_run_fixed.copy()
                current_steps = []  # Track steps for this run's FFT
                if settings.DETREND_PERIODOGRAM and settings.DETREND_TYPE.lower() != "none":
                    if settings.DETREND_TYPE.lower() == 'exponential':
                        data_proc, _ = _detrend_exponential_data(data_proc.copy(), return_trend_component=False)
                        current_steps.append("exp_detrend")
                    elif settings.DETREND_TYPE.lower() == 'linear':
                        data_proc = detrend(data_proc.copy(), type='linear')
                        current_steps.append("lin_detrend")
                if settings.APPLY_FFT_WINDOW:
                    if len(data_proc) > 0: data_proc *= np.hanning(N_SEGMENT_LENGTH)  # Use N_SEGMENT_LENGTH for Hanning
                    current_steps.append("Hann")

                if not global_steps_methC and current_steps: global_steps_methC = current_steps

                if len(data_proc) == N_SEGMENT_LENGTH:  # Ensure it's still the correct length
                    mags_fft_full = np.abs(rfft(data_proc))  # FFT of the processed segment
                    collected_mags_for_avg.append(mags_fft_full)
                    num_runs_for_avg += 1

        if num_runs_for_avg > 0:
            avg_mags_fft_full = np.mean(np.array(collected_mags_for_avg), axis=0)
            # Ensure avg_mags_fft_full has the expected length corresponding to N_SEGMENT_LENGTH FFT
            if len(avg_mags_fft_full) == len(_common_freqs_for_method_c):  # len(rfft(N)) = N//2 + 1
                mags_on_common_valid_freqs = avg_mags_fft_full[_valid_common_freqs_idx]
                plot_mags_for_c_unfiltered = mags_on_common_valid_freqs[_plot_mask_c_for_common_axis]
                if len(plot_mags_for_c_unfiltered) == len(_plot_periods_axis_for_method_c):
                    plot_mags_for_c_sorted = plot_mags_for_c_unfiltered[_sort_idx_c_methC_for_common_axis]
                    ax_method_c.plot(_plot_periods_axis_for_method_c, plot_mags_for_c_sorted,
                                     color=settings.PLOT_COLORS[i_file % len(settings.PLOT_COLORS)],
                                     linestyle=settings.PLOT_LINESTYLES[i_file % len(settings.PLOT_LINESTYLES)],
                                     label=f"{lbl_methC} (Avg of {num_runs_for_avg}/{N_total_runs} PGs)")
                    any_plot_methC = True
                    title_methC_parts.append(f"{lbl_methC}({num_runs_for_avg}/{N_total_runs} PGs)")
                else:  # Should not happen if axis setup is correct
                    title_methC_parts.append(f"{lbl_methC}(len mismatch after mask)")
            else:  # Should not happen
                title_methC_parts.append(f"{lbl_methC}(avg mags len error)")
        else:
            title_methC_parts.append(f"{lbl_methC}(0/{N_total_runs} suitable for avg)")

    processing_title_methC = f" ({', '.join(global_steps_methC)})" if global_steps_methC else " (Raw)"
    ax_method_c.set_title(
        f"Mean of Indiv. PGs (Method C: Pk+{settings.POST_PEAK_OFFSET_SAMPLES}samp, Len={N_SEGMENT_LENGTH} -> FFT -> Avg FFTs){processing_title_methC}\n"
        f"Comparing: {', '.join(title_methC_parts)}")
    ax_method_c.set_xlabel(
        f"Period ({time_unit_label_pas}/cycle, plotted: {min_p_us:.1f}-{max_p_us:.1f} {time_unit_label_pas}, FFT SegLen={N_SEGMENT_LENGTH})")
    ax_method_c.set_ylabel("Avg FFT Mag (of Indiv. Runs)")
    ax_method_c.grid(True, alpha=0.3, which='both')

    if any_plot_methC:
        ax_method_c.legend(fontsize=8, loc='upper right')
        fig_method_c.tight_layout()
        fig_method_c.canvas.manager.set_window_title(
            f"PAS: Fig 6 - Mean of PGs (Method C, Pk+{settings.POST_PEAK_OFFSET_SAMPLES}) - {settings.DEVICE_FILTER}")
        created_figs.append(fig_method_c)
    else:
        plt_instance.close(fig_method_c)
        print(
            f"I(PAS_Fig6_MethC): No averaged periodograms (Pk+{settings.POST_PEAK_OFFSET_SAMPLES}). Title parts: {', '.join(title_methC_parts)}")

    if not created_figs:  # If no figures were actually finalized for PAS
        return None
    # For PAS, we don't return a single fig, as it creates multiple.
    # The main loop will just know figs were made if plt.get_fignums() is non-empty.
    return "periodogram_analysis_steps_completed"  # Or just None, main script checks plt.get_fignums()