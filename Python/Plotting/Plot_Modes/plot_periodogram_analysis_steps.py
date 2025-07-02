# --- START OF FILE Plot_Modes/plot_periodogram_analysis_steps.py ---

import matplotlib.pyplot as plt
import numpy as np
from .common_plot_utils import get_periodogram_analysis_steps


def generate_plot_periodogram_analysis_steps(dfs, act_lbls, settings, sum_cache, plt_instance):
    """
    Generates a detailed, multi-panel plot for each dataset, showing the
    steps of periodogram calculation: trend fitting, detrended signal, and final periodogram.
    """
    generated_figs = []
    min_p_us = settings.MIN_PERIOD_PLOT * settings.SAMPLE_TIME_DELTA_US
    max_p_us = settings.MAX_PERIOD_PLOT_ABS * settings.SAMPLE_TIME_DELTA_US

    for i, lbl in enumerate(act_lbls):
        s_data = sum_cache.get(lbl, {})
        mean_trace = s_data.get("mean_trace")
        N_runs = s_data.get("N_for_mean", 0)

        if mean_trace is None or N_runs == 0 or len(mean_trace) < (settings.POST_PEAK_OFFSET_SAMPLES + 10):
            print(f"W(ANALYSIS_STEPS): Skipping '{lbl}' due to insufficient data.")
            continue

        # --- 1. Extract the same segment as other periodogram plots ---
        idx_peak = np.argmax(mean_trace)
        segment_start_idx = idx_peak + settings.POST_PEAK_OFFSET_SAMPLES
        segment_end_idx = segment_start_idx + settings.FIT_WINDOW_POST_PEAK

        segment_start_idx_clipped = min(max(0, segment_start_idx), len(mean_trace) - 1)
        segment_end_idx_clipped = min(segment_end_idx, len(mean_trace))

        data_segment = mean_trace[segment_start_idx_clipped:segment_end_idx_clipped]

        if len(data_segment) < 10:
            print(f"W(ANALYSIS_STEPS): Skipping '{lbl}' due to very short post-peak segment.")
            continue

        # --- NEW: Create a time axis for the segment plots ---
        time_axis_us = np.arange(len(data_segment)) * settings.SAMPLE_TIME_DELTA_US

        # --- 2. Get all analysis steps from the common utility function ---
        analysis_data = get_periodogram_analysis_steps(
            data_segment,
            settings.SAMPLE_TIME_DELTA_US,
            settings.DETREND_PERIODOGRAM,
            settings.DETREND_TYPE,
            settings.APPLY_FFT_WINDOW
        )

        if not analysis_data:
            continue

        # --- 3. Create the 3-panel plot for this dataset ---
        fig, axes = plt_instance.subplots(3, 1, figsize=(12, 16), sharex=False)
        fig.canvas.manager.set_window_title(f"Plot: Analysis Steps - {lbl}")

        # Panel 1: Original Segment and Trend Fit
        ax1 = axes[0]
        # --- MODIFIED: Use time axis for plotting ---
        ax1.plot(time_axis_us, analysis_data['original_segment'], color='C0', lw=2, label='Original Signal Segment')
        if analysis_data['trend_line'] is not None:
            ax1.plot(time_axis_us, analysis_data['trend_line'], color='C1', lw=2.5, linestyle='--',
                     label=f"Fitted '{settings.DETREND_TYPE}' Trend")
        ax1.set_title("Step 1: Signal Segment and Trend Identification")
        # --- MODIFIED: Update x-axis label ---
        ax1.set_xlabel(f"Time within Segment ({settings.tu_raw_lbl})")
        ax1.set_ylabel("ADC Value")
        ax1.legend()
        ax1.grid(True, alpha=0.5)

        # Panel 2: Detrended Signal (Input to FFT)
        ax2 = axes[1]
        # --- MODIFIED: Use time axis for plotting ---
        ax2.plot(time_axis_us, analysis_data['final_segment_for_fft'], color='C2', lw=2, label='Signal for FFT')
        ax2.set_title("Step 2: Processed Signal (Input for FFT)")
        # --- MODIFIED: Update x-axis label ---
        ax2.set_xlabel(f"Time within Segment ({settings.tu_raw_lbl})")
        ax2.set_ylabel("ADC Value (Processed)")
        ax2.legend()
        ax2.grid(True, alpha=0.5)

        # Panel 3: Resulting Periodogram
        ax3 = axes[2]
        periods_to_plot = analysis_data['periods']
        mags_to_plot = analysis_data['magnitudes']
        plot_mask = (periods_to_plot >= min_p_us) & (periods_to_plot <= max_p_us)
        ax3.plot(periods_to_plot[plot_mask], mags_to_plot[plot_mask], color='C3', lw=2)
        ax3.set_title("Step 3: Resulting Periodogram")
        ax3.set_xlabel(f"Period ({settings.tu_raw_lbl})")
        ax3.set_ylabel("FFT Magnitude")
        ax3.grid(True, which='both', alpha=0.5)
        ax3.set_xlim(left=min_p_us, right=max_p_us)

        # Add a main title for the entire figure
        fig.suptitle(f"Periodogram Analysis Steps for '{lbl}' (N={N_runs})\n"
                     f"Processing: {analysis_data['processing_steps_str']}", fontsize=20)

        fig.tight_layout(rect=[0, 0.03, 1, 0.94])  # Adjust layout to make room for suptitle
        generated_figs.append(fig)

    # Return a single figure if only one was made, otherwise return the list
    return generated_figs[0] if len(generated_figs) == 1 else generated_figs

# --- END OF FILE Plot_Modes/plot_periodogram_analysis_steps.py ---