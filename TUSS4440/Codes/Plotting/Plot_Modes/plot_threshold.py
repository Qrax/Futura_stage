# Plot_Mods/plot_threshold.py
import matplotlib.pyplot as plt # For linting/autocomplete
import numpy as np
from .common_plot_utils import plot_runs_common

def generate_plot_threshold(dfs, act_lbls, settings, sum_cache, plt_instance):
    """
    Generates the THRESHOLD plot.
    """
    fig_th, ax_th = plt_instance.subplots(figsize=(12, 6))
    title_th_parts = []
    any_plot_th = False

    for i, df_th_file in enumerate(dfs):
        lbl_th = act_lbls[i]
        c_th = settings.PLOT_COLORS[i % len(settings.PLOT_COLORS)]

        # plot_runs_common handles the logic based on plot_mode_str and settings
        did_plot_file_th, _ = plot_runs_common(ax_th, df_th_file, lbl_th, c_th, "THRESHOLD", settings)

        if did_plot_file_th:
            any_plot_th = True
            # For title, use sum_cache for number of *valid, aligned* runs used in mean calculation
            num_valid_aligned_runs = sum_cache[lbl_th].get('runs_count', 'N/A') if lbl_th in sum_cache else 'N/A'
            title_th_parts.append(f"{lbl_th} ({num_valid_aligned_runs} valid runs)")

    if any_plot_th:
        ax_th.set_xlabel(
            f"Sample-idx relative to Trigger (0 = first V > {settings.THRESH} ADC, "
            f"{settings.SAMPLE_TIME_DELTA_US}{settings.tu_raw_lbl}/sample)")
        ax_th.set_ylabel(f"Voltage ({'ADC' if settings.adc_to_v(1) == 1 else 'V'})")
        ax_th.set_title(
            f"{settings.DEVICE_FILTER}–THRESHOLD: Aligned Traces "
            f"(Window: [-{settings.WINDOW_BEFORE}, +{settings.WINDOW_AFTER}] samples around trigger)\n"
            f"Comparing {', '.join(title_th_parts)}. "
            f"(Up to {settings.MAX_RUNS if settings.MAX_RUNS is not None else 'all'} runs processed/file)")

        h_th, l_th = ax_th.get_legend_handles_labels()
        if h_th:
            by_l_th = dict(zip(l_th, h_th))
            ax_th.legend(by_l_th.values(), by_l_th.keys(), title="Measurements", fontsize=8, loc='upper right')

        ax_th.grid(True, alpha=0.3)
        fig_th.tight_layout()
        fig_th.canvas.manager.set_window_title(f"Plot:THRESHOLD - {settings.DEVICE_FILTER}")
        return fig_th
    else:
        plt_instance.close(fig_th)
        print("W(THRESHOLD):No threshold-aligned traces plotted from any file.")
        return None