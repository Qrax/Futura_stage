# Plot_Mods/plot_raw.py
import matplotlib.pyplot as plt # plt is passed as an argument, but good for linting
import numpy as np
from .common_plot_utils import plot_runs_common

def generate_plot_raw(dfs, act_lbls, settings, sum_cache, plt_instance):
    """
    Generates the RAW plot.
    dfs: list of pandas DataFrames, one for each CSV file.
    act_lbls: list of labels corresponding to dfs.
    settings: An object or dictionary containing all global plot settings.
    sum_cache: Dictionary with pre-calculated summary data (not used by RAW).
    plt_instance: The matplotlib.pyplot module itself.
    """
    fig_r, ax_r = plt_instance.subplots(figsize=(12, 6))
    plotted_runs_counts_r = []
    any_plot_r = False
    actual_labels_plotted_r = []

    for i, df_r_file in enumerate(dfs):
        lbl_r = act_lbls[i]
        c_r = settings.PLOT_COLORS[i % len(settings.PLOT_COLORS)]

        # plot_runs_common needs the settings object
        did_plot_file_r, _ = plot_runs_common(ax_r, df_r_file, lbl_r, c_r, "RAW", settings)

        if did_plot_file_r:
            any_plot_r = True
            actual_labels_plotted_r.append(lbl_r)
            num_runs_in_file = len(df_r_file["Run"].unique())
            max_r = settings.MAX_RUNS if settings.MAX_RUNS is not None else num_runs_in_file
            plotted_runs_counts_r.append(min(num_runs_in_file, max_r))

    if any_plot_r:
        ax_r.set_xlabel(f"Time ({settings.tu_raw_lbl})")
        ax_r.set_ylabel(f"Voltage ({'ADC' if settings.adc_to_v(1) == 1 else 'V'})") # Use settings.adc_to_v

        title_parts = [f"{actual_labels_plotted_r[j]} ({plotted_runs_counts_r[j]} runs)"
                       for j in range(len(actual_labels_plotted_r))]
        title_str = (f"{settings.DEVICE_FILTER}–RAW: Full Traces "
                     f"(up to {settings.MAX_RUNS if settings.MAX_RUNS is not None else 'all'} runs/file)\n"
                     f"Comparing {', '.join(title_parts)}")
        ax_r.set_title(title_str)

        h_r, l_r = ax_r.get_legend_handles_labels()
        if h_r: # Ensure there are handles and labels
            by_l_r = dict(zip(l_r, h_r)) # To remove duplicate labels if any run plotted with same label
            ax_r.legend(by_l_r.values(), by_l_r.keys(), title="Measurements", fontsize=8, loc='upper right')

        ax_r.grid(True, alpha=0.3)
        fig_r.tight_layout()
        fig_r.canvas.manager.set_window_title(f"Plot:RAW - {settings.DEVICE_FILTER}")
        return fig_r # Return the figure object to the main script
    else:
        # Important: Close the figure if nothing was plotted to prevent empty windows
        plt_instance.close(fig_r)
        print("W(RAW):No raw traces plotted from any file.")
        return None