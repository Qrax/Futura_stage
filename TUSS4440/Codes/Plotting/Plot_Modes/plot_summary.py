# --- START OF FILE Plot_Modes/plot_summary.py ---

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t  # For confidence intervals


def generate_plot_summary(dfs, act_lbls, settings, sum_cache, plt_instance):
    """
    Generates the SUMMARY plot (mean traces with CI/PI).
    Handles both fixed and dynamic window lengths.
    """
    mean_trace_data_exists = any(
        label in sum_cache and sum_cache[label].get("N_for_mean", 0) > 0 for label in act_lbls
    )

    if not mean_trace_data_exists:
        print("W(SUMMARY):No mean trace data found in sum_cache for any file.")
        return None

    fig_s, ax_s = plt_instance.subplots(figsize=(12, 7))
    any_summary_plotted = False

    for i, lbl_s in enumerate(act_lbls):
        if lbl_s in sum_cache:
            cache_s = sum_cache[lbl_s]
            mt_s = cache_s.get("mean_trace")
            N_s = cache_s.get("N_for_mean", 0)
            sem_s = cache_s.get("sem_trace")
            std_s = cache_s.get("std_trace")

            if not (N_s and N_s > 0 and mt_s is not None and len(mt_s) > 0):
                continue

            num_points_in_trace = len(mt_s)
            time_axis = np.arange(-settings.WINDOW_BEFORE, num_points_in_trace - settings.WINDOW_BEFORE) * settings.SAMPLE_TIME_DELTA_US

            any_summary_plotted = True
            c = settings.PLOT_COLORS[i % len(settings.PLOT_COLORS)]
            ls = settings.PLOT_LINESTYLES[i % len(settings.PLOT_LINESTYLES)]

            # --- MODIFIED 1: Only label the main line, and use the simple label ---
            # This is the ONLY item that will appear in the legend.
            ax_s.plot(time_axis, mt_s, lw=2.5, color=c, linestyle=ls, label=lbl_s)

            if N_s > 1:
                tcrit = t.ppf(0.975, df=N_s - 1)

                # --- MODIFIED 2: Remove the 'label' argument from fill_between ---
                # This prevents the CI and PI from creating their own legend entries.
                if sem_s is not None and len(sem_s) == num_points_in_trace:
                    ci_m = tcrit * sem_s
                    ax_s.fill_between(time_axis, mt_s - ci_m, mt_s + ci_m, alpha=0.35, color=c)

                if std_s is not None and len(std_s) == num_points_in_trace:
                    pi_m = tcrit * std_s * np.sqrt(1 + 1 / N_s)
                    ax_s.fill_between(time_axis, mt_s - pi_m, mt_s + pi_m, alpha=0.15, color=c)

    if any_summary_plotted:
        ax_s.set_xlabel(f"Time Relative to Trigger ({settings.tu_raw_lbl})")
        ax_s.set_ylabel(f"Voltage ({'ADC' if settings.adc_to_v(1) == 1 else 'V'})")

        # Your simplified title
        title_str = "Mean Signal Traces ± 95% CI & PI"
        ax_s.set_title(title_str, pad=20)

        ax_s.grid(True, which='both', linestyle='--', alpha=0.5)

        # --- MODIFIED 3: Replace the entire complex legend logic with one simple line ---
        # This automatically finds all items that were given a 'label' and creates a clean legend.
        ax_s.legend(title="Metingen")

        fig_s.tight_layout()
        fig_s.canvas.manager.set_window_title(f"Plot:SUMMARY - {settings.DEVICE_FILTER}")
        return fig_s
    else:
        plt_instance.close(fig_s)
        print("W(SUMMARY):No valid mean trace data to plot from any file.")
        return None

# --- END OF FILE plot_summary.py ---