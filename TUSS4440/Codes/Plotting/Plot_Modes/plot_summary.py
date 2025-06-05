# Plot_Mods/plot_summary.py
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t # For confidence intervals

def generate_plot_summary(dfs, act_lbls, settings, sum_cache, plt_instance):
    """
    Generates the SUMMARY plot (mean traces with CI/PI).
    """
    mean_trace_data_exists = any(
        label in sum_cache and sum_cache[label].get("N_for_mean", 0) > 0 for label in act_lbls
    )

    if not mean_trace_data_exists:
        print("W(SUMMARY):No mean trace data found in sum_cache for any file.")
        return None

    fig_s, ax_s = plt_instance.subplots(figsize=(12, 6))
    # Time axis relative to trigger point (sample 0)
    t_rel_s = np.arange(-settings.WINDOW_BEFORE, settings.WINDOW_AFTER + 1)
    title_s_parts = []
    any_summary_plotted = False

    for i, lbl_s in enumerate(act_lbls):
        if lbl_s in sum_cache:
            cache_s = sum_cache[lbl_s]
            mt_s = cache_s.get("mean_trace")
            N_s = cache_s.get("N_for_mean", 0)
            sem_s = cache_s.get("sem_trace")
            # std_s = cache_s.get("std_trace") # For PI, if you re-enable it

            if not (N_s and N_s > 0 and mt_s is not None and len(mt_s) == len(t_rel_s)):
                title_s_parts.append(f"{lbl_s}(no data)")
                continue

            any_summary_plotted = True
            title_s_parts.append(f"{lbl_s}(N={N_s})")
            c = settings.PLOT_COLORS[i % len(settings.PLOT_COLORS)]
            ls = settings.PLOT_LINESTYLES[i % len(settings.PLOT_LINESTYLES)]

            ax_s.plot(t_rel_s, mt_s, lw=2, color=c, linestyle=ls, label=f"Mean ({lbl_s})")

            if N_s > 1: # SEM and CI only make sense for N > 1
                tcrit = t.ppf(0.975, df=N_s - 1) # 95% CI
                if sem_s is not None and isinstance(sem_s, np.ndarray) and len(sem_s) == len(t_rel_s):
                    ci_m = tcrit * sem_s
                    ax_s.fill_between(t_rel_s, mt_s - ci_m, mt_s + ci_m, alpha=0.35, color=c,
                                      label=f"95%CI ({lbl_s})")
                # Prediction Interval (PI) - can be very wide
                # if std_s is not None and isinstance(std_s, np.ndarray) and len(std_s) == len(t_rel_s):
                # pi_m = tcrit * std_s * np.sqrt(1 + 1 / N_s)
                # ax_s.fill_between(t_rel_s, mt_s - pi_m, mt_s + pi_m, alpha=0.15, color=c, label=f"95%PI ({lbl_s})")

    if any_summary_plotted:
        ax_s.set_xlabel(
            f"Sample-idx relative to Trigger (0=V>{settings.THRESH}, "
            f"{settings.SAMPLE_TIME_DELTA_US}{settings.tu_raw_lbl}/sample)")
        ax_s.set_ylabel(f"Voltage ({'ADC' if settings.adc_to_v(1) == 1 else 'V'})")
        ax_s.set_title(
            f"{settings.DEVICE_FILTER}–SUMMARY: Mean Traces ± CI\n"
            f"Window: [-{settings.WINDOW_BEFORE}, +{settings.WINDOW_AFTER}] samples. Comparing: {', '.join(title_s_parts)}")
        ax_s.grid(True, alpha=0.3)

        h_s, l_s = ax_s.get_legend_handles_labels()
        if h_s: # Consolidate legend
            handles_to_show, labels_to_show = [], []
            main_plot_labels = {} # To store main line for each file
            ci_labels = {}      # To store CI for each file
            for handle, label in zip(h_s, l_s):
                if "Mean (" in label:
                    main_plot_labels[label.split("Mean (")[1].split(")")[0]] = handle
                elif "95%CI (" in label:
                    ci_labels[label.split("95%CI (")[1].split(")")[0]] = handle
            # Order for legend: Mean, then CI for each file
            for lbl_key in act_lbls: # Iterate in order of files
                if lbl_key in main_plot_labels:
                    handles_to_show.append(main_plot_labels[lbl_key])
                    labels_to_show.append(f"Mean ({lbl_key})")
                if lbl_key in ci_labels:
                    handles_to_show.append(ci_labels[lbl_key])
                    labels_to_show.append(f"95%CI ({lbl_key})")
            if handles_to_show:
                 ax_s.legend(handles_to_show, labels_to_show, fontsize=8, loc='upper right', title="Legend")

        fig_s.tight_layout()
        fig_s.canvas.manager.set_window_title(f"Plot:SUMMARY - {settings.DEVICE_FILTER}")
        return fig_s
    else:
        plt_instance.close(fig_s) # Close if fig_s was created but nothing plotted
        print("W(SUMMARY):No valid mean trace data to plot from any file.")
        return None