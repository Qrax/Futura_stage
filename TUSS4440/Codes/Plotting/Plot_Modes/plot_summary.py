# Sla dit op als: Plot_Mods/plot_summary.py (Vervang de volledige inhoud)

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t

def calculate_sem(std_dev, n):
    if n > 1: return std_dev / np.sqrt(n)
    return np.zeros_like(std_dev)

def generate_plot_summary(dfs, act_lbls, settings, sum_cache, plt_instance):
    fig_s, ax_s = plt_instance.subplots(figsize=(12, 8))
    any_summary_plotted = False

    # Loop door de actieve labels en geef elk label een unieke kleur van de standaardlijst.
    for i, lbl_s in enumerate(act_lbls):
        cache_s = sum_cache.get(lbl_s, {})
        mt_s, N_s, std_s = cache_s.get("mean_trace"), cache_s.get("N_for_mean", 0), cache_s.get("std_trace")

        if not (N_s and mt_s is not None and len(mt_s) > 0): continue

        any_summary_plotted = True
        time_axis = np.arange(-settings.WINDOW_BEFORE, len(mt_s) - settings.WINDOW_BEFORE) * settings.SAMPLE_TIME_DELTA_US

        # GEBRUIK DE KLEUR DIE HOORT BIJ DE INDEX VAN DIT LABEL
        color = settings.PLOT_COLORS[i % len(settings.PLOT_COLORS)]

        ax_s.plot(time_axis, mt_s, lw=3, color=color, label=lbl_s, zorder=5)

        if N_s > 1 and std_s is not None:
            tcrit = t.ppf(0.975, df=N_s - 1)
            sem_s = calculate_sem(std_s, N_s)
            ci_margin = tcrit * sem_s
            pi_margin = tcrit * std_s * np.sqrt(1 + 1 / N_s)
            ax_s.fill_between(time_axis, mt_s - ci_margin, mt_s + ci_margin, alpha=0.35, color=color, label='_nolegend_')
            ax_s.fill_between(time_axis, mt_s - pi_margin, mt_s + pi_margin, alpha=0.15, color=color, label='_nolegend_')

    if any_summary_plotted:
        ax_s.set_xlabel(f"Tijd relatief tot Trigger [{settings.tu_raw_lbl}]")
        ax_s.set_ylabel("Voltage [ADC]")
        ax_s.set_title("Gemiddelde Signaal Traces ± 95% CI & PI")
        ax_s.grid(True, which='both', linestyle='--', alpha=0.5)
        ax_s.legend(title="Metingen")
        fig_s.tight_layout()
        return fig_s
    else:
        plt_instance.close(fig_s)
        return None