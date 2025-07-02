# Sla dit op als: Plot_Modes/plot_summary.py

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t


# Helper functie om de SEM te berekenen (Standard Error of the Mean)
def calculate_sem(std_dev, n):
    if n > 1:
        return std_dev / np.sqrt(n)
    return np.zeros_like(std_dev)


def generate_plot_summary(dfs, act_lbls, settings, sum_cache, plt_instance):
    """
    Genereert de SUMMARY plot (gemiddelde traces met CI en PI).
    Compatibel met profiel-gebaseerde aanpak.
    """
    fig_s, ax_s = plt_instance.subplots(figsize=(8, 6))
    any_summary_plotted = False

    defect_metingen = []
    defectloos_metingen = []

    for i, lbl_s in enumerate(act_lbls):
        if 'Defectloos' in lbl_s:
            defectloos_metingen.append((i, lbl_s))
        else:
            defect_metingen.append((i, lbl_s))

    # Definieer een herbruikbare plot-functie om herhaling te voorkomen
    def plot_trace(index, label):
        nonlocal any_summary_plotted  # Zorg dat we de buitenste variabele kunnen aanpassen
        cache_s = sum_cache.get(label, {})
        mt_s = cache_s.get("mean_trace")
        N_s = cache_s.get("N_for_mean", 0)
        std_s = cache_s.get("std_trace")

        if not (N_s and N_s > 0 and mt_s is not None and len(mt_s) > 0):
            return

        any_summary_plotted = True
        num_points_in_trace = len(mt_s)

        # De `WINDOW_BEFORE` instelling uit `plotting_master.py` bepaalt het startpunt.
        start_sample_index = -settings.WINDOW_BEFORE
        end_sample_index = num_points_in_trace - settings.WINDOW_BEFORE

        # Maak een as die van negatief (vóór trigger) naar positief (na trigger) loopt.
        time_axis = np.arange(start_sample_index, end_sample_index) * settings.SAMPLE_TIME_DELTA_US

        c = settings.PLOT_COLORS[index % len(settings.PLOT_COLORS)]

        # Plot de gemiddelde lijn, deze krijgt het label voor de legenda
        ax_s.plot(time_axis, mt_s, lw=3, color=c, label=label, zorder=10 if 'Defectloos' in label else 5)

        # Toon de variabiliteit met 95% CI en PI
        if N_s > 1 and std_s is not None and len(std_s) == num_points_in_trace:
            tcrit = t.ppf(0.975, df=N_s - 1)

            # Confidence Interval
            sem_s = calculate_sem(std_s, N_s)
            ci_margin = tcrit * sem_s
            ax_s.fill_between(time_axis, mt_s - ci_margin, mt_s + ci_margin, alpha=0.35, color=c, label=f'_nolegend_')

            # Prediction Interval
            pi_margin = tcrit * std_s * np.sqrt(1 + 1 / N_s)
            ax_s.fill_between(time_axis, mt_s - pi_margin, mt_s + pi_margin, alpha=0.15, color=c, label=f'_nolegend_')

    # STAP 1: Teken eerst alle metingen MET een defect
    for index, label in defect_metingen:
        plot_trace(index, label)

    # STAP 2: Teken daarna alle metingen ZONDER defect (deze komen bovenop)
    for index, label in defectloos_metingen:
        plot_trace(index, label)
    if any_summary_plotted:
        ax_s.set_xlabel(f"Tijd relatief tot Trigger [{settings.tu_raw_lbl}]")
        # Dit repareert de crash:
        ax_s.set_ylabel("Voltage [ADC]")
        ax_s.set_title("Gemiddelde Signaal Traces ± 95% CI & PI")
        ax_s.grid(True, which='both', linestyle='--', alpha=0.5)
        ax_s.legend(title="Metingen")
        fig_s.tight_layout()
        return fig_s
    else:
        plt_instance.close(fig_s)
        return None