# Sla dit op als: Plot_Mods/plot_post_peak_periodogram.py (Vervang de volledige inhoud)

import matplotlib.pyplot as plt
import numpy as np
from .common_plot_utils import _calculate_generic_periodogram_common


def generate_plot_post_peak_periodogram(dfs, act_lbls, settings, sum_cache, plt_instance):
    """
    Genereert een 'schone', directe periodogram plot.
    CORRECTIE: Gebruikt nu directe kleurtoewijzing (1 unieke kleur per meting)
    en een context-specifieke x-as limiet.
    """
    fig, ax = plt_instance.subplots(figsize=(12, 8))
    any_pg_mean_plot = False

    # Bepaal de x-limiet op basis van de HUIDIGE data
    max_x_lim_for_this_plot = 0
    for lbl in act_lbls:
        s_data = sum_cache.get(lbl, {})
        profile_key = s_data.get("profile_key", "default")
        profile = settings.ANALYSIS_PROFILES[profile_key]
        current_max_period = profile.get('MAX_PERIOD_PLOT_US', 1800)
        if current_max_period > max_x_lim_for_this_plot:
            max_x_lim_for_this_plot = current_max_period

    # Loop simpelweg door elk label en geef het een unieke kleur.
    for i, lbl in enumerate(act_lbls):
        s_data = sum_cache.get(lbl, {})
        mean_trace, N = s_data.get("mean_trace"), s_data.get("N_for_mean", 0)

        if mean_trace is None or N == 0: continue

        # Haal de juiste profiel-instellingen op
        profile_key = s_data.get("profile_key", "default")
        profile = settings.ANALYSIS_PROFILES[profile_key]
        detrend_type = profile['DETREND_TYPE']
        max_period_us = profile['MAX_PERIOD_PLOT_US']

        # Segment extractie
        idx_peak_mean = np.argmax(mean_trace)
        segment_start_idx = idx_peak_mean + settings.POST_PEAK_OFFSET_SAMPLES
        segment_end_idx = segment_start_idx + settings.FIT_WINDOW_POST_PEAK
        data_segment = mean_trace[segment_start_idx:segment_end_idx]

        if len(data_segment) < 20: continue

        # Periodogram berekening
        periods, mags, _ = _calculate_generic_periodogram_common(
            data_segment, settings.SAMPLE_TIME_DELTA_US,
            settings.DETREND_PERIODOGRAM, detrend_type, settings.APPLY_FFT_WINDOW,
            0, max_period_us
        )

        if periods is None or len(periods) < 10: continue

        any_pg_mean_plot = True

        # Gebruik de unieke kleur die hoort bij de index van dit label
        color = settings.PLOT_COLORS[i % len(settings.PLOT_COLORS)]
        current_label = f"{lbl} ({detrend_type[0].upper()})"

        ax.plot(periods, mags, lw=2.5, color=color, linestyle='-', label=current_label)

    if any_pg_mean_plot:
        ax.set_title("Periodogram Analyse")
        ax.set_xlabel(f"Periode [{settings.tu_raw_lbl}]")
        ax.set_ylabel("FFT Magnitude [a.u.]")
        ax.grid(True, which="both", ls="--", alpha=0.5)
        ax.legend(title="Meting (Detrend Type)")
        ax.set_xlim(left=0, right=max_x_lim_for_this_plot)
        ax.set_ylim(bottom=0)
        fig.tight_layout()
        return fig
    else:
        plt_instance.close(fig)
        return None