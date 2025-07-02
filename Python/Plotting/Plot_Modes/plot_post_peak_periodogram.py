# Sla dit op als: Plot_Modes/plot_post_peak_periodogram.py (Vervang de volledige inhoud)

import matplotlib.pyplot as plt
import numpy as np
from .common_plot_utils import _calculate_generic_periodogram_common


def generate_plot_post_peak_periodogram(dfs, act_lbls, settings, sum_cache, plt_instance):
    """
    Genereert een 'schone', directe periodogram plot.
    BELANGRIJK: De berekening achter de schermen gebruikt de correcte, profiel-specifieke
    instellingen (zoals DETREND_TYPE), maar de visualisatie is minimalistisch.
    """
    fig, ax = plt_instance.subplots(figsize=(10, 7))
    any_pg_mean_plot = False

    # ----------------------------------------------------------------------------------
    # --- FIX: Bepaal de maximale x-as limiet op basis van de DAADWERKELIJK GEPLOTTE data.
    # Dit voorkomt dat een G10-plot (max 300 µs) een x-as tot 1800 µs krijgt,
    # alleen omdat het aluminium profiel ook gedefinieerd is.
    # ----------------------------------------------------------------------------------
    max_x_lim_for_this_plot = 0
    for lbl in act_lbls:
        s_data = sum_cache.get(lbl)
        if not s_data: continue

        profile_key = s_data.get("profile_key", "default")
        profile = settings.ANALYSIS_PROFILES.get(profile_key, settings.ANALYSIS_PROFILES['default'])
        current_max_period = profile.get('MAX_PERIOD_PLOT_US', 500)  # Gebruik 500 als fallback

        if current_max_period > max_x_lim_for_this_plot:
            max_x_lim_for_this_plot = current_max_period
    # ----------------------------------------------------------------------------------
    # --- EINDE FIX ---
    # ----------------------------------------------------------------------------------

    for i, lbl in enumerate(act_lbls):
        s_data = sum_cache.get(lbl, {})
        mean_trace = s_data.get("mean_trace")
        N = s_data.get("N_for_mean", 0)

        # Haal de correcte, profiel-specifieke instellingen op
        profile_key = s_data.get("profile_key", "default")
        profile = settings.ANALYSIS_PROFILES[profile_key]
        detrend_type = profile['DETREND_TYPE']
        max_period_us = profile['MAX_PERIOD_PLOT_US']  # Deze wordt gebruikt voor de BEREKENING

        if mean_trace is None or N == 0: continue

        # Segment extractie blijft hetzelfde
        idx_peak_mean = np.argmax(mean_trace)
        segment_start_idx = idx_peak_mean + settings.POST_PEAK_OFFSET_SAMPLES
        segment_end_idx = segment_start_idx + settings.FIT_WINDOW_POST_PEAK
        data_segment = mean_trace[segment_start_idx:segment_end_idx]

        if len(data_segment) < 20: continue

        # Voer de periodogram berekening uit met de JUISTE detrend_type voor dit materiaal
        periods, mags, _ = _calculate_generic_periodogram_common(
            data_segment, settings.SAMPLE_TIME_DELTA_US,
            settings.DETREND_PERIODOGRAM, detrend_type, settings.APPLY_FFT_WINDOW,
            0, max_period_us
        )

        if periods is None or len(periods) < 10: continue

        any_pg_mean_plot = True
        # Maak een duidelijk label dat het detrend-type aangeeft
        current_label = f"{lbl} ({detrend_type[0].upper()})"
        c = settings.PLOT_COLORS[i % len(settings.PLOT_COLORS)]

        # --- De Enige Plot-regel die nodig is ---
        # Teken de pure periodogram-lijn zonder extra's
        ax.plot(periods, mags, lw=2.5, color=c, linestyle='-', label=current_label)

    if any_pg_mean_plot:
        # Configureer de plot met simpele, duidelijke labels
        ax.set_title("Periodogram Analyse")
        ax.set_xlabel(f"Periode [{settings.tu_raw_lbl}]")
        ax.set_ylabel("FFT Magnitude [a.u.]")
        ax.grid(True, which="both", ls="--", alpha=0.5)
        ax.legend(title="Meting (Detrend Type)")

        # --- FIX: Gebruik hier de correct berekende x-limiet ---
        ax.set_xlim(left=0, right=max_x_lim_for_this_plot)
        ax.set_ylim(bottom=0)
        fig.tight_layout()

        return fig
    else:
        plt_instance.close(fig)
        return None