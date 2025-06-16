# Sla dit op als: Plot_Modes/plot_peak_prominence.py

import matplotlib.pyplot as plt
import numpy as np
from .common_plot_utils import _calculate_generic_periodogram_common


def generate_plot_peak_prominence(dfs, act_lbls, settings, sum_cache, plt_instance):
    """
    Berekent de 'prominentie' van de hoofdpiek in het periodogram voor elke meting
    en toont de resultaten in een bar chart. Gebruikt materiaal-specifieke profielen.
    """
    fig, ax = plt_instance.subplots(figsize=(14, 8))
    prominence_data = []

    for i, lbl in enumerate(act_lbls):
        s_data = sum_cache.get(lbl, {})
        mean_trace = s_data.get("mean_trace")
        N = s_data.get("N_for_mean", 0)

        # Haal de juiste profiel-instellingen op
        profile_key = s_data.get("profile_key", "default")
        profile = settings.ANALYSIS_PROFILES[profile_key]
        detrend_type = profile['DETREND_TYPE']
        max_period_us = profile['MAX_PERIOD_PLOT_US']

        if mean_trace is None or N == 0:
            continue

        # Gebruik dezelfde segment-extractie als voorheen
        idx_peak_mean = np.argmax(mean_trace)
        segment_start_idx = idx_peak_mean + settings.POST_PEAK_OFFSET_SAMPLES
        segment_end_idx = segment_start_idx + settings.FIT_WINDOW_POST_PEAK
        data_segment = mean_trace[segment_start_idx:segment_end_idx]

        if len(data_segment) < 20:
            continue

        # Bereken periodogram met de profiel-specifieke instellingen
        periods, mags, _ = _calculate_generic_periodogram_common(
            data_segment, settings.SAMPLE_TIME_DELTA_US,
            settings.DETREND_PERIODOGRAM, detrend_type, settings.APPLY_FFT_WINDOW,
            0,  # min period
            max_period_us  # max period
        )

        if periods is None or len(periods) < 10:
            ratio = 0
        else:
            # Bereken de Piek-Prominentie Ratio
            peak_amp = np.max(mags)
            peak_idx = np.argmax(mags)

            # Definieer een 'vloer'-regio (alles behalve een klein venster rond de piek)
            window_around_peak = 5  # +/- 5 datapunten
            floor_mask = np.ones_like(mags, dtype=bool)
            floor_mask[max(0, peak_idx - window_around_peak):min(len(mags), peak_idx + window_around_peak)] = False

            floor_mags = mags[floor_mask]
            if len(floor_mags) > 0:
                floor_avg = np.mean(floor_mags)
                ratio = peak_amp / floor_avg if floor_avg > 1e-6 else 0
            else:
                ratio = 0

        prominence_data.append(
            {"label": lbl, "ratio": ratio, "color": settings.PLOT_COLORS[i % len(settings.PLOT_COLORS)]})

    # Als er geen data is, sluit de lege plot en stop
    if not prominence_data:
        plt_instance.close(fig)
        return None

    # Sorteer voor een nettere plot
    prominence_data.sort(key=lambda x: x["ratio"], reverse=True)

    labels = [d["label"] for d in prominence_data]
    ratios = [d["ratio"] for d in prominence_data]
    colors = [d["color"] for d in prominence_data]

    ax.bar(labels, ratios, color=colors)
    ax.set_ylabel("Piek-Prominentie Ratio (Piek / Gem. Vloer)")
    ax.set_title("Betrouwbaarheid van Defect-Detectie: Piek-Prominentie in Periodogram")
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt_instance.xticks(rotation=45, ha="right")
    fig.tight_layout()

    # De cruciale ontbrekende regel
    return fig