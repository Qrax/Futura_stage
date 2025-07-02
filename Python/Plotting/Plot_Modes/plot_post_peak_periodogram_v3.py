# Sla dit op als: Plot_Modes/plot_post_peak_periodogram_v3.py

import matplotlib.pyplot as plt
import numpy as np
from .common_plot_utils import _calculate_generic_periodogram_common
from scipy.signal import find_peaks  # Importeer de slimme piekzoeker


def generate_plot_post_peak_periodogram_v3(dfs, act_lbls, settings, sum_cache, plt_instance):
    """
    DE FINALE, MEEST ROBUUSTE VERSIE.
    - Gebruikt profiel-specifieke detrending.
    - Gebruikt scipy.find_peaks om de piek met de GROOTSTE PROMINENTIE te vinden.
    - Visualiseert deze berekende, ware prominentie.
    """
    fig, ax = plt_instance.subplots(figsize=(14, 8))
    any_pg_mean_plot = False

    max_x_lim = max(p['MAX_PERIOD_PLOT_US'] for p in settings.ANALYSIS_PROFILES.values())

    for i, lbl in enumerate(act_lbls):
        s_data = sum_cache.get(lbl, {})
        mean_trace = s_data.get("mean_trace")
        N = s_data.get("N_for_mean", 0)
        profile_key = s_data.get("profile_key", "default")
        profile = settings.ANALYSIS_PROFILES[profile_key]
        detrend_type = profile['DETREND_TYPE']
        max_period_us = profile['MAX_PERIOD_PLOT_US']

        if mean_trace is None or N == 0: continue

        idx_peak_mean = np.argmax(mean_trace)
        segment_start_idx = idx_peak_mean + settings.POST_PEAK_OFFSET_SAMPLES
        segment_end_idx = segment_start_idx + settings.FIT_WINDOW_POST_PEAK
        data_segment = mean_trace[segment_start_idx:segment_end_idx]

        if len(data_segment) < 20: continue

        periods, mags, _ = _calculate_generic_periodogram_common(
            data_segment, settings.SAMPLE_TIME_DELTA_US,
            settings.DETREND_PERIODOGRAM, detrend_type, settings.APPLY_FFT_WINDOW,
            0, max_period_us
        )

        if periods is None or len(periods) < 10: continue

        any_pg_mean_plot = True
        current_label = f"{lbl} ({detrend_type[0].upper()})"
        c = settings.PLOT_COLORS[i % len(settings.PLOT_COLORS)]

        # --- DE NIEUWE, SLIMME PIEK-DETECTIE ---
        # 1. Vind alle pieken en hun prominenties
        peak_indices, properties = find_peaks(mags, prominence=(None, None))

        # Als er geen pieken worden gevonden, plot alleen de lijn
        if len(peak_indices) == 0:
            ax.plot(periods, mags, lw=2, color=c, linestyle='-', label=current_label)
            continue

        # 2. Vind de piek met de GROOTSTE prominentie
        prominences = properties['prominences']
        most_prominent_peak_i_in_list = np.argmax(prominences)

        # 3. Haal de eigenschappen van deze specifieke piek op
        true_peak_idx = peak_indices[most_prominent_peak_i_in_list]
        true_peak_prominence = prominences[most_prominent_peak_i_in_list]
        true_peak_amp = mags[true_peak_idx]
        true_peak_period = periods[true_peak_idx]

        # --- PLOT DE WARE PROMINENTIE ---
        ax.plot(periods, mags, lw=1.5, color=c, linestyle='-', label=current_label, alpha=0.7)
        ax.plot(true_peak_period, true_peak_amp, 'o', markersize=8, color=c, markeredgecolor='black', zorder=3)

        # De verticale lijn toont nu de BEREKENDE, WARE prominentie
        ymin = true_peak_amp - true_peak_prominence
        ax.vlines(x=true_peak_period, ymin=ymin, ymax=true_peak_amp,
                  colors=c, linestyles='solid', lw=3, zorder=2)
        # Teken een klein horizontaal streepje aan de onderkant voor duidelijkheid
        ax.hlines(y=ymin, xmin=true_peak_period - 5, xmax=true_peak_period + 5, colors=c, lw=2)

    if any_pg_mean_plot:
        ax.set_title("Finale Periodogram Analyse met Piek Prominentie Detectie")
        ax.set_xlabel(f"Periode ({settings.tu_raw_lbl})")
        ax.set_ylabel("FFT Magnitude")
        ax.grid(True, which="both", ls="--", alpha=0.5)
        ax.legend(title="Meting (Detrend Type)")
        ax.set_xlim(left=0, right=max_x_lim)
        ax.set_ylim(bottom=0)
        fig.tight_layout()
        return fig
    else:
        plt_instance.close(fig)
        return None