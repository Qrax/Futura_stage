# Sla dit op als: Plot_Modes/plot_post_peak_periodogram_v2.py

import matplotlib.pyplot as plt
import numpy as np
from .common_plot_utils import _calculate_generic_periodogram_common
from scipy.signal import find_peaks  # Importeer de slimme piekzoeker


def generate_plot_post_peak_periodogram_v2(dfs, act_lbls, settings, sum_cache, plt_instance):
    """
    Een verbeterde versie die de 'ware' piek vindt met scipy.find_peaks,
    gebaseerd op prominentie, in plaats van de absolute maximale waarde.
    Dit voorkomt het foutief selecteren van het laatste datapunt.
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

        # --- NIEUWE, SLIMME PIEK-DETECTIE ---
        peak_indices, properties = find_peaks(mags, prominence=(None, None))

        if len(peak_indices) == 0:
            ax.plot(periods, mags, lw=2, color=c, linestyle='-', label=current_label)
            continue

        # Vind de piek met de GROOTSTE prominentie
        prominences = properties['prominences']
        most_prominent_peak_i_in_list = np.argmax(prominences)

        # Haal de eigenschappen van deze correcte piek op
        true_peak_idx = peak_indices[most_prominent_peak_i_in_list]
        true_peak_amp = mags[true_peak_idx]
        true_peak_period = periods[true_peak_idx]

        # --- DE OUDE 'VLOER' BEREKENING (voor vergelijk) ---
        window_around_peak = 5
        floor_mask = np.ones_like(mags, dtype=bool)
        floor_mask[
        max(0, true_peak_idx - window_around_peak):min(len(mags), true_peak_idx + window_around_peak)] = False
        floor_avg = np.mean(mags[floor_mask]) if len(mags[floor_mask]) > 0 else 0

        # --- PLOT DE RESULTATEN ---
        ax.plot(periods, mags, lw=2, color=c, linestyle='-', label=current_label, zorder=2)
        # Plot de marker op de CORRECT gevonden piek
        ax.plot(true_peak_period, true_peak_amp, '*', markersize=12, color=c, markeredgecolor='black', zorder=3)
        # Plot de 'globale vloer' als een horizontale lijn
        ax.axhline(y=floor_avg, color=c, linestyle='--', linewidth=1.5, zorder=1)

    if any_pg_mean_plot:
        ax.set_title("Periodogram Analyse met Correcte Piek Detectie")
        ax.set_xlabel(f"Periode ({settings.tu_raw_lbl})")
        ax.set_ylabel("FFT Magnitude")
        ax.grid(True, which="both", ls="--", alpha=0.5)
        ax.legend(title="Meting (Piek ★, Globale Vloer ---)")
        ax.set_xlim(left=0, right=max_x_lim)
        ax.set_ylim(bottom=0)
        fig.tight_layout()
        return fig
    else:
        plt_instance.close(fig)
        return None