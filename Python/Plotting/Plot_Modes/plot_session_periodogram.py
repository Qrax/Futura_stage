# Sla dit op als: Plot_Modes/plot_session_periodogram.py (NIEUW BESTAND)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from .common_plot_utils import _calculate_generic_periodogram_common


# Helper-functie om sessienaam te bepalen, identiek aan de scatter plot
def _get_session_from_filename(filename):
    """ Bepaalt de sessienaam op basis van de bestandsnaam. """
    if not filename: return 'Onbekende Sessie'
    fn_lower = filename.lower()
    if 'please' in fn_lower: return 'Sessie 1'
    if 'gleuf_teuf' in fn_lower: return 'Sessie 2'
    if 'ff_opnieuw' in fn_lower: return 'Sessie 3'
    return 'Onbekende Sessie'


def generate_plot_session_periodogram(dfs, act_lbls, settings, sum_cache, plt_instance):
    """
    Genereert een periodogram plot waarbij de kleuren per SESSIE worden gegroepeerd,
    identiek aan de scatter plot, voor een overzicht op hoger niveau.
    """
    fig, ax = plt_instance.subplots(figsize=(10, 7))
    any_pg_mean_plot = False

    # --- Maak de kleur-mapping op basis van sessies ---
    all_sessions = sorted(list(set(
        _get_session_from_filename(sum_cache.get(lbl, {}).get("filename", ""))
        for lbl in act_lbls if sum_cache.get(lbl)
    )))
    colors = plt.get_cmap('Set1').colors
    session_color_map = {session: colors[i % len(colors)] for i, session in enumerate(all_sessions)}

    # --- Bepaal de maximale x-as ---
    max_x_lim_for_this_plot = 0
    for lbl in act_lbls:
        s_data = sum_cache.get(lbl)
        if not s_data: continue
        profile_key = s_data.get("profile_key", "default")
        profile = settings.ANALYSIS_PROFILES.get(profile_key, settings.ANALYSIS_PROFILES['default'])
        current_max_period = profile.get('MAX_PERIOD_PLOT_US', 500)
        if current_max_period > max_x_lim_for_this_plot:
            max_x_lim_for_this_plot = current_max_period

    # --- Teken de periodogrammen met de juiste sessie-kleur ---
    for lbl in act_lbls:
        s_data = sum_cache.get(lbl, {})
        mean_trace = s_data.get("mean_trace")
        if mean_trace is None or len(mean_trace) < 20: continue

        profile_key = s_data.get("profile_key", "default")
        profile = settings.ANALYSIS_PROFILES[profile_key]

        data_segment = mean_trace[
                       np.argmax(mean_trace) + settings.POST_PEAK_OFFSET_SAMPLES:
                       np.argmax(mean_trace) + settings.POST_PEAK_OFFSET_SAMPLES + settings.FIT_WINDOW_POST_PEAK
                       ]

        if len(data_segment) < 20: continue

        periods, mags, _ = _calculate_generic_periodogram_common(
            data_segment=data_segment,
            sample_delta_us=settings.SAMPLE_TIME_DELTA_US,
            do_detrend_gate=settings.DETREND_PERIODOGRAM,
            current_detrend_type=profile['DETREND_TYPE'],
            do_apply_fft_window=settings.APPLY_FFT_WINDOW,
            min_period_us_plot=0,
            max_period_us_plot=profile['MAX_PERIOD_PLOT_US']
        )

        if periods is None or len(periods) == 0: continue

        any_pg_mean_plot = True

        # Haal de juiste kleur op uit de map
        filename = s_data.get("filename", "")
        session = _get_session_from_filename(filename)
        c = session_color_map.get(session, 'grey')

        # Teken de lijn met lagere alpha om overlappingen te tonen
        ax.plot(periods, mags, lw=2.0, color=c, linestyle='-', alpha=0.7)

    if any_pg_mean_plot:
        # --- Maak de handmatige legenda en configureer de plot ---
        ax.set_title("Periodogram Analyse per Sessie")
        ax.set_xlabel(f"Periode [{settings.tu_raw_lbl}]")
        ax.set_ylabel("FFT Magnitude [a.u.]")
        ax.grid(True, which="both", ls="--", alpha=0.5)

        legend_handles = [Line2D([0], [0], color=color, lw=2.5, label=session)
                          for session, color in session_color_map.items()]

        ax.legend(handles=legend_handles, title="Meetsessie")

        ax.set_xlim(left=0, right=max_x_lim_for_this_plot)
        ax.set_ylim(bottom=0)
        fig.tight_layout()

        return fig
    else:
        plt_instance.close(fig)
        return None