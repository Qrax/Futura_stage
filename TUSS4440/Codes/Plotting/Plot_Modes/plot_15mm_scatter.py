# Sla dit op als: Plot_Modes/plot_15mm_scatter.py (UPDATE)

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from matplotlib.lines import Line2D
from .common_plot_utils import get_periodogram_analysis_steps

# --- CONFIGURATIE ---
MIN_PEAK_PROMINENCE_INITIAL = 1000
RELATIVE_PROMINENCE_THRESHOLD_PERCENT = 30
BUBBLE_SIZE_SCALING_FACTOR = 0.05


# --------------------

def _get_session_from_filename(filename):
    fn_lower = filename.lower()
    if 'please' in fn_lower: return 'Sessie 1 (please)'
    if 'gleuf_teuf' in fn_lower: return 'Sessie 2 (gleuf_teuf)'
    if 'ff_opnieuw' in fn_lower: return 'Sessie 3 (ff_opnieuw)'
    return 'Onbekende Sessie'


def generate_plot_15mm_scatter(dfs, act_lbls, settings, sum_cache, plt_instance):
    """ Genereert de bubble chart met een adaptieve filter en een nette, handmatige legenda (Stap 9). """
    print("\n" + "=" * 80);
    print("START: Generating Appendix Figure (Step 9) - Scatter Plot");
    print(f"       Relative threshold set to: {RELATIVE_PROMINENCE_THRESHOLD_PERCENT}%")
    print("=" * 80)

    plot_data = []

    for lbl in act_lbls:
        if '15mm' not in lbl: continue
        s_data = sum_cache.get(lbl, {})
        mean_trace, filename = s_data.get("mean_trace"), s_data.get("filename", "")
        if mean_trace is None or len(mean_trace) < 20: continue

        peak_amplitude = np.max(mean_trace)
        profile = settings.ANALYSIS_PROFILES[s_data.get("profile_key", "default")]

        idx_peak = np.argmax(mean_trace)
        data_segment = mean_trace[
                       idx_peak + settings.POST_PEAK_OFFSET_SAMPLES: idx_peak + settings.POST_PEAK_OFFSET_SAMPLES + settings.FIT_WINDOW_POST_PEAK]

        analysis_result = get_periodogram_analysis_steps(
            data_segment, settings.SAMPLE_TIME_DELTA_US, settings.DETREND_PERIODOGRAM,
            profile['DETREND_TYPE'], settings.APPLY_FFT_WINDOW)
        if analysis_result is None: continue

        periods_us_full, mags_full = analysis_result['periods'], analysis_result['magnitudes']
        peak_indices, properties = find_peaks(mags_full, prominence=MIN_PEAK_PROMINENCE_INITIAL)
        if len(peak_indices) == 0: continue

        found_peaks = []
        max_period_plot = profile.get('MAX_PERIOD_PLOT_US', 1800)
        for i, idx in enumerate(peak_indices):
            period = periods_us_full[idx]
            if 0 < period <= max_period_plot:
                found_peaks.append({'period': period, 'prominence': properties['prominences'][i]})

        if not found_peaks: continue
        found_peaks.sort(key=lambda p: p['prominence'], reverse=True)

        primary_prominence = found_peaks[0]['prominence']
        prominence_threshold = primary_prominence * (RELATIVE_PROMINENCE_THRESHOLD_PERCENT / 100.0)
        significant_peaks = [p for p in found_peaks if p['prominence'] >= prominence_threshold]

        plot_data.append({'label': lbl, 'session': _get_session_from_filename(filename), 'amplitude': peak_amplitude,
                          'peaks': significant_peaks})

    if not plot_data:
        print("\nE: No data left to generate the plot.\n")
        return None

    print("\n--- Analysetabel: Significante Piekprominentie per Meting ---")
    header = f"{'Label':<15} | {'Sessie':<20} | {'Piek Ampl.':>12} | {'Significante Pieken (Periode µs / Prominentie)':<50}"
    print(header);
    print("-" * (len(header) + 5))
    for data in sorted(plot_data, key=lambda x: x['label']):
        peak_info_str = "".join([f"{p['period']:.0f}µs/{p['prominence']:.0f} | " for p in data['peaks'][:3]])
        print(f"{data['label']:<15} | {data['session']:<20} | {data['amplitude']:>12.0f} | {peak_info_str}")
    print("-" * (len(header) + 5));
    print("\n")

    fig, ax = plt_instance.subplots(figsize=(14, 9))
    sessions = sorted(list(set(p['session'] for p in plot_data)))
    colors = plt.get_cmap('Set1').colors

    for i, session in enumerate(sessions):
        session_data = [p for p in plot_data if p['session'] == session]
        c = colors[i % len(colors)]

        periods_to_plot, amplitudes_to_plot, sizes_to_plot = [], [], []
        for data in session_data:
            for peak in data['peaks']:
                periods_to_plot.append(peak['period'])
                amplitudes_to_plot.append(data['amplitude'])
                sizes_to_plot.append(peak['prominence'] * BUBBLE_SIZE_SCALING_FACTOR)

        ax.scatter(periods_to_plot, amplitudes_to_plot, s=sizes_to_plot,
                   color=c, alpha=0.8, edgecolors='black', linewidths=1.2)

    legend_handles = []
    for i, session in enumerate(sessions):
        c = colors[i % len(colors)]
        handle = Line2D([0], [0], marker='o', color='w', label=session,
                        markerfacecolor=c, markeredgecolor='black',
                        markersize=15, alpha=0.8)
        legend_handles.append(handle)

    ax.legend(handles=legend_handles, title="Meetsessie", loc='upper right')

    # *** AANGEPASTE TITEL ***
    ax.set_title(
        f"Stap 9: Finale Analyse - Significante Pieken (>{RELATIVE_PROMINENCE_THRESHOLD_PERCENT}% Relatieve Drempel)")

    ax.set_xlabel(f"Periode van Significante Piek [{settings.tu_raw_lbl}]")
    ax.set_ylabel("Signaal Piekamplitude [ADC]")
    ax.grid(True, which='both', linestyle='--', alpha=0.6)
    fig.tight_layout()

    print("--- Appendix Figure (Step 9) generated successfully. ---")
    print("=" * 80)
    return fig