# Sla dit op als: Plot_Modes/plot_appendix_peak_finding_explained.py (FINALE VERSIE MET CORRECTE STAPPEN)

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from matplotlib.lines import Line2D
from .common_plot_utils import get_periodogram_analysis_steps

# --- CONFIGURATIE ---
TARGET_LABEL = "AL 15mm (S1-M1)"
MIN_PEAK_PROMINENCE_INITIAL = 1000
RELATIVE_PROMINENCE_THRESHOLD_PERCENT = 30
BUBBLE_SIZE_SCALING_FACTOR = 0.05


# --------------------

def generate_plot_appendix_peak_finding_explained(dfs, act_lbls, settings, sum_cache, plt_instance):
    """
    Genereert alle figuren (Stap 7, 8, 9 en 10) voor de bijlage
    over piekanalyse, gebaseerd op één enkel voorbeeldsignaal.
    """
    print("\n" + "=" * 80)
    print(f"START: Generating full appendix for Peak Finding using '{TARGET_LABEL}'")
    print("=" * 80)

    s_data = sum_cache.get(TARGET_LABEL)
    if not s_data or s_data.get("mean_trace") is None:
        print(f"E: Kon data voor '{TARGET_LABEL}' niet vinden.");
        return None

    mean_trace = s_data["mean_trace"]
    peak_amplitude = np.max(mean_trace)
    profile = settings.ANALYSIS_PROFILES[s_data["profile_key"]]
    detrend_type = profile['DETREND_TYPE']
    max_period_plot = profile['MAX_PERIOD_PLOT_US']

    data_segment = mean_trace[np.argmax(mean_trace) + settings.POST_PEAK_OFFSET_SAMPLES: np.argmax(
        mean_trace) + settings.POST_PEAK_OFFSET_SAMPLES + settings.FIT_WINDOW_POST_PEAK]
    analysis_result = get_periodogram_analysis_steps(data_segment, settings.SAMPLE_TIME_DELTA_US,
                                                     settings.DETREND_PERIODOGRAM, detrend_type,
                                                     settings.APPLY_FFT_WINDOW)
    if analysis_result is None: return None

    periods_us, mags = analysis_result['periods'], analysis_result['magnitudes']
    plot_mask = (periods_us >= 0) & (periods_us <= max_period_plot)
    plot_periods, plot_mags = periods_us[plot_mask], mags[plot_mask]
    peak_indices, properties = find_peaks(plot_mags, prominence=MIN_PEAK_PROMINENCE_INITIAL)

    # Genereer de figuren in de juiste volgorde
    _generate_step7_figure(plot_periods, plot_mags, max_period_plot, settings, plt_instance)
    _generate_step8_figure(plot_periods, plot_mags, peak_indices, properties, max_period_plot, settings, plt_instance)
    _generate_step9_figure(plot_periods, plot_mags, peak_indices, properties, max_period_plot, settings, plt_instance)
    _generate_step10_figure(plot_periods, peak_indices, properties, peak_amplitude, settings, plt_instance)

    print("\n" + "=" * 80);
    print("SUCCESS: All 4 appendix figures generated.");
    print("=" * 80)
    return None


def _generate_step7_figure(plot_periods, plot_mags, max_period_plot, settings, plt_instance):
    fig, ax = plt_instance.subplots(figsize=(12, 7))
    ax.plot(plot_periods, plot_mags, lw=2.5, color="red")
    ax.set_title(f"Stap 7: Input voor Piekanalyse (Periodogram van '{TARGET_LABEL}')")
    ax.set_xlabel(f"Periode [{settings.tu_raw_lbl}]");
    ax.set_ylabel("FFT Magnitude [a.u.]")
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.set_xlim(left=0, right=max_period_plot);
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    print("Generated Appendix Figure (Step 7)")


def _generate_step8_figure(plot_periods, plot_mags, peak_indices, properties, max_period_plot, settings, plt_instance):
    fig, ax = plt_instance.subplots(figsize=(15, 10))
    fig.suptitle(f"Stap 8: Initiële Piek-detectie voor '{TARGET_LABEL}'", fontsize=20)
    ax.plot(plot_periods, plot_mags, label='Periodogram', color='grey', alpha=0.8, zorder=1)
    ax.scatter(plot_periods[peak_indices], plot_mags[peak_indices], color='red', s=120, zorder=5,
               label=f'Alle Gevonden Pieken (Prom. > {MIN_PEAK_PROMINENCE_INITIAL})')
    for i, idx in enumerate(peak_indices):
        ax.text(plot_periods[idx], plot_mags[idx], f' P={properties["prominences"][i]:.0f}', va='bottom', ha='left',
                bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.2'))
    ax.set_title('Alle kandidaten op basis van absolute prominentiedrempel')
    ax.set_xlabel(f"Periode [{settings.tu_raw_lbl}]");
    ax.set_ylabel('FFT Magnitude [a.u.]')
    ax.set_xlim(0, max_period_plot)
    ax.legend(loc='upper right');
    ax.grid(True, which='both', linestyle='--', alpha=0.6)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    print("Generated Appendix Figure (Step 8)")


def _generate_step9_figure(plot_periods, plot_mags, peak_indices, properties, max_period_plot, settings, plt_instance):
    fig, ax = plt_instance.subplots(figsize=(15, 10))
    prominences = properties['prominences']
    if len(prominences) == 0: return

    max_prom = np.max(prominences)
    threshold = max_prom * (RELATIVE_PROMINENCE_THRESHOLD_PERCENT / 100.0)

    fig.suptitle(f"Stap 9: Toepassing van Relatieve Drempel ({threshold:.0f})", fontsize=20)
    ax.plot(plot_periods, plot_mags, color='grey', alpha=0.5, zorder=1)

    for i, idx in enumerate(peak_indices):
        prom = prominences[i]
        is_significant = prom >= threshold
        color = 'green' if is_significant else 'dimgray'
        marker_size = 140 if is_significant else 70
        z_order = 10 if is_significant else 5
        ax.scatter(plot_periods[idx], plot_mags[idx], color=color, s=marker_size, zorder=z_order, edgecolors='black',
                   linewidth=1 if is_significant else 0.5)
        ax.text(plot_periods[idx], plot_mags[idx], f' P={prom:.0f}', va='bottom', ha='left',
                color='black' if is_significant else 'gray',
                bbox=dict(facecolor='white', alpha=0.6 if is_significant else 0.3, boxstyle='round,pad=0.2'))

    ax.set_title(f"Significante pieken (groen) vs. niet-significante pieken (grijs)")
    ax.set_xlabel(f"Periode [{settings.tu_raw_lbl}]");
    ax.set_ylabel('FFT Magnitude [a.u.]')
    ax.set_xlim(0, max_period_plot)
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Significante Piek', markerfacecolor='green',
                              markeredgecolor='black', markersize=12),
                       Line2D([0], [0], marker='o', color='w', label='Niet-Significante Piek',
                              markerfacecolor='dimgray', markersize=10)]
    ax.legend(handles=legend_elements, loc='upper right')
    ax.grid(True, which='both', linestyle='--', alpha=0.6)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    print("Generated Appendix Figure (Step 9)")


def _generate_step10_figure(plot_periods, peak_indices, properties, peak_amplitude, settings, plt_instance):
    fig, ax = plt_instance.subplots(figsize=(14, 9))
    prominences = properties.get('prominences')
    if prominences is not None and len(prominences) > 0:
        max_prom = np.max(prominences)
        threshold = max_prom * (RELATIVE_PROMINENCE_THRESHOLD_PERCENT / 100.0)

        periods_to_plot, sizes_to_plot = [], []
        for i, prom in enumerate(prominences):
            if prom >= threshold:
                periods_to_plot.append(plot_periods[peak_indices[i]])
                sizes_to_plot.append(prom * BUBBLE_SIZE_SCALING_FACTOR)

        if periods_to_plot:
            amplitudes_to_plot = [peak_amplitude] * len(periods_to_plot)
            color = settings.PLOT_COLORS[0]
            ax.scatter(periods_to_plot, amplitudes_to_plot, s=sizes_to_plot, color=color, alpha=0.8, edgecolors='black',
                       linewidths=1.2)

    ax.set_title(f"Stap 10: Finale Analyse - Alleen Significante Pieken voor '{TARGET_LABEL}'")
    ax.set_xlabel(f"Periode van Significante Piek [{settings.tu_raw_lbl}]")
    ax.set_ylabel("Signaal Piekamplitude [ADC]")
    ax.grid(True, which='both', linestyle='--', alpha=0.6)
    fig.tight_layout()
    print("Generated Appendix Figure (Step 10)")