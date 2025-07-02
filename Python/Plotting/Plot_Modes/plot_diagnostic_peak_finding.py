# Sla dit op als: Plot_Modes/plot_diagnostic_peak_finding.py (UPDATE)

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from .common_plot_utils import get_periodogram_analysis_steps

# --- CONFIGURATIE ---
TARGET_LABEL_FOR_DIAGNOSIS = "AL 15mm (S1-M1)"
MIN_PEAK_PROMINENCE_INITIAL = 1000


# ----------------------------------------------------

def generate_plot_diagnostic_peak_finding(dfs, act_lbls, settings, sum_cache, plt_instance):
    """
    Genereert een UITGEBREIDE diagnostische plot die het proces van piek-vinden
    visueel demonstreert (Stap 8).
    """
    print("\n" + "=" * 80)
    print(f"START: Generating Appendix Figure (Step 8) for Peak Finding")
    print(f"       Target measurement: '{TARGET_LABEL_FOR_DIAGNOSIS}'")
    print("=" * 80)

    s_data = sum_cache.get(TARGET_LABEL_FOR_DIAGNOSIS)
    if not s_data:
        print(f"E: Could not find data for '{TARGET_LABEL_FOR_DIAGNOSIS}'.");
        return None
    mean_trace = s_data.get("mean_trace")
    if mean_trace is None or len(mean_trace) == 0:
        print(f"E: No 'mean_trace' found for '{TARGET_LABEL_FOR_DIAGNOSIS}'.");
        return None

    profile_key = s_data.get("profile_key", "default")
    profile = settings.ANALYSIS_PROFILES[profile_key]
    idx_peak_mean = np.argmax(mean_trace)
    segment_start_idx = idx_peak_mean + settings.POST_PEAK_OFFSET_SAMPLES
    end_idx = segment_start_idx + settings.FIT_WINDOW_POST_PEAK
    data_segment = mean_trace[segment_start_idx:end_idx]

    analysis_result = get_periodogram_analysis_steps(
        data_segment, settings.SAMPLE_TIME_DELTA_US,
        settings.DETREND_PERIODOGRAM, profile['DETREND_TYPE'], settings.APPLY_FFT_WINDOW
    )
    if analysis_result is None:
        print("E: Periodogram calculation failed.");
        return None

    periods_us, mags = analysis_result['periods'], analysis_result['magnitudes']
    max_period_plot = profile.get('MAX_PERIOD_PLOT_US', 1800)
    plot_mask = (periods_us >= 0) & (periods_us <= max_period_plot)
    periods_us, mags = periods_us[plot_mask], mags[plot_mask]

    peak_indices, properties = find_peaks(mags, prominence=MIN_PEAK_PROMINENCE_INITIAL)

    if len(peak_indices) == 0:
        print("W: No peaks found with the current threshold.");
        return None

    fig, ax = plt_instance.subplots(figsize=(15, 10))

    # *** AANGEPASTE TITEL ***
    fig.suptitle(f"Stap 8: Visuele Analyse van Piekprominentie voor '{TARGET_LABEL_FOR_DIAGNOSIS}'", fontsize=20)

    ax.plot(periods_us, mags, label='Periodogram', color='grey', alpha=0.8, zorder=1)
    ax.scatter(periods_us[peak_indices], mags[peak_indices], color='red', s=120, zorder=5,
               label=f'Gevonden Pieken (Prominentie > {MIN_PEAK_PROMINENCE_INITIAL})')

    contour_heights = mags[peak_indices] - properties["prominences"]
    ax.vlines(x=periods_us[peak_indices], ymin=contour_heights, ymax=mags[peak_indices],
              color="red", lw=2, linestyle='--', label='Berekende Prominentie')

    left_bases_indices = properties['left_bases']
    right_bases_indices = properties['right_bases']
    ax.plot(periods_us[left_bases_indices], mags[left_bases_indices], "x", markersize=10, color='blue',
            label='Linker Dal (Base)')
    ax.plot(periods_us[right_bases_indices], mags[right_bases_indices], "x", markersize=10, color='green',
            label='Rechter Dal (Base)')
    ax.hlines(y=contour_heights, xmin=periods_us[left_bases_indices], xmax=periods_us[right_bases_indices],
              color="purple", lw=2, linestyle=':', label='Prominentie Contourlijn')

    for i, idx in enumerate(peak_indices):
        period_val = periods_us[idx]
        prominence_val = properties["prominences"][i]
        ax.text(period_val, mags[idx], f' P={prominence_val:.0f}',
                verticalalignment='bottom', horizontalalignment='left', color='black',
                bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.2'))

    ax.set_title('Periodogram met Gedetailleerde Prominentie-Analyse')
    ax.set_xlabel(f"Periode [{settings.tu_raw_lbl}]")
    ax.set_ylabel('FFT Magnitude [a.u.]')
    ax.set_xlim(0, max_period_plot)
    ax.set_ylim(bottom=np.min(mags) - np.std(mags) * 0.1, top=np.max(mags) + np.std(mags) * 0.1)
    ax.legend(loc='upper right')
    ax.grid(True, which='both', linestyle='--', alpha=0.6)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    print("--- Appendix Figure (Step 8) generated successfully. ---")
    print("=" * 80 + "\n")
    return fig