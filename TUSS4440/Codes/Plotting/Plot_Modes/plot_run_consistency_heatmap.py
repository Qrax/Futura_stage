# Sla dit op als: Plot_Modes/plot_run_consistency_heatmap.py

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import detrend


def generate_plot_run_consistency_heatmap(dfs, act_lbls, settings, sum_cache, plt_instance):
    """
    Visualiseert de run-to-run consistentie van het (gedetrende) signaal
    met behulp van een heatmap. Genereert een aparte plot voor elke meting.
    """
    generated_figs = []

    for lbl in act_lbls:
        s_data = sum_cache.get(lbl, {})
        matrix_list = s_data.get("matrix_list", [])
        N = s_data.get("runs_count", 0)

        profile_key = s_data.get("profile_key", "default")
        detrend_type = settings.ANALYSIS_PROFILES[profile_key]['DETREND_TYPE']

        if N < 2: continue

        detrended_traces = []
        for trace in matrix_list:
            if detrend_type == 'linear':
                detrended_traces.append(detrend(trace, type='linear'))
            else:  # Voeg hier eventueel andere detrend-types toe, bv 'exponential'
                # Note: scipy.signal.detrend ondersteunt geen 'exponential' direct.
                # Voor nu gebruiken we linear voor alle heatmaps.
                detrended_traces.append(detrend(trace, type='linear'))

        max_len = max(len(t) for t in detrended_traces) if detrended_traces else 0
        if max_len == 0: continue

        padded_matrix = np.full((len(detrended_traces), max_len), np.nan)
        for i, trace in enumerate(detrended_traces):
            padded_matrix[i, :len(trace)] = trace

        fig, ax = plt_instance.subplots(figsize=(12, 8))
        im = ax.imshow(padded_matrix, aspect='auto', cmap='viridis',
                       interpolation='nearest', origin='lower',
                       extent=[0, max_len * settings.SAMPLE_TIME_DELTA_US, 0, N])

        ax.set_title(f"Run-voor-Run Consistentie Heatmap\n'{lbl}' (N={N})")
        ax.set_xlabel(f"Tijd binnen signaal-envelop ({settings.tu_raw_lbl})")
        ax.set_ylabel("Run Nummer")

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Amplitude na Detrending (ADC)")

        fig.tight_layout()
        generated_figs.append(fig)

    return generated_figs if generated_figs else None