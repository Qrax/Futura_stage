# Sla dit op als: Plot_Modes/plot_appendix_figures.py

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t

def generate_plot_appendix_figures(dfs, act_lbls, settings, sum_cache, plt_instance):
    """
    Genereert VIER APARTE, verslag-klare figuren voor de bijlage.
    Elke plot krijgt zijn eigen venster voor makkelijke screenshots.
    """
    # --- Data Selectie ---
    if not dfs:
        return None
    df_to_illustrate = dfs[0]
    label_to_illustrate = act_lbls[0]
    cache = sum_cache.get(label_to_illustrate)
    if not cache:
        return None

    # --- Haal benodigde data op ---
    num_runs_to_show = 10
    runs_to_plot_raw = sorted(df_to_illustrate.Run.unique())[:num_runs_to_show]
    aligned_traces = cache.get("matrix_list", [])
    mean_trace = cache.get("mean_trace")
    std_trace = cache.get("std_trace")
    N = cache.get("N_for_mean", 0)

    if not aligned_traces or mean_trace is None or std_trace is None or N == 0:
        return None

    # --- Plot 1: Ruwe Data (Niet-uitgelijnd) ---
    fig1, ax1 = plt_instance.subplots(figsize=(12, 8))
    for run_id in runs_to_plot_raw:
        run_data = df_to_illustrate[df_to_illustrate.Run == run_id]
        ax1.plot(run_data['Timestamp_us'], run_data['Voltage'], alpha=0.8)
    ax1.set_title(f'Stap 1: Ruwe Data\n({label_to_illustrate})')
    ax1.set_xlabel('Tijd [µs]')
    ax1.set_ylabel('Voltage [ADC]')
    ax1.grid(True, linestyle='--', alpha=0.6)
    fig1.tight_layout()
    fig1.canvas.manager.set_window_title("BIJLAGE - Stap 1: Ruwe Data")

    # --- Plot 2: Uitgelijnde Data (na Triggering) ---
    fig2, ax2 = plt_instance.subplots(figsize=(12, 8))
    for trace in aligned_traces[:num_runs_to_show]:
        time_axis = np.arange(-settings.WINDOW_BEFORE, len(trace) - settings.WINDOW_BEFORE) * settings.SAMPLE_TIME_DELTA_US
        ax2.plot(time_axis, trace, alpha=0.8)
    ax2.set_title(f'Stap 2: Signaaluitlijning (Triggering)\n({label_to_illustrate})')
    ax2.set_xlabel('Tijd relatief tot Trigger [µs]')
    ax2.set_ylabel('Voltage [ADC]')
    ax2.grid(True, linestyle='--', alpha=0.6)
    fig2.tight_layout()
    fig2.canvas.manager.set_window_title("BIJLAGE - Stap 2: Uitgelijnde Data")

    # --- Plot 3: Gemiddelde en Spreidingsanalyse ---
    fig3, ax3 = plt_instance.subplots(figsize=(12, 8))
    time_axis_final = np.arange(-settings.WINDOW_BEFORE, len(mean_trace) - settings.WINDOW_BEFORE) * settings.SAMPLE_TIME_DELTA_US
    ax3.plot(time_axis_final, mean_trace, color='black', lw=2.5, label=f'Gemiddelde (N={N})')
    if N > 1:
        tcrit = t.ppf(0.975, df=N - 1)
        sem = std_trace / np.sqrt(N)
        ci_margin = tcrit * sem
        pi_margin = tcrit * std_trace * np.sqrt(1 + 1 / N)
        ax3.fill_between(time_axis_final, mean_trace - pi_margin, mean_trace + pi_margin, color='gray', alpha=0.3, label='95% Prediction Interval')
        ax3.fill_between(time_axis_final, mean_trace - ci_margin, mean_trace + ci_margin, color='gray', alpha=0.6, label='95% Confidence Interval')
    ax3.set_title(f'Stap 3: Gemiddelde en Spreidingsanalyse\n({label_to_illustrate})')
    ax3.set_xlabel('Tijd relatief tot Trigger [µs]')
    ax3.set_ylabel('Voltage [ADC]')
    ax3.grid(True, linestyle='--', alpha=0.6)
    ax3.legend()
    fig3.tight_layout()
    fig3.canvas.manager.set_window_title("BIJLAGE - Stap 3: Gemiddelde Data")

    # --- Plot 4: Uitleg Exponentiële Detrending ---
    fig4, ax4 = plt_instance.subplots(figsize=(12, 8))
    profile_key = cache.get('profile_key', 'default')
    current_profile = settings.ANALYSIS_PROFILES[profile_key]
    peak_idx = np.argmax(mean_trace)
    offset = settings.POST_PEAK_OFFSET_SAMPLES
    fit_window_samples = current_profile.get('MAX_WINDOW_AFTER', 1000) - offset
    start_fit = peak_idx + offset
    end_fit = start_fit + fit_window_samples
    segment = mean_trace[start_fit:end_fit]
    time_segment = time_axis_final[start_fit:end_fit]
    if len(segment) > 1:
        log_segment = np.log(segment)
        coeffs = np.polyfit(time_segment, log_segment, 1)
        exp_fit = np.exp(coeffs[1]) * np.exp(coeffs[0] * time_segment)
        detrended_signal = segment - exp_fit
        ax4.plot(time_segment, segment, label='Origineel signaal (post-piek)', lw=2.5)
        ax4.plot(time_segment, exp_fit, 'r--', label='Exponentiële Fit (Trend)', lw=2.5)
        ax4.plot(time_segment, detrended_signal, 'g-', label='Resultaat na Detrending', lw=2.5)
    ax4.set_title(f'Stap 4: Voorbereiding voor FFT (Detrending)\n({label_to_illustrate})')
    ax4.set_xlabel('Tijd relatief tot Trigger [µs]')
    ax4.set_ylabel('Voltage [ADC]')
    ax4.grid(True, linestyle='--', alpha=0.6)
    ax4.legend()
    fig4.tight_layout()
    fig4.canvas.manager.set_window_title("BIJLAGE - Stap 4: Detrending")

    # De functie retourneert niks, omdat de plots via plt.show() worden getoond.
    return None