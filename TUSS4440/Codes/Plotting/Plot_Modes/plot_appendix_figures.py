# Sla dit op als: Plot_Modes/plot_appendix_figures.py (Vervang de volledige inhoud)

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t
# Pas de import aan om de functie voor exponentiële detrending direct te kunnen gebruiken
from .common_plot_utils import get_periodogram_analysis_steps, _detrend_exponential_data
from scipy.signal import detrend as sp_detrend


def generate_plot_appendix_figures(dfs, act_lbls, settings, sum_cache, plt_instance):
    """
    Genereert een reeks verslag-klare figuren voor de bijlage die het analyseproces
    illustreren, met consistente kleuren en formules in de legenda van Stap 4.
    """
    if not dfs: return None
    df_to_illustrate, label_to_illustrate = dfs[0], act_lbls[0]
    cache = sum_cache.get(label_to_illustrate)
    if not cache: return None

    # --- Haal data op ---
    num_runs_to_show = 10
    runs_to_plot_raw = sorted(df_to_illustrate.Run.unique())[:num_runs_to_show]
    aligned_traces, mean_trace, std_trace, N = cache.get("matrix_list", []), cache.get("mean_trace"), cache.get(
        "std_trace"), cache.get("N_for_mean", 0)
    if not aligned_traces or mean_trace is None: return None
    time_axis_final = np.arange(-settings.WINDOW_BEFORE,
                                len(mean_trace) - settings.WINDOW_BEFORE) * settings.SAMPLE_TIME_DELTA_US

    # --- Kleurconsistentie ---
    color_original = 'blue'
    color_exp = 'red'
    color_lin = 'green'

    # --- Plot 1 & 2 (onveranderd) ---
    fig1, ax1 = plt_instance.subplots(figsize=(12, 8));
    ax1.set_title(f'Stap 1: Ruwe Data\n({label_to_illustrate})')  # etc...
    fig2, ax2 = plt_instance.subplots(figsize=(12, 8));
    ax2.set_title(f'Stap 2: Signaaluitlijning\n({label_to_illustrate})')  # etc...

    # Plot 1
    ax1.set_xlabel(f'Tijd [{settings.tu_raw_lbl}]');
    ax1.set_ylabel('Voltage [ADC]')
    for run_id in runs_to_plot_raw: ax1.plot(df_to_illustrate[df_to_illustrate.Run == run_id]['TimePlot'],
                                             df_to_illustrate[df_to_illustrate.Run == run_id]['Voltage'], alpha=0.8)
    ax1.grid(True, linestyle='--', alpha=0.6);
    fig1.tight_layout();
    fig1.canvas.manager.set_window_title("BIJLAGE - Stap 1")
    # Plot 2
    ax2.set_xlabel('Tijd relatief tot Trigger [µs]');
    ax2.set_ylabel('Voltage [ADC]')
    for trace in aligned_traces[:num_runs_to_show]: ax2.plot(
        np.arange(-settings.WINDOW_BEFORE, len(trace) - settings.WINDOW_BEFORE) * settings.SAMPLE_TIME_DELTA_US, trace,
        alpha=0.8)
    ax2.grid(True, linestyle='--', alpha=0.6);
    fig2.tight_layout();
    fig2.canvas.manager.set_window_title("BIJLAGE - Stap 2")

    # --- Plot 3: Gemiddelde met CI en PI ---
    fig3, ax3 = plt_instance.subplots(figsize=(12, 8))  # etc...
    ax3.set_title(f'Stap 3: Gemiddelde en Spreidingsanalyse\n({label_to_illustrate})')
    ax3.set_xlabel('Tijd relatief tot Trigger [µs]');
    ax3.set_ylabel('Voltage [ADC]')
    ax3.plot(time_axis_final, mean_trace, color=color_original, lw=2.5, label=f'Gemiddelde (N={N})')
    if N > 1 and std_trace is not None and len(std_trace) == len(mean_trace):
        tcrit = t.ppf(0.975, df=N - 1);
        sem = std_trace / np.sqrt(N);
        ci_margin = tcrit * sem
        ax3.fill_between(time_axis_final, mean_trace - ci_margin, mean_trace + ci_margin, color=color_original,
                         alpha=0.6, label='95% CI')
        pi_margin = tcrit * std_trace * np.sqrt(1 + 1 / N);
        ax3.fill_between(time_axis_final, mean_trace - pi_margin, mean_trace + pi_margin, color=color_original,
                         alpha=0.3, label='95% PI')
    ax3.grid(True, linestyle='--', alpha=0.6);
    ax3.legend();
    fig3.tight_layout();
    fig3.canvas.manager.set_window_title("BIJLAGE - Stap 3")

    # --- Voorbereiding en berekeningen voor Stap 4 ---
    peak_idx = np.argmax(mean_trace)
    start_fit = peak_idx + settings.POST_PEAK_OFFSET_SAMPLES
    end_fit = start_fit + settings.FIT_WINDOW_POST_PEAK
    segment_to_analyze = mean_trace[start_fit:end_fit]
    time_segment = time_axis_final[start_fit:end_fit]

    # --- Plot 4: Vergelijking Trends MET FORMULES ---
    fig4, ax4 = plt_instance.subplots(figsize=(12, 8))
    ax4.set_title(f'Stap 4: Vergelijking van Detrending Methoden\n({label_to_illustrate})')
    ax4.set_xlabel('Tijd relatief tot Trigger [µs]')
    ax4.set_ylabel('Voltage [ADC]')

    # Plot het originele signaal
    ax4.plot(time_segment, segment_to_analyze, color=color_original, linestyle='-',
             label='Origineel Signaal (Post-Piek)', lw=2.5)

    # Bereken, plot en label de EXPONENTIËLE trend
    _, exp_trend_line, exp_params = _detrend_exponential_data(segment_to_analyze, return_trend_component=True)
    if exp_trend_line is not None:
        A = exp_params.get('A_amplitude', 0)
        B = exp_params.get('B_slope', 0)
        C = exp_params.get('C_offset', 0)
        # Formatteer de formule netjes voor de legenda
        exp_label = f'Exp. Trend: ${A:.1f} \cdot e^{{{B:.4f} \cdot t}} + {C:.1f}$'
        ax4.plot(time_segment, exp_trend_line, color=color_exp, linestyle='--', label=exp_label, lw=2.5)

    # Bereken, plot en label de LINEAIRE trend
    # De scipy detrend functie geeft de coëfficiënten niet direct, dus we berekenen ze zelf.
    x_local = np.arange(len(segment_to_analyze))  # Gebruik lokale as voor stabiliteit
    lin_coeffs = np.polyfit(x_local, segment_to_analyze, 1)
    lin_trend_line = np.poly1d(lin_coeffs)(x_local)
    m, c = lin_coeffs[0], lin_coeffs[1]
    # Formatteer de formule. Let op de 'c' (intercept)
    lin_label = f'Lin. Trend: ${m:.2f} \cdot t_{{local}} + {c:.1f}$'
    ax4.plot(time_segment, lin_trend_line, color=color_lin, linestyle='--', label=lin_label, lw=2.5)

    ax4.grid(True, linestyle='--', alpha=0.6)
    ax4.legend(fontsize='medium')  # Iets kleinere font size voor leesbaarheid
    fig4.tight_layout()
    fig4.canvas.manager.set_window_title("BIJLAGE - Stap 4")

    # --- Plots 5 en 6 (gebruiken de reeds berekende analyses) ---
    analysis_exp = get_periodogram_analysis_steps(segment_to_analyze, settings.SAMPLE_TIME_DELTA_US, True,
                                                  'exponential', False)
    analysis_lin = get_periodogram_analysis_steps(segment_to_analyze, settings.SAMPLE_TIME_DELTA_US, True, 'linear',
                                                  False)
    analysis_none = get_periodogram_analysis_steps(segment_to_analyze, settings.SAMPLE_TIME_DELTA_US, False, 'none',
                                                   False)
    if not all([analysis_exp, analysis_lin, analysis_none]): return None

    fig5, ax5 = plt_instance.subplots(figsize=(12, 8));
    ax5.set_title(f'Stap 5: Resultaten na Detrending\n({label_to_illustrate})')  # etc...
    ax5.set_xlabel('Tijd relatief tot Trigger [µs]');
    ax5.set_ylabel('Voltage [ADC]')
    ax5.plot(time_segment, analysis_exp['detrended_segment'], color=color_exp, label='Resultaat na Exp. Detrend',
             lw=2.5)
    ax5.plot(time_segment, analysis_lin['detrended_segment'], color=color_lin, label='Resultaat na Lin. Detrend',
             lw=2.5)
    ax5.plot(time_segment, analysis_none['detrended_segment'], color=color_original,
             label='Resultaat zonder Detrend (Origineel)', lw=2.5)
    ax5.axhline(0, color='black', linestyle='--', alpha=0.7);
    ax5.grid(True, linestyle='--', alpha=0.6);
    ax5.legend();
    fig5.tight_layout();
    fig5.canvas.manager.set_window_title("BIJLAGE - Stap 5")

    profile_key = cache.get('profile_key', 'default');
    max_period_plot = settings.ANALYSIS_PROFILES.get(profile_key, {})['MAX_PERIOD_PLOT_US']

    def filter_periodogram(analysis_result, max_period):
        if analysis_result is None: return None, None
        periods, mags = analysis_result['periods'], analysis_result['magnitudes']
        mask = (periods <= max_period) & (periods >= 0);
        return periods[mask], mags[mask]

    exp_p, exp_m = filter_periodogram(analysis_exp, max_period_plot)
    lin_p, lin_m = filter_periodogram(analysis_lin, max_period_plot)
    none_p, none_m = filter_periodogram(analysis_none, max_period_plot)

    fig6, ax6 = plt_instance.subplots(figsize=(12, 8));
    ax6.set_title(f'Stap 6: Impact op Periodogram\n({label_to_illustrate})')  # etc...
    ax6.set_xlabel('Periode [µs]');
    ax6.set_ylabel('FFT Magnitude [a.u.]')
    if none_p is not None: ax6.plot(none_p, none_m, color=color_original, label='Periodogram zonder Detrend', lw=2.5)
    if lin_p is not None: ax6.plot(lin_p, lin_m, color=color_lin, label='Periodogram na Lin. Detrend', lw=2.5)
    if exp_p is not None: ax6.plot(exp_p, exp_m, color=color_exp, label='Periodogram na Exp. Detrend', lw=2.5)
    ax6.set_xlim(left=0, right=max_period_plot);
    ax6.set_ylim(bottom=0);
    ax6.grid(True, which='both', alpha=0.6);
    ax6.legend();
    fig6.tight_layout();
    fig6.canvas.manager.set_window_title("BIJLAGE - Stap 6")

    return None