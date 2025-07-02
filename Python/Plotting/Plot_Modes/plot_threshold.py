# Plot_Mods/plot_threshold.py
import matplotlib.pyplot as plt
import numpy as np


# common_plot_utils is niet langer nodig voor deze specifieke plot

def generate_plot_threshold(dfs, act_lbls, settings, sum_cache, plt_instance):
    """
    Generates the THRESHOLD plot by visualizing the individual, aligned traces
    from the summary_cache. This ensures consistency with the SUMMARY plot.
    """
    fig_th, ax_th = plt_instance.subplots(figsize=(12, 6))
    title_th_parts = []
    any_plot_th = False
    plotted_legend_for_file = {lbl: False for lbl in act_lbls}

    for i, lbl_th in enumerate(act_lbls):
        c_th = settings.PLOT_COLORS[i % len(settings.PLOT_COLORS)]

        # --- MODIFIED: Gebruik de voorgeladen data uit de cache ---
        if lbl_th in sum_cache and sum_cache[lbl_th].get("matrix_list"):
            aligned_traces = sum_cache[lbl_th]["matrix_list"]
            num_valid_aligned_runs = len(aligned_traces)

            if num_valid_aligned_runs > 0:
                any_plot_th = True

                # Plot elke individuele (mogelijk dynamische lengte) trace
                for trace in aligned_traces:
                    # Maak de x-as voor deze specifieke trace
                    t_rel_trace = np.arange(-settings.WINDOW_BEFORE, len(trace) - settings.WINDOW_BEFORE)

                    # Zorg voor één legend entry per bestand
                    current_label = None
                    if not plotted_legend_for_file[lbl_th]:
                        current_label = lbl_th
                        plotted_legend_for_file[lbl_th] = True

                    ax_th.plot(t_rel_trace, trace, '-', lw=1, alpha=0.5, c=c_th, label=current_label)

                title_th_parts.append(f"{lbl_th} ({num_valid_aligned_runs} valid runs)")
            else:
                title_th_parts.append(f"{lbl_th} (0 valid runs)")
        else:
            title_th_parts.append(f"{lbl_th} (no data)")

    if any_plot_th:
        ax_th.set_xlabel(
            f"Sample-idx relative to Trigger (0 = first V > {settings.THRESH} ADC, "
            f"{settings.SAMPLE_TIME_DELTA_US}{settings.tu_raw_lbl}/sample)")
        ax_th.set_ylabel(f"Voltage ({'ADC' if settings.adc_to_v(1) == 1 else 'V'})")

        # --- MODIFIED: Dynamic title ---
        window_title_str = (f"Dynamic Window (end @ <{settings.END_THRESH})"
                            if settings.USE_DYNAMIC_WINDOW_END
                            else f"Fixed Window: [-, +{settings.WINDOW_AFTER}]")

        ax_th.set_title(
            f"{settings.DEVICE_FILTER}–THRESHOLD: Aligned Traces ({window_title_str})\n"
            f"Comparing {', '.join(title_th_parts)}. "
            f"(Up to {settings.MAX_RUNS if settings.MAX_RUNS is not None else 'all'} runs processed/file)")

        h_th, l_th = ax_th.get_legend_handles_labels()
        if h_th:
            by_l_th = dict(zip(l_th, h_th))  # Verwijdert dubbele labels
            ax_th.legend(by_l_th.values(), by_l_th.keys(), title="Measurements", fontsize=8, loc='upper right')

        ax_th.grid(True, alpha=0.3)
        fig_th.tight_layout()
        fig_th.canvas.manager.set_window_title(f"Plot:THRESHOLD - {settings.DEVICE_FILTER}")
        return fig_th
    else:
        plt_instance.close(fig_th)
        print("W(THRESHOLD):No threshold-aligned traces plotted from any file.")
        return None