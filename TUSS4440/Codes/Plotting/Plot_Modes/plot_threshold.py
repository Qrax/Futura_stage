# Sla dit op als: Plot_Mods/plot_threshold.py (Vervang de volledige inhoud)

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

def generate_plot_threshold(dfs, act_lbls, settings, sum_cache, plt_instance):
    """
    Genereert de THRESHOLD plot door de individuele, uitgelijnde traces te visualiseren.
    DEZE VERSIE: Geeft elke meta-run (elk label) een eigen, duidelijke kleur
    zodat de spreiding TUSSEN de metingen zichtbaar wordt.
    """
    fig_th, ax_th = plt_instance.subplots(figsize=(12, 8))
    any_plot_th = False

    # De logica is nu veel simpeler: loop door elk label en geef het een unieke kleur.
    for i, lbl_th in enumerate(act_lbls):

        # Haal de unieke kleur voor deze specifieke meting (meta-run) op.
        color = settings.PLOT_COLORS[i % len(settings.PLOT_COLORS)]

        # Gebruik de voorgeladen en uitgelijnde data uit de cache
        if lbl_th in sum_cache and sum_cache[lbl_th].get("matrix_list"):
            aligned_traces = sum_cache[lbl_th]["matrix_list"]
            if not aligned_traces: continue

            any_plot_th = True

            # Eenmalig label voor de legenda voor deze hele meting
            legend_label = lbl_th

            # Plot elke individuele run (de 'rauwe' uitgelijnde traces)
            for j, trace in enumerate(aligned_traces):
                # Maak de x-as in tijd (microseconden) voor consistentie
                time_axis = np.arange(-settings.WINDOW_BEFORE,
                                      len(trace) - settings.WINDOW_BEFORE) * settings.SAMPLE_TIME_DELTA_US

                # Alleen de EERSTE trace van een meting krijgt het label voor een schone legenda.
                # De rest krijgt geen label.
                current_label = legend_label if j == 0 else "_nolegend_"

                # Plot de lijn met een lichte transparantie om overlap te tonen
                ax_th.plot(time_axis, trace, '-', lw=1, alpha=0.4, color=color, label=current_label)

    if any_plot_th:
        ax_th.set_xlabel(f"Tijd relatief tot Trigger [{settings.tu_raw_lbl}]")
        ax_th.set_ylabel("Voltage [ADC]")
        ax_th.set_title("Spreiding van Individuele Uitgelijnde Traces")
        ax_th.grid(True, which='both', linestyle='--', alpha=0.5)

        # Genereer de legenda. Deze zal nu correct zijn.
        handles, labels = ax_th.get_legend_handles_labels()
        if handles:
            ax_th.legend(handles, labels, title="Metingen")

        fig_th.tight_layout()
        return fig_th
    else:
        plt_instance.close(fig_th)
        print("W(THRESHOLD): Geen uitgelijnde traces gevonden om te plotten.")
        return None