# Sla dit op als: Plot_Modes/plot_variability_table.py

import pandas as pd
import numpy as np


def generate_plot_variability_table(dfs, act_lbls, settings, sum_cache, plt_instance):
    """
    Kwantificeert de run-to-run variabiliteit door de gemiddelde standaarddeviatie
    van het signaal te berekenen en presenteert dit in een tabel.
    Genereert geen plot, maar print de output.
    """
    print("\n--- Kwantitatieve Variabiliteitsanalyse ---")

    variability_data = []

    # We analyseren een vast segment na de piek om de vergelijking eerlijk te maken
    SEGMENT_LENGTH_FOR_ANALYSIS = 1000  # Aantal samples na de piek

    for lbl in act_lbls:
        cache_data = sum_cache.get(lbl, {})
        mean_trace = cache_data.get("mean_trace")
        std_trace = cache_data.get("std_trace")
        N = cache_data.get("N_for_mean", 0)

        if N < 2 or mean_trace is None or std_trace is None:
            variability_data.append({"Meting": lbl, "Gem. Standaarddeviatie (ADC)": np.nan, "N": N})
            continue

        # Vind de piek om het analyse-segment te bepalen
        if len(mean_trace) > 0:
            peak_idx = np.argmax(mean_trace)
        else:
            peak_idx = 0

        start_idx = min(peak_idx, len(std_trace) - 1)
        end_idx = min(start_idx + SEGMENT_LENGTH_FOR_ANALYSIS, len(std_trace))

        if start_idx >= end_idx:
            avg_std = np.nan
        else:
            # Bereken het gemiddelde van de standaarddeviatie in het interessante segment
            segment_of_interest = std_trace[start_idx:end_idx]
            avg_std = np.mean(segment_of_interest)

        variability_data.append({"Meting": lbl, "Gem. Standaarddeviatie (ADC)": avg_std, "N": N})

    # Maak en print een nette tabel met pandas
    df_variability = pd.DataFrame(variability_data)
    df_variability = df_variability.sort_values(by="Gem. Standaarddeviatie (ADC)", ascending=True)

    print("De tabel toont de gemiddelde standaarddeviatie in het signaalsegment na de piek.")
    print("Een HOGERE waarde duidt op MEER variabiliteit (instabiliteit) tussen de runs.")
    print(df_variability.to_string(index=False))

    # Deze functie genereert geen figuur, dus we retourneren None
    return None