# Sla dit op als: Plot_Modes/plot_peak_amplitude_boxplot.py (VERVANGENDE VERSIE)

import pandas as pd
import seaborn as sns
import re


def _get_condition_from_label(label_str):
    """Helper functie om een nette, gegroepeerde conditie-naam te maken."""
    material = "G10" if "g10" in label_str.lower() else "Aluminium"
    match = re.search(r'\((.*?)\)', label_str)
    if match:
        defect_info = match.group(1)
        if 'defectloos' in defect_info.lower():
            defect_info = '0mm'
        return f"{material} ({defect_info})"
    return label_str


def generate_plot_peak_amplitude_boxplot(dfs, act_lbls, settings, sum_cache, plt_instance):
    """
    NIEUWE AANPAK (GEBASEERD OP GEBRUIKERSFEEDBACK):
    Genereert een boxplot die de piekamplitudes van ELKE INDIVIDUELE RUN vergelijkt.
    Dit geeft een veel beter beeld van de consistentie van de metingen.
    """
    print("Peak Amplitude Boxplot (per run): Data verzamelen...")

    peak_data_per_run = []

    # Stap 1: Loop door elk geladen CSV-bestand (elke 'meta-meting')
    for df, label in zip(dfs, act_lbls):
        if df.empty:
            continue

        # Bepaal de nette conditie-naam voor groepering
        condition_label = _get_condition_from_label(label)

        # Stap 2: Extraheer de piekamplitude van ELKE run BINNEN dit bestand
        # We groeperen op 'Run' en vinden de max 'Voltage' voor elke groep.
        peak_amplitudes_this_file = df.groupby('Run')['Voltage'].max()

        for peak_amplitude in peak_amplitudes_this_file:
            peak_data_per_run.append({
                'Conditie': condition_label,
                'Piekamplitude [ADC]': peak_amplitude
            })

    if not peak_data_per_run:
        print("W: Geen data gevonden om de boxplot te genereren.")
        return None

    # Stap 3: Maak een Pandas DataFrame van alle verzamelde run-pieken
    peak_df = pd.DataFrame(peak_data_per_run)
    print(f"Totaal {len(peak_df)} individuele run-pieken gevonden.")

    # <<< NIEUW: BEREKEN EN PRINT DE MEDIAAN PER CONDITIE >>>
    # We groeperen de data op 'Conditie' en berekenen de mediaan van de piekamplitude.
    median_values = peak_df.groupby('Conditie')['Piekamplitude [ADC]'].median()

    print("\n--- Mediaan Piekamplitudes ---")
    # Sorteer op de gewenste plot-volgorde voor een logische output
    plot_order = [
        'Aluminium (0mm)', 'Aluminium (5mm)', 'Aluminium (15mm)',
        'G10 (0mm)', 'G10 (5mm)', 'G10 (15mm)'
    ]
    # Filter de medianen zodat alleen de aanwezige condities worden getoond, in de juiste volgorde
    for conditie in plot_order:
        if conditie in median_values.index:
            mediaan = median_values[conditie]
            print(f"{conditie:<20} Mediaan: {mediaan:.2f} [ADC]")
    print("------------------------------\n")
    # <<< EINDE NIEUWE CODE >>>

    # Stap 4: Maak de plot
    fig, ax = plt_instance.subplots(figsize=(12, 8))

    # Gebruik de al gedefinieerde plot_order variabele
    sns.boxplot(
        x='Conditie',
        y='Piekamplitude [ADC]',
        data=peak_df,
        ax=ax,
        order=plot_order,
        palette="Set2",
        showfliers=False  # Optioneel: verberg uitschieters als het te druk wordt
    )

    # We kunnen de stippen nog steeds toevoegen voor detail, maar met veel
    # datapunten kan het er rommelig uitzien. We maken ze kleiner en transparanter.
    sns.stripplot(
        x='Conditie',
        y='Piekamplitude [ADC]',
        data=peak_df,
        ax=ax,
        order=plot_order,
        color='black',
        alpha=0.2,  # Maak ze transparanter
        size=3,  # Maak ze kleiner
        jitter=True
    )

    # Stap 5: Opmaak van de plot
    ax.set_title('Vergelijking van Piekamplitudes per Run', pad=20)
    ax.set_xlabel('Meetconditie')
    ax.set_ylabel('Piekamplitude [ADC]')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt_instance.xticks(rotation=15, ha='right')
    fig.tight_layout()

    return fig