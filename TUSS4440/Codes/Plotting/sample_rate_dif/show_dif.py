import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Deel 1: Data Inladen en Focuspunt Bepalen ---

try:
    df = pd.read_csv("data.txt")
except FileNotFoundError:
    print("FOUT: Het bestand 'data.txt' werd niet gevonden.")
    print("Zorg ervoor dat de data is opgeslagen in 'data.txt' in dezelfde map als dit script.")
    exit()

# Bepaal het punt van de steilste stijging voor het centreren van de plots
steepest_rise_index = df['ADC_Value'].diff().idxmax()
steepest_rise_time = df.loc[steepest_rise_index, 'Timestamp_us']

print(f"Focuspunt voor de plots: Tijdstip = {steepest_rise_time:.2f} µs")
print("-" * 40)

# --- Deel 2: Definieer Plot-venster en Simuleer Data ---

# Uitgebreid tijdvenster
time_window_before = 120
time_window_after = 370
start_time = steepest_rise_time - time_window_before
end_time = steepest_rise_time + time_window_after

# Filter de originele data voor het plot-venster
plot_data_original = df[(df['Timestamp_us'] >= start_time) & (df['Timestamp_us'] <= end_time)]


# Functie voor simulatie
def resample_data(original_df, new_sample_time):
    new_timestamps = np.arange(0, original_df['Timestamp_us'].max(), new_sample_time)
    indices = [original_df['Timestamp_us'].sub(ts).abs().idxmin() for ts in new_timestamps]
    unique_indices = sorted(list(set(indices)))
    resampled_df = original_df.loc[unique_indices]
    # Filter meteen voor het plot-venster
    return resampled_df[(resampled_df['Timestamp_us'] >= start_time) & (resampled_df['Timestamp_us'] <= end_time)]


# --- Deel 3: Plotfunctie voor een Enkele Grafiek ---

def create_standalone_plot(data, title, filename, color, label):
    """Genereert en slaat een enkele, groot opgemaakte plot op ZONDER deze direct te tonen."""
    plt.figure(figsize=(10, 8))  # Maak een nieuwe figuur aan voor elke plot

    plt.scatter(data['Timestamp_us'], data['ADC_Value'], s=50, color=color, label=label)

    # Opmaak voor presentatie
    plt.title(title, fontsize=24, pad=20)
    plt.xlabel('Tijd (µs)', fontsize=18, labelpad=15)
    plt.ylabel('ADC Waarde', fontsize=18, labelpad=15)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=16)
    plt.xlim(start_time, end_time)
    plt.ylim(plot_data_original['ADC_Value'].min() - 50, plot_data_original['ADC_Value'].max() + 50)

    # Opslaan
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Plot opgeslagen als '{filename}'")


# --- Deel 4: Genereer de Plots in Presentatie-volgorde ---

# Definieer de volgorde, titels en kleuren voor de presentatie
plots_to_generate = [
    {'rate': 100, 'title': 'Gesimuleerde Sampletijd: 100 µs', 'color': 'crimson', 'label': 'Gesimuleerde meetpunten'},
    {'rate': 46, 'title': 'Gesimuleerde Sampletijd: 46 µs', 'color': 'darkorange', 'label': 'Gesimuleerde meetpunten'},
    {'rate': 16.3, 'title': 'Gesimuleerde Sampletijd: 16.3 µs', 'color': 'forestgreen',
     'label': 'Gesimuleerde meetpunten'},
    {'rate': 4.58, 'title': 'Originele Data (Sampletijd: 4.58 µs)', 'color': 'dodgerblue',
     'label': 'Originele meetpunten'}
]

for plot_info in plots_to_generate:
    rate = plot_info['rate']

    if rate == 4.58:
        data_to_plot = plot_data_original
    else:
        data_to_plot = resample_data(df, rate)

    create_standalone_plot(
        data=data_to_plot,
        title=plot_info['title'],
        filename=f'Plot_{rate}us.png',
        color=plot_info['color'],
        label=plot_info['label']
    )

print("-" * 40)
print("Alle plots zijn succesvol gegenereerd en opgeslagen.")
print("Alle vensters worden nu tegelijk geopend.")

# Toon alle gemaakte figuren tegelijkertijd
plt.show()