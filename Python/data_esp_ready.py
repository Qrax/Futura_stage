import pandas as pd
import os
import numpy as np

# --- Instellingen ---
INPUT_FILE = 'data.txt'
OUTPUT_FILENAME = 'signal_data.h'
# Aantal samples om te gebruiken voor een soepele overgang van/naar het nulpunt
PADDING_SAMPLES = 50

# --- Script ---
print("--- TUSS Data Converter v3.0 (met Padding) ---")

if not os.path.exists(INPUT_FILE):
    print(f"\nFOUT: Inputbestand '{INPUT_FILE}' niet gevonden.")
    exit()

print(f"Lezen van het bestand: {INPUT_FILE}...")
df = pd.read_csv(INPUT_FILE)
adc_values = df['ADC_Value'].tolist()
avg_delay = round(df['Timestamp_us'].diff().mean())

min_adc = min(adc_values)
max_adc = max(adc_values)
adc_range = max_adc - min_adc

if adc_range == 0:
    scaled_values = [128] * len(adc_values)
else:
    scaled_values = [int(((val - min_adc) * 255) / adc_range) for val in adc_values]

# --- VOEG PADDING TOE ---
NEUTRAL_VALUE = 128
# Padding aan het begin: van 128 naar de eerste sample waarde
start_padding = np.linspace(NEUTRAL_VALUE, scaled_values[0], PADDING_SAMPLES, dtype=int)
# Padding aan het eind: van de laatste sample waarde naar 128
end_padding = np.linspace(scaled_values[-1], NEUTRAL_VALUE, PADDING_SAMPLES, dtype=int)

# Voeg alles samen tot het definitieve signaal
final_signal = list(start_padding) + scaled_values + list(end_padding)

print(f"Signaal voorbereid: {PADDING_SAMPLES} padding, {len(scaled_values)} data, {PADDING_SAMPLES} padding.")
print(f"Totaal aantal samples: {len(final_signal)}")

# --- Genereer het .h bestand ---
# (De rest van het script is hetzelfde als voorheen, maar gebruikt 'final_signal')
header_content = []
header_content.append("#ifndef SIGNAL_DATA_H\n#define SIGNAL_DATA_H\n\n#include <pgmspace.h>\n")
header_content.append(f"const int numSamples = {len(final_signal)};")
header_content.append(f"const int SAMPLE_DELAY_US = {avg_delay};\n")
header_content.append(f"const byte signalData[numSamples] PROGMEM = {{")
line = "  "
for i, val in enumerate(final_signal):
    line += f"{val},"
    if (i + 1) % 20 == 0:
        header_content.append(line)
        line = "  "
    else:
        line += " "
if line.strip() != "":
    header_content.append(line.strip().rstrip(','))
header_content.append("};\n\n#endif // SIGNAL_DATA_H")

with open(OUTPUT_FILENAME, 'w') as f:
    f.write('\n'.join(header_content))

print(f"\nSUCCESS: Data met padding weggeschreven naar '{OUTPUT_FILENAME}'")