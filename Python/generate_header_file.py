import pandas as pd
import os

# --- Instellingen ---
# Zorg dat dit bestand in dezelfde map staat als het script
INPUT_FILE = 'data.txt'
# De naam van het C++ header-bestand dat we gaan maken
OUTPUT_FILENAME = 'signal_data.h'

# --- Script ---
print("--- TUSS Data Converter v2.0 ---")

# Controleer of het inputbestand bestaat
if not os.path.exists(INPUT_FILE):
    print(f"\nFOUT: Inputbestand '{INPUT_FILE}' niet gevonden.")
    print("Zorg ervoor dat het script in dezelfde map staat als je datafile.")
    exit()

print(f"Lezen van het bestand: {INPUT_FILE}...")
df = pd.read_csv(INPUT_FILE)

# Haal de ADC data eruit en bereken de gemiddelde delay
adc_values = df['ADC_Value'].tolist()
avg_delay = round(df['Timestamp_us'].diff().mean())

# Bepaal min en max voor het schalen om de volledige 8-bit DAC range te gebruiken
min_adc = min(adc_values)
max_adc = max(adc_values)
adc_range = max_adc - min_adc

print(f"Data gevonden. Aantal samples: {len(adc_values)}")
print(f"Originele 12-bit ADC range in dit signaal: {min_adc} tot {max_adc}")
print(f"Berekende sample delay: {avg_delay} µs")

# Voorkom delen door nul als alle waardes hetzelfde zijn
if adc_range == 0:
    scaled_values = [128] * len(adc_values) # Zet alles op het middenpunt
else:
    # Schaal de ADC waardes naar 8-bit DAC waardes (0-255)
    scaled_values = [int(((val - min_adc) * 255) / adc_range) for val in adc_values]

# --- Genereer de inhoud van het .h bestand ---
header_content = []
# Header guard: voorkomt problemen als het bestand per ongeluk vaker wordt ge-include
header_content.append("#ifndef SIGNAL_DATA_H")
header_content.append("#define SIGNAL_DATA_H\n")
header_content.append("#include <pgmspace.h>\n") # Nodig voor PROGMEM

# Schrijf de array definitie
header_content.append(f"// Data gegenereerd uit {INPUT_FILE}")
header_content.append(f"// Originele ADC range was {min_adc} - {max_adc}")
header_content.append(f"const int numSamples = {len(scaled_values)};")
header_content.append(f"const int SAMPLE_DELAY_US = {avg_delay};\n")
header_content.append(f"const byte signalData[numSamples] PROGMEM = {{")

# Voeg data toe, 20 waardes per regel voor de leesbaarheid
line = "  "
for i, val in enumerate(scaled_values):
    line += f"{val},"
    if (i + 1) % 20 == 0:
        header_content.append(line)
        line = "  "
    else:
        line += " "
if line.strip() != "":
    header_content.append(line.strip().rstrip(','))

# Sluit de array en de header guard af
header_content.append("};\n")
header_content.append("#endif // SIGNAL_DATA_H")

# Schrijf alles naar het .h bestand
try:
    with open(OUTPUT_FILENAME, 'w') as f:
        f.write('\n'.join(header_content))
    print(f"\nSUCCESS: Data succesvol weggeschreven naar '{OUTPUT_FILENAME}'")
except Exception as e:
    print(f"\nFOUT: Kon bestand niet schrijven. Error: {e}")