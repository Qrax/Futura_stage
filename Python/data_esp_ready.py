import pandas as pd
import matplotlib.pyplot as plt

# --- CONFIGURATIE: CONTROLEER DEZE WAARDEN ---

# 1. De naam van jouw CSV-bestand
CSV_FILE_PATH = 'data.txt'

# 2. De naam van de kolom met de meetwaarden
COLUMN_NAME = 'ADC_Value'

# 3. Na het zien van de grafiek, vul hier de start- en eind-index in.
#    Laat op 0 staan om eerst de grafiek te bekijken.
START_INDEX = 0
END_INDEX = 0

# --- EINDE CONFIGURATIE ---


# -------- SCRIPT START --------
print("--- Puls Converter Script ---")
# Laad de CSV-data
try:
    df = pd.read_csv(CSV_FILE_PATH)
    print(f"'{CSV_FILE_PATH}' succesvol geladen met {len(df)} datapunten.")
except FileNotFoundError:
    print(f"FOUT: Kan het bestand '{CSV_FILE_PATH}' niet vinden. Zorg dat het in dezelfde map staat.")
    exit()

if COLUMN_NAME not in df.columns:
    print(f"FOUT: Kolom '{COLUMN_NAME}' niet gevonden. Beschikbare kolommen: {list(df.columns)}")
    exit()

# Als de gebruiker nog geen range heeft gekozen, toon alleen de grafiek en stop
if START_INDEX == 0 and END_INDEX == 0:
    print("Grafiek van de volledige dataset wordt nu getoond...")
    plt.figure(figsize=(15, 7))
    plt.plot(df.index, df[COLUMN_NAME])
    plt.title('Volledige Dataset - Zoek de start en het einde van je puls!')
    plt.xlabel('Index (Sample Nummer)')
    plt.ylabel('ADC Waarde')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    print("\nInstructie: Kijk op de grafiek en noteer de index-nummers.")
    print("Pas daarna START_INDEX en END_INDEX aan in het script en draai het opnieuw.")
    exit()

# Selecteer het interessante deel van de data
if END_INDEX <= START_INDEX:
    print("FOUT: END_INDEX moet groter zijn dan START_INDEX.")
    exit()

puls_data = df[COLUMN_NAME][START_INDEX:END_INDEX].values
print(f"Puls geselecteerd van index {START_INDEX} tot {END_INDEX} ({len(puls_data)} samples).")

# Normaliseer de data naar 8-bit (0-255) voor de ESP32 DAC
min_val = puls_data.min()
max_val = puls_data.max()

# Cruciale check om delen door nul te voorkomen
if max_val == min_val:
    print("FOUT: Alle waarden in de selectie zijn hetzelfde. Kan niet normaliseren.")
    exit()

# Centreer de data rond het midden (128) voor een betere AC-koppeling in de DRV2700
# Dit is een geavanceerde stap die je signaal verbetert!
normalized_data = [int(((x - min_val) / (max_val - min_val)) * 255) for x in puls_data]

# Genereer de C++ array code
cpp_array = "const int numSamples = {};\n".format(len(normalized_data))
cpp_array += "byte signalData[numSamples] = {"
cpp_array += ", ".join(map(str, normalized_data))
cpp_array += "};"

print("\n--- ✅ SUCCES! KOPIEER DE VOLGENDE CODE NAAR JE ARDUINO SKETCH ---\n")
print(cpp_array)
print("\n--------------------------------------------------------------------\n")