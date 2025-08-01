import serial
import matplotlib.pyplot as plt
import numpy as np
import time

# --- CONFIGURATIE ---
SERIAL_PORT = 'COM4'  # <-- VERANDER DIT!
BAUD_RATE = 921600
MAX_ADC_VALUE = 4095.0
MAX_VOLTAGE = 3.0
SAMPLE_RATE_HZ_EXPECTED = 400000

# --- SCRIPT ---
samples = []
metadata = {} # Gebruik een dictionary om alle metadata op te slaan
is_connected = False

print(f"Poging tot verbinden met poort {SERIAL_PORT} op {BAUD_RATE} baud...")

try:
    esp32 = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=5) # Iets langere timeout
    is_connected = True
    time.sleep(2) 
    print("Succesvol verbonden met de ESP32.")

    print("\nVerstuur meetcommando 'm'...")
    esp32.write(b'm')

    print("Lezen van data en statusberichten...")
    while True:
        try:
            line = esp32.readline().decode('utf-8', errors='ignore').strip()
            
            if not line:
                # Als de timeout is bereikt, stop de lus
                print("Timeout bereikt, stoppen met lezen.")
                break

            # Check voor metadata die een ':' bevat
            if ':' in line:
                parts = line.split(':', 1)
                key = parts[0]
                value = parts[1]
                metadata[key] = value # Sla op in de dictionary
                print(f"Metadata gevonden -> {key}: {value}")
                continue # Ga naar de volgende regel

            # Check voor het eindsignaal
            if line == "END":
                print("\nEindsignaal 'END' ontvangen.")
                break

            # Als het geen metadata of END is, moet het een sample zijn
            if line.isdigit():
                samples.append(int(line))
            else:
                print(f"ESP32 Status: {line}")

        except Exception as e:
            print(f"Fout tijdens het lezen: {e}")
            break
            
finally:
    if is_connected and esp32.is_open:
        esp32.close()
        print("Seriële poort gesloten.")

# --- DATA VERWERKING EN PLOTTEN ---
print("\n--- SAMENVATTING ---")
print(f"Totaal aantal ontvangen samples: {len(samples)}")
print("Ontvangen Metadata:")
for key, value in metadata.items():
    print(f"  - {key}: {value}")

# --- DATA VERWERKING EN PLOTTEN ---
if samples:
    num_samples = len(samples)
    
    # Haal de rate uit de metadata
    actual_sample_rate = float(metadata.get('RATE', SAMPLE_RATE_HZ_EXPECTED))
    print(f"\nGebruik van de sample rate: {actual_sample_rate:.2f} S/s")

    # --- DE FINALE TWEE-PUNTS KALIBRATIE ---
    # Dit zijn de waarden berekend op basis van jouw metingen.
    ADC_GAIN = 0.00081641     # <-- JOUW BEREKENDE GAIN
    ADC_OFFSET = 0.179        # <-- JOUW BEREKENDE OFFSET
    
    samples_np = np.array(samples)
    # Pas de finale, gekalibreerde formule toe
    voltages_np = (samples_np * ADC_GAIN) + ADC_OFFSET
    
    average_voltage = np.mean(voltages_np)
    print(f"Gemiddelde GEKALIBREERDE spanning: {average_voltage:.3f} V")

    time_axis_ms = np.arange(num_samples) / actual_sample_rate * 1000

    # --- PLOT MET HET FINALE VOLTAGE ---
    plt.figure(figsize=(15, 7))
    plt.plot(time_axis_ms, voltages_np)
    plt.title(f"ESP32 ADC Meting - 2-Punts Gekalibreerd\nSample Rate: {actual_sample_rate/1000:.2f} kS/s")
    plt.xlabel("Tijd (ms)")
    plt.ylabel("Spanning (V)")
    plt.grid(True)
    # Zet een limiet op de y-as voor beter zicht
    plt.ylim(min(voltages_np) - 0.1, max(voltages_np) + 0.1) 
    plt.axhline(y=average_voltage, color='r', linestyle='--', label=f'Gemiddeld: {average_voltage:.3f} V')
    plt.legend()
    plt.show()
else:
    print("Geen samples ontvangen.")