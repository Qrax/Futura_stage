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

if samples:
    num_samples = len(samples)
    
    # Haal de rate uit de metadata, als deze bestaat
    if 'RATE' in metadata:
        actual_sample_rate = float(metadata['RATE'])
        print(f"\nGebruik van de daadwerkelijk gemeten sample rate: {actual_sample_rate:.2f} S/s")
    else:
        actual_sample_rate = SAMPLE_RATE_HZ_EXPECTED
        print(f"\nWAARSCHUWING: RATE niet gevonden in metadata. Gebruik van de verwachte waarde: {actual_sample_rate} S/s")

    # Voorkom delen door nul als de rate 0 is
    if actual_sample_rate == 0:
        print("Fout: Sample rate is 0, kan geen tijd-as berekenen.")
    else:
        voltages_np = (np.array(samples) / MAX_ADC_VALUE) * MAX_VOLTAGE
        time_axis_ms = np.arange(num_samples) / actual_sample_rate * 1000

        plt.figure(figsize=(15, 7))
        plt.plot(time_axis_ms, voltages_np)
        plt.title(f"ESP32 ADC Meting\nBerekende Sample Rate: {actual_sample_rate/1000:.2f} kS/s")
        plt.xlabel("Tijd (ms)")
        plt.ylabel("Spanning (V)")
        plt.grid(True)
        plt.axhline(y=np.mean(voltages_np), color='r', linestyle='--', label=f'Gemiddeld: {np.mean(voltages_np):.3f} V')
        plt.legend()
        plt.show()
else:
    print("Geen samples ontvangen. Kan geen grafiek maken.")