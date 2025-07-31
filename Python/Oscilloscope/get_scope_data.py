# -*- coding: utf-8 -*-
import pyvisa
import numpy as np
import json
import matplotlib.pyplot as plt
import re

# --- 1. CONFIGURATIE EN VERBINDING ---
# Het door jou opgegeven, correcte VISA-adres
SCOPE_ADDRESS = 'USB0::0x5345::0x1234::2306226::RAW'

try:
    rm = pyvisa.ResourceManager()
    scope = rm.open_resource(SCOPE_ADDRESS)
    scope.timeout = 20000
    print(f"Verbonden met: {scope.query('*IDN?')}")
except Exception as e:
    print(f"Fout bij het verbinden. Foutmelding: {e}")
    exit()

# --- 2. DATA OPHALEN ---
print("\nData ophalen...")
scope.write(':STOP')

try:
    scope.write(':DATa:WAVe:SCReen:HEAD?')
    header_bytes = scope.read_raw()
    header_str = header_bytes.decode('utf-8', errors='ignore')
    
    match = re.search(r'{', header_str)
    if match:
        clean_header_str = header_str[match.start():]
        header = json.loads(clean_header_str)
        print("Header succesvol verwerkt.")
    else:
        raise ValueError("Geen geldige JSON-start '{' gevonden in header.")

    scope.write(':DATa:WAVe:SCReen:CH1?')
    data_bytes = scope.read_raw()
    raw_data = np.frombuffer(data_bytes, dtype='h')
    print(f"Succesvol {len(raw_data)} datapunten overgezet.")

except Exception as e:
    print(f"\nFOUT tijdens data-acquisitie: {e}")
    scope.close()
    exit()

scope.close()
print("Verbinding gesloten.")


# --- 3. DATA OPSCHONEN EN VERWERKEN ---
print("\nData opschonen en verwerken...")
try:
    # Verwijder de eerste paar corrupte datapunten (de 'hoge piek')
    slice_offset = 10
    clean_raw_data = raw_data[slice_offset:]
    
    # Haal de waarden uit de header
    time_info = header['TIMEBASE']
    time_scale_str = time_info['SCALE']
    time_scale = float(time_scale_str.lower().replace('ms', 'e-3').replace('us', 'e-6').replace('ns', 'e-9').replace('s', ''))
    
    ch1_info = header['CHANNEL'][0]
    volt_scale_str = ch1_info['SCALE']
    volt_scale_on_screen = float(volt_scale_str.lower().replace('mv', 'e-3').replace('v', ''))
    
    probe_attenuation = float(ch1_info.get('PROBE', '1X').replace('X', ''))

    # De correcte conversieformule
    divs_on_screen = (clean_raw_data / 32768.0) * 4.0
    voltage_on_screen = divs_on_screen * volt_scale_on_screen
    processed_volts = voltage_on_screen * probe_attenuation

    # Maak de tijd-as aan
    num_points = len(clean_raw_data)
    total_time = 10 * time_scale # 10 divisies horizontaal
    time_axis_seconds = np.linspace(0, total_time, num_points)
    
    print("Verwerking voltooid.")
    print(f"Maximale berekende spanning: {np.max(processed_volts):.4f} V")

except Exception as e:
    print(f"FOUT bij het verwerken van de data: {e}")
    exit()

# --- 4. DATA PLOTTEN ---
print("Bezig met plotten...")

# AANPASSING: Gebruik 'dark_background' stijl
plt.style.use('dark_background')

plt.figure(figsize=(12, 6))

# AANPASSING: Converteer tijd naar microseconden en zet lijnkleur op geel
plt.plot(time_axis_seconds * 1e6, processed_volts, color='yellow')

plt.title(f"Oscilloscoop Data ({ch1_info.get('NAME', 'CH1')})")

# AANPASSING: Label van de x-as
plt.xlabel("Tijd (µs)")
plt.ylabel("Spanning (V)")
plt.grid(True)
plt.show()

print("\nProgramma voltooid.")