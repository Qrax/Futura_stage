# -*- coding: utf-8 -*-
import pyvisa
import numpy as np
import json
import matplotlib.pyplot as plt
import re

# --- CONFIGURATIE ---
SCOPE_ADDRESS = 'USB0::0x5345::0x1234::2306226::RAW'

def get_header_and_data(scope, channel="CH1"):
    """Haalt zowel de header als de data op in één functie."""
    # Header ophalen
    scope.write(':DATa:WAVe:SCReen:HEAD?')
    header_bytes = scope.read_raw()
    header_str = header_bytes.decode('utf-8', errors='ignore')
    match = re.search(r'{', header_str)
    if not match:
        raise ValueError("Geen JSON in header gevonden.")
    header = json.loads(header_str[match.start():])
    
    # Data ophalen
    scope.write(f':DATa:WAVe:SCReen:{channel}?')
    data_bytes = scope.read_raw()
    raw_data = np.frombuffer(data_bytes, dtype='h')
    
    # Data opschonen
    return header, raw_data[10:]

def plot_data(time_axis, volts, ch_info, raw_data):
    """Maakt de plot en print de analyse."""
    plt.style.use('default')
    plt.figure(figsize=(12, 6))
    plt.plot(time_axis * 1e6, volts)
    plt.title(f"Oscilloscoop Data ({ch_info.get('NAME', 'CH1')})")
    plt.xlabel("Tijd (µs)")
    plt.ylabel("Spanning (V)")
    plt.grid(True)
    
    print("\n--- Analyse Berekende Data ---")
    print(f"Berekende Vmax: {np.max(volts):.4f} V (voor ADC-waarde {np.min(raw_data)})")
    print(f"Berekende Vmin: {np.min(volts):.4f} V (voor ADC-waarde {np.max(raw_data)})")
    print("----------------------------\n")
    
    plt.show()

# --- HOOFDPROGRAMMA ---
if __name__ == "__main__":
    scope = None
    try:
        rm = pyvisa.ResourceManager()
        scope = rm.open_resource(SCOPE_ADDRESS)
        scope.timeout = 20000
        print(f"Verbonden met: {scope.query('*IDN?')}")
        
        scope.write(':STOP')
        
        header, raw_data = get_header_and_data(scope, "CH1")
        ch_info = header['CHANNEL'][0]
        
        if ch_info['DISPLAY'] == 'ON':
            # --- JOUW FORMULE, GEIMPLEMENTEERD MET LIVE DATA ---
            
            # 1. Pak de ruwe waarden
            a = -raw_data  # Inversie toepassen
            d = float(ch_info.get('OFFSET', 0))
            v_screen = float(ch_info['SCALE'].lower().replace('mv', 'e-3').replace('v', ''))
            probe = float(ch_info.get('PROBE', '1X').replace('X', ''))
            v_real = v_screen * probe

            print("\n--- Berekening Volgt Jouw Formule ---")
            print(f"Ruwe ADC (geïnverteerd) loopt van {np.min(a)} tot {np.max(a)}")
            print(f"Offset 'd' uit header: {d}")
            print(f"Echte V/Div 'v': {v_real:.4f} V")
            
            # 2. Pas de formule toe
            # y = ((a / 2048) * 5 - (d / 50)) * v
            term1 = (a / 2048.0) * 5.0
            term2 = d / 50.0
            processed_volts = (term1 - term2) * v_real
            
            print(f"Term 1 (positie) loopt van {np.min(term1):.4f} tot {np.max(term1):.4f}")
            print(f"Term 2 (offset): {term2:.4f}")

            # Maak de tijd-as aan
            time_scale = float(header['TIMEBASE']['SCALE'].lower().replace('ms', 'e-3').replace('us', 'e-6').replace('ns', 'e-9').replace('s', ''))
            num_points = len(raw_data)
            total_time = 10 * time_scale
            time_axis = np.linspace(0, total_time, num_points)

            plot_data(time_axis, processed_volts, ch_info, raw_data)

    except Exception as e:
        print(f"\n--- ER IS EEN FOUT OPGETREDEN ---")
        print(e)
    finally:
        if scope:
            scope.close()
            print("\nVerbinding gesloten.")
        print("Programma voltooid.")