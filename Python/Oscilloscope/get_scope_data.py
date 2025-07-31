# -*- coding: utf-8 -*-
import pyvisa
import numpy as np
import json
import matplotlib.pyplot as plt
import re

# --- CONFIGURATIE ---
SCOPE_ADDRESS = 'USB0::0x5345::0x1234::2306226::RAW'
CHANNELS_TO_READ = ["CH1"] # Pas aan naar ["CH1", "CH2"] voor beide

def get_scope_header(scope):
    """Vraagt de master header op met de instellingen van de hele scoop."""
    scope.write(':DATa:WAVe:SCReen:HEAD?')
    header_bytes = scope.read_raw()
    header_str = header_bytes.decode('utf-8', errors='ignore')
    
    match = re.search(r'{', header_str)
    if match:
        clean_header_str = header_str[match.start():]
        return json.loads(clean_header_str)
    else:
        raise ValueError("Geen geldige JSON-start '{' gevonden in header.")

def get_channel_data(scope, channel):
    """Vraagt de ruwe data voor een specifiek kanaal op."""
    scope.write(f':DATa:WAVe:SCReen:{channel}?')
    data_bytes = scope.read_raw()
    # We lezen nog steeds 16-bit, maar we interpreteren het als 12-bit data.
    raw_data = np.frombuffer(data_bytes, dtype='h')
    return raw_data

def process_data(header, raw_data, ch_info):
    """Verwerkt de ruwe data naar Volts en Seconden op basis van de header."""
    slice_offset = 10
    clean_raw_data = raw_data[slice_offset:]
    
    time_info = header['TIMEBASE']
    time_scale_str = time_info['SCALE']
    time_scale = float(time_scale_str.lower().replace('ms', 'e-3').replace('us', 'e-6').replace('ns', 'e-9').replace('s', ''))
    
    volt_scale_str = ch_info['SCALE']
    volt_scale_on_screen = float(volt_scale_str.lower().replace('mv', 'e-3').replace('v', ''))
    
    # --- DE 12-BIT ADC CORRECTIE ---
    # Een 12-bit ADC heeft 2^12 = 4096 stappen. Het bereik is -2048 tot 2047.
    # De `raw_data` wordt hierdoor gedeeld.
    ADC_MAX_VALUE = 2048.0 
    
    # De scherm-layout heeft 8 divisies, dus +/- 4 vanuit het midden.
    NUM_DIVISIONS_VERTICAL = 8.0

    # De `volt_scale_on_screen` is de waarde die de gebruiker ziet, gecorrigeerd voor de probe.
    # De `raw_data` is de pure meting. De formule wordt:
    # V = (ADC_waarde / Max_ADC_waarde) * (Aantal_Divisies / 2) * Scherm_V_per_Div
    processed_volts = (clean_raw_data / ADC_MAX_VALUE) * (NUM_DIVISIONS_VERTICAL / 2.0) * volt_scale_on_screen
    # ------------------------------------

    # Maak de tijd-as aan
    num_points = len(clean_raw_data)
    total_time = 10 * time_scale
    time_axis_seconds = np.linspace(0, total_time, num_points)
    
    return time_axis_seconds, processed_volts

def print_scope_settings(header, all_plot_data):
    # ... (Deze functie blijft hetzelfde)
    print("\n" + "="*40)
    print("Gedetecteerde Scoop Instellingen")
    print("="*40)
    time_info = header['TIMEBASE']
    print(f"  Horizontaal:")
    print(f"    Tijd/Div: {time_info['SCALE']}")
    sample_info = header['SAMPLE']
    print(f"  Acquisitie:")
    print(f"    Sample Rate: {sample_info['SAMPLERATE']}")
    print(f"    Memory Depth: {sample_info['DEPMEM']}")
    for channel_name, data in all_plot_data.items():
        ch_info = data[2]
        print(f"  Kanaal {channel_name}:")
        print(f"    Status: {ch_info['DISPLAY']}")
        print(f"    Volt/Div: {ch_info['SCALE']}")
        print(f"    Probe: {ch_info['PROBE']}")
        print(f"    Koppeling: {ch_info['COUPLING']}")
    print("="*40 + "\n")


def plot_all_data(all_plot_data):
    # ... (Deze functie blijft hetzelfde)
    print("Bezig met plotten...")
    plt.style.use('default')
    plt.figure(figsize=(14, 7))
    colors = ['#1f77b4', '#ff7f0e']
    for i, (channel_name, data) in enumerate(all_plot_data.items()):
        time_axis, volts, ch_info = data
        plt.plot(time_axis * 1e6, volts, label=channel_name, color=colors[i % len(colors)])
    plt.title("Oscilloscoop Data")
    plt.xlabel("Tijd (µs)")
    plt.ylabel("Spanning (V)")
    plt.grid(True)
    plt.legend()
    plt.show()

# --- HOOFDPROGRAMMA ---
if __name__ == "__main__":
    scope = None
    all_plot_data = {}
    try:
        rm = pyvisa.ResourceManager()
        scope = rm.open_resource(SCOPE_ADDRESS)
        scope.timeout = 20000
        print(f"Verbonden met: {scope.query('*IDN?')}")
        
        scope.write(':STOP')
        header = get_scope_header(scope)
        
        for i, channel in enumerate(CHANNELS_TO_READ):
            ch_info = header['CHANNEL'][i]
            if ch_info['DISPLAY'] == 'ON':
                print(f"\nKanaal {channel} staat AAN. Data wordt opgehaald.")
                raw_data = get_channel_data(scope, channel)
                time_axis, volts = process_data(header, raw_data, ch_info)
                all_plot_data[channel] = (time_axis, volts, ch_info)
            else:
                print(f"\nKanaal {channel} staat UIT. Wordt overgeslagen.")

        if not all_plot_data:
            print("\nGeen actieve kanalen gevonden om te plotten. Programma stopt.")
        else:
            print_scope_settings(header, all_plot_data)
            plot_all_data(all_plot_data)

    except Exception as e:
        print(f"\n--- ER IS EEN FOUT OPGETREDEN ---")
        print(e)
    finally:
        if scope:
            scope.close()
            print("\nVerbinding gesloten.")
        print("Programma voltooid.")