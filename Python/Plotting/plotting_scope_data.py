# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from scipy.fft import fft, fftfreq

# --- CONFIGURATIE ---
DATA_FOLDER = os.path.join('.', 'Oscilloscope','Data', 'Scope_Data')
FILENAME = '20000104_002326.csv'
FFT_RESOLUTION_BOOST_FACTOR = 10 # 10x meer punten in de FFT plot voor een gladder resultaat

def load_and_prepare_data(filepath):
    """Leest en bereidt de data voor analyse."""
    HEADER_ROW_INDEX = 10
    print(f"\n--- Data inlezen (skiprows={HEADER_ROW_INDEX}) ---")
    df = pd.read_csv(filepath, skiprows=HEADER_ROW_INDEX)
    
    print("\n--- Data voorbereiden en DC-component filteren ---")
    voltage_mV = df.iloc[:, 1].to_numpy()
    voltage_V_original = voltage_mV / 1000.0
    dc_component = np.mean(voltage_V_original)
    voltage_V_filtered = voltage_V_original - dc_component
    print(f"DC-component van {dc_component:.4f} V verwijderd.")
    
    # Metadata hardcoded voor nu, kan later uit parse_metadata gehaald worden
    time_interval_s = 1e-9 # Gebaseerd op 0.00100uS -> 1ns
    sample_rate = 1 / time_interval_s
    num_points = len(voltage_V_filtered)
    time_axis_s = np.arange(num_points) * time_interval_s
    print(f"Sample Rate: {sample_rate / 1e6:.2f} MS/s, Aantal Punten: {num_points}")
    
    return time_axis_s, voltage_V_filtered, num_points, time_interval_s

def plot_time_domain(time_axis, voltage):
    """Plot de data in het tijddomein."""
    print("\n--- Plot Tijddomein wordt gegenereerd ---")
    plt.figure(figsize=(14, 7))
    
    # Downsample voor snellere plot: teken maar max 100,000 punten
    step = max(1, len(voltage) // 100000)
    plt.plot(time_axis[::step] * 1e6, voltage[::step])
    
    plt.title("Tijddomein (DC-component verwijderd)")
    plt.xlabel("Tijd (µs)")
    plt.ylabel("Spanning (V)")
    plt.grid(True)
    plt.show()

def plot_fft_domain(voltage, num_points, time_interval):
    """Berekent en plot de FFT."""
    print("\n--- FFT wordt berekend ---")
    
    # --- ZERO PADDING voor hogere resolutie in de plot ---
    # We voegen nullen toe om de FFT langer te maken
    n_fft = num_points * FFT_RESOLUTION_BOOST_FACTOR
    
    yf = fft(voltage, n=n_fft)
    xf = fftfreq(n_fft, time_interval)
    
    # Bereken de resolutie
    resolution = (1/time_interval) / n_fft
    print(f"FFT Berekend. Plot resolutie is nu ~{resolution:.2f} Hz.")
    
    print("\n--- Plot Frequentiedomein wordt gegenereerd ---")
    plt.figure(figsize=(14, 7))
    
    N_half = n_fft // 2
    # Zoek de index voor 100 kHz
    max_freq_index = np.where(xf >= 100000)[0][0]
    
    plt.plot(xf[0:max_freq_index] / 1000, 2.0/num_points * np.abs(yf[0:max_freq_index]))
    plt.title("Frequentiedomein (FFT)")
    plt.xlabel("Frequentie (kHz)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    
    # Vind de piek in het 0-100kHz bereik
    peak_idx = np.argmax(np.abs(yf[1:max_freq_index])) + 1
    peak_freq = xf[peak_idx] / 1000
    print(f"Piekfrequentie gedetecteerd: {peak_freq:.3f} kHz")
    
    plt.show()


# --- HOOFDPROGRAMMA ---
if __name__ == "__main__":
    full_path = os.path.join(DATA_FOLDER, FILENAME)
    
    if not os.path.exists(full_path):
        print(f"FOUT: Bestand '{FILENAME}' niet gevonden.")
        exit()

    # Laad de data één keer
    time_axis, voltage, num_points, time_interval = load_and_prepare_data(full_path)

    # Vraag de gebruiker wat hij wil zien
    while True:
        choice = input("\nWelke plot wil je zien? (1=Tijd, 2=FFT, q=Stoppen): ").strip()
        if choice == '1':
            plot_time_domain(time_axis, voltage)
        elif choice == '2':
            plot_fft_domain(voltage, num_points, time_interval)
        elif choice.lower() == 'q':
            break
        else:
            print("Ongeldige keuze, probeer opnieuw.")

    print("\nProgramma voltooid.")