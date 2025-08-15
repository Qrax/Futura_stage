import pyvisa
import numpy as np
import json
import matplotlib.pyplot as plt
import re
from scipy.signal import hilbert

# --- CONFIGURATIE ---
SCOPE_ADDRESS = 'USB0::0x5345::0x1234::2306226::RAW'
CHANNELS_TO_READ = ["CH1", "CH2"]
DIFF_CHANNELS = ["CH1", "CH2"] 

def get_all_channel_data(scope, channels):
    """Haalt de header en de data voor een lijst van kanalen op."""
    scope.write(':DATa:WAVe:SCReen:HEAD?')
    header_bytes = scope.read_raw()
    header_str = header_bytes.decode('utf-8', errors='ignore')
    match = re.search(r'{', header_str)
    if not match:
        raise ValueError("Geen JSON in header gevonden.")
    header = json.loads(header_str[match.start():])
    
    all_data = {'header': header, 'channel_data': {}}
    
    for channel in channels:
        scope.write(f':DATa:WAVe:SCReen:{channel}?')
        data_bytes = scope.read_raw()
        raw_data = np.frombuffer(data_bytes, dtype='h')
        all_data['channel_data'][channel] = raw_data[10:]
        
    return all_data

def plot_multi_channel_data(plot_results):
    """
    Maakt één plot met data van meerdere kanalen. Gebruikt twee Y-assen
    als er twee kanalen met verschillende schalen worden geplot.
    """
    plt.style.use('default')
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    print("\n--- Analyse Berekende Data ---")
    
    if not plot_results:
        print("Geen data om te plotten.")
        return
        
    res1 = plot_results[0]
    color1 = 'tab:blue'
    ax1.set_xlabel("Tijd (µs)")
    ax1.set_ylabel(f"Spanning (V) - {res1['name']}", color=color1)
    line1, = ax1.plot(res1['time_axis'] * 1e6, res1['volts'], color=color1, label=res1['name'])
    ax1.tick_params(axis='y', labelcolor=color1)
    
    print(f"\n--- Kanaal: {res1['name']} ---")
    print(f"Berekende Vmax: {np.max(res1['volts']):.4f} V")
    print(f"Berekende Vmin: {np.min(res1['volts']):.4f} V")

    lines = [line1]

    if len(plot_results) > 1:
        res2 = plot_results[1]
        color2 = 'tab:red'
        
        ax2 = ax1.twinx() 
        ax2.set_ylabel(f"Spanning (V) - {res2['name']}", color=color2)
        line2, = ax2.plot(res2['time_axis'] * 1e6, res2['volts'], color=color2, label=res2['name'])
        ax2.tick_params(axis='y', labelcolor=color2)
        
        print(f"\n--- Kanaal: {res2['name']} ---")
        print(f"Berekende Vmax: {np.max(res2['volts']):.4f} V")
        print(f"Berekende Vmin: {np.min(res2['volts']):.4f} V")
        lines.append(line2)

    fig.tight_layout()
    plt.title("Oscilloscoop Data Individuele Kanalen (Dubbele Y-as)")
    ax1.grid(True)
    plt.legend(lines, [l.get_label() for l in lines])
    print("\n----------------------------------\n")

def plot_differential_inputs(plot_results):
    """
    NIEUW: Maakt een plot van de twee kanalen op een ENKELE y-as om de 
    input voor de differentiaalberekening te visualiseren.
    """
    plt.figure(figsize=(12, 6))
    
    ch1_data = next((res for res in plot_results if res['name'] == DIFF_CHANNELS[0]), None)
    ch2_data = next((res for res in plot_results if res['name'] == DIFF_CHANNELS[1]), None)

    if ch1_data and ch2_data:
        # Plot beide op dezelfde as
        plt.plot(ch1_data['time_axis'] * 1e6, ch1_data['volts'], label=ch1_data['name'], color='tab:blue')
        plt.plot(ch2_data['time_axis'] * 1e6, ch2_data['volts'], label=ch2_data['name'], color='tab:red')

    plt.title("Input voor Differentieel Signaal (Op één Y-as)")
    plt.xlabel("Tijd (µs)")
    plt.ylabel("Spanning (V)")
    plt.grid(True)
    plt.legend()
    print("\n--- Visualisatie van de input voor de differentiële berekening ---")
    print("Beide kanalen worden op één as getoond om de aftreksom te verduidelijken.")
    print("-----------------------------------------------------------------\n")


def analyze_and_plot_difference(time_axis, diff_volts, name):
    """
    Maakt een plot van het verschil-signaal en berekent de enveloppe met de
    Hilbert-transformatie voor een veel soepeler resultaat.
    """
    analytic_signal = hilbert(diff_volts)
    top_envelope = np.abs(analytic_signal)
    bottom_envelope = -top_envelope
    
    vpp_over_time = top_envelope - bottom_envelope
    average_vpp = np.mean(vpp_over_time)

    print(f"--- Analyse Verschil Signaal (met Hilbert): {name} ---")
    print(f"Gemiddelde Vpp (gebaseerd op enveloppe): {average_vpp:.4f} V")
    print("------------------------------------------------------\n")

    plt.figure(figsize=(12, 6))
    plt.plot(time_axis * 1e6, diff_volts, label=name, color='purple', alpha=0.7)
    plt.plot(time_axis * 1e6, top_envelope, label='Bovenste Enveloppe', color='orange', linestyle='--')
    plt.plot(time_axis * 1e6, bottom_envelope, label='Onderste Enveloppe', color='dodgerblue', linestyle='--')
    plt.title(f"Verschil Signaal met Enveloppe: {name}")
    plt.xlabel("Tijd (µs)")
    plt.ylabel("Spanning (V)")
    plt.grid(True)
    plt.legend()
    
    plt.figure(figsize=(12, 6))
    plt.plot(time_axis * 1e6, vpp_over_time, label='Vpp over tijd', color='green')
    plt.axhline(y=average_vpp, color='red', linestyle='--', label=f'Gemiddelde Vpp ({average_vpp:.4f} V)')
    plt.title(f"Piek-Piek Spanning (Vpp) over Tijd")
    plt.xlabel("Tijd (µs)")
    plt.ylabel("Piek-Piek Spanning (V)")
    plt.ylim(bottom=0)
    plt.grid(True)
    plt.legend()

# --- HOOFDPROGRAMMA ---
if __name__ == "__main__":
    scope = None
    try:
        rm = pyvisa.ResourceManager()
        scope = rm.open_resource(SCOPE_ADDRESS)
        scope.timeout = 20000
        print(f"Verbonden met: {scope.query('*IDN?')}")
        
        scope.write(':STOP')
        
        all_data = get_all_channel_data(scope, CHANNELS_TO_READ)
        header = all_data['header']
        
        plot_results = []

        time_scale = float(header['TIMEBASE']['SCALE'].lower().replace('ms', 'e-3').replace('us', 'e-6').replace('ns', 'e-9').replace('s', ''))
        num_points = len(next(iter(all_data['channel_data'].values())))
        total_time = 10 * time_scale
        time_axis = np.linspace(0, total_time, num_points)

        for channel_name in CHANNELS_TO_READ:
            ch_info = next((ch for ch in header['CHANNEL'] if ch['NAME'] == channel_name), None)
            
            if ch_info and ch_info['DISPLAY'] == 'ON':
                raw_data = all_data['channel_data'][channel_name]
                a = raw_data
                d = float(ch_info.get('OFFSET', 0))
                v_screen = float(ch_info['SCALE'].lower().replace('mv', 'e-3').replace('v', ''))
                probe = float(ch_info.get('PROBE', '1X').replace('X', ''))
                v_real = v_screen * probe
                
                term1 = (a / 2048.0) * 5.0
                term2 = d / 50.0
                processed_volts = (term1 - term2) * v_real
                
                plot_results.append({
                    'name': channel_name,
                    'time_axis': time_axis,
                    'volts': processed_volts,
                    'raw_data': raw_data
                })
            else:
                print(f"\nKanaal {channel_name} staat uit of is niet gevonden in de header. Wordt overgeslagen.")

        if plot_results:
            # PLOT 1: Individuele kanalen met dubbele Y-as
            plot_multi_channel_data(plot_results)

            if len(DIFF_CHANNELS) == 2 and all(ch in CHANNELS_TO_READ for ch in DIFF_CHANNELS):
                ch1_name, ch2_name = DIFF_CHANNELS[0], DIFF_CHANNELS[1]
                
                ch1_data = next((res for res in plot_results if res['name'] == ch1_name), None)
                # BUGFIX: Correcte syntax voor het zoeken naar het tweede kanaal
                ch2_data = next((res for res in plot_results if res['name'] == ch2_name), None)

                if ch1_data and ch2_data:
                    # PLOT 2 (NIEUW): De inputs voor de differentiaalberekening
                    plot_differential_inputs(plot_results)

                    # Bereken en plot de rest
                    diff_volts = ch1_data['volts'] - ch2_data['volts']
                    diff_name = f"{ch1_name} - {ch2_name}"
                    
                    # PLOT 3 en 4: Het resultaat en de Vpp analyse
                    analyze_and_plot_difference(time_axis, diff_volts, diff_name)
                else:
                    print("Kon niet de data voor beide verschil-kanalen vinden.")
            
            # Toon alle gemaakte plots tegelijk
            plt.show()
            
        else:
            print("Geen actieve kanalen gevonden om te plotten.")

    except Exception as e:
        print(f"\n--- ER IS EEN FOUT OPGETREDEN ---")
        print(e)
    finally:
        if scope:
            scope.close()
            print("\nVerbinding gesloten.")
        print("Programma voltooid.")