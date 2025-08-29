import serial
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import threading
import queue
import time
import sys

# --- Configuratie ---
SERIAL_PORT = 'COM9'  # <-- VERANDER DIT!
BAUD_RATE = 2000000

data_queue = queue.Queue()

def read_serial_data(ser, data_queue, stop_event):
    """Thread 1: Leest data van de ESP32."""
    print("Serial reader thread gestart.")
    in_data_block = False
    data_lines = []
    while not stop_event.is_set():
        try:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if line == "<DATA_START>":
                    in_data_block = True
                    data_lines = []
                elif line == "<DATA_END>":
                    in_data_block = False
                    full_data_str = "".join(data_lines)
                    values = [int(v) for v in full_data_str.split(',') if v]
                    data_queue.put(values)
                elif in_data_block:
                    data_lines.append(line)
                elif line:
                    # Print alle statusberichten direct in de terminal
                    print(f"ESP32 >> {line}")
        except Exception as e:
            print(f"Fout in serial reader thread: {e}")
            break
    print("Serial reader thread gestopt.")

def send_commands(ser, stop_event):
    """Thread 3: Leest jouw input in de terminal en stuurt het naar de ESP32."""
    print("Command sender thread gestart. Typ hier en druk op Enter om te zenden.")
    while not stop_event.is_set():
        try:
            # Wacht tot de gebruiker een regel invoert (en op Enter drukt)
            command = sys.stdin.readline()
            # Stuur het commando (inclusief de newline van Enter) naar de ESP32
            ser.write(command.encode('utf-8'))
        except Exception as e:
            print(f"Fout in command sender thread: {e}")
            break
    print("Command sender thread gestopt.")

def update_plot(frame, ax):
    """Wordt aangeroepen door de plotter om de grafiek te verversen."""
    try:
        values = data_queue.get_nowait()
        if values:
            print(f"Plotting {len(values)} nieuwe datapunten...")
            ax.clear()
            ax.plot(values)
            ax.set_title("Live Time-Reversal Meting")
            ax.set_xlabel("Sample Nummer")
            ax.set_ylabel("Spanning (mV)")
            ax.grid(True)
            ax.set_ylim(0, 3300)
    except queue.Empty:
        pass # Geen nieuwe data, doe niets.

def main():
    """Hoofdfunctie: start alles."""
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print(f"Verbonden met {SERIAL_PORT}.")
        time.sleep(2)
    except serial.SerialException as e:
        print(f"FATALE FOUT: Kon poort niet openen. {e}")
        return

    stop_event = threading.Event()

    # Start de twee achtergrond-threads
    serial_reader = threading.Thread(target=read_serial_data, args=(ser, data_queue, stop_event))
    serial_writer = threading.Thread(target=send_commands, args=(ser, stop_event))
    serial_reader.daemon = True
    serial_writer.daemon = True
    serial_reader.start()
    serial_writer.start()

    # Start de plotter in de hoofd-thread
    fig, ax = plt.subplots(figsize=(12, 6))
    ani = animation.FuncAnimation(fig, update_plot, fargs=(ax,), interval=200)
    
    print("\n--- Plot venster is nu actief ---")
    print("--- Focus op de TERMINAL om commando's (zoals Enter) te sturen ---")
    plt.show()

    # Dit deel wordt pas uitgevoerd als je de plot sluit
    print("Plot venster gesloten. Programma wordt afgesloten.")
    stop_event.set()
    serial_reader.join()
    serial_writer.join()
    ser.close()

if __name__ == '__main__':
    main()