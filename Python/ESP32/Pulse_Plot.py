import serial
import numpy as np
import matplotlib.pyplot as plt
import threading
import queue
import time

# --- Configuration (MUST MATCH ESP32 CODE) ---
SERIAL_PORT = 'COM9'  # The COM port of your MASTER ESP32
BAUD_RATE = 2000000
SAMPLE_RATE = 2000000  # 2 MHz sample rate
NUM_SAMPLES = 1024     # The size of the ADC buffer

# --- Global variables for threading ---
data_queue = queue.Queue()
exit_event = threading.Event()

def serial_reader_thread(ser):
    """ This thread reads all serial lines and puts completed data packets into a queue. """
    in_data_packet = False
    current_packet = []
    while not exit_event.is_set():
        try:
            line = ser.readline().decode('utf-8').strip()
            if not line: continue
            if line == "---BEGIN-DATA---":
                in_data_packet = True
                current_packet = []
            elif line == "---END-DATA---":
                if in_data_packet:
                    in_data_packet = False
                    if len(current_packet) == NUM_SAMPLES:
                        data_queue.put(np.array(current_packet))
                    else:
                        print(f"Warning: Incomplete packet (size {len(current_packet)})")
            elif in_data_packet:
                try: current_packet.append(int(line))
                except ValueError: pass
            else:
                print(f"ESP32: {line}")
        except Exception:
            exit_event.set()

def user_input_thread(ser):
    """ This thread waits for the user to press Enter to trigger a new capture. """
    while not exit_event.is_set():
        command = input()
        if command.lower() == 'exit':
            exit_event.set()
            break
        try:
            print(">>> Triggering ESP32...")
            ser.write(b'\n')
        except serial.SerialException as e:
            print(f"Error writing to serial port: {e}")
            exit_event.set()

def update_plot(ax1, ax2, data):
    """
    Clears the axes and plots the new data and its FFT.
    This function does NOT create a new window.
    """
    # 1. Clear previous data
    ax1.clear()
    ax2.clear()

    # 2. Process new data
    time_axis = np.arange(NUM_SAMPLES) / SAMPLE_RATE * 1e6 # in microseconds (us)
    fft_vals = np.fft.rfft(data - np.mean(data))
    fft_freq = np.fft.rfftfreq(NUM_SAMPLES, 1.0 / SAMPLE_RATE)
    fft_mag = np.abs(fft_vals)

    # 3. Plot new data
    # Plot 1: Time Domain (Waveform)
    ax1.plot(time_axis, data, color='b')
    ax1.set_title('Received Waveform')
    ax1.set_xlabel('Time (μs)')
    ax1.set_ylabel('ADC Value (0-4095)')
    ax1.grid(True)
    ax1.set_ylim(1000, 2800) # Keep y-axis consistent

    # Plot 2: Frequency Domain (FFT)
    ax2.plot(fft_freq / 1000, fft_mag, color='r') # Freq in kHz
    ax2.set_title('Frequency Spectrum (FFT)')
    ax2.set_xlabel('Frequency (kHz)')
    ax2.set_ylabel('Magnitude')
    ax2.grid(True)
    ax2.set_xlim(0, 100) # Zoom in on the 0-100 kHz range

    # Find and annotate the peak frequency
    peak_freq_index = np.argmax(fft_mag[1:]) + 1 # Ignore DC component
    peak_freq = fft_freq[peak_freq_index]
    peak_mag = fft_mag[peak_freq_index]
    ax2.annotate(f'Peak: {peak_freq/1000:.2f} kHz',
                 xy=(peak_freq/1000, peak_mag),
                 xytext=(peak_freq/1000 + 5, peak_mag * 0.9),
                 arrowprops=dict(facecolor='black', shrink=0.05))

def main():
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)
        ser.flushInput()
        print(f"Connected to {SERIAL_PORT}. Ready.")
    except serial.SerialException as e:
        print(f"Error: Could not open serial port {SERIAL_PORT}. {e}")
        return

    # --- Setup for Live Plotting ---
    plt.ion() # Turn on interactive mode
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle('Live Ultrasonic Pulse Analysis', fontsize=16)

    # Function to handle closing the plot window
    def on_close(event):
        print("Plot window closed. Exiting...")
        exit_event.set()
    fig.canvas.mpl_connect('close_event', on_close)

    # Start the background threads
    reader = threading.Thread(target=serial_reader_thread, args=(ser,))
    input_handler = threading.Thread(target=user_input_thread, args=(ser,))
    reader.daemon = True
    input_handler.daemon = True
    reader.start()
    input_handler.start()

    print("\n--- Ultrasonic Capture Control ---")
    print("Press [Enter] in this window to trigger a new measurement.")
    print("Type 'exit' and press [Enter] to quit.")
    print("Or simply close the plot window.")
    print("------------------------------------\n")

    try:
        while not exit_event.is_set():
            try:
                # Wait for a complete data packet to appear in the queue
                packet = data_queue.get(timeout=0.1)
                print(f"<<< Data packet received. Updating plot... >>>")
                update_plot(ax1, ax2, packet)
                plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout
            except queue.Empty:
                # This is normal, just waiting for data
                pass
            
            # Allow the plot to update and process events
            plt.pause(0.05)

    finally:
        print("Shutting down...")
        exit_event.set()
        plt.ioff()
        plt.close()
        # Wait for threads to finish
        reader.join(timeout=1)
        input_handler.join(timeout=1)
        ser.close()

if __name__ == '__main__':
    main()