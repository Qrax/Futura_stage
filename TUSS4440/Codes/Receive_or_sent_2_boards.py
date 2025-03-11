import serial
import time
import os
from datetime import datetime


def flush_input(ser):
    ser.reset_input_buffer()


def send_command(ser, cmd):
    ser.write(cmd.encode() + b'\n')
    time.sleep(0.2)


def read_until_end(ser, timeout=5):
    """
    Reads lines from the serial port until a line exactly equal to "E" is received,
    or until the timeout (in seconds) is reached.
    Returns a list of lines (excluding the "E" marker).
    """
    data = []
    start_time = time.time()
    while True:
        if time.time() - start_time > timeout:
            print("Timeout reached; 'E' marker not found.")
            break
        line = ser.readline().decode('utf-8', errors='replace').strip()
        if line:
            if line == "E":
                break
            data.append(line)
    return data


def get_output_filenames(mode):
    """
    Create output filenames using the current date/time and mode.
    Files will be saved in a folder called "Data" one directory up from the current folder.
    """
    parent_dir = os.path.dirname(os.getcwd())
    output_folder = os.path.join(parent_dir, "Data")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tx_filename = os.path.join(output_folder, f"{timestamp}_TX_{mode}.txt")
    rx_filename = os.path.join(output_folder, f"{timestamp}_RX_{mode}.txt")
    return tx_filename, rx_filename


if __name__ == "__main__":
    # Automatically use COM4 for transmitter and COM6 for receiver.
    transmitter_port = 'COM4'
    receiver_port = 'COM6'

    print(f"Opening transmitter on {transmitter_port} and receiver on {receiver_port}...")
    ser_tx = serial.Serial(transmitter_port, 115200, timeout=1)
    ser_rx = serial.Serial(receiver_port, 115200, timeout=1)
    time.sleep(1)  # Allow boards to initialize

    # Step 1: Start the receiver.
    print("Sending RX command to receiver board...")
    flush_input(ser_rx)
    send_command(ser_rx, "RX")
    print("Receiver board (COM6) is capturing ADC data for 3 seconds...")

    # Give the receiver some baseline time.
    time.sleep(1)

    # Step 2: Send the pulse command to the transmitter.
    print("Sending TX command to transmitter board...")
    flush_input(ser_tx)
    send_command(ser_tx, "TX")
    print("Transmitter board (COM4) sent the pulse.")

    # Wait until the receiver's capture cycle should be complete.
    time.sleep(1)

    # Step 3: Read data from the receiver.
    print("Reading data from receiver board (COM6)...")
    rx_output = read_until_end(ser_rx, timeout=5)
    print("Receiver board output:")
    for line in rx_output:
        print(line)

    # Optionally, read transmitter output (if desired).
    print("\nReading data from transmitter board (COM4)...")
    tx_output = read_until_end(ser_tx, timeout=3)
    print("Transmitter board output:")
    for line in tx_output:
        print(line)

    # Create filenames based on timestamp and mode.
    tx_filename, rx_filename = get_output_filenames("forward")

    with open(tx_filename, "w", encoding="utf-8", errors="replace") as f_tx:
        f_tx.write("\n".join(tx_output))
    with open(rx_filename, "w", encoding="utf-8", errors="replace") as f_rx:
        f_rx.write("\n".join(rx_output))

    print(f"\nData saved to:\n  Transmitter: {tx_filename}\n  Receiver: {rx_filename}")

    ser_tx.close()
    ser_rx.close()
