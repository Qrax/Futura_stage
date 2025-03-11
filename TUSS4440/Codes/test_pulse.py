import serial
import time


def read_until_end(ser, timeout=5):
    """
    Reads lines from the serial port until a line equal to "E" is received,
    or until the timeout is reached.
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


if __name__ == "__main__":
    port = input("Enter COM port (e.g., COM4): ").strip()
    ser = serial.Serial(port, 115200, timeout=1)
    time.sleep(1)  # Allow board to initialize

    mode = input("Enter mode to test (TX or RX): ").strip().upper()
    if mode == "TX":
        ser.write(b"TX\n")
    elif mode == "RX":
        ser.write(b"RX\n")
    else:
        print("Unknown mode. Exiting.")
        ser.close()
        exit(1)

    time.sleep(0.5)  # Allow the board to process the command

    print("Reading board output...")
    data = read_until_end(ser, timeout=5)
    if data:
        print("Board output:")
        for line in data:
            print(line)
        filename = f"{mode}_output.txt"
        with open(filename, "w", encoding="utf-8", errors="replace") as f:
            f.write("\n".join(data))
        print(f"Data saved to {filename}")
    else:
        print("No data received.")

    ser.close()
