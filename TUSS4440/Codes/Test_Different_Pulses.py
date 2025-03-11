import serial
import time
import matplotlib.pyplot as plt

def open_serial(port, baud=115200, timeout=1):
    try:
        ser = serial.Serial(port, baud, timeout=timeout)
        time.sleep(1)  # allow the board to initialize
        return ser
    except Exception as e:
        print("Error opening serial port:", e)
        return None

def send_command(ser, cmd):
    # Append newline and send the command.
    ser.write((cmd + "\n").encode())
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
        try:
            line = ser.readline().decode('utf-8', errors='replace').strip()
        except Exception as e:
            print("Serial read error:", e)
            continue
        if line:
            if line == "E":
                break
            data.append(line)
    return data

def filter_numeric(data):
    numeric_data = []
    for d in data:
        try:
            numeric_data.append(int(d))
        except ValueError:
            # Ignore any non-numeric lines (like status messages)
            pass
    return numeric_data

def plot_data(numeric_data):
    if not numeric_data:
        print("No numeric data available to plot.")
        return
    plt.figure()
    plt.plot(numeric_data, 'b-', lw=1)
    plt.xlabel("Sample")
    plt.ylabel("ADC Value")
    plt.title("Captured ADC Data")
    plt.ylim(0, 256)
    plt.show()

def main():
    port = "COM6"  # Using COM6 by default.
    ser = open_serial(port)
    if ser is None:
        return

    print("Connected to", port)
    print("Available commands:")
    print("  P:<number>  -> Set burst pulse count (e.g., P:10)")
    print("  L:<number>  -> Set record length in ms (e.g., L:150)")
    print("  S           -> Start measurement cycle (pulse + ADC capture)")
    print("Type 'exit' to quit.")

    while True:
        cmd = input("Enter command: ").strip()
        if cmd.lower() == "exit":
            break

        send_command(ser, cmd)

        # For P and L commands, we just print any confirmation response.
        if cmd.lower().startswith("p:") or cmd.lower().startswith("l:"):
            time.sleep(0.3)
            while ser.in_waiting:
                line = ser.readline().decode('utf-8', errors='replace').strip()
                if line:
                    print(line)
        # For the measurement cycle command "S"
        elif cmd.lower() == "s":
            print("Measurement cycle started. Waiting for data...")
            data = read_until_end(ser, timeout=5)
            print("Raw data received:")
            for d in data:
                print(d)
            numeric_data = filter_numeric(data)
            print("Plotting numeric ADC data...")
            plot_data(numeric_data)
        else:
            time.sleep(0.3)
            while ser.in_waiting:
                line = ser.readline().decode('utf-8', errors='replace').strip()
                if line:
                    print(line)

    ser.close()
    print("Serial connection closed.")

if __name__ == "__main__":
    main()
