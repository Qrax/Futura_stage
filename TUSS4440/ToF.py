import serial
import time

SERIAL_PORT = 'COM6'  # Change as needed for your system
BAUD_RATE = 115200
TIMEOUT = 1  # Read timeout in seconds


def main():
    with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=TIMEOUT) as ser:
        time.sleep(2)  # Allow time for the board to reset
        ser.reset_input_buffer()
        print("Connected to board on", SERIAL_PORT)

        while True:
            # Get a command from the user.
            cmd = input("Enter command (or type 'exit' to quit): ")
            if cmd.lower() == 'exit':
                break

            # Send the command with a newline
            ser.write((cmd + "\n").encode('utf-8'))
            print(f"Sent command: {cmd}")

            # Wait briefly to let the board process the command and send back data.
            time.sleep(1)

            # Read and print all available output.
            while ser.in_waiting:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if line:
                    print(line)


if __name__ == '__main__':
    main()
