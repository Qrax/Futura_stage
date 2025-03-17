import serial
import threading
import time
import sys
import re
import matplotlib.pyplot as plt

running = True

# Global lists to store ADC samples for each board
data_A = []
data_B = []

# Regular expression to parse board data in the expected format:
# Example board data: "196643739 us, ADC: 34"
data_pattern = re.compile(r'^(\d+)\s*us,\s*ADC:\s*(\d+)$')

# Open serial ports for Board A and Board B (adjust COM port names as needed)
try:
    serA = serial.Serial('COM4', 115200, timeout=1)
    serB = serial.Serial('COM6', 115200, timeout=1)
except Exception as e:
    print("Error opening serial ports:", e)
    sys.exit(1)

def read_serial(ser, board_name):
    global running, data_A, data_B
    while running:
        try:
            line = ser.readline()
            if not line:
                continue
            # Get the host time (in microseconds) when this line is received.
            host_time = time.time() * 1e6
            line_str = line.decode('utf-8', errors='ignore').strip()
            # Print the received line with the host timestamp and board identifier
            print(f"[{host_time:.0f} us] Board {board_name}: {line_str}")
            # Try to parse the board data if it matches the expected format:
            # "<board time> us, ADC: <adc_value>"
            match = data_pattern.match(line_str)
            if match:
                # Extract the ADC value (as integer) from the message.
                adc_value = int(match.group(2))
                if board_name.upper() == "A":
                    data_A.append(adc_value)
                elif board_name.upper() == "B":
                    data_B.append(adc_value)
        except Exception as e:
            print(f"Error reading from Board {board_name}: {e}")
            time.sleep(0.1)
    ser.close()
    print(f"Exiting read thread for Board {board_name}")

def user_input_thread():
    global running
    print("Enter command:")
    print("  A = Set Board A as MASTER, Board B as SLAVE")
    print("  B = Set Board A as SLAVE, Board B as MASTER")
    print("  SYNC = Send SYNC command to both boards")
    print("  P = Print current sync pin status 30 times on both boards")
    print("  plot = Plot accumulated ADC data from both boards")
    print("  exit = Quit")
    while running:
        try:
            cmd_line = input().strip()
            if not cmd_line:
                continue
            cmd = cmd_line.upper()
            if cmd == "A":
                try:
                    serA.write("M\n".encode())  # Board A: Master
                    serB.write("S\n".encode())  # Board B: Slave
                    print("Set: Board A = MASTER, Board B = SLAVE")
                except Exception as e:
                    print(f"Error sending A command: {e}")
            elif cmd == "B":
                try:
                    serA.write("S\n".encode())  # Board A: Slave
                    serB.write("M\n".encode())  # Board B: Master
                    print("Set: Board A = SLAVE, Board B = MASTER")
                except Exception as e:
                    print(f"Error sending B command: {e}")
            elif cmd == "SYNC":
                try:
                    serA.write("SYNC\n".encode())
                    serB.write("SYNC\n".encode())
                    print("Sent 'SYNC' command to both boards")
                except Exception as e:
                    print(f"Error sending SYNC command: {e}")
            elif cmd == "P":
                try:
                    serA.write("P\n".encode())
                    serB.write("P\n".encode())
                    print("Sent 'P' command to both boards")
                except Exception as e:
                    print(f"Error sending P command: {e}")
            elif cmd == "PLOT":
                try:
                    plt.figure()
                    if data_A:
                        plt.plot(range(len(data_A)), data_A, label='Board A')
                    else:
                        print("No ADC data from Board A to plot.")
                    if data_B:
                        plt.plot(range(len(data_B)), data_B, label='Board B')
                    else:
                        print("No ADC data from Board B to plot.")
                    plt.xlabel("Sample Number")
                    plt.ylabel("ADC Value")
                    plt.title("ADC Data from Boards")
                    plt.legend()
                    plt.show()
                except Exception as e:
                    print(f"Error plotting data: {e}")
            elif cmd == "EXIT":
                running = False
                break
            else:
                print("Invalid command. Please enter A, B, SYNC, P, plot, or exit.")
        except Exception as e:
            print("Error in user input thread:", e)
            break
    print("Exiting user input thread.")

# Start threads for reading from both boards and handling user input.
threading.Thread(target=read_serial, args=(serA, 'A'), daemon=True).start()
threading.Thread(target=read_serial, args=(serB, 'B'), daemon=True).start()
threading.Thread(target=user_input_thread, daemon=True).start()

# Keep the main thread alive.
try:
    while running:
        time.sleep(1)
except KeyboardInterrupt:
    running = False
    print("Exiting program...")
    time.sleep(0.5)
