import serial
import threading
import time
import sys
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

running = True

# Lists to store (timestamp, ADC value) tuples for each board.
dataA = []
dataB = []

# Open serial ports for Board A and Board B.
try:
    serA = serial.Serial('COM4', 115200, timeout=1)
    serB = serial.Serial('COM6', 115200, timeout=1)
except Exception as e:
    print("Error opening serial ports:", e)
    sys.exit(1)

def read_serial(ser, board_name):
    global dataA, dataB, running
    while running:
        try:
            line = ser.readline()
            if not line:
                continue
            # Get the host time (in microseconds) when this line is received.
            host_time = time.time() * 1e6
            line_str = line.decode('utf-8', errors='ignore').strip()
            # Print the host timestamp along with the board's message.
            print(f"[{host_time:.0f} us] Board {board_name}: {line_str}")
            # If the data is numeric, you might process it as before...
            try:
                adc_val = float(line_str)
                # Store ADC data, etc.
            except ValueError:
                # Not a numeric reading (like a sync test message); continue.
                pass
        except Exception as e:
            print(f"Error reading from Board {board_name}: {e}")
            time.sleep(0.1)
    ser.close()
    print(f"Exiting read thread for Board {board_name}")


def user_input_thread():
    print("Enter command:")
    print("  F = Set Board A as master, Board B as slave")
    print("  B = Set Board A as slave, Board B as master")
    print("  S = Send sync command to both boards (only master triggers sync)")
    print("  L:<number> = Set record length on both boards")
    while running:
        try:
            cmd_line = input().strip()
            if not cmd_line:
                continue
            cmd = cmd_line.upper()
            if cmd.startswith("L:"):
                try:
                    serA.write((cmd_line + "\n").encode())
                    serB.write((cmd_line + "\n").encode())
                    print(f"Sent '{cmd_line}' to both boards")
                except Exception as e:
                    print(f"Error sending L command: {e}")
            elif cmd == "F":
                # Set Board A as master and Board B as slave.
                try:
                    serA.write("M\n".encode())  # Board A: Master
                    time.sleep(0.1)
                    serB.write("S\n".encode())  # Board B: Slave
                    print("Set: Board A = MASTER, Board B = SLAVE")
                except Exception as e:
                    print(f"Error sending F command: {e}")
            elif cmd == "B":
                # Set Board A as slave and Board B as master.
                try:
                    serA.write("S\n".encode())  # Board A: Slave
                    time.sleep(0.1)
                    serB.write("M\n".encode())  # Board B: Master
                    print("Set: Board A = SLAVE, Board B = MASTER")
                except Exception as e:
                    print(f"Error sending B command: {e}")
            elif cmd == "S":
                # Send sync command to both boards.
                try:
                    serA.write("SYNC\n".encode())
                    #time.sleep(0.1)
                    serB.write("SYNC\n".encode())
                    print("Sent 'SYNC' command to both boards")
                except Exception as e:
                    print(f"Error sending SYNC command: {e}")
            else:
                print("Invalid command. Please enter F, B, S, or L:<number>.")
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
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    running = False
    print("Exiting program...")
    time.sleep(0.5)
#
# # Plot the stored data after stopping.
# if dataA:
#     xA = list(range(len(dataA)))
#     adcA = [val for (_, val) in dataA]
# else:
#     xA, adcA = [], []
# if dataB:
#     xB = list(range(len(dataB)))
#     adcB = [val for (_, val) in dataB]
# else:
#     xB, adcB = [], []
#
# plt.figure()
# plt.plot(xA, adcA, label='Board A')
# plt.plot(xB, adcB, label='Board B')
# plt.xlabel("Sample Number")
# plt.ylabel("ADC Value")
# plt.title("ADC Readings Over Sample Number")
# plt.legend()
# plt.show()
