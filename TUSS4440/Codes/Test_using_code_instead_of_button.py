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
            line_str = line.decode('utf-8', errors='ignore').strip()
            # Skip the end marker.
            if line_str == "E":
                continue
            try:
                adc_val = float(line_str)
            except ValueError:
                print(f"Board {board_name} received non-numeric: {line_str}")
                continue
            # Use the current time in microseconds (for logging purposes)
            timestamp = time.time() * 1e6
            print(f"Board {board_name}: Time: {timestamp:.0f} us, ADC: {adc_val}")
            # Store the data
            if board_name == 'A':
                dataA.append((timestamp, adc_val))
            else:
                dataB.append((timestamp, adc_val))
        except Exception as e:
            print(f"Error reading from Board {board_name}: {e}")
            time.sleep(0.1)
    ser.close()
    print(f"Exiting read thread for Board {board_name}")

def user_input_thread():
    print("Enter command: F (forward), B (backward), or L:<number> (set record length)")
    while running:
        try:
            cmd_line = input().strip()
            if not cmd_line:
                continue
            cmd = cmd_line.upper()
            if cmd.startswith("L:"):
                # Send the L command to both boards.
                try:
                    serA.write((cmd_line + "\n").encode())
                    serB.write((cmd_line + "\n").encode())
                    print(f"Sent '{cmd_line}' to both Board A and Board B")
                except Exception as e:
                    print(f"Error sending L command: {e}")
            elif cmd == "F":
                # Forward command: send "A" to Board A and "B" to Board B.
                try:
                    serA.write("A\n".encode())
                    time.sleep(0.1)
                    serB.write("B\n".encode())
                    print("Sent 'A' to Board A and 'B' to Board B (Forward)")
                except Exception as e:
                    print(f"Error sending forward command: {e}")
            elif cmd == "B":
                # Backward command: send "B" to Board A and "A" to Board B.
                try:
                    serA.write("B\n".encode())
                    time.sleep(0.1)
                    serB.write("A\n".encode())

                    print("Sent 'B' to Board A and 'A' to Board B (Backward)")
                except Exception as e:
                    print(f"Error sending backward command: {e}")
            else:
                print("Invalid command. Please enter F, B, or L:<number>.")
        except Exception as e:
            print("Error in user input thread:", e)
            break
    print("Exiting user input thread.")

# Start threads to read from each board and process user commands.
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

# After stopping, plot the stored data using sample number as the x-axis.
if dataA:
    xA = list(range(len(dataA)))  # Sample numbers for Board A
    adcA = [val for (_, val) in dataA]
else:
    xA, adcA = [], []

if dataB:
    xB = list(range(len(dataB)))  # Sample numbers for Board B
    adcB = [val for (_, val) in dataB]
else:
    xB, adcB = [], []

plt.figure()
plt.plot(xA, adcA, label='Board A')
plt.plot(xB, adcB, label='Board B')
plt.xlabel("Sample Number")
plt.ylabel("ADC Value")
plt.title("ADC Readings Over Sample Number")
plt.legend()
plt.show()
