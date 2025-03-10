import serial
import time

# Open the serial connection to COM6.
ser = serial.Serial('COM6', 115200, timeout=1)
time.sleep(1)  # Allow time for the board to reset and initialize

# Ask the user which command to send.
cmd = input("Enter command (TX to transmit, RX to receive): ").strip().upper()

if cmd == "TX":
    ser.write(b"TX\n")
    time.sleep(0.2)  # Brief delay for processing
    print("Board response:")
    while ser.in_waiting:
        print(ser.readline().decode().strip())

elif cmd == "RX":
    ser.write(b"RX\n")
    # Allow time for the board to capture and stream data.
    time.sleep(1.5)
    data = []
    while ser.in_waiting:
        line = ser.readline().decode().strip()
        if line:
            data.append(line)
    # Save the received data to a file.
    with open("received_data.txt", "w") as f:
        f.write("\n".join(data))
    print("Data received and saved to received_data.txt.")

else:
    print("Unknown command. Please enter 'TX' or 'RX'.")

ser.close()
