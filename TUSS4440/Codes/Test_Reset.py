import serial
import time

port = 'COM4'  # Adjust to your board's port
ser = serial.Serial(port, 115200, timeout=1)
time.sleep(1)  # Allow time for initialization

print("Reading board output for 10 seconds...")
end_time = time.time() + 10
while time.time() < end_time:
    if ser.in_waiting:
        line = ser.readline().decode('utf-8', errors='replace').strip()
        if line:
            print(line)
ser.close()
