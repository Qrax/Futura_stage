import serial
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import re
import threading
import sys
import matplotlib
matplotlib.use('TkAgg')
import numpy as np

# Configure serial port (adjust port as needed).
ser = serial.Serial('COM4', 115200, timeout=1)

# Regular expression to extract data from lines like:
# "Time: 123456 us, ADC_value: 200"
pattern = re.compile(r"Time:\s*(\d+)\s*us,\s*ADC_value:\s*(\d+)")

# Lists to store time and ADC values.
times = []
adc_values = []

# Define the scroll window (in microseconds). Example: 1,000,000 us = 1 second.
scrollWindow = 1000000

# Set up the figure and axis.
fig, ax = plt.subplots()
# Increase marker size for better visibility.
scat = ax.scatter([], [], s=30, color='blue')
ax.set_xlabel('Time (us)')
ax.set_ylabel('ADC Value')
ax.set_title('Live ADC Measurements')

#STart the plot with an y view from 15 to 50
ax.set_ylim(15, 50)


def update(frame):
    # Read all available lines from the serial port.
    while ser.in_waiting:
        try:
            raw_line = ser.readline()
            line_str = raw_line.decode('utf-8', errors='ignore').strip()
            if line_str:
                match = pattern.search(line_str)
                if match:
                    t_val = int(match.group(1))
                    adc_val = int(match.group(2))
                    times.append(t_val)
                    adc_values.append(adc_val)
                else:
                    # Debug: print line that doesn't match the expected pattern.
                    print("Received (unmatched):", line_str)
        except Exception as e:
            print("Error reading from serial:", e)

    if times:
        # Prepare data for the scatter plot.
        data = np.column_stack((times, adc_values))
        scat.set_offsets(data)

        # "Scroll" the x-axis: only show data within the last scrollWindow microseconds.
        last_time = times[-1]
        if last_time > scrollWindow:
            ax.set_xlim(last_time - scrollWindow, last_time)
        else:
            ax.set_xlim(0, scrollWindow)

        # Optionally adjust the y-axis.
        ax.relim()
        ax.autoscale_view(scaley=True)
    return scat,


def serial_command_thread():
    """Thread to read terminal input and send 's' or 'x' commands via serial."""
    print("Type 's' to send start command, 'x' to send stop command.")
    while True:
        try:
            cmd = input().strip()
            if cmd.lower() in ['s', 'x']:
                ser.write((cmd + "\n").encode())
                print(f"Sent command: {cmd}")
            else:
                print("Only 's' or 'x' are accepted.")
        except Exception as e:
            print("Error reading command:", e)
            break


# Start the command thread as a daemon.
threading.Thread(target=serial_command_thread, daemon=True).start()

# Create an animation that updates the scatter plot every 100 ms.
ani = animation.FuncAnimation(fig, update, interval=1)

plt.show()
ser.close()