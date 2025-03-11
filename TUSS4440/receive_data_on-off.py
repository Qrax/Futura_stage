import serial
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import numpy as np
from collections import deque

# Configure serial port - adjust COM port as needed
try:
    ser = serial.Serial('COM6', 115200, timeout=1)
    print("Serial port opened successfully on COM6.")
except Exception as e:
    print("Error opening serial port:", e)
    exit(1)

# Data buffers
current_data_buffer = []  # Buffer for the current measurement cycle
is_collecting_data = False  # Flag to indicate if we're currently collecting a data block

# Time series data - larger maxlen to keep more data visible
time_data = deque(maxlen=20000)
sensor_data = deque(maxlen=20000)
pulse_markers = []

# Status tracking
sampling_enabled = False
last_pulse_time = 0
start_time = time.time()

# Create the figure and axes with more space for time series
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [1, 4]})

# Top plot - Current pulse measurement
line1, = ax1.plot([], [], 'b-')
ax1.set_xlim(0, 300)
ax1.set_ylim(0, 256)
ax1.set_title('Current Pulse Measurement')
ax1.set_xlabel('Sample Number')
ax1.set_ylabel('ADC Value')
ax1.grid(True)

# Bottom plot - Time series with wider view
line2, = ax2.plot([], [], 'g-', alpha=0.7, linewidth=1.0)  # Thinner line for clarity
pulse_lines = []  # Vertical lines marking pulse measurements
ax2.set_xlim(0, 10)  # Show 10 seconds initially (will auto-adjust)
ax2.set_ylim(0, 256)
ax2.set_title('High-Speed Time Series Data')
ax2.set_xlabel('Time (seconds)')
ax2.set_ylabel('ADC Value')
ax2.grid(True)

# Status text display
status_text = ax1.text(0.02, 0.95, 'Sampling: UIT', transform=ax1.transAxes,
                       fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
pulse_text = ax1.text(0.02, 0.85, '', transform=ax1.transAxes,
                      fontsize=12, bbox=dict(facecolor='lightgreen', alpha=0.0))

# Keep track of data for dynamic y-axis scaling
min_value_seen = 255
max_value_seen = 0


def init():
    line1.set_data([], [])
    line2.set_data([], [])
    return line1, line2, status_text, pulse_text


def update_pulse_text():
    global last_pulse_time, pulse_text
    current_time = time.time()
    if current_time - last_pulse_time < 2:
        pulse_text.set_text("Puls gestuurd")
        pulse_text.set_bbox(dict(facecolor='lightgreen', alpha=0.7))
    else:
        pulse_text.set_text("")
        pulse_text.set_bbox(dict(facecolor='white', alpha=0.0))


def update_time_series():
    global time_data, sensor_data, pulse_markers, pulse_lines, min_value_seen, max_value_seen

    if len(time_data) > 0:
        current_time = time.time() - start_time

        # Dynamic timespan - Show narrower window when sampling for better detail
        timespan = 5 if sampling_enabled else 15
        ax2.set_xlim(max(0, current_time - timespan), current_time + 0.5)

        # Dynamic y-axis scaling with 10% margins above and below
        if min_value_seen < 255 and max_value_seen > 0:
            margin = max(5, (max_value_seen - min_value_seen) * 0.1)
            ax2.set_ylim(max(0, min_value_seen - margin), min(255, max_value_seen + margin))

        # Update main line
        line2.set_data(list(time_data), list(sensor_data))

        # Update pulse markers (vertical lines)
        for line in pulse_lines:
            line.remove()
        pulse_lines = []

        for marker_time in pulse_markers:
            if marker_time > current_time - timespan:
                line = ax2.axvline(x=marker_time, color='r', linestyle='-', alpha=0.7, linewidth=1.5)
                pulse_lines.append(line)


def update(frame):
    global current_data_buffer, time_data, sensor_data, sampling_enabled
    global last_pulse_time, pulse_markers, is_collecting_data, min_value_seen, max_value_seen

    while ser.in_waiting:
        try:
            line_received = ser.readline().decode('utf-8').strip()
        except Exception as e:
            print(f"Decoding error: {e}")
            continue

        # Special markers
        if line_received == "START":
            is_collecting_data = True
            current_data_buffer = []
            continue
        elif line_received == "END":
            is_collecting_data = False
            # Update current measurement plot with complete data block
            if current_data_buffer:
                x_vals = list(range(len(current_data_buffer)))
                line1.set_data(x_vals, current_data_buffer)
                ax1.set_xlim(0, len(current_data_buffer) + 10)
            continue
        elif line_received == "SAMPLE":
            # Next line will be a single sample value from continuous mode
            try:
                sample_line = ser.readline().decode('utf-8').strip()
                if sample_line.isdigit():
                    sample_value = int(sample_line)

                    # Track min/max for dynamic scaling
                    min_value_seen = min(min_value_seen, sample_value)
                    max_value_seen = max(max_value_seen, sample_value)

                    # Add to time series
                    current_time = time.time() - start_time
                    time_data.append(current_time)
                    sensor_data.append(sample_value)
            except Exception as e:
                print(f"Error reading sample value: {e}")
            continue

        # Normal data processing for pulse measurements
        if is_collecting_data and line_received.isdigit():
            try:
                value = int(line_received)
                # Add to current data buffer
                current_data_buffer.append(value)

                # Track min/max for scaling
                min_value_seen = min(min_value_seen, value)
                max_value_seen = max(max_value_seen, value)

                # Also add to time series
                current_time = time.time() - start_time
                time_data.append(current_time)
                sensor_data.append(value)
            except ValueError:
                # Skip non-integer values
                continue
        else:
            # Handle status messages
            if "Pulse sent" in line_received:
                print("*** PULS GEDETECTEERD! ***")
                last_pulse_time = time.time()
                # Mark the pulse in the time series
                pulse_markers.append(time.time() - start_time)
            elif "Continuous sampling ENABLED" in line_received:
                sampling_enabled = True
                print("*** SAMPLING AAN GEDETECTEERD! ***")
            elif "Continuous sampling DISABLED" in line_received:
                sampling_enabled = False
                print("*** SAMPLING UIT GEDETECTEERD! ***")

    # Update status displays
    status_text.set_text(f"Sampling: {'AAN' if sampling_enabled else 'UIT'}")
    update_pulse_text()
    update_time_series()

    return line1, line2, status_text, pulse_text


# Add buttons for interaction
clear_button_ax = plt.axes([0.8, 0.01, 0.08, 0.04])
clear_button = plt.Button(clear_button_ax, 'Clear Data')


def clear_data(event):
    global time_data, sensor_data, pulse_markers, pulse_lines, min_value_seen, max_value_seen
    time_data.clear()
    sensor_data.clear()
    pulse_markers.clear()
    for line in pulse_lines:
        line.remove()
    pulse_lines = []
    min_value_seen = 255
    max_value_seen = 0
    print("Time series data cleared.")


clear_button.on_clicked(clear_data)

save_button_ax = plt.axes([0.9, 0.01, 0.08, 0.04])
save_button = plt.Button(save_button_ax, 'Save')


def save_data(event):
    try:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"ultrasonic_data_{timestamp}.csv"
        with open(filename, 'w') as f:
            f.write("Time,Value,IsPulse\n")
            pulse_set = set(pulse_markers)
            for t, v in zip(time_data, sensor_data):
                is_pulse = 1 if t in pulse_set else 0
                f.write(f"{t:.6f},{v},{is_pulse}\n")
        print(f"Data saved to {filename}")
    except Exception as e:
        print(f"Error saving data: {e}")


save_button.on_clicked(save_data)

# Start animation with faster update interval
ani = animation.FuncAnimation(fig, update, init_func=init, interval=50, blit=True, cache_frame_data=False)

print("Plot gestart. Sluit het plotvenster om te stoppen.")
print("Druk op PUSH1 om continue sampling aan/uit te zetten.")
print("Druk op PUSH2 om een puls te sturen.")
plt.tight_layout()
plt.show(block=True)
print("Plot window closed. Exiting program.")