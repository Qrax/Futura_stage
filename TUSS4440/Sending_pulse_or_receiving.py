import serial
import matplotlib

matplotlib.use('TkAgg')  # Use a backend that works well with PyCharm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

try:
    ser = serial.Serial('COM6', 115200, timeout=1)
    print("Serial port opened successfully on COM6.")
except Exception as e:
    print("Error opening serial port:", e)
    exit(1)

data_buffer = []  # Buffer for the current cycle
receiving_enabled = True  # Track receiving status
last_pulse_time = 0  # Time when last pulse was detected

fig, ax = plt.subplots(figsize=(10, 6))
line, = ax.plot([], [], 'b-')
ax.set_xlim(0, 100)  # Adjust as needed
ax.set_ylim(0, 256)  # 8-bit ADC values: 0-255
ax.set_xlabel('Sample Number')
ax.set_ylabel('ADC Value')
ax.set_title('Ultrasonic Sensor ADC Data')
ax.grid(True)

# Status text display
status_text = ax.text(0.02, 0.95, 'Ontvangen: AAN', transform=ax.transAxes,
                      fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
pulse_text = ax.text(0.02, 0.90, '', transform=ax.transAxes,
                     fontsize=12, bbox=dict(facecolor='lightgreen', alpha=0.0))


def init():
    print("Animation init() called.")
    line.set_data([], [])
    return line, status_text, pulse_text


def update_pulse_text():
    global last_pulse_time, pulse_text

    # Als er recent een puls is geweest, toon deze voor 2 seconden
    current_time = time.time()
    if current_time - last_pulse_time < 2:
        pulse_text.set_text("Puls gestuurd")
        pulse_text.set_bbox(dict(facecolor='lightgreen', alpha=0.7))
    else:
        pulse_text.set_text("")
        pulse_text.set_bbox(dict(facecolor='white', alpha=0.0))


def update(frame):
    global data_buffer, receiving_enabled, last_pulse_time
    new_cycle_complete = False

    while ser.in_waiting:
        try:
            line_received = ser.readline().decode('utf-8').strip()
        except Exception as e:
            print("Decoding error:", e)
            continue

        # Debug - print alles wat binnenkomt
        if line_received:
            print(f"Ontvangen: '{line_received}'")

        if line_received == 'E':
            new_cycle_complete = True
            break  # End current cycle processing
        elif line_received.isdigit():
            data_buffer.append(int(line_received))
        else:
            # Handle status messages
            if "Pulse sent" in line_received:
                print("*** PULS GEDETECTEERD! ***")
                last_pulse_time = time.time()
            elif "Receiving ENABLED" in line_received:
                receiving_enabled = True
                print("*** ONTVANGEN AAN GEDETECTEERD! ***")
            elif "Receiving DISABLED" in line_received:
                receiving_enabled = False
                print("*** ONTVANGEN UIT GEDETECTEERD! ***")

    # Update status text
    status_text.set_text(f"Ontvangen: {'AAN' if receiving_enabled else 'UIT'}")

    # Update pulse text
    update_pulse_text()

    if new_cycle_complete:
        # Update plot with the new cycle's data
        x_vals = list(range(len(data_buffer)))
        line.set_data(x_vals, data_buffer)
        ax.set_xlim(0, len(data_buffer) + 10)
        print(f"Cyclus compleet. Plot bijgewerkt met {len(data_buffer)} samples.")

        # Clear the buffer for the next cycle
        data_buffer = []
    else:
        # If no complete cycle, update the plot with any partial data
        if data_buffer:
            x_vals = list(range(len(data_buffer)))
            line.set_data(x_vals, data_buffer)
            ax.set_xlim(0, len(data_buffer) + 10)

    return line, status_text, pulse_text


ani = animation.FuncAnimation(fig, update, init_func=init, interval=100, blit=True)

print("Plot gestart. Sluit het plotvenster om te stoppen.")
plt.show(block=True)
print("Plot window closed. Exiting program.")