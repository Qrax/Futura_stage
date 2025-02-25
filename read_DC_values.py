import serial
import matplotlib
matplotlib.use('TkAgg')  # Use a backend that works well with PyCharm
import matplotlib.pyplot as plt
import matplotlib.animation as animation

try:
    ser = serial.Serial('COM6', 115200, timeout=1)
    print("Serial port opened successfully on COM6.")
except Exception as e:
    print("Error opening serial port:", e)
    exit(1)

data_buffer = []  # Buffer for the current cycle
all_data = []     # Cumulative data across cycles

fig, ax = plt.subplots()
line, = ax.plot([], [], 'b-')
ax.set_xlim(0, 100)         # Adjust as needed
ax.set_ylim(0, 256)         # 8-bit ADC values: 0-255
ax.set_xlabel('Sample Number')
ax.set_ylabel('ADC Value')
ax.set_title('Ultrasonic Sensor ADC Data')
ax.grid(True)

def init():
    print("Animation init() called.")
    line.set_data([], [])
    return line,


def update(frame):
    global data_buffer
    new_cycle_complete = False
    while ser.in_waiting:
        try:
            line_received = ser.readline().decode('utf-8').strip()
        except Exception as e:
            print("Decoding error:", e)
            continue
        print("Received:", repr(line_received))
        if line_received == 'E':
            new_cycle_complete = True
            break  # End current cycle processing
        elif line_received.isdigit():
            data_buffer.append(int(line_received))
        else:
            print("Ignoring non-numeric message:", line_received)
    if new_cycle_complete:
        # Update plot with the new cycle's data
        x_vals = list(range(len(data_buffer)))
        line.set_data(x_vals, data_buffer)
        ax.set_xlim(0, len(data_buffer) + 10)
        print("Cycle complete. Plot updated.")
        # Clear the buffer for the next cycle
        data_buffer = []
    else:
        # If no complete cycle, update the plot with any partial data
        if data_buffer:
            x_vals = list(range(len(data_buffer)))
            line.set_data(x_vals, data_buffer)
            ax.set_xlim(0, len(data_buffer) + 10)
    return line,



ani = animation.FuncAnimation(fig, update, init_func=init, interval=100, blit=True)

print("Starting plot. Close the plot window to exit.")
plt.show(block=True)
print("Plot window closed. Exiting program.")
