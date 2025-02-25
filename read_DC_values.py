import serial
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

# Open the serial port (adjust COM port and baud rate as needed)
try:
    ser = serial.Serial('COM6', 115200, timeout=1)
    print("Serial port opened successfully.")
except Exception as e:
    print(f"Error opening serial port: {e}")
    exit(1)

data_buffer = []  # Buffer to collect ADC samples

# Set up the plot
fig, ax = plt.subplots()
line, = ax.plot([], [], 'b-')
ax.set_xlim(0, 100)
ax.set_ylim(0, 256)  # 8-bit ADC values range from 0 to 255
ax.set_xlabel('Sample Number')
ax.set_ylabel('ADC Value')
ax.set_title('Ultrasonic Sensor ADC Data')
ax.grid(True)

def init():
    print("Animation initialization.")
    line.set_data([], [])
    return line,

def update(frame):
    global data_buffer
    print("Update function called.")
    # Read all available lines from serial port
    while ser.in_waiting:
        try:
            raw_line = ser.readline().decode('utf-8').strip()
        except Exception as e:
            print("Error decoding line:", e)
            continue
        print("Raw line received:", repr(raw_line))
        if raw_line == 'E':  # End-of-cycle marker
            print("End-of-cycle detected. Updating plot.")
            x_vals = list(range(len(data_buffer)))
            line.set_data(x_vals, data_buffer)
            data_buffer = []  # Clear buffer for next cycle
            break
        else:
            try:
                value = int(raw_line)
                data_buffer.append(value)
            except ValueError:
                print("Could not parse value from:", raw_line)
    return line,

# Create the animation
ani = animation.FuncAnimation(fig, update, init_func=init, interval=100, blit=True)

print("Starting plot. Close the plot window to exit.")
plt.show()
print("Plot window closed. Exiting program.")
