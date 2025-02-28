import serial
import matplotlib

matplotlib.use('TkAgg')  # Use a backend that works well with PyCharm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import csv
import os

try:
    ser = serial.Serial('COM6', 115200, timeout=1)
    print("Serial port opened successfully on COM6.")
except Exception as e:
    print("Error opening serial port:", e)
    exit(1)

data_buffer = []  # Buffer for the current cycle
all_data = []  # Cumulative data across cycles

fig, ax = plt.subplots()
line, = ax.plot([], [], 'b-')
ax.set_xlim(0, 100)  # Adjust as needed
ax.set_ylim(0, 256)  # 8-bit ADC values: 0-255
ax.set_xlabel('Sample Number')
ax.set_ylabel('ADC Value')
ax.set_title('Ultrasonic Sensor ADC Data')
ax.grid(True)


def init():
    print("Animation init() called.")
    line.set_data([], [])
    return line,


def update(frame):
    global data_buffer, all_data
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

        # Store the current cycle's data in all_data
        all_data.append(data_buffer.copy())  # add copy so the data doesnt change when data_buffer is reset

        # Clear the buffer for the next cycle
        data_buffer = []
    else:
        # If no complete cycle, update the plot with any partial data
        if data_buffer:
            x_vals = list(range(len(data_buffer)))
            line.set_data(x_vals, data_buffer)
            ax.set_xlim(0, len(data_buffer) + 10)
    return line,


def on_close(event):
    """Callback function to save data when the plot window is closed."""
    print("Plot window closed. Saving data...")
    save_data_to_csv(all_data)
    print("Data saved. Exiting program.")
    ser.close()  # Make sure to close the serial port
    plt.close('all')  # Close all plot windows
    exit()


def process_data(data):
    """Remove the initial peak from the data."""
    if len(data) < 20:
        return data  # Not enough data to process

    # Find where the initial peak ends
    for i in range(len(data) - 10):
        if all(val < 50 for val in data[i:i + 10]):
            return data[i:]  # Return data after the initial peak

    return data  # Return original if no clear cutoff found


def save_data_to_csv(data, filename="ultrasonic_data.csv"):
    """Save the collected data to a CSV file with proper headers."""
    if not data:
        print("No data to save.")
        return

    # Create a directory for data if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')

    # Process each cycle to remove initial peaks
    processed_data = [process_data(cycle) for cycle in data]

    filepath = os.path.join('data', filename)
    print(f"Saving data to {filepath}")

    # Save the processed data in a simple format
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write headers
        writer.writerow(['capture_id', 'sample_id', 'adc_value'])

        # Write data
        for capture_id, capture_data in enumerate(processed_data):
            for sample_id, value in enumerate(capture_data):
                writer.writerow([capture_id, sample_id, value])

    print(f"Processed data saved to {filepath}")

    # Also save the raw data for backup
    raw_filepath = os.path.join('data', "raw_" + filename)
    with open(raw_filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['capture_id', 'sample_id', 'adc_value'])
        for capture_id, capture_data in enumerate(data):
            for sample_id, value in enumerate(capture_data):
                writer.writerow([capture_id, sample_id, value])

    print(f"Raw data saved to {raw_filepath}")

    return filepath


# Connect the close event to the on_close callback
fig.canvas.mpl_connect('close_event', on_close)

ani = animation.FuncAnimation(fig, update, init_func=init, interval=100, blit=True)

print("Starting plot. Close the plot window to exit.")
plt.show(block=True)