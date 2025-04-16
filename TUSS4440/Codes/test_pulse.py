import serial
import time
import pandas as pd
import matplotlib.pyplot as plt
import sys # To exit gracefully on error
import numpy as np # Import numpy for array manipulation

# --- Configuration ---
serial_port = 'COM6'  # <<<--- CHANGE THIS if your port is different
baud_rate = 115200    # Must match the Energia sketch
end_marker = 'E'
measurement_command = b'P' # Send 'P' as bytes
num_points_to_move = 250 # <<<--- Number of points from the END to move to the START
# <<<--- Average time between samples in microseconds (from Arduino output)
time_per_sample_us = 16

# --- Serial Connection Setup ---
ser = None  # Initialize ser to None
try:
    print(f"Attempting to connect to {serial_port} at {baud_rate} baud...")
    ser = serial.Serial(serial_port, baud_rate, timeout=5)
    print(f"Connected to {serial_port}.")
    print("Waiting a moment for Arduino to initialize...")
    time.sleep(2.5) # Adjust this delay if needed

    # --- Send Command and Receive Data ---
    print(f"Sending command '{measurement_command.decode()}' to start measurement...")
    ser.reset_input_buffer()
    ser.write(measurement_command)
    time.sleep(0.1)

    print("Waiting for data...")
    adc_data = []
    received_lines = [] # Store all received lines for debugging if needed

    # --- Read data (ignore Arduino's informational print lines) ---
    while True:
        try:
            line_bytes = ser.readline()
            if not line_bytes:
                print("Warning: Timeout waiting for data line. Processing collected data.")
                break

            line = line_bytes.decode('utf-8', errors='ignore').strip()
            received_lines.append(line) # Store for debugging

            # Stop reading when the end marker is found
            if line == end_marker:
                print("End marker 'E' received. Measurement complete.")
                break
            elif line:
                # Try to convert to integer, skip if it fails (ignores text lines)
                try:
                    adc_value = int(line)
                    adc_data.append(adc_value)
                except ValueError:
                    # Silently ignore lines that are not integers
                    # You could print them here if needed for debugging:
                    # print(f"Ignoring non-integer line: {line}")
                    pass # Ignore non-integer lines like the info messages

        except serial.SerialTimeoutException:
            print("Warning: Timeout occurred while reading data. Processing any data received before timeout.")
            break
        except Exception as e:
            print(f"An error occurred during data reception: {e}")
            print("Received lines before error:", received_lines)
            break

    print(f"Received {len(adc_data)} valid data points.")

    # --- Process and Rearrange Data ---
    modified_adc_data = None # Initialize
    data_was_rearranged = False

    if not adc_data:
        print("No ADC data received. Exiting.")
    elif len(adc_data) < num_points_to_move:
         print(f"Warning: Received data length ({len(adc_data)}) is less than "
               f"num_points_to_move ({num_points_to_move}). Using original data order.")
         modified_adc_data = np.array(adc_data)
         data_was_rearranged = False
    else:
        print(f"Rearranging data: Moving last {num_points_to_move} points to the beginning.")
        original_data_array = np.array(adc_data)
        total_points = len(original_data_array)
        split_index = total_points - num_points_to_move
        last_part = original_data_array[split_index:]
        first_part = original_data_array[:split_index]
        modified_adc_data = np.concatenate((last_part, first_part))
        data_was_rearranged = True
        print(f"Created rearranged dataset with {len(modified_adc_data)} points.")


    # --- Create DataFrame, Calculate Time Axis, and Plot (only if data exists) ---
    if modified_adc_data is not None and len(modified_adc_data) > 0:
        print("Creating Pandas DataFrame...")
        # Store ADC values directly
        df_modified = pd.DataFrame({'ADC_Value': modified_adc_data})

        # --- Calculate Time Axis ---
        num_points = len(modified_adc_data)
        # Create time vector starting from 0, with steps of time_per_sample_us
        time_axis_us = np.arange(num_points) * time_per_sample_us
        # Convert time to milliseconds for better readability on the plot
        time_axis_ms = time_axis_us / 1000.0
        # Add the time axis to the DataFrame (optional, but can be useful)
        df_modified['Time_ms'] = time_axis_ms

        print("\nDataFrame Head (with Time):")
        print(df_modified.head())
        print("\nDataFrame Tail (with Time):")
        print(df_modified.tail())
        print("\nDataFrame Info:")
        df_modified.info()

        # --- Plot Data with Matplotlib using Time Axis ---
        print("Plotting rearranged data against time...")
        plt.figure(figsize=(12, 6))
        # Plot Time_ms on x-axis and ADC_Value on y-axis
        plt.plot(df_modified['Time_ms'], df_modified['ADC_Value'], marker='.', linestyle='-', markersize=4) # Smaller marker '.'

        plot_title = f'TUSS44x0 ADC Readings)'
        # if data_was_rearranged:
        #      plot_title += f'\n(Last {num_points_to_move} points moved to start)'
        # elif len(adc_data) > 0 and len(adc_data) < num_points_to_move:
        #      plot_title = f'Original TUSS44x0 ADC Readings ({time_per_sample_us} µs/sample)\n(Data too short to rearrange)'

        plt.title(plot_title)
        plt.xlabel('Time (ms)') # Set x-axis label to Time in milliseconds
        plt.ylabel('ADC Value (8-bit)')
        plt.grid(True)
        min_val = np.min(modified_adc_data)
        max_val = np.max(modified_adc_data)
        plt.ylim(min(0, min_val - 5), max(255, max_val + 5)) # Give a little margin
        plt.tight_layout()
        plt.show()
        print("Plot displayed.")
    elif modified_adc_data is not None and len(modified_adc_data) == 0:
        print("Data array is empty after processing. Nothing to plot.")


except serial.SerialException as e:
    print(f"Serial Error: {e}")
    print(f"Could not open port {serial_port}. Check port and availability.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
finally:
    # --- Ensure Serial Port is Closed ---
    if ser and ser.is_open:
        ser.close()
        print(f"Serial port {serial_port} closed.")
    else:
        print("Serial port was not opened or already closed.")