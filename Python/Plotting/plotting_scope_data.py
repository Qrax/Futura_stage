import os
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Set up file path and loading parameters ---

# From your CSV, we see data starts on line 5, so we skip 4 header rows.
HEADER_ROWS_TO_SKIP = 4

# From your CSV header "Period:,23.640 uS", this is the time between samples.
# 1 microsecond (uS) = 1e-6 seconds.
SAMPLING_PERIOD_DT = 23.640e-6 # (in seconds)

file_name = "42khz_sinus_wave_dac_esp.csv"
relative_data_path = os.path.join('../python/data/scope_data/', file_name)


# --- 2. Load the data from the CSV file ---
try:
    print(f"Attempting to load data from: {relative_data_path}")
    # Load the data, skipping the 4-line header.
    # The file has two columns: Sample Index and Voltage.
    data = np.loadtxt(relative_data_path, delimiter=",", skiprows=HEADER_ROWS_TO_SKIP)
    
    # The second column (index 1) is our voltage data.
    v = data[:, 1]
    print(f"Data loaded successfully. Found {len(v)} samples.")

except (FileNotFoundError, IOError):
    print(f"Error: File not found at '{relative_data_path}'.")
    exit()
except ValueError as e:
    print(f"Error loading data: {e}. Check `skiprows` and the file's delimiter.")
    exit()
except IndexError:
    # This error happens if loadtxt reads only one column or a malformed row.
    print("Error: Could not parse two columns from the data. Check the delimiter and file format after the header.")
    exit()


# --- 3. Create the corresponding time array ---
# The first column is just an index (1, 2, 3...). We create a proper time vector.
# t = [0, dt, 2*dt, 3*dt, ...]
num_samples = len(v)
t = np.arange(num_samples) * SAMPLING_PERIOD_DT


# --- 4. Compute the Fast Fourier Transform (FFT) ---
# We use our defined sampling period 'dt' for the calculation.
F = np.fft.rfftfreq(num_samples, SAMPLING_PERIOD_DT)
V_fft = np.abs(np.fft.rfft(v))


# --- 5. Plot the results ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), tight_layout=True)

# Plot Time Series Data
ax1.plot(t, v)
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Voltage (V)")
ax1.set_title("Oscilloscope Time Series Data")
ax1.grid(True)

# Plot FFT
ax2.plot(F, V_fft)
ax2.set_xscale('log') # Log scale is better for frequency plots
ax2.set_yscale('log') # Log scale is better for magnitude
ax2.set_xlabel("Frequency (Hz)")
ax2.set_ylabel("Magnitude")
ax2.set_title(f"FFT of Voltage (Peak should be near {1/SAMPLING_PERIOD_DT:.1f} Hz)")
ax2.grid(True, which="both", ls="-")

plt.show()