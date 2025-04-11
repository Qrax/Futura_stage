import numpy as np
import matplotlib.pyplot as plt

# Simuleer de tijdsvector (bijv. 200 samples over 1 seconde)
t = np.linspace(0, 1, 200)

# Simuleer "harde" ADC-data:
# - Beginwaarden rond 25
# - Een snelle stijging naar 200 tussen sample 40 en 50
# - Een exponentiële afname van 200 naar ongeveer 25 tussen sample 50 en 150
raw_signal = np.ones_like(t) * 25  # Baseline op 25

# Snel stijgende overgang: van sample 40 tot 50
rise_start, rise_end = 40, 50
raw_signal[rise_start:rise_end] = np.linspace(25, 200, rise_end - rise_start)

# Exponentiële afname van sample 50 tot 150
decay_length = 150 - rise_end
decay_time = np.linspace(0, 1, decay_length)
# Stel een exponentiële afname in:
decay = 200 * np.exp(-3 * decay_time)  # 3 bepaalt hoe snel het afneemt
raw_signal[rise_end:150] = 25 + decay

# Na sample 150 blijft het signaal op de baseline (25)
raw_signal[150:] = 25

# Nu simuleren we een RC-laagdoorlaatfilter.
# Dit doen we met een eenvoudig discrete low-pass filter:
#    y[n] = y[n-1] + alpha * (x[n] - y[n-1])
# Waar alpha = dt/(RC + dt). We kiezen hier een waarde voor alpha die het filtergedrag simuleert.
alpha = 0.1  # Pas deze waarde aan voor meer/minder smoothing
filtered_signal = np.zeros_like(raw_signal)
filtered_signal[0] = raw_signal[0]
for i in range(1, len(raw_signal)):
    filtered_signal[i] = filtered_signal[i - 1] + alpha * (raw_signal[i] - filtered_signal[i - 1])

# Visualisatie
plt.figure(figsize=(10, 8))

# Plot 1: Harde ADC-data (zoals direct gemeten, met scherpe overgangen)
plt.subplot(2, 1, 1)
plt.plot(t, raw_signal, 'b-', lw=2, label='Harde ADC Data')
plt.title('Origineel "Harde" ADC Signaal (bijv. via bitbang burst)')
plt.ylabel('ADC Waarde / Spanning (arbitrary)')
plt.legend()
plt.grid(True)

# Plot 2: Filtered output: simulatie van het signaal na de omzetting (PWM + RC filter)
plt.subplot(2, 1, 2)
plt.plot(t, filtered_signal, 'r-', lw=2, label='Gefilterde Output')
plt.title('Verzacht Signaal na RC Filtering (analog_out)')
plt.xlabel('Tijd (s)')
plt.ylabel('Gemoduleerde Spanning (arbitrary)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
