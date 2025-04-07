import numpy as np
import matplotlib.pyplot as plt

# --- Simulatie Parameters ---
sampling_rate = 500000  # Hoge sample rate voor nauwkeurigheid (500 kHz)
duration = 0.002      # 2 milliseconden
t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
carrier_freq = 40000    # 40 kHz

# --- Fase 1: Analoge Echo Envelope (Gesimuleerd) ---
delay = 0.0005
pulse_width = 0.0002
echo_envelope = np.zeros_like(t)
mask = (t > delay) & (t < delay + pulse_width)
echo_envelope[mask] = 1.0 * np.exp(-(t[mask] - delay) * 5000) # Exponentieel verval
echo_envelope += np.random.normal(0, 0.05, echo_envelope.shape)
echo_envelope[echo_envelope < 0] = 0

# --- Fase 2: Digitale Samples (Gesimuleerd) ---
adc_sampling_rate = 50000 # 50 kHz sample rate voor ADC
adc_indices = np.round(np.linspace(0, len(t) - 1, int(adc_sampling_rate * duration))).astype(int)
t_adc = t[adc_indices]
digital_samples = echo_envelope[adc_indices]

# --- Fase 3: Time-Reversed Digitale Samples ---
time_reversed_digital_samples = digital_samples[::-1] # Keer de volgorde om

# --- Fase 4: Analoge Time-Reversed Envelope (DAC Output - Gesimuleerd) ---
# We gebruiken dezelfde tijd-as voor het plotten, maar de *waarden* zijn omgekeerd
# De DAC reconstrueert de golfvorm op basis van de omgekeerde sample-volgorde
time_reversed_analog_envelope = np.interp(t, t_adc, time_reversed_digital_samples)

# --- Fase 5: Gemoduleerd Signaal (Modulator Output - Gesimuleerd) ---
carrier_wave = 0.5 * np.sin(2 * np.pi * carrier_freq * t) + 0.5
modulated_signal = time_reversed_analog_envelope * carrier_wave

# --- Fase 8: Ontvangen Envelope op Bord A (Gesimuleerd) ---
received_time_reversed_envelope = time_reversed_analog_envelope + np.random.normal(0, 0.03, time_reversed_analog_envelope.shape)
received_time_reversed_envelope[received_time_reversed_envelope < 0] = 0

# --- Plotten ---
plt.figure(figsize=(12, 15))

plt.subplot(5, 1, 1)
plt.plot(t * 1000, echo_envelope)
plt.title("Fase 1: Analoge Echo Envelope (Ontvangen op VOUT Bord B)")
plt.xlabel("Tijd (ms)")
plt.ylabel("Amplitude")
plt.grid(True)

plt.subplot(5, 1, 2)
plt.plot(t_adc * 1000, digital_samples, 'o-', label='Originele Samples')
# Plot de omgekeerde samples tegen de originele tijd-as om de omkering te zien
plt.plot(t_adc * 1000, time_reversed_digital_samples, 'rx-', label='Time-Reversed Samples')
plt.title("Fase 2 & 3: Digitale Samples & Time-Reversed Samples (MSP430 Bord B)")
plt.xlabel("Tijd (ms)")
plt.ylabel("Digitale Waarde")
plt.legend()
plt.grid(True)

plt.subplot(5, 1, 3)
plt.plot(t * 1000, time_reversed_analog_envelope)
plt.title("Fase 4: Analoge Time-Reversed Envelope (Output Externe DAC)")
plt.xlabel("Tijd (ms)")
plt.ylabel("Amplitude")
plt.grid(True)

plt.subplot(5, 1, 4)
plt.plot(t * 1000, modulated_signal)
plt.title("Fase 5: Gemoduleerd 40kHz Signaal (met Time-Reversed Envelope)")
plt.xlabel("Tijd (ms)")
plt.ylabel("Amplitude")
plt.grid(True)

plt.subplot(5, 1, 5)
plt.plot(t * 1000, received_time_reversed_envelope)
plt.title("Fase 8: Ontvangen Time-Reversed Envelope (VOUT Bord A)")
plt.xlabel("Tijd (ms)")
plt.ylabel("Amplitude")
plt.grid(True)

plt.tight_layout()
plt.show()