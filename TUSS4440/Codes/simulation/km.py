import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import detrend
from numpy.fft import rfft, rfftfreq

# --- Simulatieparameters ---
sample_rate = 100  # Hz (samples per seconde) -> Tijd tussen samples = 0.01 s
duration = 20  # seconden (totale duur van het signaal)
n_points = int(sample_rate * duration)

# Tijdvector
time = np.linspace(0, duration, n_points, endpoint=False)

# 1. Genereer het langzaam afnemende signaal (lineaire trend)
amplitude_start = 100
amplitude_end = 0  # Laat het signaal afnemen naar 0
linear_trend = np.linspace(amplitude_start, amplitude_end, n_points)

# 2. Genereer de sinus met een periode van 1 seconde
sinus_amplitude = 20  # Amplitude van de sinus
sinus_frequency = 1  # Hz (dus periode = 1/1 = 1 seconde)
sinus_signal = sinus_amplitude * np.sin(2 * np.pi * sinus_frequency * time)

# 3. Combineer de signalen
combined_signal = linear_trend + sinus_signal

# --- Periodogram Berekening (vergelijkbaar met jouw script) ---
# Parameters voor periodogram (kunnen aangepast worden)
apply_detrend = True  # Cruciaal hier vanwege de sterke lineaire trend
apply_fft_window = True  # Goede gewoonte

# Segment voor analyse (we nemen het hele signaal)
signal_for_periodogram = combined_signal.copy()
n_segment = len(signal_for_periodogram)

# Voorbewerking
if apply_detrend:
    signal_for_periodogram = detrend(signal_for_periodogram, type='linear')
    print("Signaal gedetrend voor periodogram.")

if apply_fft_window:
    if n_segment > 0:
        fft_window = np.hanning(n_segment)
        signal_for_periodogram = signal_for_periodogram * fft_window
        print("Hanning-venster toegepast.")
    else:
        print("Segmentlengte 0, Hanning-venster overgeslagen.")

# FFT berekening
fft_magnitudes = np.abs(rfft(signal_for_periodogram))
# Sample-interval 'd' is 1.0 / sample_rate voor frequenties in Hz
sample_interval = 1.0 / sample_rate
fft_frequencies_hz = rfftfreq(n_segment, d=sample_interval)

# Converteer frequenties naar periodes (in seconden)
# Vermijd deling door nul als er een DC-component (freq=0) is
valid_indices = fft_frequencies_hz > 1e-9  # Kleine epsilon
if not np.any(valid_indices):
    print("Geen valide non-zero frequenties gevonden. Kan periodes niet berekenen.")
    periods_sec = np.array([])
    magnitudes_for_periods = np.array([])
else:
    periods_sec = 1.0 / fft_frequencies_hz[valid_indices]
    magnitudes_for_periods = fft_magnitudes[valid_indices]

# --- Plotten ---
fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=False)  # sharex=False want x-assen verschillen

# Plot 1: Originele en bewerkte signalen
axs[0].plot(time, combined_signal, label='Origineel Signaal (Trend + Sinus)', alpha=0.7)
axs[0].plot(time, linear_trend, label='Lineaire Trend', linestyle='--', color='gray')
if apply_detrend:
    # Om het gedetrende signaal te plotten dat de FFT inging (na windowing):
    # Herbereken het effect van detrending op het originele signaal voor visualisatie
    # (Let op: het *signaal_voor_periodogram* is al gewindowd als dat aanstond)
    detrended_for_plot = detrend(combined_signal, type='linear')
    axs[0].plot(time, detrended_for_plot, label='Gedetrend Signaal (voor FFT input, zonder window)', linestyle=':',
                color='purple', alpha=0.7)
axs[0].set_xlabel('Tijd (seconden)')
axs[0].set_ylabel('Amplitude')
axs[0].set_title('Gesimuleerd Signaal')
axs[0].legend()
axs[0].grid(True, alpha=0.3)

# Plot 2: Periodogram
if len(periods_sec) > 0:
    # Sorteer voor een nette plot
    sort_order = np.argsort(periods_sec)
    axs[1].plot(periods_sec[sort_order], magnitudes_for_periods[sort_order], label='Periodogram Magnitude')

    # Markeer de verwachte periode van 1 seconde
    expected_period = 1.0  # seconde
    axs[1].axvline(expected_period, color='red', linestyle='--', label=f'Verwachte Periode ({expected_period} s)')

    # Zoom in op een relevant periodebereik indien nodig, bijv. 0.1s tot 5s
    axs[1].set_xlim(0.1, max(5, duration / 4))  # Laat tot een kwart van de signaalduur zien of 5s
    # axs[1].set_xscale('log') # Kan handig zijn als periodes sterk variëren

    axs[1].set_xlabel('Periode (seconden)')
    axs[1].set_ylabel('FFT Magnitude')
    title_periodogram = 'Periodogram'
    if apply_detrend: title_periodogram += ' (Gedetrend)'
    if apply_fft_window: title_periodogram += ' (Hann Window)'
    axs[1].set_title(title_periodogram)
    axs[1].legend()
    axs[1].grid(True, which="both", ls="-", alpha=0.3)
else:
    axs[1].text(0.5, 0.5, "Periodogram kon niet berekend worden.", ha='center', va='center')

plt.tight_layout()
plt.show()

# Print de sterkste periodes (optioneel)
if len(periods_sec) > 0:
    num_top_periods = 5
    top_indices = np.argsort(magnitudes_for_periods)[-num_top_periods:][::-1]
    print(f"\nTop {num_top_periods} gedetecteerde periodes (s) en hun magnitudes:")
    for idx in top_indices:
        print(f"  Periode: {periods_sec[idx]:.4f} s, Magnitude: {magnitudes_for_periods[idx]:.2f}")