import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import matplotlib
matplotlib.use('TkAgg') # Kan nodig zijn

# --- Simulatie Parameters ---
fs_sim = 1000000 # Hoge sample rate (1 MHz) voor nauwkeurige 40kHz simulatie
duration = 0.0015 # Simuleer 1.5 ms
t_sim = np.linspace(0, duration, int(fs_sim * duration), endpoint=False)
carrier_freq = 40000 # 40 kHz

# --- STAP 1 & 2: Simuleer het zwakke 40kHz signaal van de transducer op INN/INP ---
# Maak een echo envelope (een gaussische puls als voorbeeld)
echo_delay = 0.0005 # Echo komt na 0.5 ms
echo_width_time = 0.0001 # Duur van de echo
echo_center = echo_delay + echo_width_time / 2
echo_envelope_sim = np.exp(-((t_sim - echo_center)**2) / (2 * (echo_width_time/4)**2))
# Maak het signaal erg zwak (millivolt range)
echo_amplitude_mV = 2.0 # Piek van 2 mV
echo_envelope_scaled = echo_envelope_sim * (echo_amplitude_mV / 1000.0) # Schaal naar Volt

# Genereer 40kHz draaggolf
carrier_wave = np.sin(2 * np.pi * carrier_freq * t_sim)

# Het signaal zoals het bij INN/INP aankomt
signal_at_inp_inn = echo_envelope_scaled * carrier_wave
# Voeg wat achtergrondruis toe
noise_std_dev = 0.0001 # 0.1 mV ruis standaard deviatie
signal_at_inp_inn += np.random.normal(0, noise_std_dev, signal_at_inp_inn.shape)


# --- Plotting Setup ---
plt.figure(figsize=(10, 15)) # Aangepast formaat voor 5 plots

# --- Plot STAP 3: Signaal op INN/INP ---
plt.subplot(5, 1, 1)
plt.plot(t_sim * 1000, signal_at_inp_inn * 1000) # Plot in mV
plt.title("(Stap 1-3) Gesimuleerd Elektrisch Signaal op INN/INP (Zwakke 40kHz Echo + Ruis)")
plt.xlabel("Tijd (ms)")
plt.ylabel("Amplitude (mV)")
plt.ylim(-echo_amplitude_mV*1.5, echo_amplitude_mV*1.5) # Zoom in op Y-as
plt.grid(True)

# --- STAP 4: Simuleer LNA ---
lna_gain = 15.0 # Typische gain (V/V), instelbaar via LNA_GAIN
signal_after_lna = signal_at_inp_inn * lna_gain
# LNA voegt ook wat ruis toe (iets verhoogd)
signal_after_lna += np.random.normal(0, noise_std_dev * 2, signal_after_lna.shape)

plt.subplot(5, 1, 2)
plt.plot(t_sim * 1000, signal_after_lna * 1000) # Plot in mV
plt.title(f"(Stap 4) Signaal NA LNA (Gain ≈ {lna_gain:.1f})")
plt.xlabel("Tijd (ms)")
plt.ylabel("Amplitude (mV)")
# Y limiet geschaald met gain
plt.ylim(-echo_amplitude_mV*lna_gain*1.5, echo_amplitude_mV*lna_gain*1.5)
plt.grid(True)

# --- STAP 5: Simuleer Band Pass Filter (BPF) ---
f_center = 40000
bw = 10000 # Bandbreedte = 10kHz (Q=4 bij 40kHz)
low_cutoff = f_center - bw/2
high_cutoff = f_center + bw/2
nyquist_sim = fs_sim / 2
low = low_cutoff / nyquist_sim
high = high_cutoff / nyquist_sim
order = 4 # Orde van het Butterworth filter
try:
    b, a = signal.butter(order, [low, high], btype='band')
    signal_after_bpf = signal.filtfilt(b, a, signal_after_lna)
except ValueError as e:
    print(f"BPF Error: {e}")
    signal_after_bpf = signal_after_lna # Sla filter over bij fout

plt.subplot(5, 1, 3)
plt.plot(t_sim * 1000, signal_after_bpf * 1000) # Plot in mV
plt.title("(Stap 5) Signaal NA Band Pass Filter (Rond 40kHz)")
plt.xlabel("Tijd (ms)")
plt.ylabel("Amplitude (mV)")
plt.ylim(-echo_amplitude_mV*lna_gain*1.5, echo_amplitude_mV*lna_gain*1.5) # Behoud Y-schaal
plt.grid(True)

# --- STAP 6 & 7: Simuleer Log Amp & Demodulator (Resulteert in VOUT) ---
# De LogAmp en Demodulator samen produceren de VOUT envelope.
# We simuleren dit door de envelope te nemen van het signaal NA BPF
# en daar een log-achtige compressie op toe te passen.

# 1. Bereken envelope na BPF
analytic_signal_bpf = signal.hilbert(signal_after_bpf)
envelope_after_bpf = np.abs(analytic_signal_bpf)

# 2. Pas log-achtige compressie toe op de envelope
#    Dit is een vereenvoudiging! Echte Log Amps zijn complexer.
log_offset = 1e-9 # Voorkom log(0)
# Normaliseer input t.o.v. het geschatte ruisniveau na LNA voor log functie
noise_after_lna_approx_mV = (noise_std_dev * 2 * 1000)
compressed_envelope = np.log10( (envelope_after_bpf*1000) / noise_after_lna_approx_mV + log_offset )
compressed_envelope[compressed_envelope < 0] = 0 # Log kan negatief worden voor input < ruis

# 3. Schaal resultaat naar ADC range (0-255), met een baseline/pedestal
#    Dit bootst de Demodulator + Output Buffer na.
min_log_val = np.min(compressed_envelope)
max_log_val = np.max(compressed_envelope)
pedestal = 35 # Geschatte ADC waarde voor baseline/ruis (uit jouw data)
peak_val_sim = 65 # Geschatte ADC piekwaarde voor de zwakke echo (uit jouw data)
if max_log_val > min_log_val:
     simulated_vout = pedestal + (compressed_envelope - min_log_val) * (peak_val_sim - pedestal) / (max_log_val - min_log_val)
else:
     simulated_vout = np.ones_like(compressed_envelope) * pedestal

simulated_vout[simulated_vout < 0] = 0 # Ondergrens
simulated_vout[simulated_vout > 255] = 255 # Bovengrens

plt.subplot(5, 1, 4)
plt.plot(t_sim * 1000, simulated_vout)
plt.title("(Stap 6&7) Gesimuleerde VOUT Envelope (Na Log Amp & Demodulator)")
plt.xlabel("Tijd (ms)")
plt.ylabel("Geschatte VOUT (ADC)")
plt.ylim(0, 260) # Zet Y-limiet op ADC range
plt.grid(True)

# --- STAP 8 & 9: Simuleer ADC Sampling ---
# Neem samples van de gesimuleerde VOUT op de ADC sample rate
time_between_samples_s_adc = 15e-6 # Jouw 15 µs
fs_adc_sim = 1 / time_between_samples_s_adc
num_adc_samples = int(duration * fs_adc_sim)
t_adc_sim = np.linspace(0, duration, num_adc_samples, endpoint=False)

# Interpoleer VOUT naar ADC tijdstippen
digital_data_msp = np.interp(t_adc_sim, t_sim, simulated_vout).astype(int)

plt.subplot(5, 1, 5)
plt.plot(t_adc_sim * 1000, digital_data_msp, 'o-', markersize=3)
plt.title("(Stap 8&9) Gesimuleerde Digitale Data in MSP430 (Na ADC)")
plt.xlabel("Tijd (ms)")
plt.ylabel("Digitale ADC Waarde")
plt.ylim(0, 260) # Zet Y-limiet op ADC range
plt.grid(True)


plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.suptitle("Simulatie Interne TUSS4440 Ontvangst Signaalverwerking", fontsize=16)
plt.show()