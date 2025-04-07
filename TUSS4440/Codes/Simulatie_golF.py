import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# --- Jouw Ruwe ADC Data (Volledige meting) ---
raw_adc_data = np.array([
    111, 130, 143, 154, 165, 172, 179, 186, 192, 196, 201, 205, 208, 209, 212,
    214, 215, 216, 217, 218, 218, 219, 220, 221, 219, 221, 221, 222, 220, 220,
    222, 220, 220, 221, 220, 220, 220, 219, 217, 218, 215, 214, 213, 211, 209,
    206, 204, 202, 199, 197, 194, 192, 190, 186, 184, 181, 178, 175, 172, 169,
    166, 165, 161, 159, 156, 153, 150, 148, 144, 142, 137, 136, 134, 129, 128,
    123, 121, 118, 116, 113, 111, 108, 106, 103, 101, 99, 96, 94, 91, 88, 85,
    82, 79, 74, 70, 68, 65, 60, 56, 51, 48, 47, 47, 47, 46, 46, 45, 45, 42, 41,
    40, 37, 36, 36, 36, 36, 36, 37, 39, 39, 41, 41, 40, 38, 37, 35, 35, 37, 38,
    39, 41, 41, 41, 41, 39, 37, 37, 38, 38, 37, 38, 38, 37, 37, 38, 36, 35, 35,
    35, 33, 35, 37, 38, 38, 39, 39, 39, 39, 38, 37, 37, 36, 37, 40, 41, 42, 43,
    46, 46, 47, 46, 45, 43, 40, 41, 39, 38, 38, 39, 41, 41, 41, 41, 43, 43, 42,
    41, 41, 42, 41, 41, 42, 44, 45, 46, 46, 45, 44, 42, 40, 39, 41, 41, 42, 42,
    42, 43, 43, 43, 43, 46, 47, 47, 46, 47, 46, 45, 43, 44, 43, 42, 40, 38, 38,
    36, 36, 36, 38, 38, 38, 39, 40, 39, 39, 40, 42, 42, 40, 39, 41, 41, 39, 38,
    37, 36, 37, 36, 37, 39, 40, 40, 42, 41, 41, 40, 37, 35, 35, 35, 34, 32, 32,
    33, 34, 36, 36, 37, 38, 41, 41, 41, 42, 43, 43, 42, 40, 39, 40, 40, 40, 40,
    38, 36, 35, 35, 36, 38, 40, 41, 42, 46, 47, 50, 50, 50, 51, 51, 50, 48, 48,
    48, 47, 45, 44, 43, 41, 39, 40, 41, 40, 40, 41, 40, 39, 38, 40, 40, 42, 43,
    44, 45, 44, 43, 42, 41, 39, 38, 39, 40, 42, 42, 43, 41, 40, 37, 35, 33, 33,
    33, 34, 35, 35, 37, 39, 39, 40, 42, 41, 41, 42, 43, 44, 44, 45, 46, 47, 46,
    48, 50, 51, 50, 49, 48, 46, 44, 42, 44, 44, 44, 43, 43, 42, 44, 44, 43, 41,
    39, 39, 38, 38, 39, 39, 38, 35, 35, 35, 34, 35, 34, 34, 36, 37, 37, 39, 39,
    37, 37, 37, 37, 38, 40, 40, 39, 38, 40, 41, 42, 40, 40, 39, 37, 37, 36, 35,
    34, 34, 33, 34, 34, 33, 32, 31, 31, 31, 32, 30, 32, 32, 34, 34, 35, 35, 35,
    36, 36, 35, 33, 33, 33, 35, 37, 37, 37, 39, 38, 37, 37, 38, 37, 37, 38, 39,
    38, 36, 37, 36, 36, 38, 39, 38, 37, 39, 42, 42, 40, 40, 40, 41, 40, 39, 38,
    37, 38, 38, 38, 36, 36, 38, 38, 36, 37, 37, 36, 35, 37, 39, 38, 37, 39, 40,
    40, 39, 38, 36, 36, 37, 37, 38, 36, 39, 41, 43, 43, 42, 42, 41, 41, 40, 39,
    38, 38, 38, 39, 41, 41, 42, 39, 37, 34, 33, 32, 31, 29, 31, 31, 31, 34, 33,
    32, 32, 32, 33, 33, 32, 35, 37, 39, 38, 37, 37, 36, 34, 32, 32, 35, 34, 33,
    33, 32, 33, 34, 35, 36, 36, 37, 38, 37, 35, 35, 36, 37, 37, 38, 40, 42, 42,
    43, 42, 41, 39, 36, 35, 35, 33, 33, 33, 34, 35, 36, 34, 34, 37, 39, 40, 39,
    37, 36, 36, 37, 39, 39, 39, 37, 35, 34, 34, 35, 36, 36, 36, 37, 36, 35, 35,
    36, 38, 40, 39, 38, 39, 39, 38, 36, 34, 35, 38, 41, 43, 44, 46, 50, 54, 56,
    58, 61, 62, 63, 64, 64, 65, 65, 64, 64, 64, 63, 63, 61, 59, 56, 55, 53, 54,
    53, 51, 49, 48, 47, 46, 44, 44, 44, 43, 41, 38, 36, 37, 37, 39, 41, 42, 40,
    41, 43, 43, 40, 40, 42, 43, 41, 40, 40, 40, 38, 39, 39, 38, 36, 35, 35, 33,
    34, 33, 31, 32, 31, 31, 31, 30, 29, 28, 28, 28, 27, 27, 27, 29, 31, 30, 30,
    31, 33, 33, 33, 32, 33, 34, 34, 32, 31, 31, 30, 29, 30, 31, 31, 31, 32, 32,
    31, 32, 34, 33, 34, 34, 35, 36, 36, 34, 34, 33, 33, 32, 31, 30, 33, 32, 32,
    31, 31, 32, 32, 31, 31, 30, 29, 30, 31, 34, 36, 36, 35, 34, 34, 34, 35, 34,
    34, 35, 36, 37, 37, 37, 36, 35, 34, 35, 36, 37, 38, 39, 40, 40, 38, 36, 36,
    37, 36, 35, 33, 32, 32, 32, 33, 34, 36, 39, 40, 40, 39, 39, 40, 39, 40, 42,
    43, 41, 39, 39, 37, 36, 36, 34, 35, 36, 38, 41, 41, 41, 42, 43, 43, 42, 41,
    42, 41, 40, 39, 39, 40, 39, 36, 35, 33, 34, 35, 36, 34, 35, 35, 34, 33, 34,
    37, 39, 39, 37, 39, 38, 37, 36, 37, 40, 40, 38, 38, 36, 35, 33, 32, 32, 32,
    33, 32, 32, 33, 34, 35, 34, 36, 36, 37, 36, 36, 37, 39, 40, 39, 37, 37, 36,
    35, 37, 38, 38, 39, 40, 42, 41, 39, 38, 36, 36, 37, 36, 36, 36, 37, 36, 35,
    35, 34, 35, 34, 34, 34, 37, 37, 39, 41, 42, 42, 39, 39, 38, 38, 36, 36, 35,
    34, 34, 34, 33, 33, 32, 32, 31, 29, 30, 31, 33, 33, 34, 35, 37, 36, 36, 38,
    39, 38, 37, 37, 37, 40, 40, 40, 41, 42, 42, 42, 41, 39, 39, 39, 38, 39, 38,
    38, 40, 41, 41, 40, 41, 41, 40, 41, 40, 40, 38, 38, 38, 37, 37, 36, 35, 35,
    34, 34, 34, 34, 35, 36, 37, 36, 35, 33, 32, 32, 32, 31, 31, 30, 30, 32, 33,
    34, 35, 35, 36, 37, 39, 39, 38, 36, 36, 37, 39, 38, 38, 38, 39, 38, 37, 35,
    34, 34, 33, 34, 33, 33, 33, 34, 33, 32, 32, 34, 36, 37, 40, 41, 41, 42, 43,
    43, 43
])

# --- Parameters ---
time_between_samples_us = 15 # µs
time_between_samples_s = time_between_samples_us / 1_000_000 # s
carrier_freq = 40000 # Hz
total_samples = len(raw_adc_data)
total_duration_s = total_samples * time_between_samples_s
t_full = np.linspace(0, total_duration_s, total_samples, endpoint=False)

# --- Fase 1: Plot Originele Volledige Data ---
plt.figure(figsize=(12, 15)) # Maak figure groter
plt.subplot(4, 1, 1) # Aangepast naar 4 subplots
plt.plot(t_full * 1000, raw_adc_data)
plt.title("Fase 1: Volledige Ontvangen Data (VOUT Bord B)")
plt.xlabel("Tijd (ms)")
plt.ylabel("ADC Waarde")
plt.grid(True)

# --- Fase 3: Keer het *volledige* signaal om in tijd ---
time_reversed_full_signal = raw_adc_data[::-1]

# --- Fase 4: Plot de Tijd-Omgekeerde Envelope (DAC Output Simulatie) ---
plt.subplot(4, 1, 2)
plt.plot(t_full * 1000, time_reversed_full_signal, 'r-') # Gebruik t_full als tijd-as
plt.title("Fase 4: Tijd-Omgekeerde Volledige Envelope (Gesimuleerde DAC Output)")
plt.xlabel("Tijd (ms)")
plt.ylabel("ADC Waarde")
plt.grid(True)

# --- Fase 5: Simuleer Modulatie van het Volledige Tijd-Omgekeerde Signaal ---
# Hoge sample rate nodig voor visualisatie 40kHz
sim_sampling_rate = 1000000 # 1 MHz simulatie sample rate
t_sim_fine = np.linspace(0, total_duration_s, int(sim_sampling_rate * total_duration_s), endpoint=False)

# Interpoleer de tijd-omgekeerde envelope naar de fijnere tijd-as
time_reversed_envelope_fine = np.interp(t_sim_fine, t_full, time_reversed_full_signal)

# Normaliseer envelope (ADC waarden naar 0-1 range, gebaseerd op PIEK)
max_adc = 255 # Aanname 8-bit ADC
# Of gebruik de max waarde uit de data voor betere schaling
# max_adc = np.max(raw_adc_data)
# baseline_adc = 35 # Geschatte baseline
normalized_envelope = time_reversed_envelope_fine / max_adc # Simpele normalisatie op max ADC
normalized_envelope[normalized_envelope < 0] = 0 # Ondergrens
normalized_envelope[normalized_envelope > 1] = 1 # Bovengrens

# Genereer draaggolf
carrier_wave = 0.5 * np.sin(2 * np.pi * carrier_freq * t_sim_fine) + 0.5

# Moduleren
modulated_signal = normalized_envelope * carrier_wave

plt.subplot(4, 1, 3)
plt.plot(t_sim_fine * 1000, modulated_signal)
# Plot de envelope eroverheen om de modulatie te zien
#plt.plot(t_sim_fine * 1000, normalized_envelope, 'k--', alpha=0.5, label='Norm. Envelope')
plt.title("Fase 5: Gesimuleerd Gemoduleerd 40kHz Signaal (met Tijd-Omgekeerde Volledige Envelope)")
plt.xlabel("Tijd (ms)")
plt.ylabel("Amplitude (gesim.)")
#plt.legend()
plt.grid(True)


# --- Fase 8: Simuleer Ontvangst op Bord A ---
# Dit is een *vereenvoudigde* weergave.
# We tonen het directe artefact (begin van originele data)
# GEVOLGD door het tijd-omgekeerde signaal (na ToF)

# Schatting artefact duur (visueel, tot het signaal laag wordt, bv tot sample 100)
artifact_length = 100
transmit_artifact_shape = raw_adc_data[:artifact_length]

# Schatting ToF (ruw: piek echo - piek artefact)
artifact_peak_idx = np.argmax(raw_adc_data[:artifact_length]) # Index v/d piek in artefact
# Vind de echo piek NA het artefact
echo_search_start = artifact_length + 50 # Begin met zoeken na het artefact + marge
echo_peak_idx = echo_search_start + np.argmax(raw_adc_data[echo_search_start:])
tof_samples = echo_peak_idx - artifact_peak_idx
print(f"Geschatte ToF: {tof_samples} samples = {tof_samples * time_between_samples_us} µs")

# Maak het ontvangen signaal array
vout_A = np.ones(total_samples) * np.mean(raw_adc_data[-100:]) # Start met baseline ruis (gemiddelde laatste 100)

# Plaats artefact aan het begin
vout_A[:artifact_length] = transmit_artifact_shape

# Plaats het tijd-omgekeerde signaal na de ToF
start_idx_reversed = tof_samples # Begin na round trip tijd
end_idx_reversed = start_idx_reversed + total_samples

# Zorg dat we niet buiten de array grenzen gaan bij het plakken
len_to_paste = min(total_samples, total_samples - start_idx_reversed)
if len_to_paste > 0:
     # Voeg ruis toe aan het omgekeerde signaal voor realisme
    noise = np.random.normal(0, 5, time_reversed_full_signal[:len_to_paste].shape)
    signal_to_paste = time_reversed_full_signal[:len_to_paste] + noise
    signal_to_paste[signal_to_paste < 0] = 0 # Clip noise
    vout_A[start_idx_reversed : start_idx_reversed + len_to_paste] += signal_to_paste - np.mean(raw_adc_data[-100:]) # Voeg signaal toe relatief aan baseline

# Clip eventuele negatieve waarden door ruis
vout_A[vout_A < 0] = 0

plt.subplot(4, 1, 4)
plt.plot(t_full * 1000, vout_A, 'm-')
plt.title("Fase 8: Gesimuleerde Ontvangst op VOUT Bord A (Artefact + Time-Reversed Signal)")
plt.xlabel("Tijd (ms)")
plt.ylabel("ADC Waarde (gesim.)")
plt.grid(True)


plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Voorkom overlap titels
plt.suptitle("Simulatie Tijd-Omkering met Volledige Data", fontsize=16)
plt.show()