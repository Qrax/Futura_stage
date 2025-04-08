import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import matplotlib
matplotlib.use('TkAgg')

# --- Simulatie Parameters ---
fs_sim = 2000000 # Hoge sample rate (2 MHz) voor nauwkeurige 40kHz simulatie
carrier_freq = 40000 # 40 kHz
num_pulses = 8     # Aantal pulsen (uit BURST_PULSE register)
vdrv_voltage = 12.0 # Voorbeeld VDRV spanning (uit VDRV_CTRL register)
half_bridge_mode = False # Normale mode (uit HALF_BRG_MODE bit)

# Bereken burst duur
pulse_period_s = 1.0 / carrier_freq
burst_duration_s = num_pulses * pulse_period_s

# Tijd-as voor simulatie (iets langer dan de burst)
total_duration_s = burst_duration_s * 1.5
t_sim = np.linspace(0, total_duration_s, int(fs_sim * total_duration_s), endpoint=False)

# --- STAP 3 & 4: Simuleer OUTA en OUTB signalen ---
# Maak een 40kHz blokgolf (50% duty cycle)
# Signal is 'actief' (laag, schakelt FET aan) tijdens de puls
# Gebruik scipy.signal.square: amplitude 1, dus * 0.5 + 0.5 -> 0 tot 1
# We inverteren het zodat het 0 is tijdens 'aan', 1 tijdens 'uit' (actief laag)
# En schalen naar VDRV spanning
square_wave_40k = 0.5 * signal.square(2 * np.pi * carrier_freq * t_sim, duty=0.5) + 0.5
# Maak een 'masker' dat alleen hoog is tijdens de burst
burst_mask = (t_sim >= 0) & (t_sim < burst_duration_s)

# OUTA signaal (actief = 0V, inactief = VDRV)
signal_outa = np.ones_like(t_sim) * vdrv_voltage # Beginnen op VDRV
signal_outa[burst_mask] = (1 - square_wave_40k[burst_mask]) * vdrv_voltage # 0V tijdens actieve deel burst

# OUTB signaal
if half_bridge_mode:
    # In phase: OUTB doet hetzelfde als OUTA
    signal_outb = signal_outa.copy()
else:
    # Out of phase (complementair): OUTB is omgekeerde van OUTA tijdens burst
    signal_outb = np.ones_like(t_sim) * vdrv_voltage # Beginnen op VDRV
    signal_outb[burst_mask] = square_wave_40k[burst_mask] * vdrv_voltage # 0V als OUTA VDRV is etc.

# --- Plotting ---
plt.figure(figsize=(12, 15))

# Plot OUTA
plt.subplot(5, 1, 1)
plt.plot(t_sim * 1000, signal_outa)
plt.title("Stap 4a: Elektrisch Signaal op OUTA Pin")
plt.xlabel("Tijd (ms)")
plt.ylabel("Spanning (V)")
plt.ylim(-1, vdrv_voltage * 1.1)
plt.grid(True)

# Plot OUTB
plt.subplot(5, 1, 2)
plt.plot(t_sim * 1000, signal_outb)
plt.title(f"Stap 4b: Elektrisch Signaal op OUTB Pin ({'In Phase' if half_bridge_mode else 'Out of Phase'})")
plt.xlabel("Tijd (ms)")
plt.ylabel("Spanning (V)")
plt.ylim(-1, vdrv_voltage * 1.1)
plt.grid(True)

# --- STAP 5a: Simuleer Differentieel Signaal (voor Center-Tap Transformer) ---
# Dit is het signaal over de primaire wikkeling(en)
if not half_bridge_mode:
    signal_diff_primary = signal_outa - signal_outb
    plt.subplot(5, 1, 3)
    plt.plot(t_sim * 1000, signal_diff_primary)
    plt.title("Stap 5a (Optie A): Differentieel Signaal (OUTA - OUTB) voor Primaire Wikkeling")
    plt.xlabel("Tijd (ms)")
    plt.ylabel("Spanning (V)")
    plt.ylim(-vdrv_voltage * 1.1, vdrv_voltage * 1.1)
    plt.grid(True)
else:
     plt.subplot(5, 1, 3)
     plt.text(0.5, 0.5, 'Differentieel signaal niet relevant\nin Half-Bridge Mode', horizontalalignment='center', verticalalignment='center')
     plt.title("Stap 5a (Optie B): Half-Bridge Mode")

# --- STAP 5b: Simuleer Signaal over de Transducer (na transformator) ---
# Dit is conceptueel. We nemen aan dat de transformator ideaal is
# en de spanning verhoogt (turns_ratio > 1)
turns_ratio = 8.42 # Voorbeeld ratio (zoals in jouw project info)
if not half_bridge_mode:
    signal_at_transducer = signal_diff_primary * turns_ratio
else:
    # In half-bridge mode drijf je misschien direct tussen VDRV en OUTA/B
    # Voorbeeld: neem OUTA als referentie
    signal_at_transducer = (vdrv_voltage - signal_outa) * turns_ratio

plt.subplot(5, 1, 4)
plt.plot(t_sim * 1000, signal_at_transducer)
plt.title(f"Stap 5b: Gesimuleerd Elektrisch Signaal óver de Transducer (Turns Ratio ≈ {turns_ratio:.2f})")
plt.xlabel("Tijd (ms)")
plt.ylabel("Spanning (V)")
# Pas Y limiet aan voor hogere spanning
plt.ylim(-vdrv_voltage * turns_ratio * 1.1, vdrv_voltage * turns_ratio * 1.1)
plt.grid(True)

# --- STAP 6: Conceptuele Akoestische Puls ---
# We plotten de *envelope* van de burst om het geluid te representeren
acoustic_envelope = np.zeros_like(t_sim)
acoustic_envelope[burst_mask] = 1 # Simpelweg AAN tijdens de burst

plt.subplot(5, 1, 5)
plt.plot(t_sim * 1000, acoustic_envelope, 'g-')
plt.title("Stap 6: Conceptuele Ultrasone Geluidspuls (Envelope)")
plt.xlabel("Tijd (ms)")
plt.ylabel("Geluidsdruk (Aan/Uit)")
plt.ylim(-0.1, 1.1)
plt.grid(True)


plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.suptitle(f"Simulatie TUSS4440 Normale Zendpuls ({num_pulses} pulsen @ {carrier_freq/1000}kHz)", fontsize=16)
plt.show()