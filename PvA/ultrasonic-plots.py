import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sg
from scipy.fft import fft, fftfreq
import matplotlib.ticker as ticker

np.random.seed(42)  # Voor reproduceerbaarheid

# Parameters for simulation
sample_count = 270
base_noise_level = 5


# Functie om het ontvangen signaal van de tweede transducer te simuleren
def create_receiver_signal(samples=sample_count, with_defect=False, defect_position=150):
    # Creëer basisruisniveau
    signal_data = np.random.normal(0, base_noise_level, samples)

    # Parameters voor de ontvangen puls
    pulse_position = 50  # Later in de tijd dan de zendpuls
    pulse_width = 15

    # Pulshoogte zonder defect is sterker
    if not with_defect:
        pulse_height = 60
        pulse_decay = 0.3
    else:
        # Met defect is de puls zwakker, maar de vorm kan ook veranderen door materiaaleigenschappen
        pulse_height = 35
        pulse_decay = 0.4  # Iets snellere demping door het defect
        pulse_position += 5  # Kleine vertraging door het defect

    # Creëer de ontvangen puls (geleidelijke opbouw, gevolgd door exponentieel verval)
    for i in range(pulse_width):
        if i < 3:
            # Opbouwfase
            signal_data[pulse_position + i] += pulse_height * (i + 1) / 3
        elif i < 5:
            # Plateau
            signal_data[pulse_position + i] += pulse_height
        else:
            # Afname
            signal_data[pulse_position + i] += pulse_height * np.exp(-pulse_decay * (i - 4))

    # Als er een defect is, kunnen we extra reflecties of veranderingen in het signaal zien
    if with_defect:
        # Extra pulsje door reflectie van defect
        echo_position = pulse_position + 30
        echo_height = 15
        echo_width = 8

        for i in range(echo_width):
            if i < 2:
                signal_data[echo_position + i] += echo_height * i / 2
            else:
                signal_data[echo_position + i] += echo_height * np.exp(-0.6 * (i - 1))

    return signal_data


# Creëer de signalen
signal_no_defect = create_receiver_signal(with_defect=False)
signal_with_defect = create_receiver_signal(with_defect=True)

# Plot 1: Tijddomein plot (ADC waarden)
plt.figure(figsize=(12, 6))
plt.plot(signal_no_defect, 'b-', linewidth=2, label='Zonder defect')
plt.plot(signal_with_defect, 'r--', linewidth=2, label='Met defect')

plt.title('Vergelijking van ultrasone signalen met en zonder defect', fontsize=16)
plt.xlabel('Sample nummer', fontsize=14)
plt.ylabel('ADC waarde', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)

# Annotatie voor hoofdeffecten
plt.annotate('Hoofdpuls (verzwakt door defect)',
             xy=(55, signal_with_defect[55]),
             xytext=(80, signal_with_defect[55] + 20),
             arrowprops=dict(facecolor='red', shrink=0.05, width=1.5),
             fontsize=10, color='red')

plt.annotate('Defect echo',
             xy=(85, signal_with_defect[85]),
             xytext=(105, signal_with_defect[85] + 15),
             arrowprops=dict(facecolor='red', shrink=0.05, width=1.5),
             fontsize=10, color='red')

plt.tight_layout()
plt.savefig('dual_transducer_tijddomein.png', dpi=300)
plt.show()


# Plot 2: Frequentiedomein analyse
def compute_fft(signal_data, fs=1e6):
    n = len(signal_data)
    # Pas window toe om spectral leakage te verminderen
    window = sg.windows.hann(n)
    windowed_signal = signal_data * window

    # Bereken FFT
    yf = fft(windowed_signal)
    xf = fftfreq(n, 1 / fs)

    # Neem alleen positieve frequenties en normaliseer
    pos_mask = xf >= 0
    xf = xf[pos_mask]
    yf = 2.0 / n * np.abs(yf[pos_mask])

    return xf, yf


# Samplefrequentie van 1 MHz voor demonstratie
fs = 1e6  # 1 MHz samplefrequentie
xf_no_defect, yf_no_defect = compute_fft(signal_no_defect, fs)
xf_with_defect, yf_with_defect = compute_fft(signal_with_defect, fs)

# Converteer Hz naar kHz voor weergave
xf_no_defect_khz = xf_no_defect / 1000
xf_with_defect_khz = xf_with_defect / 1000

plt.figure(figsize=(12, 6))
plt.plot(xf_no_defect_khz, yf_no_defect, 'b-', linewidth=2, label='Zonder defect')
plt.plot(xf_with_defect_khz, yf_with_defect, 'r--', linewidth=2, label='Met defect')

plt.title('Frequentiespectrum van ultrasone signalen met en zonder defect', fontsize=16)
plt.xlabel('Frequentie (kHz)', fontsize=14)
plt.ylabel('Amplitude', fontsize=14)
plt.xlim(0, 100)  # Plot van 0 tot 100 kHz
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)

# Annotatie voor frequentieverschuiving
peak_freq_no_defect = xf_no_defect_khz[np.argmax(yf_no_defect[:100])]
peak_freq_with_defect = xf_with_defect_khz[np.argmax(yf_with_defect[:100])]

plt.annotate(f'Hoofdpiek: {peak_freq_no_defect:.1f} kHz',
             xy=(peak_freq_no_defect, np.max(yf_no_defect[:100])),
             xytext=(peak_freq_no_defect + 10, np.max(yf_no_defect[:100])),
             arrowprops=dict(facecolor='blue', shrink=0.05, width=1.5),
             fontsize=10, color='blue')

plt.annotate(f'Verschoven piek: {peak_freq_with_defect:.1f} kHz',
             xy=(peak_freq_with_defect, np.max(yf_with_defect[:100])),
             xytext=(peak_freq_with_defect - 20, np.max(yf_with_defect[:100]) - 0.5),
             arrowprops=dict(facecolor='red', shrink=0.05, width=1.5),
             fontsize=10, color='red')

plt.tight_layout()
plt.savefig('dual_transducer_frequentiedomein.png', dpi=300)
plt.show()

# Plot 3: Histogram van meerdere metingen
# Simuleer variaties in pulshoogte en -vertraging
n_measurements = 100
no_defect_amplitude = []
with_defect_amplitude = []
no_defect_delay = []
with_defect_delay = []

for _ in range(n_measurements):
    # Variaties in pulshoogte
    no_defect_amp = np.random.normal(60, 5)
    no_defect_amplitude.append(no_defect_amp)

    with_defect_amp = np.random.normal(35, 8)
    with_defect_amplitude.append(with_defect_amp)

    # Variaties in pulsvertraging
    no_defect_d = np.random.normal(50, 2)
    no_defect_delay.append(no_defect_d)

    with_defect_d = np.random.normal(55, 3)
    with_defect_delay.append(with_defect_d)

# Histogram voor pulshoogte
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist([no_defect_amplitude, with_defect_amplitude], bins=20,
         color=['blue', 'red'], alpha=0.7,
         label=['Zonder defect', 'Met defect'])

plt.title('Histogram van pulshoogte', fontsize=14)
plt.xlabel('Pulshoogte (ADC waarde)', fontsize=12)
plt.ylabel('Aantal metingen', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)

# Histogram voor pulsvertraging
plt.subplot(1, 2, 2)
plt.hist([no_defect_delay, with_defect_delay], bins=20,
         color=['blue', 'red'], alpha=0.7,
         label=['Zonder defect', 'Met defect'])

plt.title('Histogram van pulsvertraging', fontsize=14)
plt.xlabel('Vertraging (sample nummer)', fontsize=12)
plt.ylabel('Aantal metingen', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)

plt.tight_layout()
plt.savefig('dual_transducer_histogrammen.png', dpi=300)
plt.show()

# Plot 4: Gecombineerde visualisatie
plt.figure(figsize=(18, 12))

# Tijddomein
plt.subplot(2, 2, 1)
plt.plot(signal_no_defect, 'b-', linewidth=2, label='Zonder defect')
plt.plot(signal_with_defect, 'r--', linewidth=2, label='Met defect')
plt.title('A-Scan: ADC waarden', fontsize=14)
plt.xlabel('Sample nummer', fontsize=12)
plt.ylabel('ADC waarde', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)

# Frequentiedomein
plt.subplot(2, 2, 2)
plt.plot(xf_no_defect_khz, yf_no_defect, 'b-', linewidth=2, label='Zonder defect')
plt.plot(xf_with_defect_khz, yf_with_defect, 'r--', linewidth=2, label='Met defect')
plt.title('FFT: Frequentiedomein analyse', fontsize=14)
plt.xlabel('Frequentie (kHz)', fontsize=12)
plt.ylabel('Amplitude', fontsize=12)
plt.xlim(0, 100)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)

# Histogram voor pulshoogte
plt.subplot(2, 2, 3)
plt.hist([no_defect_amplitude, with_defect_amplitude], bins=20,
         color=['blue', 'red'], alpha=0.7,
         label=['Zonder defect', 'Met defect'])
plt.title('Histogram: Pulshoogte', fontsize=14)
plt.xlabel('Pulshoogte (ADC waarde)', fontsize=12)
plt.ylabel('Aantal metingen', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)

# Uitvergroting van interessant gebied
plt.subplot(2, 2, 4)
zoom_start = 40
zoom_end = 100
plt.plot(range(zoom_start, zoom_end), signal_no_defect[zoom_start:zoom_end], 'b-', linewidth=2, label='Zonder defect')
plt.plot(range(zoom_start, zoom_end), signal_with_defect[zoom_start:zoom_end], 'r--', linewidth=2, label='Met defect')
plt.title('Uitvergroting van pulsbereik', fontsize=14)
plt.xlabel('Sample nummer', fontsize=12)
plt.ylabel('ADC waarde', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)

plt.suptitle('Ultrasoondetectie van defecten in composietmateriaal (Dual-Transducer)', fontsize=18)
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Maak ruimte voor hoofdtitel
plt.savefig('dual_transducer_overzicht.png', dpi=300)
plt.show()

# Plot 5: Visualisatie van systeemgevoeligheid voor detectie
# Simuleer reeks metingen met toenemende defectgrootte
defect_sizes = np.linspace(0, 1, 20)  # Van 0 (geen defect) tot 1 (ernstig defect)
amplitude_response = []
delay_response = []
frequency_shift = []

for size in defect_sizes:
    # Pulshoogte neemt af met defectgrootte
    amp = 60 - 30 * size + np.random.normal(0, 3)
    amplitude_response.append(amp)

    # Vertraging neemt toe met defectgrootte
    delay = 50 + 10 * size + np.random.normal(0, 1)
    delay_response.append(delay)

    # Frequentieverschuiving met defectgrootte
    freq = 40 - 5 * size + np.random.normal(0, 0.5)
    frequency_shift.append(freq)

plt.figure(figsize=(15, 6))
plt.subplot(1, 3, 1)
plt.plot(defect_sizes, amplitude_response, 'bo-')
plt.title('Pulshoogte vs. defectgrootte', fontsize=14)
plt.xlabel('Relatieve defectgrootte', fontsize=12)
plt.ylabel('Pulshoogte (ADC waarde)', fontsize=12)
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.plot(defect_sizes, delay_response, 'ro-')
plt.title('Pulsvertraging vs. defectgrootte', fontsize=14)
plt.xlabel('Relatieve defectgrootte', fontsize=12)
plt.ylabel('Vertraging (sample nummer)', fontsize=12)
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
plt.plot(defect_sizes, frequency_shift, 'go-')
plt.title('Piekfrequentie vs. defectgrootte', fontsize=14)
plt.xlabel('Relatieve defectgrootte', fontsize=12)
plt.ylabel('Frequentie (kHz)', fontsize=12)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('dual_transducer_sensitivity.png', dpi=300)
plt.show()

# Plot 6: Detectiedrempelwaarde visualisatie met SNR
plt.figure(figsize=(12, 6))

# Definieer detectiedrempel gebaseerd op signaal-ruisverhouding (SNR)
noise_floor = np.mean(np.abs(signal_no_defect[0:40]))  # Ruisniveau bepalen van vroege samples
snr_threshold = 3  # Signaal moet minstens 3x ruisniveau zijn voor betrouwbare detectie
detection_threshold = noise_floor * snr_threshold

# Plot signalen
plt.plot(signal_no_defect, 'b-', linewidth=2, label='Zonder defect')
plt.plot(signal_with_defect, 'r--', linewidth=2, label='Met defect')

# Plot drempelwaarde
plt.axhline(y=detection_threshold, color='g', linestyle='-', linewidth=2,
            label=f'Detectiedrempel (SNR={snr_threshold})')

# Markeer gebied boven drempelwaarde
for i in range(len(signal_with_defect)):
    if signal_with_defect[i] > detection_threshold and signal_with_defect[i] > signal_no_defect[i]:
        plt.fill_between([i - 0.5, i + 0.5], [detection_threshold, detection_threshold],
                         [signal_with_defect[i], signal_with_defect[i]],
                         color='yellow', alpha=0.5)

plt.title('Defectdetectie met signaal-ruisverhouding (SNR) drempel', fontsize=16)
plt.xlabel('Sample nummer', fontsize=14)
plt.ylabel('ADC waarde', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)

plt.tight_layout()
plt.savefig('dual_transducer_snr_detectie.png', dpi=300)
plt.show()