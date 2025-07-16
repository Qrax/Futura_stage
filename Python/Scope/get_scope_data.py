import pyvisa
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
#  DEEL 1: VERBINDEN VIA NI-VISA (DE STANDAARD MANIER)
# =============================================================================

# We creëren een ResourceManager.
# Geen '@py' meer nodig! PyVISA zal automatisch de superieure
# NI-VISA backend vinden die nu geïnstalleerd is.
rm = pyvisa.ResourceManager()

# We kunnen de gevonden resources printen. De string hieronder zou moeten matchen.
print(f"Gevonden apparaten: {rm.list_resources()}")

# Plak hier de 'VISA Resource Name' die u in NI MAX heeft gevonden.
# Het is belangrijk dat deze exact klopt.
VISA_RESOURCE_STRING = 'USB0::0x5345::0x1234::SN-dummy::INSTR'  # <--- VERVANG DEZE STRING

# =============================================================================
#  DEEL 2: COMMUNICEER EN PLOT DE DATA
# =============================================================================

scope = None
try:
    # Open de verbinding
    scope = rm.open_resource(VISA_RESOURCE_STRING)
    scope.timeout = 10000

    # Test de verbinding
    identity = scope.query('*IDN?')
    print(f"Succesvol verbonden met: {identity.strip()}")

    # Haal de data op
    print("Data ophalen...")
    scope.write(':STOP')
    scope.write(':WAV:SOUR CHAN1')
    scope.write(':WAV:FORM BYTE')
    scope.write(':WAV:MODE NORM')

    preamble_str = scope.query(':WAV:PRE?')
    preamble = preamble_str.strip().split(',')

    points, x_increment, x_origin, y_increment, y_origin, y_reference = (
        int(preamble[2]), float(preamble[4]), float(preamble[5]),
        float(preamble[7]), float(preamble[8]), int(preamble[9])
    )

    raw_data = scope.query_binary_values(':WAV:DATA?', datatype='b', container=np.array)
    scope.write(':RUN')
    print("Data succesvol ontvangen.")

    # Converteer en plot
    voltages = (raw_data - y_origin - y_reference) * y_increment
    time = np.arange(0, points * x_increment, x_increment) + x_origin

    plt.figure(figsize=(12, 6))
    plt.plot(time * 1e3, voltages)
    plt.title("Waveform (via NI-VISA)")
    plt.xlabel("Tijd (ms)")
    plt.ylabel("Voltage (V)")
    plt.grid(True)
    plt.show()

except Exception as e:
    print(f"\nFOUT: Er is een fout opgetreden: {e}")
    print("Controleer of de VISA Resource String exact klopt met wat in NI MAX staat.")

finally:
    if scope:
        scope.close()
        print("Verbinding gesloten.")