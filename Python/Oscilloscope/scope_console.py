# -*- coding: utf-8 -*-
import pyvisa

# --- CONFIGURATIE ---
SCOPE_ADDRESS = 'USB0::0x5345::0x1234::2306226::RAW'
TIMEOUT = 5000  # Timeout in milliseconden (5 seconden)

# --- VERBINDING MAKEN ---
try:
    rm = pyvisa.ResourceManager()
    scope = rm.open_resource(SCOPE_ADDRESS)
    scope.timeout = TIMEOUT
    # Controleer of we verbonden zijn
    print(f"Verbonden met: {scope.query('*IDN?')}")
    print("Type een SCPI commando en druk op Enter. Type 'exit' om te stoppen.")
    print("------------------------------------------------------------------")
except Exception as e:
    print(f"FATALE FOUT: Kon geen verbinding maken. Foutmelding: {e}")
    exit()

# --- INTERACTIEVE TERMINAL LOOP ---
while True:
    try:
        # Vraag de gebruiker om input
        command = input("SCPI > ")

        # Stop de loop als de gebruiker 'exit' typt
        if command.lower() == 'exit':
            break

        # Controleer of het een query is (eindigt op '?')
        if '?' in command:
            # Stuur de query en print het antwoord
            response = scope.query(command)
            print(f"Antwoord: {response.strip()}")
        else:
            # Stuur het commando (dit geeft geen antwoord terug)
            scope.write(command)
            print("Commando verzonden.")

    except pyvisa.errors.VisaIOError as e:
        # Vang specifiek de timeout fout op en geef een duidelijke melding
        print(f"--- FOUT: Timeout opgetreden! De scoop heeft niet op tijd geantwoord. ---")
    except Exception as e:
        # Vang alle andere mogelijke fouten op
        print(f"--- FOUT: Er is iets misgegaan. Foutmelding: {e} ---")

# --- VERBINDING SLUITEN ---
scope.close()
print("------------------------------------------------------------------")
print("Verbinding gesloten. Programma beëindigd.")