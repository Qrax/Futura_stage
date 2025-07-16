import os
import sys
import pyvisa # Belangrijk voor het eindresultaat
import usb.core
import usb.backend.libusb1

# --- DEEL 1: ONZE WERKende ZOEKFUNCTIE ---
def find_usbtmc_resources_working():
    print("--- Uitvoeren van de 'Monkey Patch' zoekfunctie ---")
    dll_path = os.path.join(os.path.dirname(__file__), "libusb-1.0.dll")
    backend = usb.backend.libusb1.get_backend(find_library=lambda x: dll_path)
    # Vind het apparaat
    dev = usb.core.find(idVendor=0x5345, idProduct=0x1234, backend=backend)
    if dev is None:
        return []
    # Construeer de VISA string
    visa_string = f"USB0::0x{dev.idVendor:04X}::0x{dev.idProduct:04X}::SN-dummy::INSTR"
    print(f"PATCH: Apparaat gevonden! String geconstrueerd: {visa_string}")
    return [visa_string]

# --- DEEL 2: DE PATCH TOEPASSEN ---
import pyvisa_py.protocols.usbtmc
pyvisa_py.protocols.usbtmc.list_resources = find_usbtmc_resources_working
print("Monkey patch toegepast. De standaard zoekfunctie is vervangen.")

# --- DEEL 3: NU DE CODE NORMAAL GEBRUIKEN ---
print("\nNu wordt de ResourceManager aangeroepen (die onze patch zal gebruiken)...")
rm = pyvisa.ResourceManager('@py')
resources = rm.list_resources()
print(f"Resultaat van rm.list_resources(): {resources}")

if not resources:
    raise ConnectionError("FATALE FOUT: Zelfs met de patch is het apparaat niet gevonden.")

scope = rm.open_resource(resources[0])
print(f"Succesvol verbonden met: {scope.query('*IDN?')}")
scope.close()
print("Verbinding succesvol getest en gesloten.")