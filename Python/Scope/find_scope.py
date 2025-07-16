# Probeer deze code NA het updaten van de pakketten
import pyvisa
import os
import usb.backend.libusb1

print("--- Poging met bijgewerkte pakketten ---")

dll_path = os.path.join(os.path.dirname(__file__), "libusb-1.0.dll")
backend = usb.backend.libusb1.get_backend(find_library=lambda x: dll_path)

if not backend:
    raise ImportError("Kon de libusb backend niet initialiseren.")

# Deze regel zou nu moeten werken na de update
rm = pyvisa.ResourceManager('@py', backend=backend)

resources = rm.list_resources()
print(f"Gevonden resources: {resources}")

if not resources:
    print("\nNog steeds geen succes. Probeer Plan B.")
else:
    print("\nSUCCES! De update heeft het opgelost!")