import os
import sys
import usb.core
import usb.backend.libusb1

print("--- Force-Loading PyUSB Backend Test ---")

# --- Step 1: Define the EXACT path to the DLL that we know works ---
# This gets the directory where the script is located
script_dir = os.path.dirname(__file__)
# This creates the full, absolute path to the DLL
dll_path = os.path.join(script_dir, "libusb-1.0.dll")

print(f"DLL path to be used: {dll_path}")

if not os.path.exists(dll_path):
    print("\n[CRITICAL FAIL] The DLL is not next to the script. Please copy it here.")
    sys.exit()

# --- Step 2: Get the backend, but force it to use our specific DLL ---
# We provide a simple lambda function to the 'find_library' argument.
# This function ignores whatever pyusb is trying to find and just returns our known-good path.
backend = usb.backend.libusb1.get_backend(find_library=lambda x: dll_path)

if not backend:
    print("\n[CRITICAL FAIL] Even with a direct path, the backend could not be initialized.")
    print("This is a very deep and unusual error.")
    sys.exit()
else:
    print("\n[SUCCESS] Backend initialized successfully using the specified DLL!")

# --- Step 3: Now, try to find devices using this specific, correctly-loaded backend ---
print("\nSearching for all USB devices with the correctly loaded backend...")

# The 'backend' argument tells usb.core.find to use our custom-loaded backend
# instead of trying to find one on its own.
all_devices = list(usb.core.find(find_all=True, backend=backend))

if not all_devices:
    print("\n[FAIL] Found 0 devices. The issue is likely OS permissions.")
    print("Please try this one last time by running your IDE or terminal as Administrator.")
else:
    print(f"\n[VICTORY!] Successfully found {len(all_devices)} device(s)!\n")
    scope_found = False
    for i, dev in enumerate(all_devices):
        try:
            print(f"Device #{i}: VID=0x{dev.idVendor:04X}, PID=0x{dev.idProduct:04X} | Product: {dev.product}")
            if dev.idVendor == 0x5345 and dev.idProduct == 0x1234:
                scope_found = True
                print("  ===> THIS IS YOUR OSCILLOSCOPE! <===")
        except Exception:
            # Some devices don't have product strings or are protected.
            print(f"Device #{i}: VID=0x{dev.idVendor:04X}, PID=0x{dev.idProduct:04X} | (Could not get product string)")

    if scope_found:
        print("\nProblem solved. You can now use this method to connect with pyvisa.")
    else:
        print("\nPartial success. Other devices are visible, but the scope is not.")
        print("This would mean the WinUSB driver is not bound correctly, but that is unlikely.")