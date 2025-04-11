import serial
import time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg') # Kan nodig zijn

# --- Configuratie ---
COM_PORT = 'COM6'  # Jouw COM poort
BAUD_RATE = 115200
# Verhoog timeout omdat het lezen van 1024 samples even kan duren
READ_TIMEOUT = 10 # Seconden totale wachttijd
PAUSE_BETWEEN_TESTS = 3 # Seconden wachten tussen tests

# --- Globale Variabelen ---
ser = None

# --- Functies ---
def open_serial_port():
    """Probeert de seriële poort te openen."""
    global ser
    try:
        ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=0.1) # Korte timeout per read
        print(f"Verbonden met bord op {COM_PORT} met baudrate {BAUD_RATE}")
        time.sleep(2.5)
        ser.reset_input_buffer() # Betere manier om buffer te legen
        print("--- Initiële Berichten van Bord ---")
        initial_lines = read_multi_line_response(timeout=2.0) # Lees start-up berichten
        if initial_lines:
            print("\n".join(initial_lines))
        else:
             print("(Geen initiële berichten ontvangen)")
        print("---------------------------------")
        return True
    except serial.SerialException as e:
        print(f"Fout bij openen van seriële poort {COM_PORT}: {e}")
        return False
    except Exception as e:
        print(f"Onverwachte fout bij openen poort: {e}")
        return False

def close_serial_port():
    """Sluit de seriële poort."""
    global ser
    if ser and ser.is_open:
        ser.close()
        print(f"Seriële poort {COM_PORT} gesloten.")

def read_serial_line(timeout=0.1):
     """Leest één regel met een korte timeout."""
     global ser
     if ser and ser.is_open:
        try:
            # Stel de timeout specifiek voor deze leesactie in
            ser.timeout = timeout
            line = ser.readline()
            if line:
                return line.decode('utf-8', errors='ignore').strip()
            else:
                return None # Timeout
        except Exception as e:
            print(f"Fout bij lezen van seriële poort: {e}")
            return None
     return None

def read_multi_line_response(timeout=1.0):
     """Leest meerdere regels tot een timeout optreedt."""
     lines = []
     start_time = time.time()
     while (time.time() - start_time) < timeout:
         line = read_serial_line(timeout=0.1) # Korte individuele timeout
         if line is not None:
             lines.append(line)
             # Reset hoofdtimer als we data ontvangen
             start_time = time.time()
         else:
             # Als we al data hadden en nu niks meer, stop
             if lines:
                 break
             # Als we nog niks hadden en de timeout bijna voorbij is, stop ook
             if (time.time() - start_time) > (timeout - 0.1):
                 break
     return lines if lines else None

def send_command_process_response(command):
    """Verstuurt commando, leest ALLE output tot volgende prompt, extraheert data."""
    global ser
    if not ser or not ser.is_open:
        print("Fout: Seriële poort is niet open.")
        return False, None # Return (command_ok, data_list)

    print(f"\n>>> Versturen Commando: '{command}'")
    extracted_adc_data = []
    reading_adc = False
    command_ok = False # Nog niet bevestigd
    received_wait_prompt = False
    all_received_lines = [] # Voor debug

    try:
        ser.reset_input_buffer() # Begin met een schone buffer
        ser.write(f"{command}\n".encode('utf-8'))
        time.sleep(0.05) # Geef commando even tijd

        overall_start_time = time.time()

        # Blijf lezen tot de "Wacht op commando's" prompt of hoofdtijdout
        while (time.time() - overall_start_time) < READ_TIMEOUT:
            line = read_serial_line(timeout=0.5) # Timeout voor elke readline

            if line is not None:
                all_received_lines.append(line) # Log alles
                # print(f"<<< Ontvangen: {line}") # Uncomment voor live debug

                # Check status en prompts
                if "OK:" in line: command_ok = True
                if "ERROR:" in line: command_ok = False # Markeer als fout
                if "Wacht op commando's:" in line:
                    received_wait_prompt = True
                    # Geef nog heel even (50ms) de kans voor laatste data
                    time.sleep(0.05)
                    break # Einde van de cyclus

                # Data extractie logica (alleen actief NA start marker)
                if "--- ADC DATA START ---" in line:
                    if not reading_adc:
                         reading_adc = True
                         print("--- Start lezen ADC waarden (marker gevonden) ---")
                    continue # Marker zelf is geen data

                if reading_adc:
                    if line == "E":
                        print("Einde ADC data block ('E' marker gevonden).")
                        reading_adc = False # Stop met parsen als getal
                        # Breek nog niet, wacht op prompt
                    else:
                        try:
                            adc_value = int(line.strip())
                            extracted_adc_data.append(adc_value)
                        except ValueError:
                            # Negeer bekende niet-data regels
                            known_texts = ["--- ADC DATA END ---", "Actual Captured Samples:", "Finished capturing.", "Burst sent.", "Starting measurement cycle..."]
                            is_known_text = any(txt in line for txt in known_texts)
                            if not is_known_text and line: # Alleen waarschuwen voor onbekende, niet-lege regels
                                print(f"Waarschuwing: Kon regel niet parsen als ADC waarde: '{line}'")
                        except Exception as e:
                             print(f"Onverwachte fout bij parsen ADC waarde: {e}")
            else:
                # Geen data ontvangen binnen de readline timeout
                # Als we al lang wachten na de start, stop dan
                if (time.time() - overall_start_time) > (READ_TIMEOUT - 1):
                     print("Waarschuwing: Hoofdtimeout bereikt.")
                     break

        # Na de lus: print samenvatting en bepaal return waarde
        print("--- Respons Verwerking Klaar ---")
        if not received_wait_prompt:
            print("Waarschuwing: 'Wacht op commando's:' prompt niet gezien.")
            # Als we geen prompt zagen maar wel een OK, beschouw commando toch als ok?
            # command_ok = command_ok # Houd de status die we zagen

        # Geef data alleen terug als het commando 'P' was en we data vonden
        if command.upper() == 'P':
             print(f"ADC Samples Gevonden: {len(extracted_adc_data)}")
             # Alleen OK als we ook echt de prompt zagen
             return received_wait_prompt and command_ok, extracted_adc_data
        else:
             # Voor LNA commando, alleen status is relevant
             return received_wait_prompt and command_ok, None

    except Exception as e:
        print(f"Algemene fout bij versturen/ontvangen: {e}")
        # print("Alle ontvangen regels voor fout:") # Debug
        # for l in all_received_lines: print(l)  # Debug
        return False, None

# --- Hoofdprogramma ---
if __name__ == "__main__":
    if open_serial_port():
        all_plots_data = {}

        tests_to_run = {
            "1 (10 V/V)": 1,
            "2 (20 V/V)": 2
        }

        for label, gain_value in tests_to_run.items():
            print(f"\n===== Test: LNA Gain Setting {label} =====")
            # Verstuur LNA commando
            ok_lna, _ = send_command_process_response(f"LNA {gain_value}")
            if ok_lna:
                # Verstuur 'P' commando en vang data op
                ok_p, received_data = send_command_process_response("P")
                if ok_p and received_data is not None and len(received_data) > 0:
                    print(f"Data succesvol ontvangen voor {label}.")
                    all_plots_data[label] = received_data
                elif ok_p:
                     print(f"Meting OK voor {label}, maar geen ADC samples gevonden in de output.")
                else:
                     print(f"Meting NIET OK of geen data ontvangen na 'P' commando voor {label}.")
            else:
                print(f"Kon LNA Gain {gain_value} niet instellen (geen OK/prompt ontvangen).")

            print(f"\nWacht {PAUSE_BETWEEN_TESTS} seconden...")
            time.sleep(PAUSE_BETWEEN_TESTS)


        # --- Plot alles samen ---
        print("\n--- Plotten van alle resultaten ---")
        print(f"Verzamelde data dictionary (aantal keys: {len(all_plots_data)}):")
        for k, v in all_plots_data.items():
             print(f"  Label: '{k}', Aantal samples: {len(v)}")

        if not all_plots_data or all(not v for v in all_plots_data.values()):
             print("Geen valide data verzameld om te plotten.")
        else:
            plt.figure(figsize=(12, 7))
            plot_found = False
            for label, data in all_plots_data.items():
                if data:
                     print(f"--> Plotting {len(data)} points for {label}")
                     plt.plot(data, marker='.', linestyle='-', markersize=2, label=f"LNA={label}")
                     plot_found = True
                else:
                     print(f"--> Geen data om te plotten voor {label}")

            if plot_found:
                plt.title("Vergelijking ADC Data bij Verschillende LNA Gains")
                plt.xlabel("Sample Nummer")
                plt.ylabel("ADC Waarde")
                plt.legend()
                plt.grid(True)
                plt.show()
            else:
                 print("Kon geen plot genereren, geen valide datasets gevonden.")

        close_serial_port()
    else:
        print("Kon geen verbinding maken met het bord. Programma stopt.")

    print("\nScript voltooid.")