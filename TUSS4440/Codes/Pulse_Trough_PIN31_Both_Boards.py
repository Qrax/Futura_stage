import serial
import time
import threading
import queue
import matplotlib.pyplot as plt
import sys

# --- Configuratie ---
MASTER_PORT = 'COM6'
SLAVE_PORT = 'COM8'
BAUD_RATE = 115200
SERIAL_TIMEOUT = 0.5
CONNECT_DELAY = 0.5
CONFIRM_READ_DURATION = 0.5
DATA_READ_TIMEOUT = 15  # Max wachttijd voor data
END_MARKER = "E"

# --- Globale Variabelen ---
master_ser = None
slave_ser = None
data_queue_master = queue.Queue()
data_queue_slave = queue.Queue()


# --- Functies ---
def connect_ports():
    global master_ser, slave_ser;
    print("--- Verbinding starten ---")
    try:
        master_ser = serial.Serial(MASTER_PORT, BAUD_RATE, timeout=SERIAL_TIMEOUT);
        print(f"Verbonden: {MASTER_PORT}. Wacht {CONNECT_DELAY}s...");
        time.sleep(CONNECT_DELAY);
        master_ser.reset_input_buffer()
    except serial.SerialException as e:
        print(f"FOUT Master ({MASTER_PORT}): {e}");
        return False
    try:
        slave_ser = serial.Serial(SLAVE_PORT, BAUD_RATE, timeout=SERIAL_TIMEOUT);
        print(f"Verbonden: {SLAVE_PORT}. Wacht {CONNECT_DELAY}s...");
        time.sleep(CONNECT_DELAY);
        slave_ser.reset_input_buffer()
    except serial.SerialException as e:
        print(f"FOUT Slave ({SLAVE_PORT}): {e}");
        (master_ser.close() if master_ser and master_ser.is_open else None);
        return False
    print("--- Beide poorten verbonden ---");
    return True


def set_modes_and_show_initial_output():
    if not master_ser or not slave_ser or not master_ser.is_open or not slave_ser.is_open: return False
    print("\n--- Versturen M/S commando's ---");
    time.sleep(0.1)
    try:
        master_ser.reset_input_buffer();
        slave_ser.reset_input_buffer();
        master_ser.write(b'M\n');
        slave_ser.write(b'S\n')
    except serial.SerialException as e:
        print(f"Seriële FOUT M/S: {e}");
        return False
    except Exception as e:
        print(f"Algemene FOUT M/S: {e}");
        return False
    print(f"\n--- Ruwe output (max {CONFIRM_READ_DURATION}s) ---");
    start_time = time.time()
    while time.time() - start_time < CONFIRM_READ_DURATION:
        output_found = False
        try:
            if master_ser.in_waiting > 0: m_bytes = master_ser.read(master_ser.in_waiting); print(
                f"COM6 (M): {m_bytes.decode(errors='replace').strip()}"); output_found = True
            if slave_ser.in_waiting > 0: s_bytes = slave_ser.read(slave_ser.in_waiting); print(
                f"COM8 (S): {s_bytes.decode(errors='replace').strip()}"); output_found = True
            if not output_found: time.sleep(0.02)
        except serial.SerialException as e:
            print(f"\nSeriële FOUT init read: {e}");
            break
        except Exception as e:
            print(f"\nAlgemene FOUT init read: {e}");
            break
    print("--- Einde ruwe output check ---");
    return True


def read_serial_data(ser_port, data_q, stop_event):
    """Leest data, print raw, zoekt avg time diff, parseert ADC, en print debug info."""
    port_name = ser_port.port
    print(f"\n--- Start lezen van {port_name} ---") # Newline voor duidelijkheid
    adc_values = []
    buffer = ""
    avg_diff_us = None
    start_time = time.time()
    line_counter = 0 # Teller voor aantal verwerkte lijnen

    while not stop_event.is_set() and (time.time() - start_time < DATA_READ_TIMEOUT) :
        try:
            if ser_port.in_waiting > 0:
                new_bytes = ser_port.read(ser_port.in_waiting)
                if new_bytes:
                    # Print RAW bytes direct
                    # Gebruik repr() om speciale tekens zoals \r \n zichtbaar te maken
                    print(f"\nRAW Bytes {port_name}: {repr(new_bytes)}")
                    decoded_chunk = new_bytes.decode(errors='replace')
                    print(f"RAW Decoded {port_name}: {decoded_chunk}", end='') # Print zonder extra newline

                    # Voeg toe aan buffer voor lijn-parsing
                    buffer += new_bytes.decode(errors='ignore')

                    # Verwerk complete lijnen in de buffer
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        line = line.strip() # Verwijder \r en witruimte
                        line_counter += 1
                        # print(f"\nDEBUG Line {line_counter} ({port_name}): '{line}'") # Optioneel: print elke lijn

                        if not line: continue # Skip lege lijnen

                        # Check voor eindmarker
                        if line == END_MARKER:
                            print(f"\n>>> END '{END_MARKER}' ({port_name}). AvgT: {avg_diff_us} <<<")
                            # DEBUG PRINT VOOR QUEUE
                            print(f"\n>>> {port_name} putting to queue: AvgT={avg_diff_us}, Samples={len(adc_values)}, First 5 ADC: {adc_values[:5]} <<<")
                            data_q.put((avg_diff_us, adc_values))
                            return # Stop deze thread succesvol

                        # Check voor de gemiddelde tijd lijn
                        avg_prefix = "Average time between samples (us):"
                        if line.startswith(avg_prefix):
                            try:
                                avg_val_str = line[len(avg_prefix):].strip()
                                avg_diff_us = int(avg_val_str)
                                print(f"\n>>> AvgT Found ({port_name}): {avg_diff_us} <<<")
                            except ValueError:
                                print(f"\nWARN ({port_name}): Kon AvgT niet parsen uit: {line}")
                            continue # Dit was geen data lijn

                        # Probeer ADC data te parsen (Timestamp, ADC Value)
                        try:
                            if ',' in line:
                                parts = line.split(',')
                                if len(parts) == 2:
                                    # We nemen alleen de ADC waarde (tweede deel)
                                    adc_str = parts[1].strip()
                                    if adc_str.isdigit():
                                        adc_int = int(adc_str)
                                        adc_values.append(adc_int)
                                        # --- DEBUG PRINT GEPARSEERDE ADC ---
                                        # Print elke 50e sample om console niet te overspoelen
                                        if len(adc_values) % 50 == 1 or len(adc_values) < 5:
                                             print(f"\n---PARSED {port_name} ADC: {adc_int} (Sample #: {len(adc_values)})---")
                                        # --- END DEBUG ---
                        except ValueError:
                             # print(f"\nDEBUG ({port_name}): ValueError bij parsen lijn: '{line}'") # Optioneel
                             pass # Negeer parse errors voor ADC
            else:
                # Wacht heel even als er niets binnenkomt om CPU te sparen
                time.sleep(0.01)

        except serial.SerialException as e:
            print(f"\nSeriële fout tijdens lezen van {port_name}: {e}")
            stop_event.set()
            break
        except Exception as e:
            print(f"\nAlgemene fout tijdens lezen {port_name}: {e}")
            stop_event.set()
            break

    # Als de loop eindigt door timeout of stop_event ZONDER de END_MARKER te zien
    print(f"\nLezen van {port_name} gestopt (Timeout/Stop Event). Geen END marker gezien na {line_counter} lijnen.")
    # DEBUG PRINT VOOR QUEUE (ook bij timeout)
    print(f"\n>>> {port_name} putting to queue (TIMEOUT/STOP): AvgT={avg_diff_us}, Samples={len(adc_values)}, First 5 ADC: {adc_values[:5]} <<<")
    data_q.put((avg_diff_us, adc_values)) # Stuur wat we hebben (mogelijk leeg)


def plot_data(master_data, slave_data, title_extra=""):
    master_avg_diff, master_adc = master_data
    slave_avg_diff, slave_adc = slave_data

    if not master_adc and not slave_adc:
        print("Geen ADC data ontvangen om te plotten.")
        return

    plt.figure(figsize=(12, 6))
    plot_occurred = False
    time_axis_label = 'Sample Index' # Default label

    # Plot Master data
    if master_adc:
        if master_avg_diff is not None and master_avg_diff > 0:
            master_ts = [i * master_avg_diff for i in range(len(master_adc))]
            plt.plot(master_ts, master_adc, label=f'Master ({MASTER_PORT}) - {len(master_adc)} (Avg: {master_avg_diff} us)', marker='.', linestyle='-')
            time_axis_label = 'Berekende Tijd (us) / Index' # Update label
            print(f"Master data geplot ({len(master_adc)} punten, avg {master_avg_diff} us).") # Duidelijkere print
            plot_occurred = True
        else:
             # Fallback naar index voor Master
             print(f"WARN: Master avg time niet ok ({master_avg_diff}). Plot Master tegen sample index.")
             plt.plot(range(len(master_adc)), master_adc, label=f'Master ({MASTER_PORT}) - {len(master_adc)} samples (index)', marker='.', linestyle='-')
             plot_occurred = True

    # Plot Slave data
    if slave_adc:
        if slave_avg_diff is not None and slave_avg_diff > 0:
            # --- CORRECTE IF BLOK ---
            slave_ts = [i * slave_avg_diff for i in range(len(slave_adc))]
            plt.plot(slave_ts, slave_adc, label=f'Slave ({SLAVE_PORT}) - {len(slave_adc)} (Avg: {slave_avg_diff} us)', marker='.', linestyle='-')
            if time_axis_label == 'Sample Index': # Update label alleen als Master index gebruikte
                 time_axis_label = 'Berekende Tijd (us) / Index'
            print(f"Slave data geplot ({len(slave_adc)} punten, avg {slave_avg_diff} us).") # Duidelijkere print
            plot_occurred = True
        else:
             # --- CORRECTE ELSE BLOK (Fallback naar index voor Slave) ---
             print(f"WARN: Slave avg time niet ok ({slave_avg_diff}). Plot Slave tegen sample index.")
             plt.plot(range(len(slave_adc)), slave_adc, label=f'Slave ({SLAVE_PORT}) - {len(slave_adc)} samples (index)', marker='.', linestyle='-')
             plot_occurred = True # Er wordt wel geplot (op index)

    # Toon plot als er iets geplot is
    if plot_occurred:
        plt.title(f'ADC Metingen {title_extra}')
        plt.xlabel(time_axis_label)
        plt.ylabel('ADC Waarde')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        # Deze melding zou nu niet meer moeten verschijnen als er samples zijn
        print("Niets geplot omdat er geen valide ADC data was (dit is onverwacht).")


def close_ports():
    global master_ser, slave_ser;
    print("\n--- Sluiten Poorten ---")
    try:
        (master_ser.close() if master_ser and master_ser.is_open else None);
        print(f"{MASTER_PORT} gesloten.")
    except Exception as e:
        print(f"Fout sluiten master: {e}")
    try:
        (slave_ser.close() if slave_ser and slave_ser.is_open else None);
        print(f"{SLAVE_PORT} gesloten.")
    except Exception as e:
        print(f"Fout sluiten slave: {e}")


# --- Hoofdprogramma ---
if __name__ == "__main__":
    if not connect_ports(): sys.exit(1)
    if not set_modes_and_show_initial_output(): close_ports(); sys.exit(1)

    print("\nKlaar voor commando's.")
    print("  sync   - Start: Master meet direct, Slave pulseert na 50ms.")
    print("  measure- Laat Master nu meten (geen sync/puls).")
    print("  pulse  - Laat Slave nu pulsen en meten (geen sync).")
    print("  L <ms> - Set default record length.")
    print("  exit   - Stoppen.")

    try:
        while True:
            # --- CORRECTED INDENTATION BLOCK ---
            try:
                # Niveau 3 inspringing (bv. 12 spaties)
                cmd_input = input("Commando: ").strip()  # Houd hoofdletters voor L
                cmd_lower = cmd_input.lower()  # Zelfde niveau als regel erboven
            except EOFError:
                # Niveau 3 inspringing
                cmd_lower = "exit"
            # --- END CORRECTED BLOCK ---

            master_result = (None, [])
            slave_result = (None, [])
            plot_title = ""

            # --- Commando Afhandeling (Niveau 2 inspringing, bv. 8 spaties) ---
            if cmd_lower == "sync":
                # Niveau 3 inspringing
                print("\n--- Start SYNC Cyclus (Master Rec / Slave Delayed Pulse) ---")
                plot_title = "(Sync Sequence)"
                stop_event = threading.Event()
                master_thread = threading.Thread(target=read_serial_data,
                                                 args=(master_ser, data_queue_master, stop_event), daemon=True)
                slave_thread = threading.Thread(target=read_serial_data, args=(slave_ser, data_queue_slave, stop_event),
                                                daemon=True)
                master_thread.start();
                slave_thread.start();
                time.sleep(0.1)
                print("\n>>> Versturen SYNC commando naar Master (start meting & HW sync)... <<<")
                try:
                    master_ser.reset_input_buffer();
                    master_ser.write(b'SYNC\n')
                except Exception as e:
                    print(f"Fout SYNC -> Master: {e}");
                    stop_event.set();
                    continue
                print(f"\n>>> Wachten op data (max ~{DATA_READ_TIMEOUT}s)... Raw output volgt: <<<")
                master_thread.join(timeout=DATA_READ_TIMEOUT + 1)
                slave_thread.join(timeout=DATA_READ_TIMEOUT + 1)
                if master_thread.is_alive() or slave_thread.is_alive(): print(
                    "\nWaarschuwing: Lees-thread(s) timeout."); stop_event.set(); master_thread.join(
                    0.5); slave_thread.join(0.5)
                try:
                    master_result = data_queue_master.get_nowait()
                except queue.Empty:
                    print("INFO: Geen data Master.")
                try:
                    slave_result = data_queue_slave.get_nowait()
                except queue.Empty:
                    print("INFO: Geen data Slave.")
                print("\n--- Einde SYNC Cyclus ---")

            elif cmd_lower == "measure":
                # Niveau 3 inspringing
                print("\n--- Start MEASURE Commando (Master Only) ---")
                plot_title = "(Master Measure Only)"
                stop_event = threading.Event()
                master_thread = threading.Thread(target=read_serial_data,
                                                 args=(master_ser, data_queue_master, stop_event), daemon=True)
                master_thread.start();
                time.sleep(0.1)
                print("\n>>> Versturen MEASURE commando naar Master... <<<")
                try:
                    master_ser.reset_input_buffer();
                    master_ser.write(b'MEASURE\n')
                except Exception as e:
                    print(f"Fout MEASURE -> Master: {e}");
                    stop_event.set();
                    continue
                print(f"\n>>> Wachten op Master data (max ~{DATA_READ_TIMEOUT}s)... Raw output volgt: <<<")
                master_thread.join(timeout=DATA_READ_TIMEOUT + 1)
                if master_thread.is_alive(): print(
                    "\nWaarschuwing: Master lees-thread timeout."); stop_event.set(); master_thread.join(0.5)
                try:
                    master_result = data_queue_master.get_nowait()
                except queue.Empty:
                    print("INFO: Geen data Master.")
                print("\n--- Einde MEASURE Commando ---")

            elif cmd_lower == "pulse":
                # Niveau 3 inspringing
                print("\n--- Start PULSE Commando (Slave Only) ---")
                plot_title = "(Slave Pulse Only)"
                stop_event = threading.Event()
                slave_thread = threading.Thread(target=read_serial_data, args=(slave_ser, data_queue_slave, stop_event),
                                                daemon=True)
                slave_thread.start();
                time.sleep(0.1)
                print("\n>>> Versturen PULSE commando naar Slave... <<<")
                try:
                    slave_ser.reset_input_buffer();
                    slave_ser.write(b'PULSE\n')
                except Exception as e:
                    print(f"Fout PULSE -> Slave: {e}");
                    stop_event.set();
                    continue
                print(f"\n>>> Wachten op Slave data (max ~{DATA_READ_TIMEOUT}s)... Raw output volgt: <<<")
                slave_thread.join(timeout=DATA_READ_TIMEOUT + 1)
                if slave_thread.is_alive(): print(
                    "\nWaarschuwing: Slave lees-thread timeout."); stop_event.set(); slave_thread.join(0.5)
                try:
                    slave_result = data_queue_slave.get_nowait()
                except queue.Empty:
                    print("INFO: Geen data Slave.")
                print("\n--- Einde PULSE Commando ---")

            elif cmd_input.upper().startswith("L "):
                # Niveau 3 inspringing
                value_str = cmd_input[2:].strip()
                if value_str.isdigit():
                    print(f"\n>>> Versturen L {value_str} naar BEIDE borden... <<<")
                    try:
                        master_ser.reset_input_buffer();
                        master_ser.write(f"L {value_str}\n".encode());
                        time.sleep(0.1)
                        if master_ser.in_waiting > 0: print(
                            f"COM6 (M): {master_ser.read(master_ser.in_waiting).decode(errors='replace').strip()}")
                        slave_ser.reset_input_buffer();
                        slave_ser.write(f"L {value_str}\n".encode());
                        time.sleep(0.1)
                        if slave_ser.in_waiting > 0: print(
                            f"COM8 (S): {slave_ser.read(slave_ser.in_waiting).decode(errors='replace').strip()}")
                    except Exception as e:
                        print(f"Fout bij L commando: {e}")
                else:
                    print("Ongeldig formaat. Gebruik 'L <ms>'.")
                continue  # Niet plotten na L commando

            elif cmd_lower == "exit":
                # Niveau 3 inspringing
                print("Programma stoppen...");
                break
            else:
                # Niveau 3 inspringing
                print("Onbekend commando.")
                continue  # Niet plotten

            # Plot alleen als er een commando was dat data genereert (Niveau 2)
            if cmd_lower in ["sync", "measure", "pulse"]:
                plot_data(master_result, slave_result, title_extra=plot_title)

    except KeyboardInterrupt:
        print("\nCtrl+C gedetecteerd...")  # Niveau 1
    finally:  # Niveau 1
        close_ports();
        print("Programma beëindigd.")
