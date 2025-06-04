#!/usr/bin/env python3
"""
meerdere_metingen.py – logt ADC‑data van zowel MASTER (COM6) als SLAVE (COM8)
in meerdere blokken (meta-runs). Elk blok wordt in een apart CSV-bestand opgeslagen.
Kolommen CSV: Device, Run, SampleIndex, Timestamp_us, ADC_Value, Avg_Time_us.
Schrijft elke SAVE_INTERVAL runs tussentijds naar disk.
"""
# gebaseerd op sync_logger.py (origineel) :contentReference[oaicite:0]{index=0}​:contentReference[oaicite:1]{index=1}

# --- Imports --------------------------------------------------------------
import serial
import time
import threading
import queue
import sys
import pandas as pd
import os

# --- Config ---------------------------------------------------------------
MASTER_PORT = 'COM6'
SLAVE_PORT = 'COM8'
BAUD_RATE = 115200
SERIAL_TIMEOUT = 1  # algemeen lees‑timeout
CONNECT_DELAY = 0.5
CONFIRM_READ_DURATION = 1
DATA_READ_TIMEOUT = 2  # wachttijd per meetrun (s)
END_MARKER = "E"

# Nieuwe configuratie voor meta-runs
NUM_META_RUNS = 3  # Aantal "grote" runs/blokken
RUNS_PER_META = 25  # Aantal individuele metingen per meta-run
INTER_META_RUN_DELAY_S = 15  # Pauze tussen meta-runs in seconden
INTER_RUN_DELAY = 2  # Pauze tussen individuele metingen binnen een meta-run (s)
SAVE_INTERVAL = 10  # Aantal runs tussen tussentijdse CSV-saves (binnen een meta-run)

BASE_FILENAME_PROMPT = "Voer een basisnaam in voor de CSV-bestanden (bv. experiment_X): "
DEFAULT_BASE_FILENAME = "meting_data"
TARGET_DATA_SUBFOLDER = os.path.join("..", "data", "UltraSoon_Measurements")

# --- Globale variabelen ---------------------------------------------------
master_ser = slave_ser = None
data_queue_master = queue.Queue()
data_queue_slave = queue.Queue()


# --- Helper: tussentijds en eind‑save -------------------------------------
def save_csv(master_runs, slave_runs, fname):
    """Flatten alle runs en sla op als CSV (overschrijft steeds)."""
    if not master_runs and not slave_runs:
        print(f"\n[Save Info] Geen data om op te slaan voor '{fname}'.")
        return

    records = []

    def flatten(run_dicts, device_name):
        for rd in run_dicts:
            run_no = rd["Run"]
            avg_dt = rd["Avg_Time_us"]
            ts_ok = avg_dt is not None and avg_dt > 0
            for i, adc in enumerate(rd["ADC_Values"]):
                records.append({
                    "Device": device_name,
                    "Run": run_no,
                    "SampleIndex": i,
                    "Timestamp_us": i * avg_dt if ts_ok else None,
                    "ADC_Value": adc,
                    "Avg_Time_us": avg_dt
                })

    flatten(master_runs, "Master")
    flatten(slave_runs, "Slave")
    df = pd.DataFrame(records)

    # Zorg ervoor dat de map bestaat als de bestandsnaam een pad bevat
    output_dir = os.path.dirname(fname)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n[Save Info] Map '{output_dir}' aangemaakt.")

    df.to_csv(fname, index=False)
    print(f"\n[Save Success] {len(df)} rijen naar '{fname}' geschreven.")


# --- Serial‑helperfuncties (ongewijzigd) ----------------------------------
def connect_ports() -> bool:
    """Open beide seriële poorten."""
    global master_ser, slave_ser
    print("--- Verbinding starten ---")
    try:
        master_ser = serial.Serial(MASTER_PORT, BAUD_RATE, timeout=SERIAL_TIMEOUT)
        print(f"Verbonden: {MASTER_PORT}. Wacht {CONNECT_DELAY}s …")
        time.sleep(CONNECT_DELAY)
        master_ser.reset_input_buffer()
    except serial.SerialException as e:
        print(f"FOUT Master ({MASTER_PORT}): {e}")
        return False

    try:
        slave_ser = serial.Serial(SLAVE_PORT, BAUD_RATE, timeout=SERIAL_TIMEOUT)
        print(f"Verbonden: {SLAVE_PORT}. Wacht {CONNECT_DELAY}s …")
        time.sleep(CONNECT_DELAY)
        slave_ser.reset_input_buffer()
    except serial.SerialException as e:
        print(f"FOUT Slave ({SLAVE_PORT}): {e}")
        if master_ser and master_ser.is_open:
            master_ser.close()
        return False

    print("--- Beide poorten verbonden ---")
    return True


def set_modes_and_show_initial_output() -> bool:
    """Stel Master/Slave‑modus in op de boards en toon eventuele start‑output."""
    if not (master_ser and slave_ser and master_ser.is_open and slave_ser.is_open):
        return False

    print("\n--- Versturen M/S commando's ---")
    time.sleep(0.1)
    try:
        master_ser.reset_input_buffer()
        slave_ser.reset_input_buffer()
        master_ser.write(b'M\n')
        slave_ser.write(b'S\n')
    except serial.SerialException as e:
        print(f"Seriële FOUT M/S: {e}")
        return False

    print(f"\n--- Initiële output ({CONFIRM_READ_DURATION}s) ---")
    start_time = time.time()
    while time.time() - start_time < CONFIRM_READ_DURATION:
        output_found = False
        if master_ser.in_waiting > 0:
            print(f"RAW INIT {MASTER_PORT}: {master_ser.read(master_ser.in_waiting).decode(errors='replace')}", end='')
            output_found = True
        if slave_ser.in_waiting > 0:
            print(f"RAW INIT {SLAVE_PORT}: {slave_ser.read(slave_ser.in_waiting).decode(errors='replace')}", end='')
            output_found = True
        if not output_found:
            time.sleep(0.02)
    print("--- Einde init check ---")
    return True


def robust_read_serial_data(ser_port: serial.Serial,
                            data_q: queue.Queue,
                            stop_event: threading.Event):
    """
    Leest ADC-data uit één seriële poort en zet ze als
        (avg_time_us, adc_list)
    in de meegegeven queue.
    (Logica ongewijzigd, alleen print statements aangepast voor duidelijkheid)
    """
    port_name = ser_port.port
    buffer = ''
    adc_values = []
    avg_diff_us = None
    found_avg = False
    total_time_us = None
    num_samples = None
    reading_samples = False
    start_time = time.time()

    while not stop_event.is_set() and (time.time() - start_time < DATA_READ_TIMEOUT):
        try:
            if ser_port.in_waiting:
                # Kleine optimalisatie: lees niet te veel tegelijk om buffer niet te overspoelen
                # bij snelle data, maar lees wel genoeg om efficiënt te zijn.
                chunk = ser_port.read(min(ser_port.in_waiting, 1024)).decode(errors='ignore')
                # print(f"RAW {port_name}: {chunk}", end='', flush=True) # Kan veel output geven
                buffer += chunk

                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    line = line.strip()
                    if not line:
                        continue

                    if line == END_MARKER:
                        if avg_diff_us is None and total_time_us and num_samples and num_samples > 0:  # Voorkom ZeroDivisionError
                            avg_diff_us = total_time_us / num_samples
                        data_q.put((avg_diff_us, adc_values))
                        # print(f"DEBUG {port_name}: END_MARKER, data queued ({avg_diff_us}, {len(adc_values)} samples)")
                        return

                    if line.startswith("Average time between samples (us):"):
                        try:
                            avg_diff_us = float(line.split(':', 1)[1].strip())
                            found_avg = True
                            # print(f"\n>>> {port_name} AvgT={avg_diff_us:.1f} µs <<<")
                        except ValueError:
                            print(f"\nWARN {port_name}: kan AvgT (oud formaat) niet parsen: '{line}'")
                        continue

                    if line.startswith("Total time from first to last sample:"):
                        try:
                            total_time_us = float(line.split(':', 1)[1].split()[0])
                        except (ValueError, IndexError):
                            print(f"\nWARN {port_name}: kan TotalTime niet parsen: '{line}'")
                        continue

                    if line.startswith("Number of samples"):
                        try:
                            num_samples = int(line.split(':', 1)[1].strip())
                        except (ValueError, IndexError):
                            print(f"\nWARN {port_name}: kan NumSamples niet parsen: '{line}'")
                        continue

                    if line.startswith("Average per sample"):
                        try:
                            avg_diff_us = float(line.split(':', 1)[1].split()[0])
                            found_avg = True
                            # print(f"\n>>> {port_name} AvgT={avg_diff_us:.1f} µs <<<")
                        except (ValueError, IndexError):
                            print(f"\nWARN {port_name}: kan AvgT (nieuw formaat) niet parsen: '{line}'")
                        continue

                    if line.lower().startswith("samples"):
                        reading_samples = True
                        continue

                    if reading_samples:
                        try:
                            adc_values.append(int(line))
                        except ValueError:
                            # print(f"\nWARN {port_name}: kon ADC-waarde niet parsen: '{line}' (nieuw formaat)")
                            pass  # Kan een lege regel zijn na "Samples:", negeer.
                        continue

                    if ',' in line:  # Oud formaat: "timestamp, adc"
                        try:
                            _, adc_str = line.split(',', 1)
                            adc_values.append(int(adc_str.strip()))
                        except ValueError:
                            # print(f"\nWARN {port_name}: kon ADC-waarde niet parsen: '{line}' (oud formaat)")
                            pass
            else:
                time.sleep(0.005)  # Kortere slaap voor snellere reactie

        except serial.SerialException as e:
            print(f"\nSeriële FOUT tijdens lezen op {port_name}: {e}")
            stop_event.set()  # Signaleer andere thread ook te stoppen
            break
        except Exception as e:
            print(f"\nAlgemene FOUT tijdens lezen op {port_name}: {e}")
            stop_event.set()  # Signaleer andere thread ook te stoppen
            break

    # Timeout of stop_event
    if not stop_event.is_set():  # Alleen als niet al gestopt door error
        if avg_diff_us is None and total_time_us and num_samples and num_samples > 0:
            avg_diff_us = total_time_us / num_samples
        data_q.put((avg_diff_us, adc_values))
        # print(f"DEBUG {port_name}: Timeout/Stop, data queued ({avg_diff_us}, {len(adc_values)} samples)")


def close_ports():
    """Sluit beide seriële poorten."""
    print("\n--- Sluiten poorten ---")
    closed_master = False
    closed_slave = False
    try:
        if master_ser and master_ser.is_open:
            master_ser.close()
            print(f"{MASTER_PORT} gesloten.")
            closed_master = True
    except Exception as e:
        print(f"Fout bij sluiten {MASTER_PORT}: {e}")
    finally:
        try:
            if slave_ser and slave_ser.is_open:
                slave_ser.close()
                print(f"{SLAVE_PORT} gesloten.")
                closed_slave = True
        except Exception as e:
            print(f"Fout bij sluiten {SLAVE_PORT}: {e}")
    if not closed_master and master_ser: print(f"{MASTER_PORT} was niet open of kon niet gesloten worden.")
    if not closed_slave and slave_ser: print(f"{SLAVE_PORT} was niet open of kon niet gesloten worden.")


# --------------------------------------------------------------------------
if __name__ == "__main__":
    base_filename_user = input(BASE_FILENAME_PROMPT).strip()
    if not base_filename_user:
        base_filename_user = DEFAULT_BASE_FILENAME
        print(f"Geen invoer, standaard basisnaam gebruikt: '{base_filename_user}'")

    if not connect_ports():
        sys.exit(1)
    if not set_modes_and_show_initial_output():
        close_ports()
        sys.exit(1)

    print(f"\nKlaar om {NUM_META_RUNS} meta-run(s) uit te voeren, elk met {RUNS_PER_META} metingen.")
    input("Druk op Enter om te beginnen met de eerste meta-run…")

    # Huidige meta-run data (wordt mogelijk opgeslagen bij interrupt)
    current_meta_all_master_runs = []
    current_meta_all_slave_runs = []
    current_meta_filename = ""

    try:
        for meta_run_count in range(1, NUM_META_RUNS + 1):
            base_csv_name = f"{base_filename_user}_meta_{meta_run_count}.csv"
            current_meta_filename = os.path.join(TARGET_DATA_SUBFOLDER, base_csv_name)
            print(f"\n{'=' * 10} START META-RUN {meta_run_count}/{NUM_META_RUNS} {'=' * 10}")
            print(f"Data voor deze meta-run wordt opgeslagen in: '{current_meta_filename}'")

            current_meta_all_master_runs = []  # Reset voor nieuwe meta-run
            current_meta_all_slave_runs = []  # Reset voor nieuwe meta-run

            for run in range(1, RUNS_PER_META + 1):
                print(f"\n--- Meta-Run {meta_run_count} | Run {run}/{RUNS_PER_META} ---")

                for q in (data_queue_master, data_queue_slave):
                    while not q.empty():
                        try:
                            q.get_nowait()
                        except queue.Empty:
                            break

                stop_event = threading.Event()
                # Maak de threads opnieuw aan voor elke run om problemen met herstarten te voorkomen
                t_master = threading.Thread(target=robust_read_serial_data,
                                            args=(master_ser, data_queue_master, stop_event),
                                            daemon=True)
                t_slave = threading.Thread(target=robust_read_serial_data,
                                           args=(slave_ser, data_queue_slave, stop_event),
                                           daemon=True)

                # Reset input buffers vlak voor SYNC
                master_ser.reset_input_buffer()
                slave_ser.reset_input_buffer()  # Ook slave voor de zekerheid

                t_master.start()
                t_slave.start()
                time.sleep(0.1)  # Geef threads even tijd om te starten

                print(">>> SYNC naar master …")
                master_ser.write(b'SYNC\n')

                # Wachten op threads met een gecombineerde timeout
                # De individuele read timeout is DATA_READ_TIMEOUT
                # De join timeout hier is iets ruimer om threads de kans te geven
                # hun eigen timeout af te handelen en data in de queue te plaatsen.
                join_timeout = DATA_READ_TIMEOUT + 0.5
                t_master.join(join_timeout)
                t_slave.join(join_timeout)

                # Als threads nog draaien na join_timeout, forceer stop
                if t_master.is_alive() or t_slave.is_alive():
                    print("Waarschuwing: Een of beide lees-threads zijn niet op tijd gestopt. Forceren...")
                    stop_event.set()
                    t_master.join(0.5)  # Geef nog even de kans om netjes af te sluiten
                    t_slave.join(0.5)

                m_avg, m_adc = None, []
                s_avg, s_adc = None, []
                try:
                    m_avg, m_adc = data_queue_master.get_nowait()
                except queue.Empty:
                    print(f"Master ({MASTER_PORT}): Geen data ontvangen in wachtrij.")
                try:
                    s_avg, s_adc = data_queue_slave.get_nowait()
                except queue.Empty:
                    print(f"Slave ({SLAVE_PORT}): Geen data ontvangen in wachtrij.")

                if m_adc:
                    current_meta_all_master_runs.append({"Run": run, "Avg_Time_us": m_avg, "ADC_Values": m_adc})
                    print(f"Master: {len(m_adc)} samples ontvangen." + (
                        f" (AvgT: {m_avg:.1f} µs)" if m_avg is not None else " (AvgT: N/A)"))
                else:
                    print("Master: GEEN data voor run.")
                    current_meta_all_master_runs.append(
                        {"Run": run, "Avg_Time_us": None, "ADC_Values": []})  # placeholder

                if s_adc:
                    current_meta_all_slave_runs.append({"Run": run, "Avg_Time_us": s_avg, "ADC_Values": s_adc})
                    print(f"Slave : {len(s_adc)} samples ontvangen." + (
                        f" (AvgT: {s_avg:.1f} µs)" if s_avg is not None else " (AvgT: N/A)"))
                else:
                    print("Slave : GEEN data voor run.")
                    current_meta_all_slave_runs.append(
                        {"Run": run, "Avg_Time_us": None, "ADC_Values": []})  # placeholder

                if run % SAVE_INTERVAL == 0 and run < RUNS_PER_META:  # Tussentijds save
                    if current_meta_all_master_runs or current_meta_all_slave_runs:
                        save_csv(current_meta_all_master_runs, current_meta_all_slave_runs, current_meta_filename)
                    else:
                        print(f"\n[Tussentijdse save] Geen data om op te slaan voor '{current_meta_filename}'.")

                print(f"--- Einde Meta-Run {meta_run_count} | Run {run} ---")
                if run < RUNS_PER_META:  # Niet pauzeren na de allerlaatste run van een meta-run
                    time.sleep(INTER_RUN_DELAY)

            # Einde van een meta-run: definitieve save voor deze meta-run
            print(f"\n{'=' * 10} EINDE META-RUN {meta_run_count} {'=' * 10}")
            if current_meta_all_master_runs or current_meta_all_slave_runs:
                save_csv(current_meta_all_master_runs, current_meta_all_slave_runs, current_meta_filename)
            else:
                print(f"Geen data verzameld in meta-run {meta_run_count} om op te slaan in '{current_meta_filename}'.")

            # Pauze tussen meta-runs (als het niet de laatste is)
            if meta_run_count < NUM_META_RUNS:
                print(f"\nMeta-run {meta_run_count} voltooid.")
                print(f"Pauze van {INTER_META_RUN_DELAY_S} seconden voor de volgende meta-run.")

                countdown_start_time = time.time()
                remaining_time = INTER_META_RUN_DELAY_S
                while remaining_time > 0:
                    sys.stdout.write(
                        f"\rVolgende meta-run start over {int(remaining_time)} seconden... (Ctrl+C om af te breken) ")
                    sys.stdout.flush()
                    time.sleep(1)
                    remaining_time = INTER_META_RUN_DELAY_S - (time.time() - countdown_start_time)
                sys.stdout.write("\r" + " " * 80 + "\r")  # Clear line
                sys.stdout.flush()

                input(f"Druk op Enter om meta-run {meta_run_count + 1} te starten, of Ctrl+C om te stoppen...")

        print("\nAlle geplande meta-runs zijn voltooid.")

    except KeyboardInterrupt:
        print("\nProgramma onderbroken door gebruiker (Ctrl+C).")
        if current_meta_all_master_runs or current_meta_all_slave_runs:
            print(f"Poging om data van de huidige (onderbroken) meta-run op te slaan in '{current_meta_filename}'...")
            save_csv(current_meta_all_master_runs, current_meta_all_slave_runs, current_meta_filename)
        else:
            print("Geen data van huidige meta-run om op te slaan.")
    except Exception as e:
        print(f"\nOnverwachte FOUT opgetreden: {e}")
        import traceback

        traceback.print_exc()
        if current_meta_all_master_runs or current_meta_all_slave_runs and current_meta_filename:
            print(f"Poging om data van de huidige meta-run op te slaan in '{current_meta_filename}'...")
            save_csv(current_meta_all_master_runs, current_meta_all_slave_runs, current_meta_filename)
    finally:
        close_ports()
        print("Programma beëindigd.")