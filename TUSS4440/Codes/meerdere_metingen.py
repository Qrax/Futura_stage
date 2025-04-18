#!/usr/bin/env python3
"""
sync_logger.py – logt ADC‑data van zowel MASTER (COM6) als SLAVE (COM8)
en bewaart ze in één CSV met kolommen:
    Device, Run, SampleIndex, Timestamp_us, ADC_Value, Avg_Time_us
en schrijft elke SAVE_INTERVAL runs tussentijds naar disk.
"""
# gebaseerd op sync_logger.py (origineel) :contentReference[oaicite:0]{index=0}&#8203;:contentReference[oaicite:1]{index=1}

# --- Imports --------------------------------------------------------------
import serial
import time
import threading
import queue
import sys
import pandas as pd

# --- Config ---------------------------------------------------------------
MASTER_PORT           = 'COM6'
SLAVE_PORT            = 'COM8'
BAUD_RATE             = 115200
SERIAL_TIMEOUT        = 1          # algemeen lees‑timeout
CONNECT_DELAY         = 0.5
CONFIRM_READ_DURATION = 1
DATA_READ_TIMEOUT     = 2          # wachttijd per meetrun (s)
END_MARKER            = "E"

NUM_RUNS              = 1000
INTER_RUN_DELAY       = 2          # pauze tussen runs (s)
SAVE_INTERVAL         = 10         # aantal runs tussen tussentijdse CSV-saves
FILENAME              = "sync_data.csv"  # bestandsnaam voor CSV

# --- Globale variabelen ---------------------------------------------------
master_ser = slave_ser = None
data_queue_master = queue.Queue()
data_queue_slave  = queue.Queue()

# --- Helper: tussentijds en eind‑save -------------------------------------
def save_csv(master_runs, slave_runs, fname):
    """Flatten alle runs en sla op als CSV (overschrijft steeds)."""
    records = []
    def flatten(run_dicts, device_name):
        for rd in run_dicts:
            run_no  = rd["Run"]
            avg_dt  = rd["Avg_Time_us"]
            ts_ok   = avg_dt is not None and avg_dt > 0
            for i, adc in enumerate(rd["ADC_Values"]):
                records.append({
                    "Device"      : device_name,
                    "Run"         : run_no,
                    "SampleIndex" : i,
                    "Timestamp_us": i * avg_dt if ts_ok else None,
                    "ADC_Value"   : adc,
                    "Avg_Time_us" : avg_dt
                })
    flatten(master_runs, "Master")
    flatten(slave_runs,  "Slave")
    df = pd.DataFrame(records)
    df.to_csv(fname, index=False)
    print(f"\n[Tussentijdse save] {len(df)} rijen naar '{fname}' geschreven.")

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
    """Leest ADC‑data uit één poort; zet (avg_time, adc_list) in queue."""
    port_name = ser_port.port
    buffer = ''
    adc_values = []
    avg_diff_us = None
    found_avg = False
    start_time = time.time()

    while not stop_event.is_set() and (time.time() - start_time < DATA_READ_TIMEOUT):
        try:
            if ser_port.in_waiting:
                chunk = ser_port.read(ser_port.in_waiting).decode(errors='ignore')
                print(f"RAW {port_name}: {chunk}", end='')      # live raw print
                buffer += chunk

                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    line = line.strip()
                    if not line:
                        continue
                    if line == END_MARKER:
                        data_q.put((avg_diff_us, adc_values))
                        return

                    prefix = "Average time between samples (us):"
                    if not found_avg and line.startswith(prefix):
                        try:
                            avg_diff_us = float(line[len(prefix):].strip())
                            found_avg = True
                            print(f"\n>>> {port_name} AvgT={avg_diff_us:.1f} µs <<<")
                        except ValueError:
                            print(f"\nWARN {port_name}: kan AvgT niet parsen: '{line}'")
                        continue

                    # ADC‑regel: "timestamp, value"
                    if ',' in line:
                        try:
                            _, adc_str = line.split(',', 1)
                            adc = int(adc_str.strip())
                            adc_values.append(adc)
                        except ValueError:
                            pass
            else:
                time.sleep(0.01)
        except serial.SerialException as e:
            print(f"\nSeriële fout {port_name}: {e}")
            break
        except Exception as e:
            print(f"\nAlgemene fout {port_name}: {e}")
            break

    # timeout – wat we hebben toch opsturen
    data_q.put((avg_diff_us, adc_values))

def close_ports():
    """Sluit beide seriële poorten."""
    print("\n--- Sluiten poorten ---")
    try:
        if master_ser and master_ser.is_open:
            master_ser.close()
            print(f"{MASTER_PORT} gesloten.")
    finally:
        if slave_ser and slave_ser.is_open:
            slave_ser.close()
            print(f"{SLAVE_PORT} gesloten.")

# --------------------------------------------------------------------------
if __name__ == "__main__":
    if not connect_ports():
        sys.exit(1)
    if not set_modes_and_show_initial_output():
        close_ports()
        sys.exit(1)

    print(f"\nKlaar om {NUM_RUNS} sync‑cycli uit te voeren.")
    input("Druk op Enter om te beginnen …")

    all_master_runs = []
    all_slave_runs  = []

    try:
        for run in range(1, NUM_RUNS + 1):
            print(f"\n========== Run {run}/{NUM_RUNS} ==========")

            # queues leegmaken
            for q in (data_queue_master, data_queue_slave):
                while not q.empty():
                    q.get_nowait()

            # lees‑threads starten
            stop_event = threading.Event()
            t_master = threading.Thread(target=robust_read_serial_data,
                                        args=(master_ser, data_queue_master, stop_event),
                                        daemon=True)
            t_slave = threading.Thread(target=robust_read_serial_data,
                                       args=(slave_ser, data_queue_slave, stop_event),
                                       daemon=True)
            t_master.start()
            t_slave.start()
            time.sleep(0.1)

            # SYNC versturen
            print(">>> SYNC naar master …")
            master_ser.reset_input_buffer()
            master_ser.write(b'SYNC\n')

            # wachten op threads
            t_master.join(DATA_READ_TIMEOUT)
            t_slave.join(DATA_READ_TIMEOUT)
            stop_event.set()

            # data ophalen
            try:
                m_avg, m_adc = data_queue_master.get_nowait()
            except queue.Empty:
                m_avg, m_adc = None, []
            try:
                s_avg, s_adc = data_queue_slave.get_nowait()
            except queue.Empty:
                s_avg, s_adc = None, []

            if m_adc:
                all_master_runs.append({"Run": run, "Avg_Time_us": m_avg, "ADC_Values": m_adc})
                print(f"Master: {len(m_adc)} samples")
            else:
                print("Master: GEEN data")

            if s_adc:
                all_slave_runs.append({"Run": run, "Avg_Time_us": s_avg, "ADC_Values": s_adc})
                print(f"Slave : {len(s_adc)} samples")
            else:
                print("Slave : GEEN data")

            # Tussentijdse save
            if run % SAVE_INTERVAL == 0:
                save_csv(all_master_runs, all_slave_runs, FILENAME)

            print(f"========== einde run {run} ==========")
            time.sleep(INTER_RUN_DELAY)

    except KeyboardInterrupt:
        print("\nOnderbroken door gebruiker.")
    finally:
        close_ports()

    # Eindsave (ook als NUM_RUNS niet deelbaar is)
    if all_master_runs or all_slave_runs:
        save_csv(all_master_runs, all_slave_runs, FILENAME)
        print(f"Einddata opgeslagen als '{FILENAME}'")
    else:
        print("Geen data verzameld – programma stopt.")
