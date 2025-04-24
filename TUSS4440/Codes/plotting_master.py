#!/usr/bin/env python3
"""
plotting_master.py – kies snel tussen:
  1) THRESHOLD-plot: spanning rond eerste V>THRESH  (relatief sample-index)
  2) RAW-plot     : volledige ruwe spanning tegen tijd

Kies de modus door precies één van de twee PLOT_MODE-regels
hieronder te de-commentariëren.
"""

# ---------------- PLOTMODUS – comment/uncomment -----------------
#PLOT_MODE = "THRESHOLD"      # spanning rond drempel   ◄─ aktief
#PLOT_MODE = "RAW"           # ruwe data tegen tijd
PLOT_MODE = "XCORR"
#PLOT_MODE = "SHIFT_EXTREMES"
#PLOT_MODE = "SHIFT_EXTREMES_MULTI"
# ----------------------------------------------------------------

# --------- Algemene instellingen --------------------------------
CSV_FILE       = "sync_data_10_runs.csv"
DEVICE_FILTER  = "Master"      # alleen dit board tonen
ADC_BITS       = 12
V_REF          = 3.3

# THRESHOLD-modus 
THRESH         = 0.94          # drempelspanning in V
WINDOW_BEFORE  =  -10           # hoeveel samples vóór crossing
WINDOW_AFTER   =  200          # hoeveel samples na  crossing
MAX_RUNS       = 100            # max runs om te plotten

# RAW-modus
TIME_UNIT      = "ms"          # "us", "ms" of "s"
MAX_POINTS     = None          # None = alles plotten

# ----------------
THRESH     = 0.94    # V-drempel om de flank te lokaliseren
SEG_LEN    = 50    # aantal samples na de flank om te alignen
MAX_LAG    = 5      # maximale integer-shift (in samples)
shift_list = []

# ---- extra instellingen voor SHIFT_WINDOW ----
WINDOW_IDX_START = 95
WINDOW_IDX_END   = 110
SHIFT_VALUES     = [0, 5, 10, 15]

# ------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate

def adc_to_v(adc):
    return adc / (2 ** ADC_BITS - 1) * V_REF

# Data inlezen
df = pd.read_csv(CSV_FILE)
df = df[df["Device"] == DEVICE_FILTER].copy()
df["Voltage"] = adc_to_v(df["ADC_Value"])


# --------- MODE 1: plot rond eerste crossing --------------------
if PLOT_MODE.upper() == "THRESHOLD":
    all_runs = sorted(df["Run"].unique())[:MAX_RUNS]
    plt.figure(figsize=(9, 4))
    for run in all_runs:
        g = df[df["Run"] == run].reset_index(drop=True)
        above = np.where(g["Voltage"] > THRESH)[0]
        if not len(above):
            continue                        # geen crossing in deze run
        onset = above[0]
        g["idx_rel"] = g.index - onset

        mask = (g["idx_rel"] >= WINDOW_BEFORE) & (g["idx_rel"] <= WINDOW_AFTER)
        plt.plot(g.loc[mask, "idx_rel"],
                 g.loc[mask, "Voltage"],
                 "-o", ms=5, alpha=0.8, label=f"Run {run}")

    plt.xlabel(f"Sample-index (0 = 1e V>{THRESH} V)")
    plt.ylabel("Spanning (V)")
    plt.title(f"{DEVICE_FILTER} – Voltage rond V>{THRESH} V "
              f"({WINDOW_BEFORE}…{WINDOW_AFTER} samples, max {MAX_RUNS} runs)")


# --------- MODE XCORR: integer shift per run -----------------------
elif PLOT_MODE.upper() == "XCORR":
    from scipy.signal import correlate

    runs = sorted(df.Run.unique())[:MAX_RUNS]

    # 1) bepaal bij referentie-run het startpunt (eerste > THRESH)
    ref_run = runs[0]
    ref_v   = df[df.Run == ref_run].Voltage.values
    ref_on  = np.argmax(ref_v > THRESH)
    ref_seg = ref_v[ref_on : ref_on + SEG_LEN]

    plt.figure(figsize=(9,4))

    for run in runs:
        g = df[df.Run == run].reset_index(drop=True)
        v = g.Voltage.values

        # 2) vind onset d.m.v. drempel crossing
        on = np.argmax(v > THRESH)
        seg = v[on : on + SEG_LEN]

        if len(seg) < len(ref_seg):
            continue  # skip als er te weinig samples zijn

        # 3) zero-mean segmenten
        seg  = seg  - seg.mean()
        ref  = ref_seg - ref_seg.mean()

        # 4) cross-corr & beste integer lag in ±MAX_LAG
        full = correlate(seg, ref_seg, mode="full")
        centre = len(seg) - 1
        window = full[centre - MAX_LAG: centre + MAX_LAG + 1]
        shift = np.argmax(window) - MAX_LAG

        shift_list.append((run, shift))

        # 5) verschuif X-as en plot
        idx     = np.arange(len(v))
        idx_corr= idx - (on + shift - ref_on)
        plt.plot(idx_corr, v, "-o", ms=4, alpha=0.7,
                 label=f"Run {run} (Δ={shift})")

    plt.xlabel("Sample-index (verschuiving per run)")
    plt.ylabel("Spanning (V)")
    plt.title(f"{DEVICE_FILTER} – XCORR-only rising edge")
    # plt.legend(ncol=2, fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

elif PLOT_MODE.upper() == "SHIFT_EXTREMES_MULTI":
    # basis‐parameters
    runs        = sorted(df.Run.unique())[:MAX_RUNS]
    onsets      = {run: np.argmax(df[df.Run==run].Voltage.values > THRESH)
                   for run in runs}
    ref_run     = runs[0]
    ref_on      = onsets[ref_run]
    v_ref       = df[df.Run==ref_run].Voltage.values
    ref_seg     = v_ref[ref_on : ref_on + SEG_LEN]

    # de vier N_extreme waarden die je wilde
    extreme_values = [0, 5, 10, 15]

    # 2×2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)

    for idx, N_EXTREME in enumerate(extreme_values):
        # bereken rijnummer en kolomnummer
        row, col = divmod(idx, 2)
        ax = axes[row, col]

        # 1) shifts bepalen
        shifts = {}
        for run in runs:
            v   = df[df.Run == run].Voltage.values
            on  = onsets[run]
            seg = v[on : on + SEG_LEN]
            if len(seg) < len(ref_seg):
                continue
            seg0   = seg - seg.mean()
            ref0   = ref_seg - ref_seg.mean()
            full   = correlate(seg0, ref0, mode="full")
            centre = len(seg0) - 1
            window = full[centre - MAX_LAG : centre + MAX_LAG + 1]
            shifts[run] = (np.argmax(window) - MAX_LAG)

        # 2) selecteer onder- en boventoppen
        df_s      = (pd.DataFrame.from_dict(shifts, orient="index", columns=["shift"])
                       .reset_index().rename(columns={"index":"run"})
                       .sort_values("shift"))
        low_runs  = set(df_s.head(N_EXTREME)["run"])
        high_runs = set(df_s.tail(N_EXTREME)["run"])

        # 3) plot alle runs met extra ±1 voor extremes
        for run in runs:
            v     = df[df.Run == run].Voltage.values
            on    = onsets.get(run)
            shift = shifts.get(run, 0)
            if on is None:
                continue

            idx_raw  = np.arange(len(v))
            idx_corr = idx_raw - (on + shift - ref_on)

            extra = 0
            if run in low_runs:
                extra = -1
            elif run in high_runs:
                extra = +1

            ax.plot(idx_corr + extra, v,
                    "-o", ms=3, alpha=0.6,
                    label=f"Run {run}" if run == runs[0] else None)

        # labels en titel per subplot
        ax.set_title(f"N_extreme = {N_EXTREME}")
        ax.grid(alpha=0.3)
        if row == 1:
            ax.set_xlabel("Sample-index (verschoven)")
        if col == 0:
            ax.set_ylabel("Spanning (V)")

    # één legend-veld voor de hele figuur
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels,
               title="Runs", ncol=6, loc="upper center", fontsize=8)
    fig.suptitle(f"{DEVICE_FILTER} – SHIFT_EXTREMES_MULTI", y=1.02)
    fig.tight_layout()
    plt.show()



# ---------- MODE SHIFT_EXTRMES : Shift hoogste en laagste runs extra ----------
elif PLOT_MODE.upper() == "SHIFT_EXTREMES":
    # parameters
    MAX_LAG    = 50               # ± samples om te schuiven
    SEG_LEN    = WINDOW_AFTER     # hoeveel samples ná de onset
    N_EXTREME  = 0          # top / bottom runs
    OFFSET_EXT = 1                # extra samples

    runs = sorted(df.Run.unique())[:MAX_RUNS]

    # 1) onset bepalen per run
    onsets = {}
    for run in runs:
        v = df[df.Run == run].Voltage.values
        on = np.argmax(v > THRESH)
        onsets[run] = on

    # referentie‐run
    ref_run = runs[0]
    ref_on  = onsets[ref_run]
    v_ref   = df[df.Run == ref_run].Voltage.values
    ref_seg = v_ref[ref_on : ref_on + SEG_LEN]

    # 2) shift bepalen per run
    shifts = {}
    for run in runs:
        v = df[df.Run == run].Voltage.values
        on = onsets[run]
        seg = v[on : on + SEG_LEN]
        if len(seg) < len(ref_seg):
            continue
        # zero‐mean
        seg_zero = seg - seg.mean()
        ref_zero = ref_seg - ref_seg.mean()
        # cross‐correlatie
        full    = correlate(seg_zero, ref_zero, mode="full")
        center  = len(seg_zero) - 1
        window  = full[center - MAX_LAG : center + MAX_LAG + 1]
        shift   = np.argmax(window) - MAX_LAG
        shifts[run] = shift

    # 3) top/bottom N_EXTREME runs
    df_s = (pd.DataFrame.from_dict(shifts, orient="index", columns=["shift"])
              .reset_index().rename(columns={"index":"run"})
              .sort_values("shift"))
    low_runs  = set(df_s.head(N_EXTREME)["run"])
    high_runs = set(df_s.tail(N_EXTREME)["run"])

    # 4) plot met extra offset voor extremes
    plt.figure(figsize=(9,4))
    for run in runs:
        v     = df[df.Run == run].Voltage.values
        on    = onsets.get(run)
        shift = shifts.get(run, 0)
        if on is None:
            continue

        idx = np.arange(len(v))
        # basis‐verschuiving: align op ref_on + shift
        idx_corr = idx - (on + shift - ref_on)

        # extra ±OFFSET_EXT voor extreem‐runs
        extra = 0
        if run in low_runs:
            extra = -OFFSET_EXT
        elif run in high_runs:
            extra = +OFFSET_EXT

        plt.plot(idx_corr + extra, v, "-o", ms=4, alpha=0.7,
                 label=f"Run {run} (shift={shift:+d}, extra={extra:+d})")

    plt.xlabel("Sample-index (verschoven + extremes)")
    plt.ylabel("Spanning (V)")
    plt.title(f"{DEVICE_FILTER} – SHIFT_EXTREMES modus")
    #plt.legend(ncol=2, fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

# --------- MODE 2: ruwe data tegen tijd -------------------------
elif PLOT_MODE.upper() == "RAW":
    if TIME_UNIT == "s":
        scale = 1e-6
        xlabel = "Tijd (s)"
    elif TIME_UNIT == "ms":
        scale = 1e-3
        xlabel = "Tijd (ms)"
    else:           # "us"
        scale = 1.0
        xlabel = "Tijd (µs)"

    df["TimePlot"] = df["Timestamp_us"] * scale
    if MAX_POINTS:
        df = df.iloc[:MAX_POINTS]

    plt.figure(figsize=(9, 4))
    for run, g in df.groupby("Run"):
        plt.plot(g["TimePlot"], g["Voltage"],
                 ".", ms=3, alpha=0.6, label=f"Run {run}")

    plt.xlabel(xlabel)
    plt.ylabel("Spanning (V)")
    plt.title(f"{DEVICE_FILTER} – ruwe spanning versus tijd")

else:
    raise ValueError(f"PLOT_MODE '{PLOT_MODE}' ongeldig (kies 'THRESHOLD' of 'RAW').")

# --------- Afwerking --------------------------------------------
plt.grid(True, alpha=0.3)
# commentarieer onderstaande regel uit als je de legenda wilt:
# plt.legend(ncol=2, fontsize=8)
plt.tight_layout()
plt.show()
