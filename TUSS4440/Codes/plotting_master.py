#!/usr/bin/env python3
# align_on_threshold.py – ΔV tegen relatieve datapunt‑index, uitgelijnd op V>0.84 V

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Instellingen
CSV_FILE = "sync_data.csv"
ADC_BITS = 12
V_REF = 3.3
THRESH = 0.94  # voltage threshold in V


def adc_to_v(adc):
    return adc / (2 ** ADC_BITS - 1) * V_REF


# Data inlezen en converteren
df = pd.read_csv(CSV_FILE)
df = df[df["Device"] == "Master"].copy()
df["Voltage"] = adc_to_v(df["ADC_Value"])

# Bereken ΔV per run
df["dV"] = df.groupby("Run")["Voltage"].diff()

plt.figure(figsize=(9, 4))

for run, g in df.groupby("Run"):
    # herindexeer van 0..N-1
    g = g.reset_index(drop=True)

    # eerste crossing van THRESH
    above = np.where(g["Voltage"] > THRESH)[0]
    if len(above) == 0:
        # overslaan als nooit boven THRESH
        continue
    onset = above[0]

    # relatieve datapunt‑index
    g["idx_rel"] = g.index - onset

    # plot ΔV (vanaf tweede rij, want diff geeft NaN in eerste)
    x = g["idx_rel"].iloc[1:]
    y = g["dV"].iloc[1:]
    # x = relatieve datapunt‐index, y = ΔV
    mask = (x >= -10) & (x <= 20)
    plt.plot(x[mask], y[mask], '-o', ms=6, alpha=0.8, label=f'Run {run}')

# Opmaak
plt.xlabel("Datapunt (relatief aan eerste V>0.84 V)")
plt.ylabel("ΔVoltage (V per sample)")
plt.title("Master – ΔVoltage rond eerste V>0.84 V")
plt.grid(True)
#plt.legend(loc='best')
plt.tight_layout()
plt.show()
