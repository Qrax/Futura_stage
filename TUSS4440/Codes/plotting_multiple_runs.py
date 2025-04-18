#!/usr/bin/env python3
"""
sync_plotter.py – interactieve visualisatie van de CSV die door
sync_logger.py is aangemaakt (kolom 'Device','Run','SampleIndex',
'ADC_Value','Timestamp_us' aanwezig).

Bij opstart:
  • Kies device (Master/Slave/Both), default Master
  • Stel referentiespanning in (V_ref), default 3.3 V

Menu:
  a) Ruwe data – alle runs
  b) Gemiddelde runs
  c) Statistiek max‑amplitudes
  d) Steepest‑slope onset
  e) Threshold‑onset
  x) Stoppen
"""
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Config ---------------------------------------------------------------
ADC_BITS = 12            # 12‑bit ADC

# --- Utilities ------------------------------------------------------------
def load_data(fname="sync_data.csv"):
    df = pd.read_csv(fname)
    req = {"Device","Run","SampleIndex","ADC_Value","Timestamp_us"}
    if not req.issubset(df.columns):
        raise KeyError(f"CSV mist kolommen: {req - set(df.columns)}")
    df["Run"] = df["Run"].astype(int)
    df["SampleIndex"] = df["SampleIndex"].astype(int)
    df["Device"] = df["Device"].astype(str)
    return df

def select_device(df):
    """
    Kies device: Master, Slave of Both. Default Master bij Enter.
    """
    devices = sorted(df["Device"].unique())
    valid = ["Master","Slave"]
    if len(devices) > 1:
        valid.append("Both")
    prompt = f"Kies device ({'/'.join(valid)}) [Master]: "
    choice = input(prompt).strip()
    if choice == "" or choice == "Master":
        return df[df["Device"]=="Master"]
    if choice == "Slave":
        return df[df["Device"]=="Slave"]
    if choice == "Both" and "Both" in valid:
        return df.copy()
    print("Onbekende keuze, default Master.")
    return df[df["Device"]=="Master"]

def finalize_plot(n_runs):
    """Tight layout altijd, legend alleen bij ≤10 runs."""
    if n_runs <= 10:
        plt.legend(fontsize="small", ncol=1)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def adc_to_volts(series, vref):
    """Converteer ADC‑waarden naar spanning (V)."""
    return series / (2**ADC_BITS - 1) * vref

# --- Plot‑functies --------------------------------------------------------
def plot_all_runs_raw(df):
    plt.figure()
    for (_, run), grp in df.groupby(["Device","Run"]):
        plt.plot(grp["SampleIndex"], grp["Voltage"], alpha=0.6)
    plt.title("Ruwe data – alle runs")
    plt.xlabel("SampleIndex")
    plt.ylabel("Spanning (V)")
    finalize_plot(df["Run"].nunique())

def plot_average_runs(df):
    stats = df.groupby("SampleIndex")["Voltage"].agg(["mean","std"])
    plt.figure()
    x = stats.index.values
    plt.plot(x, stats["mean"], label="gemiddelde")
    plt.fill_between(x,
                     stats["mean"]-stats["std"],
                     stats["mean"]+stats["std"],
                     alpha=0.2)
    plt.title("Gemiddelde spanning over runs")
    plt.xlabel("SampleIndex")
    plt.ylabel("Spanning (V)")
    finalize_plot(1)

def plot_max_amp_stats(df, bins='fd'):
    idx = df.groupby("Run")["Voltage"].idxmax()
    max_vs = df.loc[idx, "Voltage"]
    if max_vs.empty:
        print("Geen data voor histogram.")
        return
    if bins is None:
        try: bins = int(input("Aantal bins? "))
        except: bins = 'fd'
    if bins == 'fd':
        q25,q75 = np.percentile(max_vs,[25,75])
        iqr = q75-q25
        w = 2*iqr/(len(max_vs)**(1/3))
        bins = max(1,int(np.ceil((max_vs.max()-max_vs.min())/w)))
    mu,sig = max_vs.mean(), max_vs.std(ddof=0)
    x = np.linspace(max_vs.min(), max_vs.max(), 400)
    pdf = 1/(sig*np.sqrt(2*np.pi))*np.exp(-0.5*((x-mu)/sig)**2)
    plt.figure(figsize=(6,4))
    plt.hist(max_vs, bins=bins, density=True, alpha=0.5,
             label=f"N={len(max_vs)}")
    plt.plot(x, pdf, label=f"N(μ={mu:.3f} V, σ={sig:.3f} V)")
    plt.title("Max‑spanning per run")
    plt.xlabel("Spanning (V)")
    plt.ylabel("Dichtheid")
    finalize_plot(1)

def plot_all_runs_slope_onset(df, window=5):
    plt.figure()
    for (_, run), grp in df.groupby(["Device","Run"]):
        grp = grp.sort_values("SampleIndex").copy()
        vals = grp["Voltage"].values
        slopes = (vals[window:] - vals[:-window]) / window
        j = np.argmax(slopes) + window//2
        onset = grp["SampleIndex"].iloc[j]
        grp["idx_rel"] = grp["SampleIndex"] - onset
        plt.plot(grp["idx_rel"], grp["Voltage"],
                 marker='.', linestyle='-', markersize=3, alpha=0.7)
    plt.title(f"Aligned op steepest slope (window={window})")
    plt.xlabel("SampleIndex rel (onset=0)")
    plt.ylabel("Spanning (V)")
    finalize_plot(df["Run"].nunique())

def plot_all_runs_threshold_onset(df, frac=0.1):
    plt.figure()
    for (_, run), grp in df.groupby(["Device","Run"]):
        grp = grp.sort_values("SampleIndex").copy()
        baseline = grp["Voltage"].iloc[:5].mean()
        peak     = grp["Voltage"].max()
        thresh   = baseline + frac*(peak - baseline)
        cross    = grp[grp["Voltage"] >= thresh]
        if not cross.empty:
            onset = cross["SampleIndex"].iloc[0]
        else:
            vals = grp["Voltage"].values
            slopes = np.diff(vals)
            onset = grp["SampleIndex"].iloc[np.argmax(slopes)+1]
        grp["idx_rel"] = grp["SampleIndex"] - onset
        plt.plot(grp["idx_rel"], grp["Voltage"],
                 marker='.', linestyle='-', markersize=3, alpha=0.7)
    plt.title(f"Aligned op threshold {frac*100:.0f}%")
    plt.xlabel("SampleIndex rel (onset=0)")
    plt.ylabel("Spanning (V)")
    finalize_plot(df["Run"].nunique())

# --- Mainmenu ------------------------------------------------------------
if __name__ == "__main__":
    # CSV inladen
    fname = input("CSV‑bestand (Enter=sync_data.csv): ").strip() or "sync_data.csv"
    try:
        df_total = load_data(fname)
    except Exception as e:
        print("Fout bij laden:", e)
        sys.exit(1)

    # Device selectie
    df = select_device(df_total)

    # Referentiespanning
    try:
        V_REF = float(input("Referentiespanning V_ref (V, default=3.3): ").strip() or 3.3)
    except ValueError:
        V_REF = 3.3
    df["Voltage"] = adc_to_volts(df["ADC_Value"], V_REF)

    # Menu
    while True:
        print("\nMenu:")
        print("  a) Ruwe data alle runs")
        print("  b) Gemiddelde runs")
        print("  c) Statistiek max‑amplitudes")
        print("  d) Steepest‑slope onset")
        print("  e) Threshold‑onset")
        print("  x) Stoppen")
        k = input("> ").strip().lower()
        if k == 'a':
            plot_all_runs_raw(df)
        elif k == 'b':
            plot_average_runs(df)
        elif k == 'c':
            plot_max_amp_stats(df, bins=None)
        elif k == 'd':
            w = int(input("Slope‑window (samples, def=5): ").strip() or 5)
            plot_all_runs_slope_onset(df, window=w)
        elif k == 'e':
            f = float(input("Threshold‑frac (0–1, def=0.1): ").strip() or 0.1)
            plot_all_runs_threshold_onset(df, frac=f)
        elif k == 'x':
            break
        else:
            print("Onbekende keuze.")
