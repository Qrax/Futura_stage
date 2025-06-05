# Plot_Mods/plot_explain_exp_trend.py
# ---------------------------------------------------------------------------
#   Exponential-trend explanation plot compatible with plotting_master.py
#   Fits either:
#       y = C + A*exp(B t)                         (default, USE_DOUBLE_EXP=False)
#       y = C + A1*exp(B1 t) + A2*exp(B2 t)        (if USE_DOUBLE_EXP=True)
# ---------------------------------------------------------------------------
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ─── Instellingen ───────────────────────────────────────────────────────────
USE_DOUBLE_EXP: bool = False          # ← op True voor dubbele exponent
TAIL_FRACTION_FOR_OFFSET: float = 0.10  # 10 % laatste samples ⇒ DC-offset
MAX_FIT_EVALS: int = 50_000
# ────────────────────────────────────────────────────────────────────────────

def _exp_single(t, A, B, C):            # y = C + A·e^(B t)
    return C + A * np.exp(B * t)

def _exp_double(t, A1, B1, A2, B2, C):  # y = C + A1·e^(B1 t)+A2·e^(B2 t)
    return C + A1 * np.exp(B1 * t) + A2 * np.exp(B2 * t)

def _estimate_offset(y, frac=TAIL_FRACTION_FOR_OFFSET):
    tail = y[int((1.0 - frac) * len(y)):]
    return float(np.mean(tail) if tail.size else 0.0)

def _safe_set_window_title(fig, title):
    mgr = getattr(fig.canvas, "manager", None)
    if mgr and hasattr(mgr, "set_window_title"):
        try: mgr.set_window_title(title)
        except Exception: pass

# ---------------------------------------------------------------------------
def generate_plot_explain_exp_trend(dfs, act_lbls, settings, sum_cache,
                                    plt_instance=plt):
    """
    Wordt door plotting_master.py aangeroepen; maakt per dataset één 2×2-figuur
    met (1) ruwe segment, (2) segment-minus-offset, (3) log-ruimte visualisatie,
    (4) gereconstrueerde trend.
    """
    seg_len   = settings.FIT_WINDOW_POST_PEAK
    dt_us     = settings.SAMPLE_TIME_DELTA_US
    time_lbl  = getattr(settings, "tu_raw_lbl", "µs")

    for i_file, lbl in enumerate(act_lbls):
        s   = sum_cache.get(lbl, {})
        mt  = s.get("mean_trace");   n_runs = s.get("N_for_mean", 0)
        if mt is None or n_runs == 0:
            print(f"D(EXPLAIN_EXP_TREND – {lbl}): Geen trace."); continue
        if len(mt) < settings.POST_PEAK_OFFSET_SAMPLES + 8:
            print(f"D(EXPLAIN_EXP_TREND – {lbl}): Trace te kort."); continue

        idx_peak  = int(np.argmax(mt))
        i0, i1    = idx_peak + settings.POST_PEAK_OFFSET_SAMPLES, \
                    idx_peak + settings.POST_PEAK_OFFSET_SAMPLES + seg_len
        i1        = min(i1, len(mt))
        if i1 - i0 < 8:
            print(f"D(EXPLAIN_EXP_TREND – {lbl}): Segment te kort."); continue

        y_seg     = mt[i0:i1].astype(float)
        x_idx     = np.arange(y_seg.size)
        t_s       = x_idx * dt_us * 1e-6     # seconden  (voor curve_fit)
        t_us      = x_idx * dt_us

        C0        = _estimate_offset(y_seg)
        y_corr    = y_seg - C0
        mask_pos  = y_corr > 0

        # ------------ curve_fit ------------------------------------------
        if USE_DOUBLE_EXP:
            model  = _exp_double
            p0     = [y_corr.max(), -2e3, y_corr.max()*0.3, -2e2, C0]
            bounds = ([0, -np.inf, 0, -np.inf, C0*0.5],
                      [np.inf, 0,   np.inf, 0,   C0*1.5])
        else:
            model  = _exp_single
            p0     = [y_corr.max(), -2e3, C0]
            bounds = ([0, -np.inf, C0*0.5],
                      [np.inf, 0,  C0*1.5])

        try:
            popt, _ = curve_fit(model, t_s, y_seg, p0=p0,
                                bounds=bounds, maxfev=MAX_FIT_EVALS)
            y_fit = model(t_s, *popt)
        except Exception as e:
            print(f"W(EXPLAIN_EXP_TREND – {lbl}): curve_fit mislukte ({e}).")
            continue

        # ------------ Plot ------------------------------------------------
        fig, ax = plt_instance.subplots(2, 2, figsize=(14, 10))
        col     = settings.PLOT_COLORS[i_file % len(settings.PLOT_COLORS)]

        ax[0,0].plot(t_us, y_seg, col, label="Origineel (Y)")
        ax[0,0].set(title="1. Origineel Segment",
                    xlabel=f"Tijd ({time_lbl})", ylabel="Voltage"); ax[0,0].grid()
        ax[0,0].legend(fontsize=8)

        ax[0,1].plot(t_us, y_corr, "--", color=col,
                     label=f"Y − C  (C≈{C0:.0f})")
        ax[0,1].set(title="2. Segment na Offset", xlabel=f"Tijd ({time_lbl})",
                    ylabel="Voltage (Y')"); ax[0,1].grid(); ax[0,1].legend(fontsize=8)

        if np.any(mask_pos):
            ax[1,0].plot(x_idx[mask_pos], np.log(y_corr[mask_pos]),
                         ".", ms=3, color=col, label="log(Y')")
            if not USE_DOUBLE_EXP:      # teken lineaire log-fit
                B_sec = popt[1];  A_log = np.log(popt[0])
                slope_idx = B_sec * dt_us * 1e-6   # terug naar index-schaal
                ax[1,0].plot(x_idx,
                             A_log + slope_idx*x_idx,
                             "k--", label=f"log-fit: logY ≈ {A_log:.2f}"
                                         f"+({slope_idx:.2e})·x")
        ax[1,0].set(title="3. Log-Transform & Fit",
                    xlabel="Sample-index", ylabel="log(Y')"); ax[1,0].grid()
        ax[1,0].legend(fontsize=8)

        ax[1,1].plot(t_us, y_seg, alpha=.4, color=col, label="Origineel (Y)")
        ax[1,1].plot(t_us, y_fit, "r--", lw=1.5, label="Gereconstrueerde Trend")
        ax[1,1].set(title="4. Trend vs. Origineel",
                    xlabel=f"Tijd ({time_lbl})", ylabel="Voltage")
        ax[1,1].grid(); ax[1,1].legend(fontsize=8)

        if USE_DOUBLE_EXP:
            A1,B1,A2,B2,C = popt
            txt = (f"y = C + A₁·e^(B₁t)+A₂·e^(B₂t)\\n"
                   f"A₁={A1:.0f}, B₁={B1:.2e}, "
                   f"A₂={A2:.0f}, B₂={B2:.2e}, C={C:.0f}")
        else:
            A,B,C = popt
            txt = f"y = C + A·e^(Bt);  A={A:.0f}, B={B:.2e}, C={C:.0f}"
        fig.suptitle(f"Uitleg Trend Fit – {lbl}\\n{txt}", fontsize=11, y=0.97)
        _safe_set_window_title(fig, f"Plot:EXPLAIN_EXP_TREND – {lbl}")

    return "explain_exp_trend_completed_exp"
