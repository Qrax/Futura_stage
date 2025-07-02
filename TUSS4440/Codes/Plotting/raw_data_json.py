# Sla dit op als: create_raw_data.py
import os, pandas as pd, numpy as np, json
from scipy.fft import rfft, rfftfreq
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

print("--- RAW Website Data Generation Script ---")

# --- CONFIGURATIE (blijft hetzelfde) ---
TARGET_DATA_SUBFOLDER = os.path.join("..", "..", "data", "UltraSoon_Measurements")
SAMPLE_TIME_DELTA_US = 4.63
DEVICE_FILTER = "Master"
MAX_RUNS = 20
WINDOW_BEFORE = 50
POST_PEAK_OFFSET_SAMPLES = 20

# FINALE STRUCTUUR: Met custom downsampling instellingen per conditie
CONDITIONS_TO_PROCESS = {
    'aluminium_0mm': {
        'profile': {'THRESH': 1500, 'MAX_WINDOW_AFTER': 1400, 'DETREND_TYPE': 'exponential',
                    'MAX_PERIOD_PLOT_US': 1800},
        'downsample_fft': {'high_detail_x_max': 500, 'high_detail_ratio': 0.7},
        'metingen': [
            {'file': "al_0mm_laatste_metingen_meta_1.csv", 'label': "AL Meting 1 (Defectloos)"},
            {'file': "al_0mm_laatste_metingen_meta_2.csv", 'label': "AL Meting 2 (Defectloos)"},
            {'file': "al_0mm_laatste_metingen_meta_3.csv", 'label': "AL Meting 3 (Defectloos)"},
            {'file': "al_0mm_laatste_metingen_meta_4.csv", 'label': "AL Meting 4 (Defectloos)"},
            {'file': "al_0mm_please_meta_3.csv", 'label': "AL Meting 5 (Defectloos)"},
        ]
    },
    'aluminium_5mm': {
        'profile': {'THRESH': 1500, 'MAX_WINDOW_AFTER': 1400, 'DETREND_TYPE': 'exponential',
                    'MAX_PERIOD_PLOT_US': 1800},
        'downsample_fft': {'high_detail_x_max': 500, 'high_detail_ratio': 0.8},
        'metingen': [
            {'file': "al_5mm_5_metingen_meta_1.csv", 'label': "AL Meting 1 (5mm)"},
            {'file': "al_5mm_5_metingen_meta_2.csv", 'label': "AL Meting 2 (5mm)"},
            {'file': "al_5mm_5_metingen_meta_3.csv", 'label': "AL Meting 3 (5mm)"},
            {'file': "al_5mm_5_metingen_meta_4.csv", 'label': "AL Meting 4 (5mm)"},
            {'file': "al_5mm_5_metingen_meta_5.csv", 'label': "AL Meting 5 (5mm)"},
        ]
    },
    'aluminium_15mm': {
        'profile': {'THRESH': 1500, 'MAX_WINDOW_AFTER': 1400, 'DETREND_TYPE': 'exponential',
                    'MAX_PERIOD_PLOT_US': 1800},
        'downsample_fft': {'high_detail_x_max': 1200, 'high_detail_ratio': 0.5},
        'metingen': [
            {'file': "al_15mm_please_meta_1.csv", 'label': "Sessie 1, Meting 1"},
            {'file': "al_15mm_please_meta_3.csv", 'label': "Sessie 1, Meting 3"},
            {'file': "al_15mm_gleuf_teuf_meta_2.csv", 'label': "Sessie 2, Meting 2"},
            {'file': "al_15mm_ff_opnieuw_meta_1.csv", 'label': "Sessie 3, Meting 1"},
            {'file': "al_15mm_ff_opnieuw_meta_4.csv", 'label': "Sessie 3, Meting 4"},
        ]
    },
    'g10_0mm': {
        'profile': {'THRESH': 1000, 'MAX_WINDOW_AFTER': 500, 'DETREND_TYPE': 'exponential', 'MAX_PERIOD_PLOT_US': 600},
        'downsample_fft': {'high_detail_x_max': 300, 'high_detail_ratio': 0.7},
        'metingen': [
            {'file': "g10_0mm_3e_test_meta_1.csv", 'label': "G10 Meting 1 (Defectloos)"},
            {'file': "g10_0mm_3e_test_meta_2.csv", 'label': "G10 Meting 2 (Defectloos)"},
            {'file': "g10_0mm_3e_test_meta_3.csv", 'label': "G10 Meting 3 (Defectloos)"},
            {'file': "g10_0mm_3e_test_meta_4.csv", 'label': "G10 Meting 4 (Defectloos)"},
            {'file': "g10_0mm_3e_test_meta_5.csv", 'label': "G10 Meting 5 (Defectloos)"},
        ]
    },
    'g10_5mm': {
        'profile': {'THRESH': 1000, 'MAX_WINDOW_AFTER': 500, 'DETREND_TYPE': 'exponential', 'MAX_PERIOD_PLOT_US': 600},
        'downsample_fft': {'high_detail_x_max': 200, 'high_detail_ratio': 0.8},
        'metingen': [
            {'file': "g10_5mm_final_meta_1.csv", 'label': "G10 Meting 1 (5mm)"},
            {'file': "g10_5mm_5_metingen_meta_2.csv", 'label': "G10 Meting 2 (5mm)"},
            {'file': "g10_5mm_5_metingen_meta_3.csv", 'label': "G10 Meting 3 (5mm)"},
            {'file': "g10_5mm_5_metingen_meta_4.csv", 'label': "G10 Meting 4 (5mm)"},
            {'file': "g10_5mm_5_metingen_meta_5.csv", 'label': "G10 Meting 5 (5mm)"},
        ]
    },
    'g10_15mm': {
        'profile': {'THRESH': 1000, 'MAX_WINDOW_AFTER': 500, 'DETREND_TYPE': 'exponential', 'MAX_PERIOD_PLOT_US': 600},
        'downsample_fft': {'high_detail_x_max': 300, 'high_detail_ratio': 0.7},
        'metingen': [
            {'file': "g10_15mm_5_metingen_meta_1.csv", 'label': "G10 Meting 1 (15mm)"},
            {'file': "g10_15mm_5_metingen_meta_2.csv", 'label': "G10 Meting 2 (15mm)"},
            {'file': "g10_15mm_5_metingen_meta_3.csv", 'label': "G10 Meting 3 (15mm)"},
            {'file': "g10_15mm_5_metingen_meta_4.csv", 'label': "G10 Meting 4 (15mm)"},
            {'file': "g10_15mm_5_metingen_meta_5.csv", 'label': "G10 Meting 5 (15mm)"},
        ]
    },
}


def exp_func(x, a, b, c): return a * np.exp(-b * x) + c


def get_mean_trace_from_file(csv_file, profile):
    all_aligned_traces = []
    filepath = os.path.join(TARGET_DATA_SUBFOLDER, csv_file)
    try:
        df = pd.read_csv(filepath)
        df = df[df["Device"] == DEVICE_FILTER]
    except FileNotFoundError:
        print(f"  - WARNING: File not found, skipping: {csv_file}")
        return None
    if df.empty: return None

    unique_runs = df['Run'].unique()[:MAX_RUNS]
    for run_id in unique_runs:
        run_data = df[df['Run'] == run_id]['ADC_Value'].values
        indices_above_thresh = np.where(run_data > profile['THRESH'])[0]
        if len(indices_above_thresh) > 0:
            onset = indices_above_thresh[0]
            start = onset - WINDOW_BEFORE
            end = onset + profile['MAX_WINDOW_AFTER']
            if start >= 0 and end <= len(run_data):
                all_aligned_traces.append(run_data[start:end])

    if not all_aligned_traces: return None

    max_len = max(len(t) for t in all_aligned_traces)
    padded_matrix = np.array([np.pad(t, (0, max_len - len(t)), 'edge') for t in all_aligned_traces])
    return np.mean(padded_matrix, axis=0)


def calculate_periodogram(trace, profile, get_raw_peaks=False):
    if trace is None or len(trace) < 20: return [], [], []
    idx_peak = np.argmax(trace)
    segment = trace[idx_peak + POST_PEAK_OFFSET_SAMPLES:]
    if len(segment) < 20: return [], [], []

    time_axis = np.arange(len(segment))
    try:
        popt, _ = curve_fit(exp_func, time_axis, segment, p0=(segment[0], 0.001, np.min(segment)), maxfev=5000)
        detrended_segment = segment - exp_func(time_axis, *popt)
    except RuntimeError:
        detrended_segment = segment - np.mean(segment)

    N = len(detrended_segment)
    yf = rfft(detrended_segment);
    xf = rfftfreq(N, SAMPLE_TIME_DELTA_US)
    periods = 1 / xf[1:];
    mags = np.abs(yf[1:])
    mask = periods <= profile['MAX_PERIOD_PLOT_US']

    raw_peaks_data = []
    if get_raw_peaks:
        peaks_indices, properties = find_peaks(mags, prominence=np.max(mags) * 0.1, width=3)
        for i in range(len(peaks_indices)):
            idx = peaks_indices[i]
            if periods[idx] <= profile['MAX_PERIOD_PLOT_US']:
                raw_peaks_data.append({
                    'period': round(periods[idx], 2),
                    'magnitude': round(mags[idx], 2),
                    'prominence': round(properties['prominences'][i], 2)
                })

    return periods[mask], mags[mask], raw_peaks_data


def smart_downsample(x_data, y_data, num_points, high_detail_x_max=400, high_detail_ratio=0.6):
    if len(x_data) <= num_points: return x_data.tolist(), y_data.tolist()

    num_high_detail = int(num_points * high_detail_ratio)
    num_low_detail = num_points - num_high_detail

    high_detail_mask = x_data <= high_detail_x_max
    x_high, y_high = x_data[high_detail_mask], y_data[high_detail_mask]
    x_low, y_low = x_data[~high_detail_mask], y_data[~high_detail_mask]

    indices_high = np.linspace(0, len(x_high) - 1, num_high_detail, dtype=int) if len(
        x_high) > num_high_detail else np.arange(len(x_high))
    indices_low = np.linspace(0, len(x_low) - 1, num_low_detail, dtype=int) if len(
        x_low) > num_low_detail else np.arange(len(x_low))

    final_x = np.concatenate((x_high[indices_high], x_low[indices_low]))
    final_y = np.concatenate((y_high[indices_high], y_low[indices_low]))

    sort_indices = np.argsort(final_x)
    return final_x[sort_indices].tolist(), final_y[sort_indices].tolist()


# --- Hoofdlogica ---
website_data_raw = {}
for key, config in CONDITIONS_TO_PROCESS.items():
    print(f"\nProcessing RAW data for condition: {key}...")
    profile = config['profile']
    website_data_raw[key] = []

    for meting in config['metingen']:
        print(f"  -> Processing meting: {meting['label']}")
        mean_trace = get_mean_trace_from_file(meting['file'], profile)
        if mean_trace is None: continue

        time_axis_signal = (np.arange(len(mean_trace)) - WINDOW_BEFORE) * SAMPLE_TIME_DELTA_US

        is_15mm = '15mm' in key
        periods, mags, raw_peaks = calculate_periodogram(mean_trace, profile, get_raw_peaks=is_15mm)
        if not len(periods): continue

        # BELANGRIJK: GEEN DOWNSAMPLING! We slaan de volledige arrays op.
        website_data_raw[key].append({
            'label': meting['label'],
            'signal': {'time_us': [round(t, 2) for t in time_axis_signal],
                       'amplitude': [round(a, 2) for a in mean_trace]},
            'fft': {'period_us': [round(p, 2) for p in periods], 'magnitude': [round(m, 2) for m in mags]},
            'fft_peaks': raw_peaks
        })

output_filename = 'website_data_raw.json'
# Geen indentatie voor een kleiner bestand, essentieel voor grote bestanden
with open(output_filename, 'w') as f:
    json.dump(website_data_raw, f)

print(f"\n✅ Success! FULL RAW data saved to '{output_filename}'.")