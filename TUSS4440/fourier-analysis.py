import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.fft import rfft, rfftfreq


def analyze_ultrasonic_data(filepath='data/ultrasonic_data.csv'):
    """
    Load and analyze ultrasonic data, focusing on the relevant frequency range.

    Parameters:
    filepath (str): Path to the CSV file with ultrasonic data
    """
    print(f"Looking for data in: {filepath}")

    # Check if the file exists
    if not os.path.exists(filepath):
        # Try to find any CSV file in the data directory
        data_dir = os.path.dirname(filepath)
        if os.path.exists(data_dir):
            csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv') and not f.startswith('raw_')]
            if csv_files:
                filepath = os.path.join(data_dir, csv_files[0])
                print(f"Found data file: {filepath}")
            else:
                print("No data files found. Please run the data collection script first.")
                return
        else:
            print(f"Data directory '{data_dir}' not found.")
            return

    # Load the data
    df = pd.read_csv(filepath)
    print(f"Data loaded successfully. Shape: {df.shape}")

    # Extract capture IDs
    captures = df['capture_id'].unique()
    print(f"Found {len(captures)} captures in the data")

    # Let user select a capture to analyze
    capture_to_analyze = captures[0]  # Default to first capture
    if len(captures) > 1:
        print("\nAvailable captures:")
        for i, capture in enumerate(captures):
            sample_count = len(df[df['capture_id'] == capture])
            print(f"{i}: Capture {capture} ({sample_count} samples)")

        try:
            capture_idx = int(input(f"Select a capture to analyze (0-{len(captures) - 1}) [default=0]: "))
            capture_to_analyze = captures[capture_idx]
        except (ValueError, IndexError):
            print(f"Using default capture {capture_to_analyze}")

    # Get data for the selected capture
    capture_data = df[df['capture_id'] == capture_to_analyze]
    capture_data = capture_data.sort_values(by='sample_id')
    adc_values = capture_data['adc_value'].values

    # Create the figure for plotting
    plt.figure(figsize=(12, 8))

    # Plot the original data
    plt.subplot(3, 1, 1)
    plt.plot(adc_values, 'b-')
    plt.title(f'Original Ultrasonic Data (Capture {capture_to_analyze})')
    plt.xlabel('Sample Number')
    plt.ylabel('ADC Value')
    plt.grid(True)

    # Remove the initial peak
    # Find where values stabilize after initial peak
    cut_index = 0
    for i in range(len(adc_values) - 10):
        if all(val < 50 for val in adc_values[i:i + 10]):
            cut_index = i
            break

    # If no clear cutoff was found, use a default value
    if cut_index == 0 and len(adc_values) > 20:
        cut_index = 20

    processed_data = adc_values[cut_index:]
    print(f"Removed initial {cut_index} samples")

    # Plot the processed data
    plt.subplot(3, 1, 2)
    plt.plot(processed_data, 'g-')
    plt.title('Processed Data (Initial Peak Removed)')
    plt.xlabel('Sample Number')
    plt.ylabel('ADC Value')
    plt.grid(True)

    # Calculate and plot FFT on the processed data
    n_samples = len(processed_data)
    if n_samples < 10:
        print("Not enough data points to perform FFT analysis")
        return

    # Use a sample rate appropriate for ultrasonic signals (40-400 kHz)
    # Estimate based on your hardware constraints
    estimated_sample_rate = 40000  # Conservative estimate for a 40kHz signal

    # Perform FFT
    yf = rfft(processed_data)
    xf = rfftfreq(n_samples, 1 / estimated_sample_rate)

    # Convert to magnitude
    magnitude = np.abs(yf) / n_samples * 2

    # Plot FFT
    plt.subplot(3, 1, 3)
    plt.plot(xf, magnitude, 'r-')
    plt.title('Frequency Spectrum (FFT)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.grid(True)

    # Focus on the meaningful frequency range (0-20kHz is usually sufficient)
    # Don't go above Nyquist frequency (sample_rate/2)
    nyquist = estimated_sample_rate / 2
    plt.xlim(0, min(20000, nyquist))

    # Save the analysis results
    save_dir = 'analysis_results'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt_filename = os.path.join(save_dir, f'fft_analysis_capture_{capture_to_analyze}.png')
    plt.tight_layout()
    plt.savefig(plt_filename)
    print(f"Analysis plot saved to: {plt_filename}")

    plt.show()


if __name__ == "__main__":
    analyze_ultrasonic_data()