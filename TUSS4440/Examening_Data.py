import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.signal import find_peaks


def visualize_ultrasonic_data(csv_filename):
    # Read CSV file
    df = pd.read_csv(csv_filename)

    # Basic data inspection
    print(f"Data shape: {df.shape}")
    print(f"Time range: {df['Time'].min():.3f}s to {df['Time'].max():.3f}s")
    print(f"Value range: {df['Value'].min()} to {df['Value'].max()}")
    print(f"Number of pulse events: {df['IsPulse'].sum()}")

    # Set up the figure and axes
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})

    # Plot the ultrasonic signal values
    ax1.plot(df['Time'], df['Value'], 'b-', linewidth=1, label='Ultrasonic Signal')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Ultrasonic Data Visualization')
    ax1.grid(True, alpha=0.3)

    # Set y-axis limits with some padding
    max_value = df['Value'].max() * 1.1
    min_value = max(0, df['Value'].min() * 0.9)  # Ensure we don't go below 0
    ax1.set_ylim(min_value, max_value)

    # Find pulses
    pulse_df = df[df['IsPulse'] == 1]

    # Create a continuous time series dataset with smaller step size for smooth visualization
    if not pulse_df.empty:
        # Get unique pulse times
        pulse_times = pulse_df['Time'].unique()

        # Plot each pulse with a smoother curve
        for i, pulse_time in enumerate(pulse_times):
            # Do not plot vertical lines for pulses as they don't represent physical reality
            # Instead, highlight the region where the pulse occurs

            # Add annotations for the pulses without arrows
            if i % 3 == 0:  # Only annotate every third pulse to avoid clutter
                ax1.annotate(f"Pulse {i + 1}",
                             xy=(pulse_time, max_value * 0.95),
                             xytext=(pulse_time, max_value * 0.95),
                             ha='center',
                             fontsize=9,
                             bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.6))

    # Plot the pulse events in the bottom subplot as a stem plot
    if not pulse_df.empty:
        ax2.stem(pulse_df['Time'], pulse_df['IsPulse'], linefmt='r-', markerfmt='ro', basefmt='r-',
                 label='Pulse Events')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Pulse')
    ax2.set_yticks([0, 1])
    ax2.set_ylim(-0.1, 1.1)
    ax2.grid(True, alpha=0.3)

    # Calculate and display time between pulses if there are multiple pulses
    if len(pulse_times) > 1:
        time_diffs = np.diff(pulse_times)
        avg_time_diff = np.mean(time_diffs)

        # Add a text box with statistics
        textbox_content = (
            f"Average time between pulses: {avg_time_diff:.3f}s\n"
            f"Number of pulses: {len(pulse_times)}"
        )
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.7)
        ax2.text(0.05, 0.5, textbox_content, transform=ax2.transAxes, fontsize=10,
                 verticalalignment='center', bbox=props)

    # Add legend to both subplots
    ax1.legend(loc='upper right')
    ax2.legend(loc='upper right')

    # Now let's add a zoomed-in view of pulse and echo patterns
    # Find significant peaks that may represent echoes
    threshold = df['Value'].mean() + df['Value'].std() * 2
    peaks, _ = find_peaks(df['Value'], height=threshold, distance=10)

    if len(peaks) > 0:
        # Create a new figure for detailed pulse analysis
        fig2, axs = plt.subplots(min(4, len(pulse_times)), 1, figsize=(10, 12), sharex=True)
        if len(pulse_times) == 1:
            axs = [axs]  # Handle the case of a single subplot

        for i, pulse_time in enumerate(pulse_times[:min(4, len(pulse_times))]):
            # Get a window of data around the pulse
            window_start = pulse_time - 0.02
            window_end = pulse_time + 0.2  # Look for echoes up to 200ms after pulse

            window_data = df[(df['Time'] >= window_start) & (df['Time'] <= window_end)]

            # Plot the signal in this window
            axs[i].plot(window_data['Time'], window_data['Value'], 'b-', linewidth=1.5)
            axs[i].set_title(f'Pulse {i + 1} and Echo Pattern')
            axs[i].grid(True, alpha=0.3)
            axs[i].set_ylabel('Amplitude')

            # Mark the pulse
            pulse_marker = window_data[window_data['IsPulse'] == 1]
            if not pulse_marker.empty:
                axs[i].axvline(x=pulse_marker['Time'].iloc[0], color='r', linestyle='--', alpha=0.5)
                axs[i].text(pulse_marker['Time'].iloc[0], axs[i].get_ylim()[1] * 0.9, 'Pulse',
                            ha='center', bbox=dict(facecolor='white', alpha=0.7))

            # Find and mark potential echoes in this window
            window_peaks = window_data.iloc[find_peaks(window_data['Value'], height=threshold, distance=5)[0]]
            for j, (t, v) in enumerate(zip(window_peaks['Time'], window_peaks['Value'])):
                if t > pulse_time + 0.01:  # Skip the pulse itself
                    axs[i].plot(t, v, 'go', markersize=6)
                    echo_time = t - pulse_time
                    axs[i].text(t, v * 1.1, f'Echo {j + 1}\n{echo_time * 1000:.1f}ms',
                                ha='center', va='bottom', fontsize=8,
                                bbox=dict(facecolor='white', alpha=0.7))

        if len(pulse_times) >= 1:
            axs[-1].set_xlabel('Time (s)')

        plt.tight_layout()

    # Adjust layout and save plot
    plt.figure(fig.number)
    plt.tight_layout()

    # Save the figures
    output_path = Path(csv_filename).with_suffix('.png')
    plt.savefig(output_path)

    if 'fig2' in locals():
        details_path = Path(csv_filename).stem + '_details.png'
        fig2.savefig(details_path)
        print(f"Detail visualization saved to {details_path}")

    print(f"Visualization saved to {output_path}")

    # Show the plots
    plt.show()

    # Perform analysis
    analyze_ultrasonic_data(df, pulse_times if len(pulse_df) > 0 else [])


def analyze_ultrasonic_data(df, pulse_times):
    """Perform analysis on the ultrasonic data"""
    print("\n----- Data Analysis -----")

    # Calculate the timing between pulses
    if len(pulse_times) > 1:
        time_diffs = np.diff(pulse_times)
        print(f"Time between pulses (seconds):")
        for i, diff in enumerate(time_diffs):
            print(f"  Pulse {i + 1} to {i + 2}: {diff:.6f}s")
        print(f"Average time between pulses: {np.mean(time_diffs):.6f}s")
        print(f"Standard deviation: {np.std(time_diffs):.6f}s")

    # Analyze the signal characteristics
    threshold = df['Value'].mean() + df['Value'].std() * 2
    peaks, properties = find_peaks(df['Value'], height=threshold, distance=10)

    if len(peaks) > 0:
        print(f"\nDetected {len(peaks)} significant peaks in the signal")
        peak_values = properties['peak_heights']
        print(f"Average peak value: {np.mean(peak_values):.2f}")
        print(f"Max peak value: {np.max(peak_values):.2f}")

        # Calculate noise floor (estimated from non-peak regions)
        noise_level = np.median(df['Value'])
        print(f"Estimated noise floor: {noise_level:.2f}")

        if noise_level > 0:
            snr = np.mean(peak_values) / noise_level
            print(f"Estimated signal-to-noise ratio: {snr:.2f}")

    # Echo detection analysis
    if len(pulse_times) > 0 and len(peaks) > 0:
        peak_times = df.iloc[peaks]['Time'].values

        print("\nEcho analysis:")
        all_echo_delays = []
        echo_distances = []

        # Standard speed of sound in air at room temperature (343 m/s)
        speed_of_sound = 343  # m/s

        for i, pulse_time in enumerate(pulse_times):
            # Find peaks that occur after this pulse (potential echoes)
            echo_candidates = peak_times[(peak_times > pulse_time + 0.001) &
                                         (peak_times < pulse_time + 0.1)]  # Look within 100ms

            if len(echo_candidates) > 0:
                echo_delays = echo_candidates - pulse_time
                strongest_echo_idx = np.argmax(df.iloc[peaks]['Value'][(peak_times >= echo_candidates[0]) &
                                                                       (peak_times <= echo_candidates[-1])])
                strongest_echo_delay = echo_delays[strongest_echo_idx]

                # Calculate distance based on time of flight (divide by 2 for round trip)
                distance = (strongest_echo_delay / 2) * speed_of_sound

                print(
                    f"  Pulse {i + 1}: First echo at {echo_delays[0] * 1000:.2f}ms, strongest at {strongest_echo_delay * 1000:.2f}ms")
                print(f"    Estimated distance: {distance:.2f} meters")

                all_echo_delays.extend(echo_delays)
                echo_distances.append(distance)

        if all_echo_delays:
            avg_echo_delay = np.mean(all_echo_delays)
            avg_distance = np.mean(echo_distances)

            print(f"\nAverage echo delay: {avg_echo_delay * 1000:.2f}ms")
            print(f"Average estimated distance: {avg_distance:.2f} meters")


if __name__ == "__main__":
    csv_filename = "ultrasonic_data_20250306-102258.csv"
    visualize_ultrasonic_data(csv_filename)