import serial
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from pathlib import Path
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import argparse


class TUSS44x0Interface:
    """Interface to communicate with TUSS44x0 ultrasonic sensor via serial port"""

    def __init__(self, port, baud_rate=115200, timeout=1):
        self.port = port
        self.baud_rate = baud_rate
        self.timeout = timeout
        self.ser = None
        self.connected = False

    def connect(self):
        """Connect to the serial port"""
        try:
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baud_rate,
                timeout=self.timeout
            )
            self.connected = True
            print(f"Connected to {self.port} at {self.baud_rate} baud")
            return True
        except serial.SerialException as e:
            print(f"Error connecting to {self.port}: {e}")
            self.connected = False
            return False

    def disconnect(self):
        """Disconnect from the serial port"""
        if self.ser and self.ser.is_open:
            self.ser.close()
            self.connected = False
            print(f"Disconnected from {self.port}")

    def configure_tof(self, pulse_freq=40, record_length=24, speed_of_sound=343, tof_mode=1, driver_type=0):
        """
        Configure time of flight parameters

        Parameters:
        - pulse_freq: pulse frequency in kHz (default: 40 kHz)
        - record_length: record length in ms (default: 24 ms)
        - speed_of_sound: speed of sound in m/s (default: 343 m/s)
        - tof_mode: TOF mode (0=listen only, 1=burst and listen)
        - driver_type: driver type (0=bit bang, 1=timer interrupt)
        """
        if not self.connected:
            print("Not connected to serial port")
            return False

        # Convert to byte array format expected by TUSS44x0
        pulse_freq_msb = (pulse_freq >> 8) & 0xFF
        pulse_freq_lsb = pulse_freq & 0xFF
        speed_of_sound_msb = (speed_of_sound >> 8) & 0xFF
        speed_of_sound_lsb = speed_of_sound & 0xFF

        # Create configuration array
        config = bytearray([
            pulse_freq_msb, pulse_freq_lsb,
            record_length,
            speed_of_sound_msb, speed_of_sound_lsb,
            tof_mode, driver_type
        ])

        # Send configuration command
        cmd = b'TOFCFG:' + config
        self.ser.write(cmd)
        time.sleep(0.1)  # Give the device time to process

        # Check response
        response = self.ser.readline().decode('utf-8', errors='ignore').strip()
        if "CONFIG OK" in response:
            print("TOF configuration successful")
            return True
        else:
            print(f"TOF configuration failed: {response}")
            return False

    def execute_tof(self):
        """Execute a time of flight measurement and return the results"""
        if not self.connected:
            print("Not connected to serial port")
            return None

        # Send execute command
        self.ser.write(b'TOFEXEC\n')

        # Collect data until we receive the END marker
        data_lines = []
        start_time = time.time()
        timeout = 5  # seconds

        while time.time() - start_time < timeout:
            line = self.ser.readline().decode('utf-8', errors='ignore').strip()
            if not line:
                continue

            if line == "END":
                break

            data_lines.append(line)

        # Process the received data
        times = []
        values = []
        is_pulse = []

        for line in data_lines:
            parts = line.split(',')
            if len(parts) == 3:
                try:
                    times.append(float(parts[0]))
                    values.append(int(parts[1]))
                    is_pulse.append(int(parts[2]))
                except ValueError:
                    continue

        # Create a DataFrame with the results
        if times:
            df = pd.DataFrame({
                'Time': times,
                'Value': values,
                'IsPulse': is_pulse
            })
            return df
        else:
            print("No data received from TOF execution")
            return None

    def read_register(self, register_addr):
        """Read a register value from the TUSS44x0"""
        if not self.connected:
            print("Not connected to serial port")
            return None

        cmd = f"REGREAD:{register_addr:02X}\n"
        self.ser.write(cmd.encode())
        time.sleep(0.1)

        response = self.ser.readline().decode('utf-8', errors='ignore').strip()
        if response.startswith("REG:"):
            parts = response.split(':')
            if len(parts) == 3:
                return int(parts[2], 16)

        return None

    def write_register(self, register_addr, value):
        """Write a value to a register in the TUSS44x0"""
        if not self.connected:
            print("Not connected to serial port")
            return False

        cmd = f"REGWRITE:{register_addr:02X}:{value:02X}\n"
        self.ser.write(cmd.encode())
        time.sleep(0.1)

        response = self.ser.readline().decode('utf-8', errors='ignore').strip()
        return "OK" in response


class UltrasonicDataVisualizer:
    """Real-time ultrasonic data visualization"""

    def __init__(self, tuss_interface=None):
        self.tuss = tuss_interface
        self.data = pd.DataFrame(columns=['Time', 'Value', 'IsPulse'])
        self.fig = None
        self.ax1 = None
        self.ax2 = None
        self.line1 = None
        self.scatter = None
        self.ani = None

    def setup_plots(self):
        """Set up the plots for real-time visualization"""
        self.fig = plt.figure(figsize=(12, 10))
        gs = GridSpec(3, 1, height_ratios=[2, 1, 1])

        # Signal plot
        self.ax1 = self.fig.add_subplot(gs[0])
        self.ax1.set_title('TUSS44x0 Ultrasonic Data')
        self.ax1.set_ylabel('Amplitude')
        self.ax1.grid(True, alpha=0.3)
        self.line1, = self.ax1.plot([], [], 'b-', linewidth=1)

        # Pulse events plot
        self.ax2 = self.fig.add_subplot(gs[1])
        self.ax2.set_ylabel('Pulse')
        self.ax2.set_yticks([0, 1])
        self.ax2.set_ylim(-0.1, 1.1)
        self.ax2.grid(True, alpha=0.3)
        self.scatter = self.ax2.scatter([], [], color='red', marker='o')

        # Statistics text area
        self.ax3 = self.fig.add_subplot(gs[2])
        self.ax3.axis('off')
        self.stats_text = self.ax3.text(0.05, 0.5, "", transform=self.ax3.transAxes)

        plt.tight_layout()

    def update_plot(self, frame):
        """Update function for animation"""
        if self.tuss and self.tuss.connected:
            new_data = self.tuss.execute_tof()
            if new_data is not None and not new_data.empty:
                self.data = new_data

        if not self.data.empty:
            # Update signal plot
            self.line1.set_data(self.data['Time'], self.data['Value'])
            self.ax1.relim()
            self.ax1.autoscale_view()

            # Update pulse events plot
            pulse_data = self.data[self.data['IsPulse'] == 1]
            if not pulse_data.empty:
                self.scatter.set_offsets(np.column_stack([pulse_data['Time'], pulse_data['IsPulse']]))
                self.ax2.set_xlim(self.ax1.get_xlim())

            # Update statistics
            stats = self.calculate_statistics()
            self.stats_text.set_text(stats)

        return self.line1, self.scatter, self.stats_text

    def calculate_statistics(self):
        """Calculate and format statistics from the data"""
        if self.data.empty:
            return "No data available"

        stats = []
        stats.append(f"Data points: {len(self.data)}")
        stats.append(f"Time range: {self.data['Time'].min():.3f}s - {self.data['Time'].max():.3f}s")

        # Pulse statistics
        pulse_data = self.data[self.data['IsPulse'] == 1]
        if len(pulse_data) > 1:
            pulse_times = pulse_data['Time'].values
            time_diffs = np.diff(pulse_times)
            stats.append(f"Pulses: {len(pulse_data)}")
            stats.append(f"Avg time between pulses: {np.mean(time_diffs):.3f}s")

        # Signal statistics
        stats.append(f"Max amplitude: {self.data['Value'].max()}")
        stats.append(f"Signal mean: {self.data['Value'].mean():.2f}")

        # Calculate estimated distance if we have pulses and peaks
        signal_threshold = 100  # Adjust based on your data
        peaks = self.data[self.data['Value'] > signal_threshold]

        if not pulse_data.empty and not peaks.empty:
            # Find echo delays (time from pulse to first significant peak)
            echo_delays = []
            for pulse_time in pulse_data['Time']:
                future_peaks = peaks[peaks['Time'] > pulse_time]['Time'].values
                if len(future_peaks) > 0:
                    echo_delays.append(future_peaks[0] - pulse_time)

            if echo_delays:
                avg_delay = np.mean(echo_delays)
                # Speed of sound in air (343 m/s), divide by 2 for round trip
                distance = (avg_delay * 343) / 2
                stats.append(f"Est. distance: {distance:.2f}m")

        return "\n".join(stats)

    def start_animation(self, interval=100):
        """Start the real-time visualization"""
        self.setup_plots()
        self.ani = animation.FuncAnimation(
            self.fig, self.update_plot, interval=interval,
            blit=True, cache_frame_data=False
        )
        plt.show()

    def save_data(self, filename=None):
        """Save the current data to a CSV file"""
        if self.data.empty:
            print("No data to save")
            return None

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"ultrasonic_data_{timestamp}.csv"

        self.data.to_csv(filename, index=False)
        print(f"Data saved to {filename}")
        return filename


def main():
    parser = argparse.ArgumentParser(description='TUSS44x0 Ultrasonic Sensor Interface')
    parser.add_argument('--port', '-p', required=True, help='Serial port (e.g., COM3 or /dev/ttyACM0)')
    parser.add_argument('--baud', '-b', type=int, default=115200, help='Baud rate (default: 115200)')
    parser.add_argument('--frequency', '-f', type=int, default=40, help='Pulse frequency in kHz (default: 40)')
    parser.add_argument('--record-length', '-r', type=int, default=24, help='Record length in ms (default: 24)')
    parser.add_argument('--file', type=str, help='Existing CSV file to visualize instead of connecting to hardware')

    args = parser.parse_args()

    if args.file:
        # Load and visualize existing data
        try:
            data = pd.read_csv(args.file)
            print(f"Loaded data from {args.file}")
            visualizer = UltrasonicDataVisualizer()
            visualizer.data = data
            visualizer.start_animation()
        except Exception as e:
            print(f"Error loading file: {e}")
    else:
        # Connect to hardware and start real-time visualization
        tuss = TUSS44x0Interface(args.port, args.baud)
        if tuss.connect():
            tuss.configure_tof(
                pulse_freq=args.frequency,
                record_length=args.record_length
            )

            visualizer = UltrasonicDataVisualizer(tuss)
            try:
                visualizer.start_animation()
            except KeyboardInterrupt:
                print("Stopping visualization...")
            finally:
                visualizer.save_data()
                tuss.disconnect()
        else:
            print(f"Failed to connect to {args.port}")


if __name__ == "__main__":
    main()