# TUSS44x0 Controller Script
# ---------------------
# Standalone Python script for controlling TUSS44x0 ultrasonic sensors with MSP430 boards

import time
import serial
import sys
import numpy as np
import matplotlib.pyplot as plt
import serial.tools.list_ports  # For listing available ports

class TUSS44x0Controller:
    """
    Controller class for communicating with MSP430 boards connected to TUSS44x0 ultrasonic sensors.
    Enables bidirectional communication between two boards where either can be transmitter or receiver.
    """
    
    # MSP430 register addresses for TUSS44x0 control
    TUSS_CONFIG_REG = 0x1000
    TUSS_MODE_REG = 0x1002
    TUSS_TRIGGER_REG = 0x1004
    TUSS_STATUS_REG = 0x1006
    TUSS_DATA_REG = 0x1008
    
    # Operation modes
    MODE_TRANSMITTER = 0x01
    MODE_RECEIVER = 0x02
    
    def __init__(self, port1, port2, baud_rate=115200):
        """Initialize connection to both MSP430 boards"""
        try:
            self.msp1 = serial.Serial(port1, baud_rate, timeout=1)
            self.msp2 = serial.Serial(port2, baud_rate, timeout=1)
            print(f"Connected to MSP430 boards on {port1} and {port2}")
            
            # Default assignment (can be switched later)
            self.transmitter = self.msp1
            self.receiver = self.msp2
            
            # Initialize both boards
            self._initialize_board(self.msp1)
            self._initialize_board(self.msp2)
            
            # Store measurement history
            self.measurement_history = []
            
        except serial.SerialException as e:
            print(f"Error connecting to MSP430 boards: {e}")
            raise
    
    def _initialize_board(self, board):
        """Initialize a single MSP430 board"""
        # Reset the board
        self._send_command(board, "RESET")
        time.sleep(0.5)
        
        # Configure SPI for communication with TUSS44x0
        self._send_command(board, "CONFIG_SPI")
        
        # Initial configuration of the TUSS44x0 sensor
        self._write_register(board, self.TUSS_CONFIG_REG, 0x0010)  # Basic configuration
        time.sleep(0.1)
        
        # Verify communication by reading back configuration
        config = self._read_register(board, self.TUSS_CONFIG_REG)
        if config != 0x0010:
            print(f"Warning: Configuration verification failed for board on {board.port}")
    
    def switch_roles(self):
        """Switch transmitter and receiver roles between the two MSP430 boards"""
        self.transmitter, self.receiver = self.receiver, self.transmitter
        
        # Update modes on both boards
        self._set_mode(self.transmitter, self.MODE_TRANSMITTER)
        self._set_mode(self.receiver, self.MODE_RECEIVER)
        
        print(f"Roles switched: Transmitter is now on {self.transmitter.port}, Receiver is now on {self.receiver.port}")
    
    def _set_mode(self, board, mode):
        """Set the mode (transmitter or receiver) for a specific board"""
        self._write_register(board, self.TUSS_MODE_REG, mode)
        time.sleep(0.1)
        current_mode = self._read_register(board, self.TUSS_MODE_REG)
        
        if current_mode != mode:
            print(f"Warning: Mode setting failed for board on {board.port}")
            return False
        return True
    
    def configure_transmitter(self, frequency=40000, pulse_count=8, gain=3):
        """Configure transmitter parameters"""
        # Combine parameters into a single configuration word
        config = ((frequency // 1000) << 8) | ((pulse_count & 0x0F) << 4) | (gain & 0x07)
        self._write_register(self.transmitter, self.TUSS_CONFIG_REG, config)
        print(f"Transmitter configured: {frequency}Hz, {pulse_count} pulses, gain level {gain}")
    
    def configure_receiver(self, gain=5, filter_bandwidth=2000, detection_threshold=500):
        """Configure receiver parameters"""
        # Combine parameters into configuration words
        config1 = ((gain & 0x07) << 8) | ((filter_bandwidth // 100) & 0xFF)
        config2 = detection_threshold & 0xFFFF
        
        self._write_register(self.receiver, self.TUSS_CONFIG_REG, config1)
        self._write_register(self.receiver, self.TUSS_CONFIG_REG + 2, config2)
        print(f"Receiver configured: gain level {gain}, filter bandwidth {filter_bandwidth}Hz, threshold {detection_threshold}")
    
    def send_pulse(self):
        """Trigger the transmitter to send an ultrasonic pulse"""
        # Make sure transmitter is in the right mode
        if not self._set_mode(self.transmitter, self.MODE_TRANSMITTER):
            return False
            
        # Make sure receiver is ready
        if not self._set_mode(self.receiver, self.MODE_RECEIVER):
            return False
            
        # Trigger the pulse
        self._write_register(self.transmitter, self.TUSS_TRIGGER_REG, 0x0001)
        return True
    
    def read_measurement(self, timeout=1.0):
        """Read measurement data from the receiver"""
        start_time = time.time()
        
        # Wait for data ready flag in status register
        while time.time() - start_time < timeout:
            status = self._read_register(self.receiver, self.TUSS_STATUS_REG)
            if status & 0x0001:  # Data ready bit
                # Read measurement data
                data = self._read_register(self.receiver, self.TUSS_DATA_REG)
                time_of_flight = self._read_register(self.receiver, self.TUSS_DATA_REG + 2)
                signal_strength = self._read_register(self.receiver, self.TUSS_DATA_REG + 4)
                
                # Calculate distance (assuming speed of sound is 343 m/s)
                # time_of_flight is in microseconds, so distance in meters is:
                distance = (time_of_flight * 343.0) / 2000000.0
                
                result = {
                    'distance': distance,
                    'time_of_flight': time_of_flight,
                    'signal_strength': signal_strength,
                    'raw_data': data,
                    'timestamp': time.time()
                }
                
                self.measurement_history.append(result)
                return result
            time.sleep(0.01)
        
        print("Timeout: No measurement data received")
        return None
    
    def run_continuous_measurement(self, count=10, interval=0.5):
        """Run a series of measurements with the current configuration"""
        results = []
        
        for i in range(count):
            print(f"Measurement {i+1}/{count}", end="\r")
            if self.send_pulse():
                result = self.read_measurement()
                if result:
                    results.append(result)
            time.sleep(interval)
        
        print(f"\nCompleted {len(results)} measurements")
        return results
    
    def _send_command(self, board, command):
        """Send a command string to the MSP430"""
        cmd = f"{command}\n".encode()
        board.write(cmd)
        time.sleep(0.1)
        response = board.read(100)
        return response.decode().strip()
    
    def _write_register(self, board, address, value):
        """Write a value to a register on the MSP430"""
        cmd = f"WRITE {address:04X} {value:04X}\n".encode()
        board.write(cmd)
        time.sleep(0.05)
        response = board.read(100)
        return "OK" in response.decode()
    
    def _read_register(self, board, address):
        """Read a value from a register on the MSP430"""
        cmd = f"READ {address:04X}\n".encode()
        board.write(cmd)
        time.sleep(0.05)
        response = board.read(100).decode().strip()
        
        # Parse response, expecting format like "READ 1000: XXXX"
        try:
            value = int(response.split(":")[-1].strip(), 16)
            return value
        except (ValueError, IndexError):
            print(f"Error parsing register read response: {response}")
            return None
    
    def plot_measurement_history(self):
        """Plot the history of measurements"""
        if not self.measurement_history:
            print("No measurement data available")
            return
        
        # Extract data from measurement history
        timestamps = [m['timestamp'] - self.measurement_history[0]['timestamp'] for m in self.measurement_history]
        distances = [m['distance'] for m in self.measurement_history]
        signal_strengths = [m['signal_strength'] for m in self.measurement_history]
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Distance plot
        ax1.plot(timestamps, distances, 'b-o')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Distance (m)')
        ax1.set_title('Distance Measurements')
        ax1.grid(True)
        
        # Signal strength plot
        ax2.plot(timestamps, signal_strengths, 'r-o')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Signal Strength')
        ax2.set_title('Signal Strength')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def analyze_signal(self, raw_data=None):
        """Analyze the received signal (either the latest or provided raw data)"""
        if raw_data is None:
            if not self.measurement_history:
                print("No measurement data available")
                return
            raw_data = self.measurement_history[-1]['raw_data']
        
        # Simulate a signal based on the raw_data value
        t = np.linspace(0, 1, 1000)
        signal = np.sin(2 * np.pi * 40000 * t) * np.exp(-5 * t) * (raw_data / 1000)
        
        plt.figure(figsize=(10, 6))
        plt.plot(t * 1000, signal)  # Convert time to milliseconds
        plt.xlabel('Time (ms)')
        plt.ylabel('Amplitude')
        plt.title('Ultrasonic Signal Analysis')
        plt.grid(True)
        plt.show()
    
    def clear_history(self):
        """Clear the measurement history"""
        self.measurement_history = []
        print("Measurement history cleared")
    
    def close(self):
        """Close connections to both MSP430 boards"""
        if hasattr(self, 'msp1') and self.msp1.is_open:
            self.msp1.close()
        if hasattr(self, 'msp2') and self.msp2.is_open:
            self.msp2.close()
        print("Connections to MSP430 boards closed")

def list_available_ports():
    """List all available serial ports"""
    ports = serial.tools.list_ports.comports()
    available_ports = []
    
    print("Available COM ports:")
    for port in ports:
        print(f"- {port.device}: {port.description}")
        available_ports.append(port.device)
    
    return available_ports

def main():
    """Main function to run the script"""
    print("TUSS44x0 Ultrasonic Sensor Controller")
    print("=====================================")
    
    # List available ports
    available_ports = list_available_ports()
    
    if not available_ports:
        print("No COM ports found. Please check your connections.")
        return
    
    # Get port selection from user
    print("\nSelect COM ports for the MSP430 boards:")
    
    # Show menu for first port
    print("\nSelect port for MSP430 Board 1 (Transmitter):")
    for i, port in enumerate(available_ports):
        print(f"{i+1}. {port}")
    
    try:
        choice1 = int(input("Enter number (or enter manually if not listed): "))
        if 1 <= choice1 <= len(available_ports):
            port1 = available_ports[choice1-1]
        else:
            port1 = input("Enter COM port manually (e.g., COM3): ")
    except ValueError:
        port1 = input("Enter COM port manually (e.g., COM3): ")
    
    # Show menu for second port
    print("\nSelect port for MSP430 Board 2 (Receiver):")
    for i, port in enumerate(available_ports):
        if port != port1:  # Don't show the already selected port
            print(f"{i+1}. {port}")
    
    try:
        choice2 = int(input("Enter number (or enter manually if not listed): "))
        if 1 <= choice2 <= len(available_ports):
            port2 = available_ports[choice2-1]
            if port2 == port1:
                print("Warning: You selected the same port twice.")
                port2 = input("Enter a different COM port manually (e.g., COM4): ")
        else:
            port2 = input("Enter COM port manually (e.g., COM4): ")
    except ValueError:
        port2 = input("Enter COM port manually (e.g., COM4): ")
    
    # Get baud rate
    baud_rates = [9600, 19200, 38400, 57600, 115200, 230400]
    print("\nSelect baud rate:")
    for i, rate in enumerate(baud_rates):
        print(f"{i+1}. {rate}")
    
    try:
        choice = int(input("Enter number (default is 115200): ") or "5")
        if 1 <= choice <= len(baud_rates):
            baud_rate = baud_rates[choice-1]
        else:
            baud_rate = 115200
    except ValueError:
        baud_rate = 115200
    
    print(f"\nConnecting to MSP430 boards on {port1} and {port2} at {baud_rate} baud...")
    
    try:
        # Create controller with selected ports
        controller = TUSS44x0Controller(port1, port2, baud_rate)
        
        # Simple command-line menu
        while True:
            print("\nTUSS44x0 Controller Menu:")
            print("1. Configure Transmitter")
            print("2. Configure Receiver")
            print("3. Take Single Measurement")
            print("4. Run Continuous Measurements")
            print("5. Plot Measurement History")
            print("6. Analyze Signal")
            print("7. Switch Transmitter/Receiver Roles")
            print("8. Clear Measurement History")
            print("9. Exit")
            
            choice = input("\nEnter choice (1-9): ")
            
            if choice == '1':
                # Configure transmitter
                frequency = int(input("Enter frequency in kHz (30-50, default 40): ") or "40") * 1000
                pulse_count = int(input("Enter pulse count (1-15, default 8): ") or "8")
                gain = int(input("Enter gain (0-7, default 3): ") or "3")
                controller.configure_transmitter(frequency, pulse_count, gain)
            
            elif choice == '2':
                # Configure receiver
                gain = int(input("Enter gain (0-7, default 5): ") or "5")
                bandwidth = int(input("Enter filter bandwidth in Hz (500-5000, default 2000): ") or "2000")
                threshold = int(input("Enter detection threshold (0-1000, default 500): ") or "500")
                controller.configure_receiver(gain, bandwidth, threshold)
            
            elif choice == '3':
                # Take single measurement
                print("\nTaking measurement...")
                controller.send_pulse()
                result = controller.read_measurement()
                if result:
                    print(f"Distance: {result['distance']:.3f} m")
                    print(f"Time of flight: {result['time_of_flight']} µs")
                    print(f"Signal strength: {result['signal_strength']}")
            
            elif choice == '4':
                # Run continuous measurements
                count = int(input("Enter number of measurements (default 10): ") or "10")
                interval = float(input("Enter interval in seconds (default 0.5): ") or "0.5")
                print(f"\nRunning {count} measurements with {interval}s interval...")
                results = controller.run_continuous_measurement(count, interval)
                
                for i, result in enumerate(results):
                    print(f"\nMeasurement {i+1}:")
                    print(f"  Distance: {result['distance']:.3f} m")
                    print(f"  Time of flight: {result['time_of_flight']} µs")
                    print(f"  Signal strength: {result['signal_strength']}")
            
            elif choice == '5':
                # Plot measurement history
                controller.plot_measurement_history()
            
            elif choice == '6':
                # Analyze signal
                controller.analyze_signal()
            
            elif choice == '7':
                # Switch roles
                controller.switch_roles()
            
            elif choice == '8':
                # Clear history
                controller.clear_history()
            
            elif choice == '9':
                # Exit
                break
            
            else:
                print("Invalid choice. Please try again.")
        
        # Clean up before exiting
        controller.close()
        print("Program exited.")
    
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Program terminated due to error.")

if __name__ == "__main__":
    main()
