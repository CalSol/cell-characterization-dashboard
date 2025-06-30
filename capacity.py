import serial
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import argparse
import csv
import os
from datetime import datetime

class BatteryAnalyzer:
    def __init__(self, port=None, baud_rate=9600, timeout=1):
        # Auto-detect port if none provided
        if port is None:
            self.port = self._detect_arduino_port()
        else:
            self.port = port
            
        self.baud_rate = baud_rate
        self.timeout = timeout
        self.serial_conn = None
        self.timestamps = []
        self.voltages = []
        self.currents = []
        self.capacities = []
        self.states = []
        self.status_messages = []
        self.resistance_samples = []
        self.start_time = None
        self.resistance = None
        self.open_circuit_voltage = None

    def _detect_arduino_port(self):
        """Auto-detect Arduino port"""
        try:
            import serial.tools.list_ports
            ports = list(serial.tools.list_ports.comports())
            for p in ports:
                if 'Arduino' in p.description or 'CH340' in p.description or 'USB Serial' in p.description:
                    print(f"Auto-detected Arduino on port: {p.device}")
                    return p.device
            
            # If no Arduino found, use common defaults
            if os.name == 'nt':  # Windows
                return 'COM3'
            else:  # Linux/Mac
                return '/dev/ttyACM0'
        except:
            # Fallback to default
            if os.name == 'nt':  # Windows
                return 'COM3'
            else:  # Linux/Mac
                return '/dev/ttyACM0'

    def connect(self, retry_count=3):
        """Connect to Arduino with retry mechanism"""
        for attempt in range(retry_count):
            try:
                self.serial_conn = serial.Serial(
                    port=self.port,
                    baudrate=self.baud_rate,
                    timeout=self.timeout
                )
                print(f"Connected to Arduino on {self.port}")
                time.sleep(2)  # Give time for Arduino to reset
                
                # Flush any pending data
                self.serial_conn.reset_input_buffer()
                self.serial_conn.reset_output_buffer()
                return True
            except serial.SerialException as e:
                print(f"Connect attempt {attempt+1}/{retry_count} failed: {e}")
                if attempt < retry_count - 1:
                    print("Retrying in 2 seconds...")
                    time.sleep(2)
                    
                    # Try other common ports if this one failed
                    if attempt == 1:
                        if self.port == '/dev/ttyACM0':
                            self.port = '/dev/ttyACM1'
                        elif self.port == '/dev/ttyACM1':
                            self.port = '/dev/ttyUSB0'
                        elif self.port == 'COM3':
                            self.port = 'COM4'
                    
        print("Failed to connect after multiple attempts")
        return False

    def disconnect(self):
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            print("Disconnected from Arduino")

    def read_data(self, duration=None, stop_on_completion=True):
        """
        Read data from the Arduino.
        duration: Time to read in seconds (None = indefinite)
        stop_on_completion: Stop when the test is complete (state 6)
        """
        if not self.serial_conn or not self.serial_conn.is_open:
            print("Serial connection not established")
            return False

        self.start_time = time.time()
        end_time = self.start_time + duration if duration else float('inf')

        print(f"Reading data {'for ' + str(duration) + ' seconds' if duration else 'until completion'}...")

        current_data = {}
        test_complete = False
        last_report_time = time.time()
        last_data_time = time.time()
        data_count = 0

        try:
            while time.time() < end_time and not test_complete:
                # Check for timeout with no data
                if time.time() - last_data_time > 10:  # 10 second timeout
                    print("WARNING: No data received for 10 seconds. Check connections.")
                    last_data_time = time.time()
                
                # Progress report
                if time.time() - last_report_time > 5:  # Report every 5 seconds
                    elapsed = time.time() - self.start_time
                    print(f"Reading in progress... {elapsed:.1f}s elapsed, {data_count} data points")
                    last_report_time = time.time()
                
                if self.serial_conn.in_waiting > 0:
                    last_data_time = time.time()
                    line = self.serial_conn.readline().decode('utf-8', errors='replace').strip()
                    
                    if line:  # Skip empty lines
                        print(f"← {line}")

                        # Process the line based on its content
                        if ': ' in line:
                            key, value = line.split(': ', 1)
                            
                            try:
                                if key == "Voltage":
                                    current_data['voltage'] = float(value)
                                elif key == "Current":
                                    current_data['current'] = float(value)
                                elif key == "Capacity_mAh":
                                    current_data['capacity'] = float(value)
                                elif key == "Time_ms":
                                    current_data['timestamp'] = float(value) / 1000.0  # Convert to seconds
                                elif key == "State":
                                    current_data['state'] = int(value)
                                elif key == "Status":
                                    current_data['status'] = value
                                    self.status_messages.append((time.time() - self.start_time, value))
                                elif key == "Calculated_Resistance":
                                    self.resistance = float(value)
                                    current_data['resistance'] = float(value)
                                elif key == "Open_Circuit_Voltage":
                                    self.open_circuit_voltage = float(value)
                                elif key.startswith("Resistance_Sample_") and "_V:" in key:
                                    index = int(key.split("_")[2])
                                    voltage = float(value)
                                    while len(self.resistance_samples) <= index:
                                        self.resistance_samples.append({'voltage': None, 'current': None})
                                    self.resistance_samples[index]['voltage'] = voltage
                                elif key.startswith("Resistance_Sample_") and "_I:" in key:
                                    index = int(key.split("_")[2])
                                    current = float(value)
                                    while len(self.resistance_samples) <= index:
                                        self.resistance_samples.append({'voltage': None, 'current': None})
                                    self.resistance_samples[index]['current'] = current
                            except ValueError as e:
                                print(f"Error parsing value: {e} in line: {line}")
                    
                    # When we have complete voltage and current data for one sample
                    if 'voltage' in current_data and 'current' in current_data:
                        # Add timestamp if missing
                        if 'timestamp' not in current_data:
                            current_data['timestamp'] = time.time() - self.start_time
                        
                        # Add missing fields with defaults
                        self.voltages.append(current_data.get('voltage'))
                        self.currents.append(current_data.get('current'))
                        self.timestamps.append(current_data.get('timestamp'))
                        self.capacities.append(current_data.get('capacity', 0))
                        self.states.append(current_data.get('state', 0))
                        
                        data_count += 1
                        
                        # Check if test is complete
                        if stop_on_completion and current_data.get('state') == 6:
                            print("Test completed (reached state 6)")
                            test_complete = True
                        
                        # Reset current_data but keep state information
                        state = current_data.get('state')
                        current_data = {}
                        if state is not None:
                            current_data['state'] = state
                
                time.sleep(0.05)  # Shorter sleep for more responsive reading

        except KeyboardInterrupt:
            print("\nData collection stopped by user")
        except Exception as e:
            print(f"Error reading data: {e}")
            import traceback
            traceback.print_exc()

        print(f"Data collection complete. Collected {len(self.voltages)} readings.")
        return len(self.voltages) > 0

    def calculate_resistance(self):
        """Calculate battery internal resistance based on collected data"""
        # First try to use the resistance value directly reported by Arduino
        if self.resistance is not None:
            print(f"Using Arduino-calculated resistance: {self.resistance:.4f} ohms")
            return self.resistance
            
        # If we have resistance samples, use those for a more accurate calculation
        if self.resistance_samples:
            valid_samples = [(s['current'], s['voltage']) for s in self.resistance_samples 
                            if s['current'] is not None and s['voltage'] is not None and s['current'] > 0.05]
            
            if len(valid_samples) >= 5:
                currents, voltages = zip(*valid_samples)
                slope, intercept, r_value, p_value, std_err = stats.linregress(currents, voltages)
                resistance = -slope  # Negative slope gives the resistance
                print(f"Calculated resistance from samples: {resistance:.4f} ohms (R² = {r_value**2:.4f})")
                return resistance
        
        # Fall back to using the main voltage/current data if available
        if len(self.voltages) > 10 and len(self.currents) > 10:
            # Find the maximum voltage (likely open circuit or near it)
            max_voltage = max(self.voltages)
            
            # Filter only for discharge phase (state 4)
            indices = [i for i, state in enumerate(self.states) if state == 4]
            if len(indices) >= 10:
                filtered_currents = [self.currents[i] for i in indices]
                filtered_voltages = [self.voltages[i] for i in indices]
                
                # Additional filtering for points with significant current
                valid_indices = [i for i, current in enumerate(filtered_currents) if current > 0.05]
                if len(valid_indices) >= 5:
                    valid_currents = [filtered_currents[i] for i in valid_indices]
                    valid_voltages = [filtered_voltages[i] for i in valid_indices]
                    
                    slope, intercept, r_value, p_value, std_err = stats.linregress(valid_currents, valid_voltages)
                    resistance = -slope
                    
                    # Sanity check on result
                    if 0.01 <= resistance <= 10.0:  # Reasonable range for battery resistance
                        print(f"Calculated resistance from discharge data: {resistance:.4f} ohms (R² = {r_value**2:.4f})")
                        return resistance
        
        print("Could not calculate resistance - insufficient data")
        return None

    def save_to_csv(self, filename=None):
        """Save collected data to a CSV file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"battery_test_{timestamp}.csv"
        
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Time (s)', 'Voltage (V)', 'Current (A)', 'Capacity (mAh)', 'State', 'Power (W)'])
            
            for i in range(len(self.timestamps)):
                # Calculate power
                power = self.voltages[i] * self.currents[i] if i < len(self.voltages) and i < len(self.currents) else ''
                
                writer.writerow([
                    self.timestamps[i] if i < len(self.timestamps) else '',
                    self.voltages[i] if i < len(self.voltages) else '',
                    self.currents[i] if i < len(self.currents) else '',
                    self.capacities[i] if i < len(self.capacities) else '',
                    self.states[i] if i < len(self.states) else '',
                    power
                ])
        
        print(f"Data saved to {filename}")
        
        # Save resistance samples if available
        if self.resistance_samples:
            resistance_filename = os.path.splitext(filename)[0] + "_resistance_samples.csv"
            with open(resistance_filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Sample', 'Current (A)', 'Voltage (V)'])
                
                for i, sample in enumerate(self.resistance_samples):
                    if sample['voltage'] is not None and sample['current'] is not None:
                        writer.writerow([i, sample['current'], sample['voltage']])
            
            print(f"Resistance samples saved to {resistance_filename}")
        
        return filename

    def plot_data(self, output_filename=None):
        """Generate plots of the collected data"""
        if not self.voltages or not self.currents:
            print("No data to plot")
            return

        # Create figure with subplots
        fig = plt.figure(figsize=(12, 16))
        grid = plt.GridSpec(4, 2, figure=fig)

        # Plot voltage vs time
        ax1 = fig.add_subplot(grid[0, :])
        ax1.plot(self.timestamps, self.voltages, 'b-', label='Voltage')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Voltage (V)')
        ax1.set_title('Battery Voltage vs Time')
        ax1.grid(True)
        
        # Add state transitions to the voltage plot
        last_state = -1
        for i, state in enumerate(self.states):
            if state != last_state:
                ax1.axvline(x=self.timestamps[i], color='r', linestyle='--', alpha=0.3)
                ax1.text(self.timestamps[i], min(self.voltages) + 0.1, f"S{state}", 
                         rotation=90, verticalalignment='bottom')
                last_state = state
        
        ax1.legend()

        # Plot current vs time
        ax2 = fig.add_subplot(grid[1, :])
        ax2.plot(self.timestamps, self.currents, 'r-', label='Current')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Current (A)')
        ax2.set_title('Battery Current vs Time')
        ax2.grid(True)
        ax2.legend()

        # Plot capacity vs time
        ax3 = fig.add_subplot(grid[2, 0])
        valid_capacities = [(t, c) for t, c in zip(self.timestamps, self.capacities) if c > 0]
        if valid_capacities:
            capacity_times, capacity_values = zip(*valid_capacities)
            ax3.plot(capacity_times, capacity_values, 'g-', label='Capacity')
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('Capacity (mAh)')
            ax3.set_title('Battery Capacity vs Time')
            ax3.grid(True)
            ax3.legend()
        else:
            ax3.text(0.5, 0.5, 'No capacity data available', 
                    horizontalalignment='center', verticalalignment='center')

        # Plot power vs time
        ax4 = fig.add_subplot(grid[2, 1])
        powers = [v * i for v, i in zip(self.voltages, self.currents)]
        ax4.plot(self.timestamps, powers, 'm-', label='Power')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Power (W)')
        ax4.set_title('Battery Power vs Time')
        ax4.grid(True)
        ax4.legend()

        # Plot V-I scatter for resistance calculation
        ax5 = fig.add_subplot(grid[3, :])
        
        # Filter for discharge data only and meaningful current
        valid_indices = [i for i, (c, s) in enumerate(zip(self.currents, self.states)) 
                         if c > 0.05 and s == 4]
        
        if valid_indices:
            filtered_currents = [self.currents[i] for i in valid_indices]
            filtered_voltages = [self.voltages[i] for i in valid_indices]
            
            # Plot main V-I scatter points
            ax5.scatter(filtered_currents, filtered_voltages, c='g', alpha=0.5, 
                      label='V-I During Discharge')
            
            # Add resistance samples if available
            if self.resistance_samples:
                valid_samples = [(s['current'], s['voltage']) for s in self.resistance_samples 
                                if s['current'] is not None and s['voltage'] is not None and s['current'] > 0.05]
                
                if valid_samples:
                    currents, voltages = zip(*valid_samples)
                    ax5.scatter(currents, voltages, c='b', marker='x', s=100, 
                                label='Resistance Test Samples')
            
            # Add resistance line if calculated
            resistance = self.calculate_resistance()
            if resistance is not None and resistance > 0:
                # Find voltage at zero current (OCV)
                ocv = self.open_circuit_voltage if self.open_circuit_voltage else max(self.voltages)
                
                # Plot the resistance line
                i_range = np.linspace(0, max(filtered_currents) * 1.1, 100)
                v_range = ocv - resistance * i_range
                ax5.plot(i_range, v_range, 'k--', 
                        label=f'R = {resistance:.4f} Ω')
                
                # Add text with resistance value
                ax5.text(0.7, 0.1, f'Internal Resistance: {resistance:.4f} Ω', 
                       transform=ax5.transAxes, bbox=dict(facecolor='white', alpha=0.7))
        else:
            ax5.text(0.5, 0.5, 'Insufficient data for resistance calculation', 
                   horizontalalignment='center', verticalalignment='center')
        
        ax5.set_xlabel('Current (A)')
        ax5.set_ylabel('Voltage (V)')
        ax5.set_title('Battery V-I Characteristic')
        ax5.grid(True)
        ax5.legend()

        # Add status annotations in a text box
        status_box = ''
        for time_offset, status in self.status_messages:
            status_box += f"{time_offset:.1f}s: {status}\n"
        
        if status_box:
            fig.text(0.1, 0.01, status_box, fontsize=8, 
                   bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Adjust layout to make room for status box
        
        # Add title with summary
        if len(self.capacities) > 0 and max(self.capacities) > 0:
            final_capacity = max(self.capacities)
            title = f"Battery Test Results - Capacity: {final_capacity:.2f} mAh"
            if resistance is not None:
                title += f", Resistance: {resistance:.4f} Ω"
            fig.suptitle(title, fontsize=16)
        
        # Save plot if filename provided
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"battery_test_{timestamp}.png"
        
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_filename}")
        
        plt.show()
        return output_filename

    def generate_report(self, filename=None):
        """Generate a comprehensive report of the battery test"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"battery_test_report_{timestamp}.txt"
        
        resistance = self.calculate_resistance()
        
        with open(filename, 'w') as f:
            f.write("===== Battery Test Report =====\n\n")
            f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Test Duration: {self.timestamps[-1] if self.timestamps else 0:.1f} seconds\n\n")
            
            f.write("--- Battery Characteristics ---\n")
            if self.capacities and max(self.capacities) > 0:
                f.write(f"Capacity: {max(self.capacities):.2f} mAh\n")
            else:
                f.write("Capacity: Not measured\n")
                
            if resistance is not None:
                f.write(f"Internal Resistance: {resistance:.4f} ohms\n")
            else:
                f.write("Internal Resistance: Not measured\n")
                
            if self.open_circuit_voltage is not None:
                f.write(f"Open Circuit Voltage: {self.open_circuit_voltage:.3f} V\n")
            elif self.voltages:
                f.write(f"Initial Voltage: {self.voltages[0]:.3f} V\n")
                
            if self.voltages:
                f.write(f"Final Voltage: {self.voltages[-1]:.3f} V\n")
            
            f.write("\n--- Test Status Messages ---\n")
            for time_offset, status in self.status_messages:
                f.write(f"{time_offset:.1f}s: {status}\n")
            
            f.write("\n--- Measurement Summary ---\n")
            if self.voltages:
                f.write(f"Max Voltage: {max(self.voltages):.3f} V\n")
                f.write(f"Min Voltage: {min(self.voltages):.3f} V\n")
            
            if self.currents:
                f.write(f"Max Current: {max(self.currents):.3f} A\n")
                f.write(f"Average Current: {sum(self.currents)/len(self.currents):.3f} A\n")
            
            powers = [v * i for v, i in zip(self.voltages, self.currents)]
            if powers:
                f.write(f"Max Power: {max(powers):.3f} W\n")
                f.write(f"Average Power: {sum(powers)/len(powers):.3f} W\n")
            
        print(f"Report saved to {filename}")
        return filename


def main():
    parser = argparse.ArgumentParser(description='Battery Analyzer')
    parser.add_argument('--port', default=None, help='Serial port for Arduino (auto-detect if not specified)')
    parser.add_argument('--baud', type=int, default=9600, help='Baud rate')
    parser.add_argument('--duration', type=int, default=None, help='Duration to collect data (seconds)')
    parser.add_argument('--output', default=None, help='Output CSV file prefix')
    parser.add_argument('--no-plot', action='store_true', help='Skip plotting')
    parser.add_argument('--report', action='store_true', help='Generate text report')
    args = parser.parse_args()

    analyzer = BatteryAnalyzer(port=args.port, baud_rate=args.baud)
    
    if analyzer.connect():
        try:
            print("\n==== Battery Analyzer ====")
            print("Press Ctrl+C to stop data collection at any time")
            
            if analyzer.read_data(duration=args.duration):
                print("\n==== Data Collection Complete ====")
                
                # Generate output file prefix if not provided
                output_prefix = args.output
                if not output_prefix:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_prefix = f"battery_test_{timestamp}"
                
                # Save data to CSV
                csv_file = analyzer.save_to_csv(f"{output_prefix}.csv")
                
                # Calculate battery resistance
                resistance = analyzer.calculate_resistance()
                if resistance:
                    print(f"Battery internal resistance: {resistance:.4f} ohms")
                
                # Plot data if not disabled
                if not args.no_plot:
                    plot_file = analyzer.plot_data(f"{output_prefix}.png")
                
                # Generate report if requested
                if args.report:
                    report_file = analyzer.generate_report(f"{output_prefix}_report.txt")
                
                print("\n==== Analysis Complete ====")
                print(f"Results saved with prefix: {output_prefix}")
                
                # Final summary
                capacity = max(analyzer.capacities) if analyzer.capacities else 0
                print(f"\nBattery Summary:")
                print(f"  Capacity: {capacity:.2f} mAh")
                if resistance:
                    print(f"  Internal Resistance: {resistance:.4f} ohms")
                    
                min_v = min(analyzer.voltages) if analyzer.voltages else 0
                max_v = max(analyzer.voltages) if analyzer.voltages else 0
                print(f"  Voltage range: {min_v:.2f}V - {max_v:.2f}V")
            else:
                print("No data was collected")
        except KeyboardInterrupt:
            print("\nAnalysis stopped by user")
        finally:
            analyzer.disconnect()
    else:
        print(f"Failed to connect to Arduino. Please check:")
        print("1. Is the Arduino connected to your computer?")
        print("2. Is the correct port specified? (current: {})".format(args.port if args.port else "auto-detect"))
        print("3. Is the Arduino running the battery tester code?")
        print("4. Do you have permissions to access the serial port?")

if __name__ == "__main__":
    main()