import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def plot_channel5_voltage(csv_file_path):
    """
    Plot Channel 5 voltage over time from EMG CSV data
    """
    try:
        # First, read the header to get sampling rate
        sampling_rate = 1000  # Default value
        with open(csv_file_path, 'r') as f:
            for i, line in enumerate(f):
                if i >= 11:  # Stop reading after metadata section
                    break
                if 'Device sampling rate:' in line:
                    # Extract sampling rate from line like "Device sampling rate: 1000 samples/second"
                    parts = line.split(':')[1].strip().split()
                    sampling_rate = int(parts[0])
                    break
        
        # Read CSV file, skipping the header metadata (first 11 lines)
        print(f"Reading EMG data from: {csv_file_path}")
        print(f"Detected sampling rate: {sampling_rate} Hz")
        df = pd.read_csv(csv_file_path, skiprows=11)
        
        # Check if CH5 exists
        if 'CH5' not in df.columns:
            print("Error: CH5 not found in CSV file")
            print(f"Available columns: {list(df.columns)}")
            return
        
        # Get channel 5 data
        channel5_data = df['CH5'].values
        
        # Create time axis using detected sampling rate
        time_axis = np.arange(len(channel5_data)) / sampling_rate  # Convert to seconds
        
        # Create the plot
        plt.figure(figsize=(12, 6))
        plt.plot(time_axis, channel5_data, 'b-', linewidth=1, label='Channel 5')
        
        # Set labels and title in English
        plt.xlabel('Time (seconds)')
        plt.ylabel('Voltage (uV)')
        plt.title('EMG Channel 5 - Time vs Voltage')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Add statistics
        mean_voltage = np.mean(channel5_data)
        max_voltage = np.max(channel5_data)
        min_voltage = np.min(channel5_data)
        
        # Add text box with statistics
        stats_text = f'Mean: {mean_voltage:.2f} uV\nMax: {max_voltage:.2f} uV\nMin: {min_voltage:.2f} uV\nSamples: {len(channel5_data)}'
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Adjust layout
        plt.tight_layout()
        
        # Show the plot
        plt.show()
        
        print(f"Successfully plotted Channel 5 data")
        print(f"Data points: {len(channel5_data)}")
        print(f"Duration: {time_axis[-1]:.3f} seconds")
        print(f"Sampling rate: {sampling_rate} Hz")
        print(f"Voltage range: {min_voltage:.2f} to {max_voltage:.2f} uV")
        
    except Exception as e:
        print(f"Error plotting Channel 5: {str(e)}")

if __name__ == "__main__":
    # Default CSV file path
    csv_file = "test_emg_with_events.csv"
    
    # Plot Channel 5
    plot_channel5_voltage(csv_file)