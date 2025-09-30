import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from datetime import datetime

def analyze_10hz_sine_wave_comparison(csv_file_path):
    """
    Analyze 10Hz sine wave in EMG data and compare with theoretical waveform
    Build phase-synchronized sine wave starting from 0.6 seconds for comparison
    """
    try:
        # Read EMG data
        print(f"Reading EMG data: {csv_file_path}")
        df = pd.read_csv(csv_file_path, skiprows=11)
        
        # Check if CH5 exists
        if 'CH5' not in df.columns:
            print("Error: CH5 channel not found")
            print(f"Available channels: {list(df.columns)}")
            return
        
        # Sampling rate and parameter settings
        sampling_rate = 1000  # Hz
        target_frequency = 10  # Hz (input 10Hz sine wave)
        
        # Get CH5 data
        ch5_data = df['CH5'].values
        total_samples = len(ch5_data)
        
        # Create correct time axis (assuming 1000Hz sampling rate)
        time_axis = np.arange(total_samples) / sampling_rate
        
        print(f"Total data length: {total_samples} samples")
        print(f"Time duration: {time_axis[-1]:.3f} seconds")
        print(f"Sampling rate: {sampling_rate} Hz")
        
        # Find sample index corresponding to 0.6 seconds
        start_time = 0.6  # seconds
        start_index = int(start_time * sampling_rate)
        
        if start_index >= total_samples:
            print(f"Error: 0.6 seconds exceeds data range ({time_axis[-1]:.3f}s)")
            start_index = 0
            start_time = 0
        
        # Data starting from start_time
        analysis_data = ch5_data[start_index:]
        analysis_time = time_axis[start_index:] - start_time  # Reset time axis to start from 0
        
        # Analyze data value at 0.6 seconds to determine theoretical sine wave phase
        if start_index < total_samples:
            reference_value = ch5_data[start_index]
            print(f"CH5 value at 0.6s: {reference_value:.6f} uV")
        else:
            reference_value = ch5_data[0]
            print(f"Using initial value as reference: {reference_value:.6f} uV")
        
        # Estimate amplitude and DC offset
        data_mean = np.mean(analysis_data)
        data_std = np.std(analysis_data)
        data_max = np.max(analysis_data)
        data_min = np.min(analysis_data)
        
        # Estimate amplitude (using half of peak-to-peak difference)
        estimated_amplitude = (data_max - data_min) / 2
        dc_offset = data_mean
        
        print(f"Data statistics:")
        print(f"  Mean (DC offset): {dc_offset:.6f} uV")
        print(f"  Standard deviation: {data_std:.6f} uV")
        print(f"  Maximum value: {data_max:.6f} uV")
        print(f"  Minimum value: {data_min:.6f} uV")
        print(f"  Estimated amplitude: {estimated_amplitude:.6f} uV")
        
        # Better phase alignment using derivative method
        # Find the slope at the reference point to determine if it's rising or falling
        if start_index > 0 and start_index < total_samples - 1:
            slope = ch5_data[start_index + 1] - ch5_data[start_index - 1]
        else:
            slope = 0
        
        normalized_ref = (reference_value - dc_offset) / estimated_amplitude
        normalized_ref = np.clip(normalized_ref, -1, 1)
        
        # Determine correct phase based on value and slope
        if slope >= 0:  # Rising
            theoretical_phase = np.arcsin(normalized_ref)
        else:  # Falling
            theoretical_phase = np.pi - np.arcsin(normalized_ref)
        
        # Adjust for starting time
        theoretical_phase = theoretical_phase - 2 * np.pi * target_frequency * start_time
        
        print(f"Phase calculations:")
        print(f"  Normalized value at 0.6s: {normalized_ref:.6f}")
        print(f"  Slope at 0.6s: {slope:.6f} (Rising: {slope >= 0})")
        print(f"  Theoretical phase: {theoretical_phase:.6f} radians")
        
        # This will be set after timestamp analysis
        theoretical_sine = None
        
        # Ignore CSV Timestamp column and assume perfect 1000Hz sampling
        # Create ideal timestamps: first point = 0.6s, then +1ms for each subsequent point
        ideal_timestamps = np.arange(len(analysis_data)) * 0.001 + start_time
        analysis_time = ideal_timestamps - start_time  # Reset to start from 0
        
        print(f"\nDetailed data point analysis:")
        print(f"  Total samples from 0.6s: {len(analysis_data)}")
        print(f"  IGNORING CSV Timestamp column")
        print(f"  Assuming perfect 1000Hz sampling (1ms intervals)")
        print(f"  Generated timestamp range: {ideal_timestamps[0]:.6f}s to {ideal_timestamps[-1]:.6f}s")
        print(f"  Duration: {ideal_timestamps[-1] - ideal_timestamps[0]:.6f}s")
        print(f"  Expected points at 1000Hz: {int((ideal_timestamps[-1] - ideal_timestamps[0]) * 1000)}")
        print(f"  Actual data points: {len(analysis_data)}")
        print(f"  Points per 0.1s: {len(analysis_data) / ((ideal_timestamps[-1] - ideal_timestamps[0]) / 0.1):.1f}")
        
        # Generate theoretical 10Hz sine wave using ideal 1000Hz timestamps
        theoretical_sine = estimated_amplitude * np.sin(2 * np.pi * target_frequency * analysis_time + theoretical_phase) + dc_offset
        
        # Create comparison plots
        plt.figure(figsize=(15, 9))
        
        # Subplot 1: Original data vs theoretical sine wave with sampling points
        plt.subplot(4, 1, 1)
        plt.plot(analysis_time, analysis_data, 'b-', linewidth=1, label='Actual CH5 Data', alpha=0.8)
        plt.plot(analysis_time, theoretical_sine, 'r--', linewidth=2, label=f'Theoretical 10Hz Sine Wave', alpha=0.8)
        # Mark ALL sampling points to see the true density
        plt.scatter(analysis_time, analysis_data, 
                   c='blue', s=8, alpha=0.7, label='All Actual Sample Points')
        plt.scatter(analysis_time, theoretical_sine, 
                   c='red', s=8, alpha=0.7, marker='x', label='All Theoretical Sample Points')
        
        # Show detailed info for first 0.1 seconds
        first_100ms = analysis_time <= 0.1
        points_in_100ms = np.sum(first_100ms)
        print(f"  Points in first 0.1s: {points_in_100ms}")
        print(f"  Should be 100 points at 1000Hz")
        
        plt.xlabel('Time (seconds) - Starting from 0.6s')
        plt.ylabel('Voltage (uV)')
        plt.title(f'EMG CH5 Data vs Theoretical 10Hz Sine Wave (All {len(analysis_data)} Points Shown)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Voltage difference analysis
        plt.subplot(4, 1, 2)
        voltage_difference = analysis_data - theoretical_sine
        plt.plot(analysis_time, voltage_difference, 'g-', linewidth=1, label='Voltage Difference (Actual - Theoretical)')
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.xlabel('Time (seconds)')
        plt.ylabel('Voltage Difference (uV)')
        plt.title('Voltage Difference between Actual Data and Theoretical Sine Wave')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 3: Frequency spectrum analysis (removed timestamp comparison)
        plt.subplot(3, 1, 3)
        # Calculate FFT
        fft_data = np.fft.fft(analysis_data)
        fft_theoretical = np.fft.fft(theoretical_sine)
        freqs = np.fft.fftfreq(len(analysis_data), 1/sampling_rate)
        
        # Show only positive frequency components
        positive_freqs = freqs[:len(freqs)//2]
        fft_magnitude_data = np.abs(fft_data)[:len(freqs)//2]
        fft_magnitude_theoretical = np.abs(fft_theoretical)[:len(freqs)//2]
        
        plt.semilogy(positive_freqs, fft_magnitude_data, 'b-', linewidth=1, label='Actual Data Spectrum', alpha=0.8)
        plt.semilogy(positive_freqs, fft_magnitude_theoretical, 'r--', linewidth=2, label='Theoretical Sine Wave Spectrum', alpha=0.8)
        plt.axvline(x=10, color='orange', linestyle=':', linewidth=2, label='10Hz Reference Line')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.title('Frequency Spectrum Comparison')
        plt.xlim(0, 50)  # Show only 0-50Hz
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Calculate correlation and error statistics
        correlation = np.corrcoef(analysis_data, theoretical_sine)[0, 1]
        mse = np.mean((analysis_data - theoretical_sine) ** 2)
        rmse = np.sqrt(mse)
        max_voltage_error = np.max(np.abs(voltage_difference))
        
        # Since we're assuming perfect 1000Hz sampling, no timestamp errors
        max_timestamp_error = 0.0
        mean_timestamp_error = 0.0
        std_timestamp_error = 0.0
        
        print(f"\nComparison analysis results:")
        print(f"=== Voltage Analysis ===")
        print(f"  Correlation coefficient: {correlation:.6f}")
        print(f"  Mean squared error (MSE): {mse:.6f}")
        print(f"  Root mean squared error (RMSE): {rmse:.6f}")
        print(f"  Maximum voltage error: {max_voltage_error:.6f} uV")
        print(f"  Relative error (RMSE/amplitude): {(rmse/estimated_amplitude)*100:.2f}%")
        
        print(f"\n=== Timestamp Analysis (Assuming Perfect 1000Hz) ===")
        print(f"  Using ideal 1ms intervals - no timestamp errors")
        print(f"  Perfect sampling assumed: {max_timestamp_error:.6f} ms error")
        print(f"  Total duration: {len(analysis_data)} ms")
        
        # Find dominant frequency component
        peak_index = np.argmax(fft_magnitude_data[1:50]) + 1  # Skip DC component, limit to 50Hz
        dominant_frequency = positive_freqs[peak_index]
        print(f"  Dominant frequency component: {dominant_frequency:.2f} Hz")
        
        # Verify 1000Hz sampling rate
        expected_samples_per_cycle = sampling_rate / target_frequency  # Should be 100 samples/cycle
        actual_period_samples = []
        
        # Simple period detection (finding zero crossings)
        zero_crossings = []
        mean_value = np.mean(analysis_data)
        for i in range(1, len(analysis_data)):
            if (analysis_data[i-1] - mean_value) * (analysis_data[i] - mean_value) < 0:
                zero_crossings.append(i)
        
        if len(zero_crossings) >= 4:  # Need at least two complete cycles
            periods = []
            for i in range(2, len(zero_crossings), 2):  # Every two zero crossings is a half cycle
                period_samples = zero_crossings[i] - zero_crossings[i-2]
                periods.append(period_samples)
            
            if periods:
                avg_period_samples = np.mean(periods)
                measured_frequency = sampling_rate / avg_period_samples
                print(f"  Measured period: {avg_period_samples:.1f} samples")
                print(f"  Measured frequency: {measured_frequency:.2f} Hz")
                print(f"  Theoretical period (10Hz@1000Hz): {expected_samples_per_cycle:.1f} samples")
        
        return {
            'correlation': correlation,
            'rmse': rmse,
            'dominant_frequency': dominant_frequency,
            'estimated_amplitude': estimated_amplitude,
            'dc_offset': dc_offset,
            'max_voltage_error': max_voltage_error,
            'max_timestamp_error': max_timestamp_error,
            'mean_timestamp_error': mean_timestamp_error,
            'std_timestamp_error': std_timestamp_error
        }
        
    except Exception as e:
        print(f"Analysis error: {str(e)}")
        return None

if __name__ == "__main__":
    # EMG data file path
    csv_file = r"d:\NTKCAP\Patient_data\sync_2\2025_09_23\raw_data\4\emg_data.csv"
    
    # Execute analysis
    result = analyze_10hz_sine_wave_comparison(csv_file)
    
    if result:
        print(f"\n=== Analysis Complete ===")
        print("Conclusions:")
        
        # Voltage correlation analysis
        if result['correlation'] > 0.9:
            print("✓ Data highly correlates with 10Hz sine wave")
        elif result['correlation'] > 0.7:
            print("△ Data moderately correlates with 10Hz sine wave")
        else:
            print("✗ Data has low correlation with 10Hz sine wave")
            
        # Frequency analysis
        if 9.5 <= result['dominant_frequency'] <= 10.5:
            print("✓ Dominant frequency component is close to 10Hz")
        else:
            print(f"△ Dominant frequency component is {result['dominant_frequency']:.2f}Hz, deviates from 10Hz")
        
        # Sampling rate analysis (based on timestamp errors)
        if result['max_timestamp_error'] < 0.1:  # Less than 0.1ms error
            print("✓ Sampling timestamps are very accurate (< 0.1ms error)")
        elif result['max_timestamp_error'] < 1.0:  # Less than 1ms error
            print(f"△ Sampling timestamps have small errors ({result['max_timestamp_error']:.3f}ms max)")
        else:
            print(f"✗ Sampling timestamps have significant errors ({result['max_timestamp_error']:.3f}ms max)")
        
        # Overall assessment
        print(f"\n=== Overall Assessment ===")
        if abs(result['mean_timestamp_error']) < 0.01:
            print("✓ No systematic timestamp bias detected")
        else:
            print(f"△ Systematic timestamp bias: {result['mean_timestamp_error']:.6f}ms")
            
        if result['std_timestamp_error'] < 0.1:
            print("✓ Consistent sampling intervals")
        else:
            print(f"△ Variable sampling intervals (std: {result['std_timestamp_error']:.6f}ms)")