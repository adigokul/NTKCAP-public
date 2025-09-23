import numpy as np
from scipy.signal import butter, lfilter, iirnotch, lfilter_zi
import websocket
import json
import socket
import threading
import time
import sys
import argparse
import csv
import datetime
import os

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='EMG WebSocket Data Reader',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Usage Examples:
  python emg_localhost.py                              # Auto scan, single read
  python emg_localhost.py --uri ws://localhost:31278/ws  # Direct URI specification
  python emg_localhost.py -u ws://192.168.1.100:31278   # Use short parameter
  python emg_localhost.py --uri localhost:31278         # Auto add ws:// prefix
  python emg_localhost.py -c -o emg_data.csv            # Continuous mode with output file
  python emg_localhost.py -u localhost:31278 -c         # Continuous mode auto filename
        '''
    )
    
    parser.add_argument(
        '--uri', '-u',
        type=str,
        help='Direct WebSocket URI specification, skip auto scan (e.g.: ws://localhost:31278/ws)'
    )
    
    parser.add_argument(
        '--timeout', '-t',
        type=int,
        default=5,
        help='Connection timeout in seconds (default: 5)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output CSV file path (default: auto-generated timestamp filename)'
    )
    
    parser.add_argument(
        '--continuous', '-c',
        action='store_true',
        help='Continuous mode: continuously receive and save data until manual stop'
    )
    
    return parser.parse_args()

def validate_and_format_uri(uri):
    """Validate and format WebSocket URI"""
    if not uri:
        return None
    
    # If no protocol prefix, auto add ws://
    if not uri.startswith(('ws://', 'wss://')):
        uri = f"ws://{uri}"
    
    # If no path, try to add common path
    if not uri.endswith('/ws') and '?' not in uri and '#' not in uri:
        # Check if it's just host:port format
        try:
            from urllib.parse import urlparse
            parsed = urlparse(uri)
            if parsed.path == '' or parsed.path == '/':
                uri = f"{uri}/ws"
        except:
            pass
    
    return uri

def test_direct_uri(uri, timeout=5):
    """Test directly specified URI"""
    try:
        print(f"Testing directly specified URI: {uri}")
        ws = websocket.WebSocket()
        ws.settimeout(timeout)
        ws.connect(uri)
        
        # Try to receive one data packet for validation
        try:
            data = ws.recv()
            data_dict = json.loads(data)
            
            if "contents" in data_dict:
                contents = data_dict["contents"]
                if isinstance(contents, list) and len(contents) > 0:
                    item = contents[0]
                    if "eeg" in item and isinstance(item["eeg"], list):
                        ws.close()
                        print(f"✅ URI validation successful, contains valid EMG data format")
                        return True
            
            ws.close()
            print(f"⚠️  URI connectable but data format may be incorrect")
            return True  # Still return True, let user decide whether to continue
            
        except Exception as e:
            ws.close()
            print(f"⚠️  URI connectable but cannot receive data: {e}")
            return True  # Return True if connection is successful
            
    except Exception as e:
        print(f"❌ URI connection failed: {e}")
        return False

def scan_for_websocket_servers(host_range=None, port_range=None, timeout=1):
    """Scan for available WebSocket servers"""
    if host_range is None:
        host_range = ['localhost', '127.0.0.1']
    if port_range is None:
        port_range = range(31270, 31290)  # Common EMG device port range
    
    available_servers = []
    
    print("Scanning for available WebSocket servers...")
    
    for host in host_range:
        for port in port_range:
            try:
                # Check if port is open
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(timeout)
                result = sock.connect_ex((host, port))
                sock.close()
                
                if result == 0:
                    # Try WebSocket connection
                    ws_uri = f"ws://{host}:{port}/ws"
                    try:
                        print(f"Testing WebSocket connection: {ws_uri}")
                        ws = websocket.WebSocket()
                        ws.settimeout(timeout)
                        ws.connect(ws_uri)
                        ws.close()
                        available_servers.append(ws_uri)
                        print(f"✓ Found available server: {ws_uri}")
                    except Exception as e:
                        # Try without /ws path
                        ws_uri_alt = f"ws://{host}:{port}"
                        try:
                            ws = websocket.WebSocket()
                            ws.settimeout(timeout)
                            ws.connect(ws_uri_alt)
                            ws.close()
                            available_servers.append(ws_uri_alt)
                            print(f"✓ Found available server: {ws_uri_alt}")
                        except:
                            pass
                            
            except Exception as e:
                continue
    
    return available_servers

def test_websocket_data(uri, timeout=3):
    """Test if WebSocket has valid EMG data"""
    try:
        print(f"Testing data format: {uri}")
        ws = websocket.WebSocket()
        ws.settimeout(timeout)
        ws.connect(uri)
        
        # Try to receive several data packets
        for i in range(3):
            try:
                data = ws.recv()
                data_dict = json.loads(data)
                
                # Check if contains EMG data format
                if "contents" in data_dict:
                    contents = data_dict["contents"]
                    if isinstance(contents, list) and len(contents) > 0:
                        item = contents[0]
                        if "eeg" in item and isinstance(item["eeg"], list):
                            ws.close()
                            print(f"✓ Confirmed valid EMG data format: {uri}")
                            return True
                            
            except json.JSONDecodeError:
                continue
            except Exception as e:
                continue
                
        ws.close()
        print(f"✗ No valid EMG data: {uri}")
        return False
        
    except Exception as e:
        print(f"✗ Data test failed: {uri} - {e}")
        return False

def find_emg_server():
    """Automatically find EMG WebSocket server"""
    print("=== Auto Scanning EMG WebSocket Server ===")
    
    # First scan for available WebSocket servers
    available_servers = scan_for_websocket_servers()
    
    if not available_servers:
        print("❌ No available WebSocket servers found")
        return None
    
    print(f"\nFound {len(available_servers)} available WebSocket servers")
    
    # Test which server has valid EMG data
    print("\nTesting data formats...")
    for uri in available_servers:
        if test_websocket_data(uri):
            return uri
    
    print("❌ All servers have no valid EMG data format")
    
    # If no valid data found, return the first available server
    if available_servers:
        print(f"⚠️  Returning first available server: {available_servers[0]}")
        return available_servers[0]
    
    return None

def generate_csv_filename():
    """Generate timestamped CSV filename"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"emg_data_{timestamp}.csv"

def save_emg_data_to_csv(emg_data, filename, channel_count, append=False):
    """Save EMG data to CSV file"""
    mode = 'a' if append else 'w'
    
    with open(filename, mode, newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # If new file, write headers
        if not append or not os.path.exists(filename) or os.path.getsize(filename) == 0:
            headers = ['timestamp'] + [f'channel_{i+1}' for i in range(channel_count)]
            writer.writerow(headers)
        
        # Write data
        timestamp = datetime.datetime.now().isoformat()
        if emg_data.ndim == 2:
            # Multiple data points
            for i in range(emg_data.shape[1]):
                row = [timestamp] + [emg_data[ch, i] for ch in range(channel_count)]
                writer.writerow(row)
        else:
            # Single data point
            row = [timestamp] + [emg_data[ch] for ch in range(channel_count)]
            writer.writerow(row)

def read_continuous_data_from_websocket(uri, channel_count, output_file):
    """Continuously read EMG data and save to CSV"""
    try:
        ws = websocket.WebSocket()
        ws.connect(uri)
        print(f"Start continuous data reception, using {channel_count} channels...")
        print(f"Data will be saved to: {output_file}")
        print("Press Ctrl+C to stop recording and exit\n")
        
        # Initialize filter parameters
        bp_parameter = np.zeros((channel_count, 8))
        nt_parameter = np.zeros((channel_count, 2))
        lp_parameter = np.zeros((channel_count, 4))
        
        data_count = 0
        
        while True:
            try:
                data = ws.recv()
                emg_array, bp_parameter, nt_parameter, lp_parameter = process_data_from_websocket(
                    data, bp_parameter, nt_parameter, lp_parameter, channel_count
                )
                
                if emg_array.shape[0] != 0:
                    # Save data to CSV
                    save_emg_data_to_csv(emg_array, output_file, channel_count, append=True)
                    data_count += 1
                    
                    # Update status every 100 times
                    if data_count % 100 == 0:
                        print(f"Processed {data_count} data packets, data shape: {emg_array.shape}")
                        
            except websocket.WebSocketTimeoutException:
                print("⚠️  WebSocket timeout, retrying...")
                continue
                
    except KeyboardInterrupt:
        print(f"\n\n⚠️  User interrupted, saved {data_count} data packets to {output_file}")
        ws.close()
        return True
    except Exception as e:
        print(f"❌ Continuous reading error: {e}")
        ws.close()
        return False

def read_specific_data_from_websocket(uri, bp_parameter, nt_parameter, lp_parameter, channel_count=8):
        try:
            ws = websocket.WebSocket()
            ws.connect(uri)
            print(f"Start receiving data, using {channel_count} channels...")
            while True:
                data = ws.recv()
                emg_array, bp_parameter, nt_parameter, lp_parameter = process_data_from_websocket(data, bp_parameter, nt_parameter, lp_parameter, channel_count)
                if emg_array.shape[0] != 0:
                    return emg_array, bp_parameter, nt_parameter, lp_parameter
        except Exception as e:
            print(f"WebSocket error: {e}")
            pass
        # finally:
        #     ws.close()

def detect_channel_count(uri, timeout=5):
    """Dynamically detect EMG channel count"""
    try:
        print(f"Detecting channel count: {uri}")
        ws = websocket.WebSocket()
        ws.settimeout(timeout)
        ws.connect(uri)
        
        # Try to receive several data packets to determine channel count
        for i in range(5):
            try:
                data = ws.recv()
                data_dict = json.loads(data)
                
                if "contents" in data_dict:
                    contents = data_dict["contents"]
                    if isinstance(contents, list) and len(contents) > 0:
                        # Check first item with eeg data
                        for item in contents:
                            if "eeg" in item and isinstance(item["eeg"], list):
                                channel_count = len(item["eeg"])
                                ws.close()
                                print(f"✅ Detected {channel_count} EMG channels")
                                return channel_count
                                
            except json.JSONDecodeError:
                continue
            except Exception as e:
                continue
                
        ws.close()
        print("❌ Cannot detect channel count, using default value 8")
        return 8  # Default value
        
    except Exception as e:
        print(f"❌ Channel detection failed: {e}, using default value 8")
        return 8  # Default value

def process_data_from_websocket(data, bp_parameter, nt_parameter, lp_parameter, channel_count=8):
    emg_values = np.zeros((channel_count, 50))
    j = 0
    try:
        data_dict = json.loads(data)
        if "contents" in data_dict:
            # Extract serial_number and eeg values
            serial_numbers_eegs = [(item['serial_number'][0], item['eeg']) for item in data_dict['contents'] if 'eeg' in item and len(item['eeg']) > 0]
            # Output results
            for serial_number, eeg in serial_numbers_eegs:
                # print(f"Serial Number: {serial_number}, EEG: {eeg}")
                actual_channels = min(len(eeg), channel_count)  # Use actual available channel count
                for i in range(actual_channels):
                    if j < 50:  # Ensure not exceeding array bounds
                        emg_values[i, j] = eeg[i]      # Latest 50 EMG data points
                j += 1
                if j >= 50:  # If 50 samples collected, stop
                    break
                    
            if j > 0:  # Ensure there is data
                try:
                    emg_array = np.empty((channel_count, j))  # Use actual collected data point count
                    for k in range(channel_count):
                        data_slice = emg_values[k, :j]  # Only process actually collected data
                        if len(data_slice) > 0:
                            emg_array[k], bp_parameter[k], nt_parameter[k], lp_parameter[k] = process_emg_signal(data_slice, bp_parameter[k], nt_parameter[k], lp_parameter[k])
                        else:
                            emg_array[k] = np.zeros(j)
                    return emg_array, bp_parameter, nt_parameter, lp_parameter
                except Exception as e:
                    print(f"Error occurred while processing signal: {e}")
                    return np.array([]), bp_parameter, nt_parameter, lp_parameter
            else:
                return np.array([]), bp_parameter, nt_parameter, lp_parameter
    except json.JSONDecodeError:
        print("Failed to decode JSON from WebSocket")
    except Exception as e:
        # print(f"Error processing data from WebSocket: {e}")
        return np.array([]), bp_parameter, nt_parameter, lp_parameter

# Bandpass filter design
def bandpass_filter(data, lowcut, highcut, fs, bp_filter_state, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    if bp_filter_state.all() == 0:
        bp_filter_state = lfilter_zi(b, a)
        #print("check4", bp_filter_state)
    y, bp_filter_state = lfilter(b, a, data, zi=bp_filter_state)
    return y, bp_filter_state

# Notch filter design 
def notch_filter(data, notch_freq, fs, notch_filter_state, quality_factor=30):
    nyq = 0.5 * fs
    freq = notch_freq / nyq
    b, a = iirnotch(freq, quality_factor)
    if notch_filter_state.all() == 0:
        notch_filter_state = lfilter_zi(b, a)
        #print("check5", notch_filter_state)
    y, notch_filter_state = lfilter(b, a, data, zi=notch_filter_state)
    return y, notch_filter_state

# Full-wave rectification
def full_wave_rectification(data):
    return np.abs(data)

# Low-pass filter design (envelope extraction)
def lowpass_filter(data, cutoff, fs, lp_filter_state, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    if lp_filter_state.all() == 0:
        lp_filter_state = lfilter_zi(b, a)
        #print("check8", lp_filter_state)
    y, lp_filter_state = lfilter(b, a, data, zi=lp_filter_state)
    return y, lp_filter_state

# Real-time signal processing function
def process_emg_signal(data, bp_parameter, nt_parameter, lp_parameter, fs=1000):
    # Bandpass filtering
    bandpassed, bp_parameter = bandpass_filter(data, 20, 450, fs, bp_parameter)
    # 50Hz notch filtering
    notch_filtered, nt_parameter = notch_filter(bandpassed, 50, fs, nt_parameter)
    # Full-wave rectification
    rectified = full_wave_rectification(notch_filtered)
    # Low-pass filtering for envelope extraction
    enveloped, lp_parameter = lowpass_filter(rectified, 10, fs, lp_parameter)

    return enveloped, bp_parameter, nt_parameter, lp_parameter
# EMG strength feedback
def calculate_emg_level(data, initial_max_min_rms_values, times, ta=20,rf=40,bf=25,Ga=15):
    # First 1 second for warm-up
    if times <= 1000:
        return 0, initial_max_min_rms_values
    # Use data from 1st to 10th second to determine initial min/max RMS values
    elif 1000 < times <= 5000:
        for i in range(8):
            rms_values = data[i]
            if initial_max_min_rms_values[i][0] == 0 or rms_values > initial_max_min_rms_values[i][0]:
                initial_max_min_rms_values[i][0] = rms_values
            elif initial_max_min_rms_values[i][1] == 0 or rms_values < initial_max_min_rms_values[i][1]:
                initial_max_min_rms_values[i][1] = rms_values
        return 0, initial_max_min_rms_values
    # Send reward value every 0.05 seconds
    else:
        reward = np.zeros(8)
        y = 0
        for i in range(8):
            rms_values = data[i]
            reward[i] = map_to_levels(rms_values, initial_max_min_rms_values[i])
        y = ta*reward[0]+rf*reward[1]+bf*reward[2]+Ga*reward[3]+ta*reward[4]+rf*reward[5]+bf*reward[6]+Ga*reward[7]
        print("Total: ",y/200,"Reward: ",reward)
        return y/200, initial_max_min_rms_values

def calculate_rms(signal):
    "Calculate RMS value of signal."
    return np.sqrt(np.mean(signal**2))

def map_to_levels(value, max_min_rms_values):
    """Map value to linear values beyond level 5 to -5, based on relaxation threshold and initial maximum RMS value,
    but divided into ten level intervals from 5 to -5 within upper and lower limits."""
    # Calculate value range size for each level
    try:
        level_range = (max_min_rms_values[0] - max_min_rms_values[1]) / 10
        
        if value <= max_min_rms_values[1]:
            # Calculate which level values below min_rms_values should map to
            level_diff = (max_min_rms_values[1] - value) / level_range
            return 5 + round(level_diff)
        elif value >= max_min_rms_values[0]:
            # Calculate which level values above max_rms_values should map to
            level_diff = (value - max_min_rms_values[0]) / level_range
            return -5 - round(level_diff)
        else:
            # Linear mapping to 5 to -5
            normalized_value = (value - max_min_rms_values[1]) / (max_min_rms_values[0] - max_min_rms_values[1])
            return int(round(normalized_value * (-10))) + 5
    except Exception as e:
        print(f"Error calculating reward: {e}, return 0")
        return 0
def main():
    print("EMG WebSocket Data Reader")
    print("="*40)
    
    # Parse command line arguments
    args = parse_arguments()
    
    websocket_uri = None
    
    # If URI is specified, use it directly
    if args.uri:
        print(f"Using specified URI: {args.uri}")
        websocket_uri = validate_and_format_uri(args.uri)
        print(f"Formatted URI: {websocket_uri}")
        
        if test_direct_uri(websocket_uri, args.timeout):
            print(f"✅ Using specified WebSocket server directly")
        else:
            print("❌ Specified URI cannot connect, switching to auto scan mode...")
            websocket_uri = None
    
    # If no URI specified or specified URI is invalid, perform auto scan
    if not websocket_uri:
        print("\n=== Auto Scan Mode ===")
        websocket_uri = find_emg_server()
    
    if not websocket_uri:
        print("❌ Cannot find available EMG WebSocket server")
        print("\nPlease ensure:")
        print("1. EMG device is connected and running")
        print("2. WebSocket server is started")
        print("3. Firewall allows connection")
        print("\nOr use --uri parameter to specify server address directly:")
        print("  python emg_localhost.py --uri ws://localhost:31278/ws")
        input("Press Enter to exit...")
        return
    
    print(f"\n✅ Using WebSocket server: {websocket_uri}")
    
    # Dynamically detect channel count
    print("\nDetecting EMG channel count...")
    channel_count = detect_channel_count(websocket_uri, args.timeout)
    
    # Prepare output file
    if args.output:
        output_file = args.output
    else:
        output_file = generate_csv_filename()
    
    print(f"Output file: {output_file}")
    
    # Select running mode
    if args.continuous:
        print(f"\n=== Continuous Mode ===")
        print("Continuously receiving and saving data...")
        success = read_continuous_data_from_websocket(websocket_uri, channel_count, output_file)
        if success:
            print(f"✅ Data successfully saved to {output_file}")
        else:
            print(f"❌ Error occurred during data saving process")
    else:
        print(f"\n=== Single Mode ===")
        print(f"Receiving single batch data and saving...")
        
        # Initialize parameters based on actual channel count
        bp_parameter = np.zeros((channel_count, 8))
        nt_parameter = np.zeros((channel_count, 2))
        lp_parameter = np.zeros((channel_count, 4))

        try:
            result = read_specific_data_from_websocket(websocket_uri, bp_parameter, nt_parameter, lp_parameter, channel_count)
            if result:
                emg_array, bp_parameter, nt_parameter, lp_parameter = result
                
                # Save to CSV
                save_emg_data_to_csv(emg_array, output_file, channel_count, append=False)
                
                print(f"\n✅ Successfully received and processed EMG data")
                print(f"Data shape: {emg_array.shape}")
                print(f"Channel count: {emg_array.shape[0]}")
                print(f"Data point count: {emg_array.shape[1]}")
                print(f"✅ Data saved to: {output_file}")
            else:
                print("❌ Failed to successfully process EMG data")
        except KeyboardInterrupt:
            print("\n\n⚠️  Program interrupted by user")
        except Exception as e:
            print(f"\n❌ Program execution error: {e}")
    
    print("\nProgram ended")
    print(f"Data file location: {os.path.abspath(output_file)}")

if __name__ == "__main__":
    main()

