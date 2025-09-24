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
import queue
from typing import Optional, Dict, Any

class EMGEventRecorder:
    """EMG data recorder with event marking functionality"""
    
    def __init__(self, uri: str, output_file: str, channel_count: int = 8, filter_raw_only: bool = True, use_cumulative_timestamp: bool = True):
        self.uri = uri
        self.output_file = output_file
        self.channel_count = channel_count
        self.filter_raw_only = filter_raw_only
        self.use_cumulative_timestamp = use_cumulative_timestamp  # New parameter for timestamp mode
        self.recording_active = False
        self.event_queue = queue.Queue()
        self.recording_start_time = None
        self.websocket = None
        
        # Filter parameters
        self.bp_parameter = np.zeros((channel_count, 8))
        self.nt_parameter = np.zeros((channel_count, 2))
        self.lp_parameter = np.zeros((channel_count, 4))
        
        # Data statistics
        self.sample_count = 0
        self.data_packet_count = 0
        
        # Cumulative timestamp tracking
        self.cumulative_timestamp = 0.0  # Start from 0
        self.timestamp_increment = 0.001  # 1ms increment for 1000Hz
        
    def add_event_marker(self, event_id: int, marker_name: str = "", duration: float = 0.0):
        """Add event marker to queue"""
        if not self.recording_active:
            print(f"⚠️  Recording not active, cannot add event: {event_id}")
            return False
            
        current_time = time.time()
        relative_time = current_time - self.recording_start_time if self.recording_start_time else 0
        
        event_data = {
            'event_id': event_id,
            'event_date': current_time,
            'relative_time': relative_time,
            'duration': duration,
            'marker_name': marker_name,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        self.event_queue.put(event_data)
        print(f"✅ Added EMG Event: ID={event_id}, Name='{marker_name}', Time={relative_time:.3f}s")
        return True
        
    def start_recording(self):
        """Start EMG recording"""
        if self.recording_active:
            print("⚠️  Recording already in progress")
            return False
            
        try:
            print(f"🎯 Starting EMG recording: {self.uri}")
            print(f"📁 Output file: {self.output_file}")
            
            # Ensure output directory exists
            output_dir = os.path.dirname(self.output_file)
            if output_dir:  # Only create if directory is not empty
                os.makedirs(output_dir, exist_ok=True)
            
            # Establish WebSocket connection
            self.websocket = websocket.WebSocket()
            self.websocket.connect(self.uri)
            
            self.recording_active = True
            self.recording_start_time = time.time()
            self.sample_count = 0
            self.data_packet_count = 0
            
            # Clear event queue
            while not self.event_queue.empty():
                self.event_queue.get()
            
            # Create CSV file and write header
            self._write_csv_header()
            
            print("✅ EMG recording started")
            return True
            
        except Exception as e:
            print(f"❌ EMG recording startup failed: {str(e)}")
            self.recording_active = False
            return False
    
    def stop_recording(self):
        """Stop EMG recording"""
        if not self.recording_active:
            print("⚠️  Recording not in progress")
            return False
            
        self.recording_active = False
        
        if self.websocket:
            self.websocket.close()
            self.websocket = None
            
        print(f"🛑 EMG recording stopped")
        print(f"📊 Processed {self.data_packet_count} data packets, {self.sample_count} samples")
        print(f"📁 Data saved to: {self.output_file}")
        return True
    
    def _write_csv_header(self):
        """Write Cygnus-style CSV header"""
        with open(self.output_file, 'w', newline='', encoding='utf-8') as csvfile:
            # Write file information header
            csvfile.write("Cygnus version: NTKCAP Integration\n")
            csvfile.write(f"Record datetime: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}\n")
            csvfile.write("Device name: STEEG_DG819202\n")
            csvfile.write("Device sampling rate: 1000 samples/second\n")
            csvfile.write("Data type / unit: EMG / micro-volt (uV)\n")
            csvfile.write("Filter: Bandpass 20-450Hz, Notch 50Hz, Envelope 10Hz\n")
            csvfile.write("Processing: Rectified and Low-pass filtered\n")
            csvfile.write("Timestamp format: Relative time from recording start\n")
            csvfile.write("Event synchronization: Real-time marking\n")
            csvfile.write("Integration: NTKCAP Motion Capture System\n")
            csvfile.write("Notes: Synchronized with motion capture data\n")
            
            # Write field headers
            writer = csv.writer(csvfile)
            headers = ['Timestamp', 'Serial Number']
            headers.extend([f'CH{i+1}' for i in range(self.channel_count)])
            headers.extend(['Event Id', 'Event Date', 'Event Duration', 'Software Marker', 'Software Marker Name'])
            writer.writerow(headers)
    
    def process_and_save_data(self, timeout: float = 0.1):
        """Process one data reception and save"""
        if not self.recording_active:
            return False
            
        try:
            # Set short timeout to avoid blocking
            self.websocket.settimeout(timeout)
            data = self.websocket.recv()
            
            # Process EMG data
            emg_array, self.bp_parameter, self.nt_parameter, self.lp_parameter = process_data_from_websocket(
                data, self.bp_parameter, self.nt_parameter, self.lp_parameter, self.channel_count, self.filter_raw_only
            )
            
            if emg_array.shape[0] != 0:
                # Get current events
                current_events = self._get_current_events()
                
                # Save data to CSV
                self._save_data_to_csv(emg_array, current_events)
                self.data_packet_count += 1
                
                # Display status every 100 data packets
                if self.data_packet_count % 100 == 0:
                    relative_time = time.time() - self.recording_start_time
                    print(f"📊 Processed {self.data_packet_count} data packets, recording time: {relative_time:.1f}s")
                
                return True
            
        except websocket.WebSocketTimeoutException:
            # Timeout is normal, allows checking events and status
            pass
        except Exception as e:
            print(f"❌ Data processing error: {str(e)}")
            return False
            
        return False
    
    def _get_current_events(self) -> Dict[str, Any]:
        """Get current time point events"""
        events = {}
        
        # Process all events in queue
        while not self.event_queue.empty():
            try:
                event = self.event_queue.get_nowait()
                event_time_key = f"{event['relative_time']:.3f}"
                events[event_time_key] = event
            except queue.Empty:
                break
                
        return events
    
    def _save_data_to_csv(self, emg_array: np.ndarray, events: Dict[str, Any]):
        """Save data to CSV file"""
        with open(self.output_file, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # Process each data point
            for i in range(emg_array.shape[1]):
                current_time = time.time()
                relative_time = current_time - self.recording_start_time
                
                # Check if there is corresponding event (allow 0.1 second tolerance)
                event_id = ""
                event_date = ""
                event_duration = ""
                software_marker = ""
                software_marker_name = ""
                
                for event_key, event_data in list(events.items()):
                    event_time = float(event_key)
                    if abs(relative_time - event_time) <= 0.1:
                        event_id = event_data['event_id']
                        event_date = event_data['event_date']
                        event_duration = event_data['duration']
                        software_marker = ""
                        software_marker_name = event_data['marker_name']
                        # Remove used event
                        del events[event_key]
                        break
                
                # Write data row with appropriate timestamp
                if self.use_cumulative_timestamp:
                    timestamp = f"{self.cumulative_timestamp:.3f}"
                    self.cumulative_timestamp += self.timestamp_increment
                else:
                    timestamp = f"{relative_time:.3f}"
                
                row = [timestamp, self.sample_count]
                row.extend([emg_array[ch, i] for ch in range(self.channel_count)])
                row.extend([event_id, event_date, event_duration, software_marker, software_marker_name])
                writer.writerow(row)
                
                self.sample_count += 1

def test_emg_event_recorder(data_points=3000):
    """Test EMG Event recorder
    
    Args:
        data_points: Target number of data points to collect (default: 3000)
    """
    print("🧪 Starting EMG Event recorder test")
    print("="*50)
    
    # Test parameters
    test_uri = "ws://localhost:31278/ws"
    test_output = os.path.join(os.getcwd(), "test_emg_with_events.csv")  # Use absolute path of current directory
    
    # Calculate test duration based on data points (assuming ~1000 Hz sampling rate)
    test_duration = max(10, data_points / 100)  # At least 10 seconds, or based on data points
    print(f"Target data points: {data_points}")
    print(f"Estimated test duration: {test_duration:.1f} seconds")
    
    try:
        # Detect channel count
        print("🔍 Detecting EMG channel count...")
        channel_count = detect_channel_count(test_uri)
        
        # Create recorder
        recorder = EMGEventRecorder(test_uri, test_output, channel_count)
        
        # Start recording
        if not recorder.start_recording():
            return False
        
        print(f"⏱️  Recording for {test_duration} seconds, will add Event markers at different time points...")
        print("Press Ctrl+C to stop recording early")
        
        start_time = time.time()
        
        # Add start event
        recorder.add_event_marker(130, "Recording Start")
        
        # Event marking control variables
        event1_added = False
        event2_added = False  
        event3_added = False
        
        try:
            while recorder.recording_active and recorder.sample_count < data_points:
                # Process data
                processed = recorder.process_and_save_data()
                
                # Add test events at different progress points (each event added only once)
                progress = recorder.sample_count / data_points
                elapsed = time.time() - start_time
                
                if progress >= 0.1 and not event1_added:  # At 10% progress
                    recorder.add_event_marker(141, "Test Event 1 (10% progress)")
                    event1_added = True
                elif progress >= 0.5 and not event2_added:  # At 50% progress
                    recorder.add_event_marker(142, "Test Event 2 (50% progress)")
                    event2_added = True
                elif progress >= 0.8 and not event3_added:  # At 80% progress
                    recorder.add_event_marker(143, "Test Event 3 (80% progress)")
                    event3_added = True
                
                # Show progress every 500 samples
                if recorder.sample_count > 0 and recorder.sample_count % 500 == 0:
                    progress_pct = (recorder.sample_count / data_points) * 100
                    print(f"📊 Progress: {recorder.sample_count}/{data_points} samples ({progress_pct:.1f}%)")
                
                time.sleep(0.001)  # Brief delay to avoid excessive CPU usage
                
        except KeyboardInterrupt:
            print("\n⚠️  User interrupted recording")
        
        # Add end event and stop recording
        recorder.add_event_marker(131, "Recording End")
        time.sleep(0.1)  # Let the last event have time to be processed
        recorder.stop_recording()
        
        print(f"✅ Test completed! Data saved to: {os.path.abspath(test_output)}")
        print("\n📋 Test summary:")
        print(f"   - Recording time: {time.time() - start_time:.1f} seconds")
        print(f"   - Collected samples: {recorder.sample_count}")
        print(f"   - Target samples: {data_points}")
        print(f"   - Data packets processed: {recorder.data_packet_count}")
        print(f"   - Output file: {test_output}")
        print(f"   - EMG channel count: {channel_count}")
        print("\n🔍 Please check Event fields in CSV file, should contain following events:")
        print("   - Event ID 130: Recording Start")
        print("   - Event ID 141: Test Event 1 (10% progress)")
        print("   - Event ID 142: Test Event 2 (50% progress)")
        print("   - Event ID 143: Test Event 3 (80% progress)")
        print("   - Event ID 131: Recording End")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='EMG WebSocket Data Reader',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Usage Examples:
  python emg_localhost.py                              # Auto scan, single read (quick mode)
  python emg_localhost.py --scan-frequency             # Auto scan with full frequency range
  python emg_localhost.py --uri ws://localhost:31278/ws  # Direct URI specification
  python emg_localhost.py -u ws://192.168.1.100:31278   # Use short parameter
  python emg_localhost.py --uri localhost:31278         # Auto add ws:// prefix
  python emg_localhost.py -c -o emg_data.csv            # Continuous mode with output file
  python emg_localhost.py -u localhost:31278 -c         # Continuous mode auto filename
  python emg_localhost.py --test-events                 # Test EMG with event markers (3000 samples)
  python emg_localhost.py -te --test-samples 5000       # Test with 5000 samples
  python emg_localhost.py -te -sf --test-samples 1000   # Test with frequency scan
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
    
    parser.add_argument(
        '--test-events', '-te',
        action='store_true',
        help='Test mode: test EMG recording with event markers'
    )
    
    parser.add_argument(
        '--scan-frequency', '-sf',
        action='store_true',
        help='Enable frequency scanning mode for auto-discovery'
    )
    
    parser.add_argument(
        '--test-samples', '-ts',
        type=int,
        default=3000,
        help='Number of samples for test mode (default: 3000)'
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

def scan_for_websocket_servers(host_range=None, port_range=None, timeout=1, enable_scan=True):
    """Scan for available WebSocket servers
    
    Args:
        host_range: List of hosts to scan
        port_range: Range of ports to scan
        timeout: Connection timeout
        enable_scan: If False, only try common default ports
    """
    if host_range is None:
        host_range = ['localhost', '127.0.0.1']
    
    if not enable_scan:
        # Only try common EMG device ports
        port_range = [31278, 31279, 31280]
        print("Using quick mode (limited port scan)...")
    elif port_range is None:
        port_range = range(31270, 31290)  # Full EMG device port range
        print("Using full frequency scan mode...")
    
    available_servers = []
    
    print(f"Scanning {len(host_range)} hosts and {len(list(port_range))} ports...")
    
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

def find_emg_server(enable_scan=True):
    """Automatically find EMG WebSocket server
    
    Args:
        enable_scan: If True, perform full frequency scan; if False, only try common ports
    """
    print("=== Auto Scanning EMG WebSocket Server ===")
    
    if enable_scan:
        print("🔍 Full frequency scanning mode enabled")
    else:
        print("⚡ Quick scan mode (common ports only)")
    
    # First scan for available WebSocket servers
    available_servers = scan_for_websocket_servers(enable_scan=enable_scan)
    
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

def process_data_from_websocket(data, bp_parameter, nt_parameter, lp_parameter, channel_count=8, filter_raw_only=True):
    """
    Process data from WebSocket
    
    Args:
        data: WebSocket data
        bp_parameter: Bandpass filter parameters
        nt_parameter: Notch filter parameters  
        lp_parameter: Lowpass filter parameters
        channel_count: Number of channels (default: 8)
        filter_raw_only: If True, only process raw data; if False, process all data types (default: True)
    """
    emg_values = np.zeros((channel_count, 50))
    j = 0
    try:
        data_dict = json.loads(data)
        if "contents" in data_dict:
            # Check data type filter
            data_type = data_dict.get("type", {})
            source_type = data_type.get("source_type", "")
            
            if filter_raw_only:
                # Only process raw data
                if source_type != "raw":
                    return np.array([]), bp_parameter, nt_parameter, lp_parameter
            else:
                # Only process algorithm/filtered data
                if source_type != "algorithm":
                    return np.array([]), bp_parameter, nt_parameter, lp_parameter
            
            # Extract serial_number and eeg/data values
            if filter_raw_only:
                # For raw data, use 'eeg' field
                serial_numbers_eegs = [(item['serial_number'][0], item['eeg']) for item in data_dict['contents'] if 'eeg' in item and len(item['eeg']) > 0]
            else:
                # For algorithm data, use 'data' field
                serial_numbers_eegs = [(item['sync_tick'], item['data']) for item in data_dict['contents'] if 'data' in item and len(item['data']) > 0]
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
    print("EMG WebSocket Data Reader with Event Markers")
    print("="*50)
    
    # Parse command line arguments
    args = parse_arguments()
    
    # 測試模式
    if args.test_events:
        print("🧪 EMG Event Test Mode")
        print(f"📊 Target samples: {args.test_samples}")
        success = test_emg_event_recorder(data_points=args.test_samples)
        if success:
            print("\n✅ Test completed successfully")
        else:
            print("\n❌ Test failed")
        input("Press Enter to exit...")
        return
    
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
        websocket_uri = find_emg_server(enable_scan=args.scan_frequency)
    
    if not websocket_uri:
        print("❌ Cannot find available EMG WebSocket server")
        print("\nPlease ensure:")
        print("1. EMG device is connected and running")
        print("2. WebSocket server is started")
        print("3. Firewall allows connection")
        print("\nOr use --uri parameter to specify server address directly:")
        print("  python emg_localhost.py --uri ws://localhost:31278/ws")
        print("\nOr test event recording system:")
        print("  python emg_localhost.py --test-events")
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

