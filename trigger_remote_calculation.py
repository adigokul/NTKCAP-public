"""
Remote Calculation Interface for NTKCAP
Handles remote calculation requests via JSON configuration and SSH

Usage:
    python trigger_remote_calculation.py --config remote_config.json --paths path1 path2 ...
    python trigger_remote_calculation.py --server-config config/remote_server.json --calc-config calc_config.json
"""

import sys
import os
import json
import argparse
from datetime import datetime
from typing import List, Dict, Optional

# Add GUI_source to path
gui_source_path = os.path.join(os.path.dirname(__file__), 'GUI_source')
if gui_source_path not in sys.path:
    sys.path.insert(0, gui_source_path)

from RemoteCalculationClient import RemoteCalculationClient


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='NTKCAP Remote Calculation Trigger Interface',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test connection only
  python trigger_remote_calculation.py --test-connection --server-config config/remote_server.json
  
  # Trigger calculation with paths
  python trigger_remote_calculation.py --server-config config/remote_server.json --paths "D:/NTKCAP/Patient_data/patient1/2024_11_13"
  
  # Trigger calculation with JSON config
  python trigger_remote_calculation.py --server-config config/remote_server.json --calc-config calculation_task.json
  
  # Upload data and trigger calculation
  python trigger_remote_calculation.py --server-config config/remote_server.json --paths "path1" --upload
        """
    )
    
    parser.add_argument(
        '--server-config',
        type=str,
        required=True,
        help='Remote server configuration JSON file (contains IP, username, etc.)'
    )
    
    parser.add_argument(
        '--calc-config',
        type=str,
        help='Calculation configuration JSON file (contains paths, options, etc.)'
    )
    
    parser.add_argument(
        '--paths',
        nargs='*',
        help='Calculation paths (remote server paths)'
    )
    
    parser.add_argument(
        '--fast',
        action='store_true',
        help='Enable fast calculation mode'
    )
    
    parser.add_argument(
        '--no-gait',
        action='store_true',
        help='Disable gait analysis'
    )
    
    parser.add_argument(
        '--upload',
        action='store_true',
        help='Upload data to remote server before calculation'
    )
    
    parser.add_argument(
        '--download',
        action='store_true',
        help='Download results from remote server after calculation'
    )
    
    parser.add_argument(
        '--test-connection',
        action='store_true',
        help='Test connection to remote server only (no calculation)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output mode'
    )
    
    return parser.parse_args()


def load_json_config(config_path: str) -> Dict:
    """Load JSON configuration file"""
    if not os.path.exists(config_path):
        print(f"❌ Configuration file not found: {config_path}")
        sys.exit(1)
        
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON format in {config_path}: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error loading configuration: {str(e)}")
        sys.exit(1)


def print_calculation_info(server_config: Dict, paths: List[str], fast_cal: bool, gait: bool):
    """Print calculation information"""
    print("\n" + "="*70)
    print("NTKCAP Remote Calculation Task")
    print("="*70)
    print(f"Remote Server: {server_config.get('server_ip')}:{server_config.get('server_port', 22)}")
    print(f"Username: {server_config.get('username')}")
    print(f"Remote Path: {server_config.get('remote_path')}")
    print(f"Conda Environment: {server_config.get('conda_env')}")
    print(f"\nCalculation Paths ({len(paths)} total):")
    for i, path in enumerate(paths, 1):
        print(f"  {i}. {path}")
    print(f"\nOptions:")
    print(f"  Fast Calculation: {'Yes' if fast_cal else 'No'}")
    print(f"  Gait Analysis: {'Yes' if gait else 'No'}")
    print("="*70 + "\n")


def test_connection(server_config: Dict) -> bool:
    """Test connection to remote server"""
    print("\n🔍 Testing connection to remote server...")
    
    client = RemoteCalculationClient(server_config)
    
    if not client.connect():
        return False
    
    success = client.test_connection()
    client.close()
    
    if success:
        print("\n✅ Connection test successful!\n")
    else:
        print("\n❌ Connection test failed!\n")
    
    return success


def upload_data_to_remote(client: RemoteCalculationClient, local_paths: List[str], 
                         remote_base: str, verbose: bool = False) -> bool:
    """Upload data directories to remote server"""
    print("\n📤 Uploading data to remote server...")
    
    success_count = 0
    for local_path in local_paths:
        if not os.path.exists(local_path):
            print(f"⚠️  Local path not found: {local_path}")
            continue
        
        # Calculate remote path
        # Assuming local_path is like: D:\NTKCAP\Patient_data\patient\date
        # Remote should be: remote_base/Patient_data/patient/date
        rel_path = os.path.relpath(local_path, os.getcwd())
        remote_path = f"{remote_base}/{rel_path}".replace('\\', '/')
        
        if verbose:
            print(f"  Uploading: {local_path} -> {remote_path}")
        
        if client.upload_directory(local_path, remote_path):
            success_count += 1
            print(f"  ✅ Uploaded: {rel_path}")
        else:
            print(f"  ❌ Failed: {rel_path}")
    
    print(f"\n📊 Upload completed: {success_count}/{len(local_paths)} successful\n")
    return success_count == len(local_paths)


def download_results_from_remote(client: RemoteCalculationClient, remote_paths: List[str], 
                                 local_base: str, verbose: bool = False) -> bool:
    """Download calculation results from remote server"""
    print("\n📥 Downloading results from remote server...")
    
    success_count = 0
    for remote_path in remote_paths:
        # Find calculation result folders (folders with timestamps)
        # Remote path is like: D:/NTKCAP/Patient_data/patient/date
        # Results are in: D:/NTKCAP/Patient_data/patient/date/YYYY_MM_DD_HH_MM_calculated
        
        # For now, download the entire date folder
        rel_path = remote_path.replace(client.remote_path + '/', '')
        local_path = os.path.join(local_base, rel_path.replace('/', '\\'))
        
        if verbose:
            print(f"  Downloading: {remote_path} -> {local_path}")
        
        if client.download_directory(remote_path, local_path):
            success_count += 1
            print(f"  ✅ Downloaded: {rel_path}")
        else:
            print(f"  ❌ Failed: {rel_path}")
    
    print(f"\n📊 Download completed: {success_count}/{len(remote_paths)} successful\n")
    return success_count == len(remote_paths)


def trigger_remote_calculation(client: RemoteCalculationClient, paths: List[str], 
                               fast_cal: bool = False, gait: bool = True, 
                               task_filter_dict: Optional[Dict] = None,
                               verbose: bool = False) -> bool:
    """Trigger calculation on remote server"""
    print("\n🚀 Triggering remote calculation...")
    
    start_time = datetime.now()
    print(f"⏱️  Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Execute calculation
    stdout, stderr, exit_code = client.execute_calculation(
        paths=paths,
        fast_cal=fast_cal,
        gait=gait,
        task_filter_dict=task_filter_dict
    )
    
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    
    # Print results
    print("\n" + "="*70)
    if exit_code == 0:
        print("✅ Remote calculation completed successfully!")
    else:
        print("❌ Remote calculation failed!")
    print("="*70)
    print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total time: {elapsed_time}")
    print("="*70)
    
    if verbose and stdout:
        print("\n📋 Standard Output:")
        print(stdout)
    
    if stderr and exit_code != 0:
        print("\n⚠️  Error Output:")
        print(stderr)
    
    print()
    
    return exit_code == 0


def main():
    """Main function"""
    args = parse_arguments()
    
    # Load server configuration
    print(f"📄 Loading server configuration from: {args.server_config}")
    server_config = load_json_config(args.server_config)
    
    # Test connection only
    if args.test_connection:
        success = test_connection(server_config)
        return 0 if success else 1
    
    # Determine calculation paths and options
    if args.calc_config:
        # Load from calculation config file
        print(f"📄 Loading calculation configuration from: {args.calc_config}")
        calc_config = load_json_config(args.calc_config)
        paths = calc_config.get('paths', [])
        fast_cal = calc_config.get('fast_cal', args.fast)
        gait = calc_config.get('gait', not args.no_gait)
        task_filter_dict = calc_config.get('task_filter_dict', None)
    else:
        # Use command line arguments
        if not args.paths:
            print("❌ Error: No calculation paths specified")
            print("Use --paths or --calc-config to specify paths")
            return 1
        
        paths = args.paths
        fast_cal = args.fast
        gait = not args.no_gait
        task_filter_dict = None
    
    # Print calculation info
    if args.verbose:
        print_calculation_info(server_config, paths, fast_cal, gait)
    
    # Connect to remote server
    print(f"\n🔌 Connecting to remote server...")
    client = RemoteCalculationClient(server_config)
    
    if not client.connect():
        print("❌ Failed to connect to remote server")
        return 1
    
    try:
        # Test connection
        if not client.test_connection():
            print("❌ Remote server environment validation failed")
            return 1
        
        # Upload data if requested
        if args.upload:
            # Convert remote paths to local paths for upload
            local_paths = [p.replace(server_config['remote_path'], os.getcwd()).replace('/', '\\') 
                          for p in paths]
            if not upload_data_to_remote(client, local_paths, server_config['remote_path'], args.verbose):
                print("⚠️  Some uploads failed, but continuing with calculation...")
        
        # Trigger calculation
        success = trigger_remote_calculation(
            client=client,
            paths=paths,
            fast_cal=fast_cal,
            gait=gait,
            task_filter_dict=task_filter_dict,
            verbose=args.verbose
        )
        
        if not success:
            print("❌ Remote calculation failed")
            return 1
        
        # Download results if requested
        if args.download:
            if not download_results_from_remote(client, paths, os.getcwd(), args.verbose):
                print("⚠️  Some downloads failed")
                return 1
        
        print("✅ All operations completed successfully!\n")
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        if args.verbose:
            import traceback
            print("\nDetailed traceback:")
            traceback.print_exc()
        return 1
        
    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(main())
