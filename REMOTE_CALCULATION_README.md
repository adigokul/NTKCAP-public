# NTKCAP Remote Calculation Interface

This directory contains scripts for triggering calculations on remote NTKCAP servers.

## Files

- **trigger_remote_calculation.py**: Main interface script for remote calculation
- **remote_calculate.py**: Script that runs on remote server (to be synced to remote)
- **GUI_source/RemoteCalculationClient.py**: SSH client for remote communication
- **config/remote_server_example.json**: Server configuration template
- **config/calculation_example.json**: Calculation task configuration template

## Setup

### 1. Install Dependencies

```bash
conda activate ntkcap_fast
pip install paramiko scp
```

### 2. Configure Remote Server

Edit `config/remote_server_example.json`:

```json
{
  "server_ip": "your.server.ip",
  "server_port": 22,
  "username": "ntk_cap",
  "password": "your_password_or_leave_empty_for_ssh_key",
  "ssh_key_path": "path/to/ssh/key",
  "remote_path": "D:/NTKCAP",
  "conda_env": "ntkcap_fast"
}
```

### 3. Sync Remote Server

Ensure the remote server has:
- Same NTKCAP codebase (sync via Git)
- `remote_calculate.py` in the root directory
- Conda environment `ntkcap_fast` activated

## Usage

### Test Connection

```bash
python trigger_remote_calculation.py --server-config config/remote_server_example.json --test-connection
```

### Trigger Calculation (Simple)

```bash
python trigger_remote_calculation.py \
    --server-config config/remote_server_example.json \
    --paths "D:/NTKCAP/Patient_data/patient1/2025_11_13" \
    --fast
```

### Trigger Calculation (With Config File)

1. Edit `config/calculation_example.json`:

```json
{
  "paths": [
    "D:/NTKCAP/Patient_data/patient1/2025_11_13",
    "D:/NTKCAP/Patient_data/patient2/2025_11_13"
  ],
  "fast_cal": true,
  "gait": true,
  "task_filter_dict": null
}
```

2. Run:

```bash
python trigger_remote_calculation.py \
    --server-config config/remote_server_example.json \
    --calc-config config/calculation_example.json \
    --verbose
```

### With Data Upload/Download

```bash
# Upload data, calculate, and download results
python trigger_remote_calculation.py \
    --server-config config/remote_server_example.json \
    --calc-config config/calculation_example.json \
    --upload \
    --download \
    --verbose
```

## Batch Script

Run `test_remote_trigger.bat` for interactive testing:

```cmd
test_remote_trigger.bat
```

## Workflow

1. **Local Machine**: Triggers calculation via `trigger_remote_calculation.py`
2. **SSH Connection**: Connects to remote server
3. **Remote Execution**: Runs `remote_calculate.py` on remote server
4. **Result**: Calculation results stored on remote server
5. **Download** (optional): Download results back to local machine

## Task Filtering

To calculate only specific tasks, add `task_filter_dict` to calculation config:

```json
{
  "paths": ["D:/NTKCAP/Patient_data/patient1/2025_11_13"],
  "fast_cal": true,
  "gait": true,
  "task_filter_dict": {
    "D:/NTKCAP/Patient_data/patient1/2025_11_13": ["walk", "run"]
  }
}
```

## Troubleshooting

### Connection Fails

- Check server IP and port
- Verify username and password/SSH key
- Ensure remote server SSH service is running
- Check firewall settings

### Remote Script Not Found

- Ensure `remote_calculate.py` is synced to remote server
- Check remote_path in server config

### Conda Environment Not Found

- Verify conda environment name on remote server
- Run `conda env list` on remote server to check

### Calculation Fails

- Check remote server logs
- Ensure remote server has all required dependencies
- Verify data paths exist on remote server

## Security Notes

- **Do not commit** `config/remote_server_example.json` with real passwords
- Use SSH keys instead of passwords when possible
- Keep server configuration files in `.gitignore`
