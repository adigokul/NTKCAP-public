# NTKCAP Installation Scripts

This directory contains automated installation scripts for the NTKCAP motion capture system.

## Quick Installation

```bash
# Clone the repository
git clone <repository-url>
cd NTKCAP

# Run the complete installation script
sudo ./install/scripts/setup.sh
```

## What's Included

### `setup.sh` - Complete Environment Setup
- **System Requirements Check**: Validates Ubuntu 22.04 LTS
- **Miniconda Installation**: Downloads and installs if not present
- **Poetry Setup**: Installs Poetry for dependency management
- **NVIDIA Drivers & CUDA 11.8**: Automatic GPU support setup
- **Python Environment**: Creates `ntkcap` conda environment
- **Dependencies**: Installs all required packages including MMEngine ecosystem

### `test_installation.sh` - Environment Verification
- Checks Poetry installation
- Verifies core dependencies (PyTorch, OpenCV, MMEngine, etc.)
- Validates NTKCAP main files
- Tests environment activation script

### `test_cuda.sh` - GPU Functionality Test
- Verifies NVIDIA driver status
- Checks CUDA toolkit installation
- Tests PyTorch GPU acceleration
- Performs CUDA matrix operations

## System Requirements

- **OS**: Ubuntu 22.04 LTS (recommended)
- **GPU**: NVIDIA GPU with recent drivers
- **Memory**: 8GB+ RAM recommended
- **Storage**: 10GB+ free space for dependencies

## Usage After Installation

1. **Activate Environment**:
   ```bash
   cd /home/ntk/NTKCAP
   ./activate_ntkcap.sh
   ```

2. **Run NTKCAP GUI**:
   ```bash
   python final_NTK_Cap_GUI.py
   ```

3. **Test Installation**:
   ```bash
   cd install/scripts
   ./test_installation.sh
   ./test_cuda.sh
   ```

## Troubleshooting

- **REBOOT REQUIRED**: If NVIDIA drivers were installed, reboot your system
- **Permission Issues**: Ensure you have sudo privileges
- **CUDA Not Available**: Check if system reboot is needed after driver installation
- **Dependencies Failed**: Verify internet connection and system packages

## File Structure

```
install/
├── scripts/
│   ├── setup.sh              # Main installation script
│   ├── test_installation.sh  # Environment verification
│   └── test_cuda.sh          # CUDA functionality test
└── docs/                     # Additional documentation
```

For detailed troubleshooting, refer to the main `README_POETRY.md` in the project root.