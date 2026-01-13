# NTKCAP Installation

One-command installation for the NTKCAP motion capture system.

## Quick Start

```bash
# Clone the repository
git clone -b universal https://github.com/adigokul/NTKCAP-public.git
cd NTKCAP-public/install

# Run installation (with custom environment name)
chmod +x setup.sh
./setup.sh -e your_env_name
```

## Options

```
./setup.sh [OPTIONS]

Options:
  -e, --env-name NAME    Set conda environment name (default: NTKCAP)
  -h, --help             Show help message

Examples:
  ./setup.sh                      # Use default env name 'NTKCAP'
  ./setup.sh -e ntkcap_v2         # Create env named 'ntkcap_v2'
```

## What Gets Installed

- Conda environment with Python 3.10
- PyTorch 2.0.1 with CUDA 11.8
- TensorRT 8.6.1 SDK
- mmdeploy SDK (built locally)
- ppl.cv (built locally)
- OpenSim 4.5.2
- All required dependencies

## Requirements

- Ubuntu 20.04/22.04
- NVIDIA GPU with drivers installed
- CUDA 11.8 toolkit
- cuDNN 8.x
- Miniconda/Anaconda
- 20GB+ free disk space

## After Installation

```bash
# Activate the environment
source activate_ntkcap.sh

# Run the GUI
python NTKCAP_GUI.py
```

## Troubleshooting

If you encounter issues, check:
1. CUDA is properly installed: `nvcc --version`
2. GPU is detected: `nvidia-smi`
3. cuDNN is installed: `ldconfig -p | grep cudnn`

For support, open an issue on GitHub.
