#!/bin/bash
# CUDA and GPU Status Testing Script
# Tests NVIDIA drivers, CUDA toolkit, and PyTorch GPU functionality

echo "=== CUDA and GPU Status Check ==="
echo ""

echo "1. NVIDIA Driver Status:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "❌ nvidia-smi not found - NVIDIA drivers may not be installed"
fi
echo ""

echo "2. CUDA Compiler Version:"
if command -v nvcc &> /dev/null; then
    nvcc --version
else
    echo "❌ nvcc not found - CUDA toolkit may not be installed"
fi
echo ""

echo "3. Activating conda environment and testing PyTorch:"
source ~/.bashrc
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ntkcap

python -c "
import torch
print(f'PyTorch Version: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA Version: {torch.version.cuda}')
    print(f'GPU Count: {torch.cuda.device_count()}')
    print(f'GPU Name: {torch.cuda.get_device_name(0)}')
    
    # Simple CUDA functionality test
    print('\\n=== CUDA Functionality Test ===')
    device = torch.device('cuda:0')
    x = torch.rand(5, 3).to(device)
    y = torch.rand(3, 4).to(device)
    z = torch.mm(x, y)
    print(f'CUDA matrix operation test successful! Result shape: {z.shape}')
    print('🎉 GPU acceleration fully functional!')
else:
    print('❌ CUDA still not available')
    print('\\nTroubleshooting steps:')
    print('1. Ensure NVIDIA drivers are installed: nvidia-smi')
    print('2. Check CUDA installation: nvcc --version')
    print('3. Verify environment variables are set')
    print('4. If drivers were recently installed, reboot the system')
"