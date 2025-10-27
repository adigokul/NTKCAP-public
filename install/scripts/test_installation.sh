#!/bin/bash
# NTKCAP Environment Verification Script
# Verifies Poetry environment, dependencies, and NTKCAP functionality

echo "=== NTKCAP Environment Verification ==="
echo ""

source ~/.bashrc
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ntkcap

echo "1. Checking Poetry environment:"
if command -v poetry &> /dev/null; then
    poetry --version
else
    echo "❌ Poetry not found"
fi
echo ""

echo "2. Checking core dependencies:"
python -c "
import sys
print(f'Python Version: {sys.version}')

# Check main modules
modules = [
    'torch', 'torchvision', 'cv2', 'numpy', 'PyQt6',
    'mmcv', 'mmdet', 'mmengine'
]

print('\\n=== Module Import Test ===')
for module in modules:
    try:
        __import__(module)
        print(f'✅ {module}')
    except ImportError as e:
        print(f'❌ {module}: {e}')

print('\\n=== MMEngine GPU Support Check ===')
try:
    from mmengine.device import get_device
    device = get_device()
    print(f'MMEngine Default Device: {device}')
except Exception as e:
    print(f'MMEngine device check error: {e}')

print('\\n=== PyTorch GPU Check ===')
try:
    import torch
    print(f'PyTorch CUDA Available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'GPU Device Count: {torch.cuda.device_count()}')
        print(f'Current GPU: {torch.cuda.get_device_name(0)}')
except Exception as e:
    print(f'PyTorch GPU check error: {e}')
"

echo ""
echo "3. Checking NTKCAP main files:"
cd /home/ntk/NTKCAP

if [ -f "final_NTK_Cap_GUI.py" ]; then
    echo "✅ final_NTK_Cap_GUI.py exists"
else
    echo "❌ final_NTK_Cap_GUI.py not found"
fi

if [ -f "NTKCAP_GUI.py" ]; then
    echo "✅ NTKCAP_GUI.py exists"
else
    echo "❌ NTKCAP_GUI.py not found"
fi

if [ -d "Pose2Sim" ]; then
    echo "✅ Pose2Sim directory exists"
else
    echo "❌ Pose2Sim directory not found"
fi

if [ -f "pyproject.toml" ]; then
    echo "✅ pyproject.toml exists"
else
    echo "❌ pyproject.toml not found"
fi

echo ""
echo "4. Environment activation test:"
if [ -f "activate_ntkcap.sh" ]; then
    echo "✅ activate_ntkcap.sh exists"
    echo "To manually activate environment: ./activate_ntkcap.sh"
else
    echo "❌ activate_ntkcap.sh not found"
fi

echo ""
echo "=== Verification Summary ==="
echo "If all checks passed (✅), your NTKCAP environment is ready!"
echo ""
echo "To run NTKCAP applications:"
echo "1. cd /home/ntk/NTKCAP"
echo "2. ./activate_ntkcap.sh"
echo "3. python final_NTK_Cap_GUI.py"