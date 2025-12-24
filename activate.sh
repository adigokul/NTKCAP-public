#!/bin/bash
################################################################################
# NTKCAP Environment Activation Script
# Source this script to set up the environment: source activate.sh
################################################################################

# Get the directory where this script is located
NTKCAP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║           NTKCAP Environment Activation                    ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}"

# ============================================================================
# QT PLATFORM CONFIGURATION (CRITICAL FOR PYQT6 GUI)
# ============================================================================
# Force Qt to use XCB platform plugin (system graphics)
# Without this, PyQt6 may fail with "could not load Qt platform plugin xcb"
export QT_QPA_PLATFORM=xcb

# Disable Qt's automatic scaling (prevents blurry text on HiDPI)
export QT_AUTO_SCREEN_SCALE_FACTOR=0

# Use native file dialogs
export QT_QPA_PLATFORMTHEME=gtk3

echo -e "${GREEN}✓ Qt Platform:${NC} xcb (system graphics)"

# ============================================================================
# CUDA SETUP
# ============================================================================
CUDA_PATHS=("/usr/local/cuda-11.8" "/usr/local/cuda" "/opt/cuda")
CUDA_HOME=""
for path in "${CUDA_PATHS[@]}"; do
    if [[ -d "$path" && -f "$path/bin/nvcc" ]]; then
        CUDA_HOME="$path"
        break
    fi
done

if [[ -n "$CUDA_HOME" ]]; then
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
    export CUDA_HOME="$CUDA_HOME"
    echo -e "${GREEN}✓ CUDA:${NC} $CUDA_HOME"
else
    echo -e "${YELLOW}⚠ CUDA not found in standard paths${NC}"
fi

# ============================================================================
# TENSORRT SETUP
# ============================================================================
TENSORRT_DIR="${NTKCAP_ROOT}/NTK_CAP/ThirdParty/TensorRT-8.6.1.6"
if [[ -d "$TENSORRT_DIR" && -f "$TENSORRT_DIR/lib/libnvinfer.so" ]]; then
    export TENSORRT_DIR="$TENSORRT_DIR"
    export TENSORRT_ROOT="$TENSORRT_DIR"
    export TRT_LIBPATH="${TENSORRT_DIR}/lib"
    export LD_LIBRARY_PATH="${TENSORRT_DIR}/lib:${LD_LIBRARY_PATH:-}"
    echo -e "${GREEN}✓ TensorRT:${NC} $TENSORRT_DIR"
else
    echo -e "${YELLOW}⚠ TensorRT not found at $TENSORRT_DIR${NC}"
    echo -e "${YELLOW}  Run install/scripts/setup_linux.sh to install it${NC}"
fi

# ============================================================================
# MMDEPLOY SDK (libmmdeploy.so) - CRITICAL FOR TENSORRT INFERENCE
# ============================================================================
MMDEPLOY_LIB_PATH_FILE="${NTKCAP_ROOT}/.mmdeploy_lib_path"
MMDEPLOY_SDK_LOADED=false

if [[ -f "$MMDEPLOY_LIB_PATH_FILE" ]]; then
    MMDEPLOY_LIB_DIR=$(cat "$MMDEPLOY_LIB_PATH_FILE")
    if [[ -d "$MMDEPLOY_LIB_DIR" && -f "$MMDEPLOY_LIB_DIR/libmmdeploy.so" ]]; then
        export LD_LIBRARY_PATH="${MMDEPLOY_LIB_DIR}:${LD_LIBRARY_PATH:-}"
        echo -e "${GREEN}✓ MMDeploy SDK:${NC} $MMDEPLOY_LIB_DIR"
        MMDEPLOY_SDK_LOADED=true
    fi
fi

# Fallback: check default build location
if [[ "$MMDEPLOY_SDK_LOADED" == "false" ]]; then
    MMDEPLOY_BUILD_LIB="${NTKCAP_ROOT}/NTK_CAP/ThirdParty/mmdeploy/build/lib"
    if [[ -d "$MMDEPLOY_BUILD_LIB" && -f "$MMDEPLOY_BUILD_LIB/libmmdeploy.so" ]]; then
        export LD_LIBRARY_PATH="${MMDEPLOY_BUILD_LIB}:${LD_LIBRARY_PATH:-}"
        echo -e "${GREEN}✓ MMDeploy SDK:${NC} $MMDEPLOY_BUILD_LIB"
        MMDEPLOY_SDK_LOADED=true
    fi
fi

if [[ "$MMDEPLOY_SDK_LOADED" == "false" ]]; then
    echo -e "${RED}✗ MMDeploy SDK NOT FOUND (libmmdeploy.so missing)${NC}"
    echo -e "${RED}  TensorRT inference will FAIL without this library!${NC}"
    echo -e "${YELLOW}  Run: ./install/scripts/setup_linux.sh${NC}"
fi

# ============================================================================
# CONDA ENVIRONMENT ACTIVATION
# ============================================================================
# Find and source conda
CONDA_SOURCED=false
for conda_path in "$HOME/miniconda3" "$HOME/anaconda3" "/opt/conda"; do
    if [[ -f "${conda_path}/etc/profile.d/conda.sh" ]]; then
        source "${conda_path}/etc/profile.d/conda.sh"
        CONDA_SOURCED=true
        break
    fi
done

if [[ "$CONDA_SOURCED" == "false" ]]; then
    echo -e "${RED}✗ Conda not found. Please install Miniconda or Anaconda.${NC}"
else
    # Activate the ntkcap_env environment
    if conda activate ntkcap_env 2>/dev/null; then
        echo -e "${GREEN}✓ Conda:${NC} ntkcap_env activated"
        echo -e "${GREEN}✓ Python:${NC} $(python --version 2>&1)"
    else
        echo -e "${YELLOW}⚠ Could not activate ntkcap_env${NC}"
        echo -e "${YELLOW}  Run install/scripts/setup_linux.sh to create it${NC}"
    fi
fi

# ============================================================================
# GPU VERIFICATION
# ============================================================================
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    if [[ -n "$GPU_NAME" ]]; then
        echo -e "${GREEN}✓ GPU:${NC} $GPU_NAME"
    else
        echo -e "${RED}✗ GPU not accessible via nvidia-smi${NC}"
    fi
else
    echo -e "${RED}✗ nvidia-smi not found - NVIDIA drivers not installed!${NC}"
fi

# ============================================================================
# PYTORCH CUDA VERIFICATION
# ============================================================================
if [[ "${CONDA_DEFAULT_ENV}" == "ntkcap_env" ]]; then
    python -c "
import torch
if torch.cuda.is_available():
    print(f'\033[0;32m✓ PyTorch CUDA:\033[0m {torch.cuda.get_device_name(0)}')
    print(f'\033[0;32m✓ CUDA Version:\033[0m {torch.version.cuda}')
else:
    print('\033[0;31m✗ PyTorch CUDA not available\033[0m')
" 2>/dev/null || echo -e "${YELLOW}⚠ Could not verify PyTorch CUDA${NC}"
fi

# ============================================================================
# MMDEPLOY RUNTIME VERIFICATION
# ============================================================================
if [[ "${CONDA_DEFAULT_ENV}" == "ntkcap_env" && "$MMDEPLOY_SDK_LOADED" == "true" ]]; then
    python -c "
try:
    from mmdeploy_runtime import PoseTracker
    print('\033[0;32m✓ MMDeploy Runtime:\033[0m PoseTracker available')
except ImportError as e:
    print(f'\033[0;31m✗ MMDeploy Runtime:\033[0m {e}')
except Exception as e:
    print(f'\033[0;33m⚠ MMDeploy Runtime:\033[0m {e}')
" 2>/dev/null
fi

# ============================================================================
# SET WORKING DIRECTORY
# ============================================================================
cd "$NTKCAP_ROOT"
export NTKCAP_ROOT

echo ""
echo -e "${CYAN}Working directory:${NC} $NTKCAP_ROOT"
echo ""
echo "Quick commands:"
echo "  python NTKCAP_GUI.py          # Main GUI"
echo "  python final_NTK_Cap_GUI.py   # Alternative GUI"
echo "  python remote_calculate.py    # Remote calculation"
echo ""
echo "LD_LIBRARY_PATH includes:"
echo "$LD_LIBRARY_PATH" | tr ':' '\n' | head -5
echo ""
