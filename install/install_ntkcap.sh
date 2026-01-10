#!/bin/bash
################################################################################
# NTKCAP Complete Installation Script
# ================================================================================
# This script sets up the complete NTKCAP environment with TensorRT support.
#
# PREREQUISITES:
#   - Ubuntu 20.04/22.04 with NVIDIA GPU
#   - CUDA 11.8 installed (nvcc available)
#   - cuDNN 8.x installed
#   - Miniconda/Anaconda installed
#   - Git installed
#
# USAGE:
#   1. Run: chmod +x install_ntkcap.sh
#   2. Run: ./install_ntkcap.sh
#
# WHAT THIS SCRIPT DOES:
#   1. Installs required system dependencies
#   2. Creates conda environment "NTKCAP" with pinned dependencies
#   3. Downloads TensorRT from Google Drive (automatic)
#   4. Clones and builds ppl.cv with CUDA support
#   5. Builds mmdeploy SDK with TensorRT backend
#   6. Downloads model weights and generates TensorRT engines
#   7. Fixes pipeline.json files for proper inference
#   8. Verifies the installation
#
# TIME: 30-60 minutes depending on internet and GPU
################################################################################

set -e

# ==============================================================================
# CONFIGURATION - All paths are relative to project root
# ==============================================================================

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Project root is parent of install directory
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

# Conda environment name
CONDA_ENV_NAME="NTKCAP"

# Directory structure (relative to PROJECT_ROOT)
THIRDPARTY_DIR="${PROJECT_ROOT}/NTK_CAP/ThirdParty"
MMDEPLOY_DIR="${THIRDPARTY_DIR}/mmdeploy"
TENSORRT_VERSION="8.6.1.6"
TENSORRT_DIR="${THIRDPARTY_DIR}/TensorRT-${TENSORRT_VERSION}"
PPLCV_DIR="${THIRDPARTY_DIR}/ppl.cv"

# TensorRT archive
TENSORRT_ARCHIVE="TensorRT-${TENSORRT_VERSION}.Linux.x86_64-gnu.cuda-11.8.tar.gz"
# Google Drive download link (public)
TENSORRT_GDRIVE_ID="1CoETA05oSYV44WItwEh478HCCLZFZf8g"

# Model weights URLs
RTMDET_WEIGHT_URL="https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_m_8xb32-300e_coco/rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth"
RTMPOSE_HALPE_WEIGHT_URL="https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.pth"

# CUDA Architecture - Auto-detect or override
CUDA_ARCH=""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

log() { echo -e "${GREEN}[$(date +'%H:%M:%S')] ✓ $1${NC}"; }
warn() { echo -e "${YELLOW}[$(date +'%H:%M:%S')] ⚠ $1${NC}"; }
error() { echo -e "${RED}[$(date +'%H:%M:%S')] ✗ ERROR: $1${NC}"; exit 1; }
info() { echo -e "${CYAN}[$(date +'%H:%M:%S')] ℹ $1${NC}"; }
section() {
    echo ""
    echo -e "${BLUE}══════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}══════════════════════════════════════════════════════════════════════${NC}"
    echo ""
}

ask_continue() {
    echo ""
    read -p "$(echo -e ${YELLOW}Continue with $1? [Y/n]: ${NC})" response
    case "$response" in
        [nN][oO]|[nN])
            warn "Skipping $1"
            return 1
            ;;
        *)
            return 0
            ;;
    esac
}

detect_cuda_arch() {
    # Auto-detect GPU compute capability
    if command -v nvidia-smi &>/dev/null; then
        local gpu_info=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)
        if [[ -n "${gpu_info}" ]]; then
            # Convert 8.9 to 89
            CUDA_ARCH=$(echo "${gpu_info}" | tr -d '.')
            log "Detected GPU compute capability: ${gpu_info} (sm_${CUDA_ARCH})"
            return 0
        fi
    fi

    # Fallback - ask user
    warn "Could not auto-detect GPU architecture"
    echo "Common values:"
    echo "  - RTX 40 series: 89"
    echo "  - RTX 30 series: 86"
    echo "  - RTX 20 series: 75"
    echo "  - GTX 10 series: 61"
    read -p "Enter your GPU compute capability (e.g., 89 for RTX 4060): " CUDA_ARCH

    if [[ -z "${CUDA_ARCH}" ]]; then
        error "CUDA architecture is required"
    fi
}

# ==============================================================================
# BANNER
# ==============================================================================

echo ""
echo "╔════════════════════════════════════════════════════════════════════════╗"
echo "║                                                                        ║"
echo "║     ███╗   ██╗████████╗██╗  ██╗ ██████╗ █████╗ ██████╗                ║"
echo "║     ████╗  ██║╚══██╔══╝██║ ██╔╝██╔════╝██╔══██╗██╔══██╗               ║"
echo "║     ██╔██╗ ██║   ██║   █████╔╝ ██║     ███████║██████╔╝               ║"
echo "║     ██║╚██╗██║   ██║   ██╔═██╗ ██║     ██╔══██║██╔═══╝                ║"
echo "║     ██║ ╚████║   ██║   ██║  ██╗╚██████╗██║  ██║██║                    ║"
echo "║     ╚═╝  ╚═══╝   ╚═╝   ╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝╚═╝                    ║"
echo "║                                                                        ║"
echo "║              Complete Installation Script v1.0                         ║"
echo "║                                                                        ║"
echo "╚════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Project Root: ${PROJECT_ROOT}"
echo ""

# ==============================================================================
# STEP 0: PREREQUISITES CHECK
# ==============================================================================

section "Step 0: Checking Prerequisites"

# Check if running as root (not recommended)
if [[ $EUID -eq 0 ]]; then
    warn "Running as root is not recommended. Consider running as regular user."
fi

# Check CUDA
if ! command -v nvcc &>/dev/null; then
    # Try common CUDA paths
    for cuda_path in "/usr/local/cuda-11.8" "/usr/local/cuda"; do
        if [[ -f "${cuda_path}/bin/nvcc" ]]; then
            export PATH="${cuda_path}/bin:${PATH}"
            export LD_LIBRARY_PATH="${cuda_path}/lib64:${LD_LIBRARY_PATH:-}"
            export CUDA_HOME="${cuda_path}"
            break
        fi
    done
fi

if ! command -v nvcc &>/dev/null; then
    error "CUDA not found. Please install CUDA 11.8 and ensure nvcc is in PATH.

Installation guide: https://developer.nvidia.com/cuda-11-8-0-download-archive"
fi

CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
log "CUDA version: ${CUDA_VERSION}"

# Check if CUDA 11.8
if [[ ! "${CUDA_VERSION}" == "11.8"* ]]; then
    warn "CUDA ${CUDA_VERSION} detected. This script is tested with CUDA 11.8."
    warn "Compatibility with other versions is not guaranteed."
fi

# Check GPU
if ! command -v nvidia-smi &>/dev/null; then
    error "nvidia-smi not found. Please ensure NVIDIA drivers are installed."
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
log "GPU: ${GPU_NAME}"

# Detect CUDA architecture
detect_cuda_arch

# Check conda
if ! command -v conda &>/dev/null; then
    error "Conda not found. Please install Miniconda or Anaconda.

Installation guide: https://docs.conda.io/en/latest/miniconda.html"
fi

CONDA_BASE=$(conda info --base)
log "Conda base: ${CONDA_BASE}"

# Check git
if ! command -v git &>/dev/null; then
    error "Git not found. Please install git."
fi
log "Git: $(git --version)"

# Check cmake
if ! command -v cmake &>/dev/null; then
    warn "cmake not found. Will install via pip."
fi

# Check TensorRT - will be downloaded in Step 0.5 if not present
if [[ ! -f "${SCRIPT_DIR}/${TENSORRT_ARCHIVE}" ]] && [[ ! -d "${TENSORRT_DIR}" ]]; then
    info "TensorRT not found. Will be downloaded automatically in Step 0.5."
    NEED_TENSORRT_DOWNLOAD=1
else
    NEED_TENSORRT_DOWNLOAD=0
    log "TensorRT found"
fi

log "All prerequisites satisfied"

if ! ask_continue "installation"; then
    exit 0
fi

# ==============================================================================
# STEP 0.5: INSTALL SYSTEM DEPENDENCIES
# ==============================================================================

section "Step 0.5: Installing System Dependencies"

info "Installing required system packages..."
sudo apt-get update -qq

# Build essentials and development libraries
sudo apt-get install -y \
    build-essential \
    gcc-11 g++-11 \
    cmake \
    ninja-build \
    pkg-config \
    wget \
    curl \
    git \
    unzip \
    python3-pip \
    aria2 \
    libtinfo5 2>/dev/null || sudo ln -sf /lib/x86_64-linux-gnu/libtinfo.so.6 /lib/x86_64-linux-gnu/libtinfo.so.5

# Install gdown for Google Drive downloads
pip3 install --quiet gdown || pip install --quiet gdown

# Download TensorRT from Google Drive if needed
if [[ "${NEED_TENSORRT_DOWNLOAD}" -eq 1 ]]; then
    info "Downloading TensorRT ${TENSORRT_VERSION} from Google Drive (~3GB)..."

    mkdir -p "${SCRIPT_DIR}/downloads"
    TENSORRT_DOWNLOAD_PATH="${SCRIPT_DIR}/downloads/${TENSORRT_ARCHIVE}"

    gdown "https://drive.google.com/uc?id=${TENSORRT_GDRIVE_ID}" -O "${TENSORRT_DOWNLOAD_PATH}"

    if [[ ! -f "${TENSORRT_DOWNLOAD_PATH}" ]]; then
        error "Failed to download TensorRT from Google Drive.

Please download manually from:
  https://drive.google.com/file/d/${TENSORRT_GDRIVE_ID}/view

Save as: ${TENSORRT_DOWNLOAD_PATH}
Then run this script again."
    fi

    # Move to expected location
    mv "${TENSORRT_DOWNLOAD_PATH}" "${SCRIPT_DIR}/${TENSORRT_ARCHIVE}"
    log "TensorRT downloaded successfully"
fi

# OpenGL and graphics libraries
sudo apt-get install -y \
    libgl1-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libfontconfig1 \
    libice6 \
    libxkbcommon-x11-0 \
    libegl1

# XCB libraries for PyQt6 GUI
sudo apt-get install -y \
    libxcb-cursor0 \
    libxcb-xinerama0 \
    libxcb-xfixes0 \
    libxcb-shape0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-render-util0

# OpenCV development libraries (for CMake discovery during mmdeploy build)
sudo apt-get install -y libopencv-dev

# Add user to input group for keyboard module
if ! groups | grep -q '\binput\b'; then
    info "Adding user to 'input' group for keyboard module..."
    sudo usermod -a -G input "$(whoami)" || warn "Could not add user to input group"
    warn "You may need to log out and back in for keyboard permissions to take effect"
fi

log "System dependencies installed"

# ==============================================================================
# STEP 1: CREATE CONDA ENVIRONMENT
# ==============================================================================

section "Step 1: Creating Conda Environment '${CONDA_ENV_NAME}'"

# Source conda
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# Check if environment exists
if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
    warn "Environment '${CONDA_ENV_NAME}' already exists."
    read -p "$(echo -e ${YELLOW}Recreate environment? This will delete existing. [y/N]: ${NC})" response
    case "$response" in
        [yY][eE][sS]|[yY])
            info "Removing existing environment..."
            conda deactivate 2>/dev/null || true
            conda env remove -n "${CONDA_ENV_NAME}" -y
            ;;
        *)
            info "Using existing environment"
            conda activate "${CONDA_ENV_NAME}"
            log "Activated existing environment"
            ;;
    esac
fi

# Create environment if it doesn't exist
if ! conda env list | grep -q "^${CONDA_ENV_NAME} "; then
    info "Creating conda environment with Python 3.10..."
    conda create -n "${CONDA_ENV_NAME}" python=3.10 -y
fi

# Activate environment
conda activate "${CONDA_ENV_NAME}"
log "Activated environment: ${CONDA_ENV_NAME}"

# ==============================================================================
# STEP 2: INSTALL PYTHON DEPENDENCIES
# ==============================================================================

section "Step 2: Installing Python Dependencies"

info "Installing PyTorch with CUDA 11.8..."
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

info "Installing numpy..."
pip install numpy==1.22.4  # Must match verification and working environment

info "Installing OpenMMLab packages..."
pip install openmim
mim install mmengine==0.10.7
mim install mmcv==2.1.0
mim install mmdet==3.3.0
mim install mmpose==1.3.1

info "Installing mmdeploy..."
pip install mmdeploy==1.3.1
pip install mmdeploy-runtime-gpu==1.3.1

info "Installing TensorRT Python bindings from local SDK..."
# Install TensorRT from the local SDK wheel files (more reliable than pypi.nvidia.com)
TRT_PYTHON_DIR="${TENSORRT_DIR}/python"
TRT_WHEEL=$(find "${TRT_PYTHON_DIR}" -name "tensorrt-*-cp310-*.whl" 2>/dev/null | head -1)
TRT_LEAN_WHEEL=$(find "${TRT_PYTHON_DIR}" -name "tensorrt_lean-*-cp310-*.whl" 2>/dev/null | head -1)

if [[ -f "${TRT_WHEEL}" ]]; then
    info "Installing TensorRT from local wheel: $(basename ${TRT_WHEEL})"
    pip install "${TRT_WHEEL}"
else
    warn "Local TensorRT wheel not found, trying PyPI..."
    pip install tensorrt==8.6.1 --index-url https://pypi.nvidia.com
fi

if [[ -f "${TRT_LEAN_WHEEL}" ]]; then
    info "Installing TensorRT-lean from local wheel: $(basename ${TRT_LEAN_WHEEL})"
    pip install "${TRT_LEAN_WHEEL}"
fi

info "Installing CuPy for GPU acceleration..."
pip install cupy-cuda11x==13.6.0

info "Installing ONNX and ONNXRuntime..."
pip install onnx==1.17.0
pip install onnxruntime-gpu==1.17.1

info "Installing OpenCV with GUI support..."
# Install headless first (some packages depend on it), then full version (takes precedence)
pip install opencv-python-headless==4.9.0.80
pip install opencv-python==4.11.0.86  # Full version with highgui for cv2.imshow()
pip install scipy==1.13.0
pip install matplotlib==3.8.4
pip install pandas==1.4.4
pip install tqdm==4.65.2
pip install pyyaml
pip install cmake
pip install toml
pip install ipython
pip install keyboard==0.13.5
pip install beautifulsoup4 lxml
pip install pyserial
pip install pygltflib==1.16.5
pip install natsort==8.4.0
pip install openpyxl
pip install websocket-client
pip install func_timeout==4.3.5
pip install protobuf==3.20.2
pip install ultralytics==8.2.40
pip install Pose2Sim==0.4.0
pip install multiprocess==0.70.18

info "Installing OpenSim (biomechanics library)..."
conda install -c opensim-org opensim -y

info "Installing PyQt5 and PyQt6 (GUI dependencies)..."
# PyQt5 is needed for some components, PyQt6 for others
pip install PyQt5==5.15.11 PyQt5-Qt5==5.15.18 PyQt5-sip==12.17.2
pip install PyQt6==6.5.3 PyQt6-Qt6==6.5.3 PyQt6-sip==13.6.0
pip install PyQt6-WebEngine==6.5.0 PyQt6-WebEngine-Qt6==6.5.3
pip install pyqtgraph==0.13.7

info "Installing OpenNI2 Python bindings (for PoE cameras)..."
pip install openni

log "Python dependencies installed"

# Verify numpy version
NUMPY_VERSION=$(python -c "import numpy; print(numpy.__version__)")
log "numpy version: ${NUMPY_VERSION}"

if ! ask_continue "TensorRT setup"; then
    exit 0
fi

# ==============================================================================
# STEP 3: SETUP TENSORRT
# ==============================================================================

section "Step 3: Setting up TensorRT ${TENSORRT_VERSION}"

mkdir -p "${THIRDPARTY_DIR}"

if [[ ! -d "${TENSORRT_DIR}" ]]; then
    if [[ -f "${SCRIPT_DIR}/${TENSORRT_ARCHIVE}" ]]; then
        info "Extracting TensorRT..."
        tar -xzf "${SCRIPT_DIR}/${TENSORRT_ARCHIVE}" -C "${THIRDPARTY_DIR}"
        log "TensorRT extracted to ${TENSORRT_DIR}"
    else
        error "TensorRT archive not found: ${SCRIPT_DIR}/${TENSORRT_ARCHIVE}"
    fi
else
    log "TensorRT already exists at ${TENSORRT_DIR}"
fi

# Verify TensorRT
if [[ ! -f "${TENSORRT_DIR}/lib/libnvinfer.so" ]]; then
    error "TensorRT installation invalid. libnvinfer.so not found."
fi

# Set environment variables
export TENSORRT_DIR="${TENSORRT_DIR}"
export LD_LIBRARY_PATH="${TENSORRT_DIR}/lib:${LD_LIBRARY_PATH:-}"

log "TensorRT configured: ${TENSORRT_DIR}"

if ! ask_continue "ppl.cv build"; then
    exit 0
fi

# ==============================================================================
# STEP 4: BUILD PPL.CV
# ==============================================================================

section "Step 4: Building ppl.cv with CUDA Support"

if [[ ! -d "${PPLCV_DIR}" ]]; then
    info "Cloning ppl.cv..."
    git clone https://github.com/openppl-public/ppl.cv.git "${PPLCV_DIR}"
    log "ppl.cv cloned"
elif [[ ! -d "${PPLCV_DIR}/.git" ]]; then
    warn "ppl.cv exists but is not a git repository. Re-cloning..."
    rm -rf "${PPLCV_DIR}"
    git clone https://github.com/openppl-public/ppl.cv.git "${PPLCV_DIR}"
    log "ppl.cv re-cloned"
else
    log "ppl.cv already exists"
fi

cd "${PPLCV_DIR}"

# Create build directory
PPLCV_BUILD_DIR="${PPLCV_DIR}/cuda-build"

# Clean stale CMake caches if they exist (prevents errors when building from different locations)
if [[ -f "${PPLCV_BUILD_DIR}/CMakeCache.txt" ]]; then
    info "Cleaning stale CMake cache..."
    rm -rf "${PPLCV_BUILD_DIR}"
fi
# Also clean deps directory which contains FetchContent caches
if [[ -d "${PPLCV_DIR}/deps" ]]; then
    info "Cleaning stale dependency caches..."
    rm -rf "${PPLCV_DIR}/deps"
fi

mkdir -p "${PPLCV_BUILD_DIR}"
cd "${PPLCV_BUILD_DIR}"

info "Configuring ppl.cv with CMake..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
    -DPPLCV_USE_CUDA=ON \
    -DPPLCV_USE_X86_64=OFF \
    -DPPLCV_USE_AARCH64=OFF \
    -DPPLCV_BUILD_TESTS=OFF \
    -DPPLCV_BUILD_BENCHMARK=OFF \
    -DPPLCV_INSTALL=ON \
    -DCMAKE_INSTALL_PREFIX="${PPLCV_BUILD_DIR}/install"

info "Building ppl.cv (this may take a few minutes)..."
make -j$(nproc)

info "Installing ppl.cv..."
make install

# Verify
if [[ ! -f "${PPLCV_BUILD_DIR}/install/lib/libpplcv_static.a" ]]; then
    error "ppl.cv build failed. Library not found."
fi

log "ppl.cv built successfully"

cd "${PROJECT_ROOT}"

if ! ask_continue "mmdeploy SDK build"; then
    exit 0
fi

# ==============================================================================
# STEP 5: BUILD MMDEPLOY SDK
# ==============================================================================

section "Step 5: Building mmdeploy SDK with TensorRT"

# Clone mmdeploy if it doesn't exist or isn't a valid git repo
if [[ ! -d "${MMDEPLOY_DIR}" ]]; then
    info "Cloning mmdeploy..."
    git clone --depth 1 --branch v1.3.1 https://github.com/open-mmlab/mmdeploy.git "${MMDEPLOY_DIR}"
    log "mmdeploy cloned"
elif [[ ! -d "${MMDEPLOY_DIR}/.git" ]]; then
    warn "mmdeploy exists but is not a git repository. Re-cloning..."
    rm -rf "${MMDEPLOY_DIR}"
    git clone --depth 1 --branch v1.3.1 https://github.com/open-mmlab/mmdeploy.git "${MMDEPLOY_DIR}"
    log "mmdeploy re-cloned"
else
    log "mmdeploy already exists"
fi

cd "${MMDEPLOY_DIR}"

# Initialize submodules
info "Initializing mmdeploy submodules..."
git submodule update --init --recursive

# Create build directory
MMDEPLOY_BUILD_DIR="${MMDEPLOY_DIR}/build"

# Clean stale CMake cache if it exists (prevents errors when building from different locations)
if [[ -f "${MMDEPLOY_BUILD_DIR}/CMakeCache.txt" ]]; then
    info "Cleaning stale CMake cache..."
    rm -rf "${MMDEPLOY_BUILD_DIR}"
fi

mkdir -p "${MMDEPLOY_BUILD_DIR}"
cd "${MMDEPLOY_BUILD_DIR}"

# ============================================================================
# VALIDATE ALL REQUIRED PATHS BEFORE CMAKE
# ============================================================================

# 1. Find and validate CUDA_HOME
if [[ -z "${CUDA_HOME}" ]]; then
    for cuda_path in "/usr/local/cuda-11.8" "/usr/local/cuda"; do
        if [[ -f "${cuda_path}/bin/nvcc" ]]; then
            export CUDA_HOME="${cuda_path}"
            break
        fi
    done
fi

if [[ -z "${CUDA_HOME}" ]] || [[ ! -f "${CUDA_HOME}/bin/nvcc" ]]; then
    error "CUDA not found. Please install CUDA 11.8 and ensure nvcc is available."
fi
log "CUDA_HOME validated: ${CUDA_HOME}"

# 2. Validate TensorRT directory
if [[ ! -f "${TENSORRT_DIR}/include/NvInfer.h" ]]; then
    error "TensorRT headers not found at ${TENSORRT_DIR}/include/NvInfer.h"
fi
if [[ ! -f "${TENSORRT_DIR}/lib/libnvinfer.so" ]]; then
    error "TensorRT library not found at ${TENSORRT_DIR}/lib/libnvinfer.so"
fi
log "TENSORRT_DIR validated: ${TENSORRT_DIR}"

# 3. Find and validate pplcv cmake directory (built in Step 4)
PPLCV_CMAKE_DIR=""
PPLCV_SEARCH_PATHS=(
    "${PPLCV_BUILD_DIR}/install/lib/cmake/ppl"
    "${PPLCV_BUILD_DIR}/install/lib/cmake/pplcv"
    "/usr/local/lib/cmake/ppl"
    "/usr/local/lib/cmake/pplcv"
)
for pplcv_path in "${PPLCV_SEARCH_PATHS[@]}"; do
    if [[ -f "${pplcv_path}/pplcv-config.cmake" ]]; then
        PPLCV_CMAKE_DIR="${pplcv_path}"
        break
    fi
done

if [[ -z "${PPLCV_CMAKE_DIR}" ]]; then
    error "pplcv cmake config not found. Searched paths:
$(printf '  - %s\n' "${PPLCV_SEARCH_PATHS[@]}")

Please ensure Step 4 (ppl.cv build) completed successfully."
fi
log "pplcv_DIR validated: ${PPLCV_CMAKE_DIR}"

# 4. Find OpenCV cmake directory (installed by apt in Step 0.5)
OPENCV_DIR=""
OPENCV_SEARCH_PATHS=(
    "/usr/lib/x86_64-linux-gnu/cmake/opencv4"
    "/usr/local/lib/cmake/opencv4"
    "/usr/lib/cmake/opencv4"
)
for opencv_path in "${OPENCV_SEARCH_PATHS[@]}"; do
    if [[ -f "${opencv_path}/OpenCVConfig.cmake" ]]; then
        OPENCV_DIR="${opencv_path}"
        break
    fi
done

if [[ -z "${OPENCV_DIR}" ]]; then
    warn "OpenCV cmake directory not found. CMake will try to auto-detect."
    warn "If build fails, run: sudo apt-get install libopencv-dev"
else
    log "OpenCV_DIR found: ${OPENCV_DIR}"
fi

# 5. Find cuDNN (typically in system paths on Ubuntu with apt install)
CUDNN_DIR=""
CUDNN_SEARCH_PATHS=(
    "/usr/lib/x86_64-linux-gnu"
    "/usr/local/cuda/lib64"
    "${CUDA_HOME}/lib64"
)
for cudnn_path in "${CUDNN_SEARCH_PATHS[@]}"; do
    if [[ -f "${cudnn_path}/libcudnn.so" ]] || [[ -f "${cudnn_path}/libcudnn.so.8" ]]; then
        CUDNN_DIR="${cudnn_path}"
        break
    fi
done

if [[ -z "${CUDNN_DIR}" ]]; then
    warn "cuDNN library not found. CMake will try to auto-detect."
else
    log "cuDNN found: ${CUDNN_DIR}"
fi

# ============================================================================
# CMAKE CONFIGURATION
# ============================================================================

info "CMake Configuration Summary:"
info "  CUDA_HOME    = ${CUDA_HOME}"
info "  TENSORRT_DIR = ${TENSORRT_DIR}"
info "  pplcv_DIR    = ${PPLCV_CMAKE_DIR}"
info "  OpenCV_DIR   = ${OPENCV_DIR:-auto-detect}"
info "  CUDNN_DIR    = ${CUDNN_DIR:-auto-detect}"
info "  CUDA_ARCH    = ${CUDA_ARCH}"

info "Configuring mmdeploy with CMake..."
CMAKE_ARGS=(
    ".."
    "-DCMAKE_BUILD_TYPE=Release"
    "-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH}"
    "-DMMDEPLOY_BUILD_SDK=ON"
    "-DMMDEPLOY_BUILD_SDK_PYTHON_API=ON"
    "-DMMDEPLOY_BUILD_SDK_MONOLITHIC=ON"
    "-DMMDEPLOY_TARGET_BACKENDS=trt"
    "-DMMDEPLOY_TARGET_DEVICES=cuda"
    "-DMMDEPLOY_CODEBASES=all"
    "-DTENSORRT_DIR=${TENSORRT_DIR}"
    "-Dpplcv_DIR=${PPLCV_CMAKE_DIR}"
    "-DCMAKE_INSTALL_PREFIX=${MMDEPLOY_BUILD_DIR}/install"
)

# Add OpenCV_DIR only if found (otherwise let cmake auto-detect)
if [[ -n "${OPENCV_DIR}" ]]; then
    CMAKE_ARGS+=("-DOpenCV_DIR=${OPENCV_DIR}")
fi

# Add CUDNN_DIR only if found in non-standard location
if [[ -n "${CUDNN_DIR}" ]] && [[ "${CUDNN_DIR}" != "/usr/lib/x86_64-linux-gnu" ]]; then
    CMAKE_ARGS+=("-DCUDNN_DIR=${CUDNN_DIR}")
fi

cmake "${CMAKE_ARGS[@]}"

info "Building mmdeploy SDK (this may take 10-20 minutes)..."
make -j$(nproc)

info "Installing mmdeploy SDK..."
make install

# Verify the TensorRT ops library was built
TENSORRT_OPS_LIB="${MMDEPLOY_BUILD_DIR}/lib/libmmdeploy_tensorrt_ops.so"
if [[ ! -f "${TENSORRT_OPS_LIB}" ]]; then
    error "mmdeploy build failed. TensorRT ops library not found."
fi

log "mmdeploy SDK built successfully"
log "TensorRT ops library: ${TENSORRT_OPS_LIB}"

cd "${PROJECT_ROOT}"

if ! ask_continue "TensorRT engine generation"; then
    exit 0
fi

# ==============================================================================
# STEP 6: DOWNLOAD MODEL WEIGHTS
# ==============================================================================

section "Step 6: Downloading Model Weights"

WEIGHTS_DIR="${MMDEPLOY_DIR}/build_engines"
mkdir -p "${WEIGHTS_DIR}"

RTMDET_WEIGHT="${WEIGHTS_DIR}/rtmdet_m.pth"
RTMPOSE_WEIGHT="${WEIGHTS_DIR}/rtmpose_m_halpe.pth"

if [[ ! -f "${RTMDET_WEIGHT}" ]]; then
    info "Downloading RTMDet-m weights (~214MB)..."
    wget -q --show-progress -O "${RTMDET_WEIGHT}" "${RTMDET_WEIGHT_URL}"
    log "RTMDet weights downloaded"
else
    log "RTMDet weights already exist"
fi

if [[ ! -f "${RTMPOSE_WEIGHT}" ]]; then
    info "Downloading RTMPose-m Halpe26 weights (~53MB)..."
    wget -q --show-progress -O "${RTMPOSE_WEIGHT}" "${RTMPOSE_HALPE_WEIGHT_URL}"
    log "RTMPose weights downloaded"
else
    log "RTMPose weights already exist"
fi

# ==============================================================================
# STEP 7: GENERATE TENSORRT ENGINES
# ==============================================================================

section "Step 7: Generating TensorRT Engines"

# Set environment for engine generation
export LD_LIBRARY_PATH="${TENSORRT_DIR}/lib:${LD_LIBRARY_PATH:-}"

# Engine output directories
RTMDET_ENGINE_DIR="${MMDEPLOY_DIR}/rtmpose-trt/rtmdet-m"
RTMPOSE_ENGINE_DIR="${MMDEPLOY_DIR}/rtmpose-trt/rtmpose-m"
mkdir -p "${RTMDET_ENGINE_DIR}"
mkdir -p "${RTMPOSE_ENGINE_DIR}"

# Config paths
RTMDET_DEPLOY_CFG="${MMDEPLOY_DIR}/configs/mmdet/detection/detection_tensorrt_static-320x320.py"
RTMPOSE_DEPLOY_CFG="${MMDEPLOY_DIR}/configs/mmpose/pose-detection_simcc_tensorrt_dynamic-256x192.py"

RTMDET_MODEL_CFG="${CONDA_PREFIX}/lib/python3.10/site-packages/mmdet/.mim/configs/rtmdet/rtmdet_m_8xb32-300e_coco.py"
RTMPOSE_MODEL_CFG="${CONDA_PREFIX}/lib/python3.10/site-packages/mmpose/.mim/configs/body_2d_keypoint/rtmpose/body8/rtmpose-m_8xb512-700e_body8-halpe26-256x192.py"

# Demo images
DET_IMAGE="${MMDEPLOY_DIR}/demo/resources/det.jpg"
POSE_IMAGE="${MMDEPLOY_DIR}/demo/resources/human-pose.jpg"

# Deploy script
DEPLOY_SCRIPT="${MMDEPLOY_DIR}/tools/deploy.py"

# Check if engines already exist
if [[ -f "${RTMDET_ENGINE_DIR}/end2end.engine" ]] && [[ -f "${RTMPOSE_ENGINE_DIR}/end2end.engine" ]]; then
    log "TensorRT engines already exist"
    read -p "$(echo -e ${YELLOW}Regenerate engines? [y/N]: ${NC})" response
    case "$response" in
        [yY][eE][sS]|[yY])
            info "Regenerating engines..."
            ;;
        *)
            log "Using existing engines"
            SKIP_ENGINE_GEN=1
            ;;
    esac
fi

if [[ -z "${SKIP_ENGINE_GEN}" ]]; then
    # Build RTMDet engine
    info "Building RTMDet TensorRT engine (320x320)..."
    python "${DEPLOY_SCRIPT}" \
        "${RTMDET_DEPLOY_CFG}" \
        "${RTMDET_MODEL_CFG}" \
        "${RTMDET_WEIGHT}" \
        "${DET_IMAGE}" \
        --work-dir "${RTMDET_ENGINE_DIR}" \
        --device cuda:0 \
        --dump-info

    if [[ ! -f "${RTMDET_ENGINE_DIR}/end2end.engine" ]]; then
        error "RTMDet engine generation failed"
    fi
    log "RTMDet engine generated"

    # Build RTMPose engine
    info "Building RTMPose TensorRT engine (256x192)..."
    python "${DEPLOY_SCRIPT}" \
        "${RTMPOSE_DEPLOY_CFG}" \
        "${RTMPOSE_MODEL_CFG}" \
        "${RTMPOSE_WEIGHT}" \
        "${POSE_IMAGE}" \
        --work-dir "${RTMPOSE_ENGINE_DIR}" \
        --device cuda:0 \
        --dump-info

    if [[ ! -f "${RTMPOSE_ENGINE_DIR}/end2end.engine" ]]; then
        error "RTMPose engine generation failed"
    fi
    log "RTMPose engine generated"
fi

# ==============================================================================
# STEP 8: FIX PIPELINE.JSON FILES
# ==============================================================================

section "Step 8: Fixing Pipeline Configuration Files"

# Fix RTMDet pipeline.json
RTMDET_PIPELINE="${RTMDET_ENGINE_DIR}/pipeline.json"
if [[ -f "${RTMDET_PIPELINE}" ]]; then
    info "Fixing RTMDet pipeline.json..."

    # Create fixed version
    cat > "${RTMDET_PIPELINE}" << 'RTMDET_PIPELINE_EOF'
{
    "pipeline": {
        "input": [
            "img"
        ],
        "output": [
            "dets"
        ],
        "tasks": [
            {
                "type": "Task",
                "module": "Transform",
                "name": "Preprocess",
                "input": [
                    "img"
                ],
                "output": [
                    "prep_output"
                ],
                "transforms": [
                    {
                        "type": "LoadImageFromFile"
                    },
                    {
                        "type": "Resize",
                        "size": [
                            320,
                            320
                        ],
                        "keep_ratio": true
                    },
                    {
                        "type": "Pad",
                        "size": [
                            320,
                            320
                        ],
                        "pad_val": {
                            "img": [
                                114,
                                114,
                                114
                            ]
                        }
                    },
                    {
                        "type": "Normalize",
                        "mean": [
                            103.53,
                            116.28,
                            123.675
                        ],
                        "std": [
                            57.375,
                            57.12,
                            58.395
                        ],
                        "to_rgb": false
                    },
                    {
                        "type": "DefaultFormatBundle"
                    },
                    {
                        "type": "Collect",
                        "keys": [
                            "img"
                        ],
                        "meta_keys": [
                            "ori_shape",
                            "img_shape",
                            "scale_factor",
                            "pad_param"
                        ]
                    }
                ]
            },
            {
                "type": "Task",
                "module": "Net",
                "name": "detection",
                "is_batched": false,
                "input": [
                    "prep_output"
                ],
                "output": [
                    "infer_output"
                ],
                "input_map": {
                    "img": "input"
                },
                "output_map": {}
            },
            {
                "type": "Task",
                "module": "mmdet",
                "name": "postprocess",
                "component": "ResizeBBox",
                "params": {
                    "score_thr": 0.3
                },
                "input": [
                    "prep_output",
                    "infer_output"
                ],
                "output": [
                    "dets"
                ]
            }
        ]
    }
}
RTMDET_PIPELINE_EOF
    log "RTMDet pipeline.json fixed"
fi

# Fix RTMPose pipeline.json
RTMPOSE_PIPELINE="${RTMPOSE_ENGINE_DIR}/pipeline.json"
if [[ -f "${RTMPOSE_PIPELINE}" ]]; then
    info "Fixing RTMPose pipeline.json..."

    cat > "${RTMPOSE_PIPELINE}" << 'RTMPOSE_PIPELINE_EOF'
{
    "pipeline": {
        "input": [
            "img"
        ],
        "output": [
            "post_output"
        ],
        "tasks": [
            {
                "type": "Task",
                "module": "Transform",
                "name": "Preprocess",
                "input": [
                    "img"
                ],
                "output": [
                    "prep_output"
                ],
                "transforms": [
                    {
                        "type": "LoadImageFromFile"
                    },
                    {
                        "type": "TopDownGetBboxCenterScale",
                        "padding": 1.25,
                        "image_size": [
                            192,
                            256
                        ]
                    },
                    {
                        "type": "TopDownAffine",
                        "image_size": [
                            192,
                            256
                        ]
                    },
                    {
                        "type": "Normalize",
                        "mean": [
                            123.675,
                            116.28,
                            103.53
                        ],
                        "std": [
                            58.395,
                            57.12,
                            57.375
                        ],
                        "to_rgb": true
                    },
                    {
                        "type": "ImageToTensor",
                        "keys": [
                            "img"
                        ]
                    },
                    {
                        "type": "Collect",
                        "keys": [
                            "img"
                        ],
                        "meta_keys": [
                            "img_shape",
                            "pad_shape",
                            "ori_shape",
                            "img_norm_cfg",
                            "scale_factor",
                            "bbox_score",
                            "center",
                            "scale"
                        ]
                    }
                ]
            },
            {
                "name": "pose",
                "type": "Task",
                "module": "Net",
                "is_batched": false,
                "input": [
                    "prep_output"
                ],
                "output": [
                    "infer_output"
                ],
                "input_map": {
                    "img": "input"
                },
                "output_map": {
                    "output": "simcc_x",
                    "700": "simcc_y"
                }
            },
            {
                "type": "Task",
                "module": "mmpose",
                "name": "postprocess",
                "component": "SimCCLabelDecode",
                "params": {
                    "flip_test": false,
                    "type": "SimCCLabel",
                    "input_size": [
                        192,
                        256
                    ],
                    "sigma": [
                        6.0,
                        6.93
                    ],
                    "simcc_split_ratio": 2.0,
                    "normalize": false,
                    "use_dark": false,
                    "export_postprocess": false
                },
                "output": [
                    "post_output"
                ],
                "input": [
                    "prep_output",
                    "infer_output"
                ]
            }
        ]
    }
}
RTMPOSE_PIPELINE_EOF
    log "RTMPose pipeline.json fixed"
fi

# ==============================================================================
# STEP 9: CREATE ACTIVATION SCRIPT
# ==============================================================================

section "Step 9: Creating Activation Script"

ACTIVATE_SCRIPT="${PROJECT_ROOT}/activate_ntkcap.sh"

cat > "${ACTIVATE_SCRIPT}" << ACTIVATE_EOF
#!/bin/bash
# NTKCAP Environment Activation Script
# Source this file to set up the environment: source activate_ntkcap.sh

# Get the directory where this script is located
SCRIPT_DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"

# Set TensorRT path
export TENSORRT_DIR="\${SCRIPT_DIR}/NTK_CAP/ThirdParty/TensorRT-${TENSORRT_VERSION}"
export LD_LIBRARY_PATH="\${TENSORRT_DIR}/lib:\${LD_LIBRARY_PATH:-}"

# Set CUDA path (if not already set)
if [[ -z "\${CUDA_HOME}" ]]; then
    for cuda_path in "/usr/local/cuda-11.8" "/usr/local/cuda"; do
        if [[ -d "\${cuda_path}" ]]; then
            export CUDA_HOME="\${cuda_path}"
            export PATH="\${CUDA_HOME}/bin:\${PATH}"
            export LD_LIBRARY_PATH="\${CUDA_HOME}/lib64:\${LD_LIBRARY_PATH:-}"
            break
        fi
    done
fi

# Activate conda environment
CONDA_BASE=\$(conda info --base 2>/dev/null)
if [[ -n "\${CONDA_BASE}" ]]; then
    source "\${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate ${CONDA_ENV_NAME}
fi

# Set project root
export NTKCAP_ROOT="\${SCRIPT_DIR}"
cd "\${NTKCAP_ROOT}"

echo "NTKCAP environment activated"
echo "  Python: \$(which python)"
echo "  TensorRT: \${TENSORRT_DIR}"
echo "  Project: \${NTKCAP_ROOT}"
ACTIVATE_EOF

chmod +x "${ACTIVATE_SCRIPT}"
log "Activation script created: ${ACTIVATE_SCRIPT}"

# ==============================================================================
# STEP 10: VERIFICATION
# ==============================================================================

section "Step 10: Verifying Installation"

info "Running verification tests..."

# Test Python imports
python << 'VERIFY_EOF'
import sys
print("Python:", sys.version)

# Test numpy
import numpy as np
print(f"numpy: {np.__version__}")
assert np.__version__ == "1.22.4", f"numpy version mismatch: {np.__version__}"

# Test torch
import torch
print(f"torch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Test cupy
import cupy as cp
print(f"cupy: {cp.__version__}")

# Test tensorrt
import tensorrt as trt
print(f"tensorrt: {trt.__version__}")

# Test mmdeploy
import mmdeploy_runtime
print("mmdeploy_runtime: OK")

# Test mmpose
import mmpose
print(f"mmpose: {mmpose.__version__}")

# Test mmdet
import mmdet
print(f"mmdet: {mmdet.__version__}")

# Test OpenCV (must have GUI support for cv2.imshow)
import cv2
print(f"opencv: {cv2.__version__}")
# Verify highgui is available (not headless)
if not hasattr(cv2, 'imshow'):
    print("WARNING: cv2.imshow not available - OpenCV may be headless!")

# Test PyQt
try:
    from PyQt6 import QtWidgets
    print("PyQt6: OK")
except ImportError:
    print("WARNING: PyQt6 not available")

try:
    from PyQt5 import QtWidgets as Qt5Widgets
    print("PyQt5: OK")
except ImportError:
    print("WARNING: PyQt5 not available")

print("\n✓ All Python packages verified!")
VERIFY_EOF

if [[ $? -ne 0 ]]; then
    error "Python package verification failed"
fi

log "Python packages verified"

# Test TensorRT ops library
info "Testing TensorRT ops library..."
python << VERIFY_TRT_EOF
import ctypes
import os

# Find the library
script_dir = "${PROJECT_ROOT}"
lib_path = os.path.join(script_dir, "NTK_CAP/ThirdParty/mmdeploy/build/lib/libmmdeploy_tensorrt_ops.so")

if os.path.exists(lib_path):
    try:
        ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
        print(f"✓ TensorRT ops library loaded: {lib_path}")
    except Exception as e:
        print(f"✗ Failed to load TensorRT ops library: {e}")
        exit(1)
else:
    print(f"✗ TensorRT ops library not found: {lib_path}")
    exit(1)
VERIFY_TRT_EOF

if [[ $? -ne 0 ]]; then
    error "TensorRT ops library verification failed"
fi

log "TensorRT ops library verified"

# Test TensorRT engines
info "Testing TensorRT engines..."
if [[ -f "${RTMDET_ENGINE_DIR}/end2end.engine" ]]; then
    log "RTMDet engine exists: ${RTMDET_ENGINE_DIR}/end2end.engine"
else
    warn "RTMDet engine not found"
fi

if [[ -f "${RTMPOSE_ENGINE_DIR}/end2end.engine" ]]; then
    log "RTMPose engine exists: ${RTMPOSE_ENGINE_DIR}/end2end.engine"
else
    warn "RTMPose engine not found"
fi

# ==============================================================================
# STEP 11: LIPSEDGE POE CAMERA SETUP (OPTIONAL)
# ==============================================================================

section "Step 11: LIPSedge PoE Camera Setup (Optional)"

echo ""
echo "LIPSedge AE400/AE450 PoE cameras require additional setup."
echo ""
echo "The OpenNI2 Python bindings have been installed (pip install openni)."
echo "However, you need to install the LIPSedge SDK for full camera support."
echo ""

LIPSEDGE_SDK_DIR="${THIRDPARTY_DIR}/LIPSedge-SDK"
OPENNI2_DIR="${THIRDPARTY_DIR}/OpenNI2"

if ask_continue "LIPSedge PoE camera setup"; then
    # Install system dependencies for OpenNI2
    info "Installing system dependencies for OpenNI2..."
    sudo apt-get update
    sudo apt-get install -y libusb-1.0-0-dev libudev-dev libgtk2.0-dev freeglut3-dev

    # Create OpenNI2 directory structure
    mkdir -p "${OPENNI2_DIR}"

    echo ""
    echo "┌─────────────────────────────────────────────────────────────────────────┐"
    echo "│                    LIPSedge SDK Manual Setup Required                   │"
    echo "├─────────────────────────────────────────────────────────────────────────┤"
    echo "│                                                                         │"
    echo "│  To use LIPSedge AE400/AE450 PoE cameras, follow these steps:          │"
    echo "│                                                                         │"
    echo "│  1. Download LIPSedge SDK from:                                        │"
    echo "│     https://dev.lips-hci.com/lipsedge-ae400-ae450-sdk-release          │"
    echo "│                                                                         │"
    echo "│  2. Extract and install the SDK:                                       │"
    echo "│     tar -xzf LIPS-Linux-x64-OpenNI2.2.tar.gz                          │"
    echo "│     cd LIPS-Linux-x64-OpenNI2.2                                        │"
    echo "│     sudo ./install.sh                                                  │"
    echo "│                                                                         │"
    echo "│  3. For each camera, create a directory with its IP address:           │"
    echo "│     mkdir -p ${OPENNI2_DIR}/192.168.0.100                              │"
    echo "│                                                                         │"
    echo "│  4. Copy OpenNI2 libraries and create network.json for each camera:    │"
    echo "│     - OpenNI2.dll (or .so on Linux)                                    │"
    echo "│     - OpenNI.ini                                                       │"
    echo "│     - OpenNI2/Drivers/network.json (with camera IP configured)         │"
    echo "│                                                                         │"
    echo "│  5. Configure camera IPs in config/config.json:                        │"
    echo "│     \"poe\": {                                                           │"
    echo "│       \"ips\": [\"192.168.0.100\", \"192.168.3.100\", ...],               │"
    echo "│       \"openni2_base\": \"NTK_CAP/ThirdParty/OpenNI2\",                   │"
    echo "│       \"model\": \"ae450\"                                                │"
    echo "│     }                                                                  │"
    echo "│                                                                         │"
    echo "│  Documentation: https://dev.lips-hci.com                               │"
    echo "│  GitHub: https://github.com/lips-hci/LIPSedge-sdk-samples             │"
    echo "│                                                                         │"
    echo "└─────────────────────────────────────────────────────────────────────────┘"
    echo ""

    log "OpenNI2 directory created: ${OPENNI2_DIR}"
    log "Please complete manual SDK installation as described above"
else
    info "Skipping LIPSedge setup. You can set it up later if needed."
fi

# ==============================================================================
# DONE
# ==============================================================================

echo ""
echo "╔════════════════════════════════════════════════════════════════════════╗"
echo "║                                                                        ║"
echo "║              NTKCAP Installation Complete!                             ║"
echo "║                                                                        ║"
echo "╚════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "To activate the environment, run:"
echo ""
echo "    source ${ACTIVATE_SCRIPT}"
echo ""
echo "Or manually:"
echo ""
echo "    conda activate ${CONDA_ENV_NAME}"
echo "    export LD_LIBRARY_PATH=${TENSORRT_DIR}/lib:\$LD_LIBRARY_PATH"
echo ""
echo "Installation Summary:"
echo "  - Conda environment: ${CONDA_ENV_NAME}"
echo "  - TensorRT: ${TENSORRT_DIR}"
echo "  - ppl.cv: ${PPLCV_DIR}"
echo "  - mmdeploy: ${MMDEPLOY_DIR}"
echo "  - RTMDet engine: ${RTMDET_ENGINE_DIR}"
echo "  - RTMPose engine: ${RTMPOSE_ENGINE_DIR}"
echo "  - OpenNI2 (PoE cameras): ${OPENNI2_DIR}"
echo ""
echo "Camera Support:"
echo "  - USB Webcams: Ready to use"
echo "  - PoE Cameras (AE400/AE450): Requires LIPSedge SDK (see Step 11)"
echo ""
log "Installation completed successfully!"
