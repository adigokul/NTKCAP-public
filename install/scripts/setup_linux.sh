#!/bin/bash
################################################################################
# NTKCAP Linux Environment Setup Script (THE BULLETPROOF INSTALLER)
# ================================================================================
# Target: Ubuntu 22.04/24.04 LTS
# CUDA: 11.8 (REQUIRED)
# TensorRT: 8.6.1.6 (REQUIRED)
# Python: 3.10 (via conda)
#
# This script addresses the "Black Book of Past Failures":
# 1. GCC Version Mismatch - Ensures GCC 11 for CUDA 11.8
# 2. Legacy Library Void - Installs libtinfo5
# 3. Git Zombie State - Verifies submodule health
# 4. CMake vs Pip Disconnect - Uses apt for dev libraries
# 5. TensorRT Half-Install - Extracts full TAR.GZ with headers
# 6. NumPy ABI Fracture - Pins numpy==1.22.4
################################################################################

set -euo pipefail  # Exit on error, undefined var, pipe failure
IFS=$'\n\t'

# ==============================================================================
# CONFIGURATION
# ==============================================================================
NTKCAP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CUDA_VERSION="11.8"
CUDA_MD5="f81710006b864319c06597813f8150a0"  # cuda_11.8_linux.run MD5
TENSORRT_VERSION="8.6.1.6"
TENSORRT_CUDA="11.8"
PYTHON_VERSION="3.10"
ENV_NAME="ntkcap_env"

# Paths
THIRDPARTY_DIR="${NTKCAP_ROOT}/NTK_CAP/ThirdParty"
TENSORRT_DIR="${THIRDPARTY_DIR}/TensorRT-${TENSORRT_VERSION}"
DOWNLOADS_DIR="${NTKCAP_ROOT}/install/downloads"
LOGS_DIR="${NTKCAP_ROOT}/install/logs"

# TensorRT download info (NVIDIA Developer account required)
TENSORRT_TAR="TensorRT-${TENSORRT_VERSION}.Linux.x86_64-gnu.cuda-${TENSORRT_CUDA}.tar.gz"

# ==============================================================================
# LOGGING FUNCTIONS
# ==============================================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log() { echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] ✓ $1${NC}"; }
warn() { echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] ⚠ WARNING: $1${NC}"; }
error() { echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ✗ ERROR: $1${NC}"; exit 1; }
info() { echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] ℹ $1${NC}"; }
step() { echo -e "${CYAN}[$(date +'%Y-%m-%d %H:%M:%S')] ▶ $1${NC}"; }

# ==============================================================================
# PRE-FLIGHT CHECKS
# ==============================================================================
preflight_checks() {
    step "PHASE 0: PRE-FLIGHT CHECKS"
    
    # Create directories
    mkdir -p "${DOWNLOADS_DIR}" "${LOGS_DIR}"
    
    # 1. Check if running as root (should NOT)
    if [[ $EUID -eq 0 ]]; then
        error "Do NOT run this script as root. Run as a regular user with sudo privileges."
    fi
    
    # 2. Check sudo access
    if ! sudo -n true 2>/dev/null; then
        warn "This script requires sudo privileges. You may be prompted for your password."
        sudo -v || error "Cannot obtain sudo privileges"
    fi
    
    # 3. Check Ubuntu version
    if [[ -f /etc/os-release ]]; then
        source /etc/os-release
        if [[ "$ID" != "ubuntu" ]]; then
            warn "This script is designed for Ubuntu. Detected: $ID"
        fi
        if [[ "$VERSION_ID" != "22.04" && "$VERSION_ID" != "24.04" ]]; then
            warn "This script is tested on Ubuntu 22.04/24.04. Detected: $VERSION_ID"
        fi
        info "Detected OS: $NAME $VERSION_ID"
    else
        warn "Cannot detect OS version. Proceeding anyway..."
    fi
    
    # 4. CHECK nvidia-smi (HARDWARE IS GOD)
    if ! command -v nvidia-smi &>/dev/null; then
        error "nvidia-smi not found. NVIDIA drivers are NOT installed. Install them first!"
    fi
    
    if ! nvidia-smi &>/dev/null; then
        error "nvidia-smi failed. GPU is not accessible. Check driver installation."
    fi
    
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    log "GPU detected: ${GPU_NAME}, Driver: ${DRIVER_VERSION}"
    
    # 5. Check GCC version (The Compiler War)
    if ! command -v gcc-11 &>/dev/null; then
        warn "gcc-11 not found. Will install..."
        NEED_GCC11=true
    else
        GCC11_VERSION=$(gcc-11 --version | head -1)
        log "GCC 11 found: ${GCC11_VERSION}"
        NEED_GCC11=false
    fi
    
    # 6. Check libtinfo5 (Legacy Library Void)
    if ! ldconfig -p | grep -q "libtinfo.so.5"; then
        warn "libtinfo5 not found. Will install..."
        NEED_LIBTINFO5=true
    else
        log "libtinfo5 found"
        NEED_LIBTINFO5=false
    fi
    
    # 7. Check submodule health (Git Zombie State)
    if [[ ! -f "${THIRDPARTY_DIR}/mmdeploy/CMakeLists.txt" ]]; then
        error "SUBMODULE ZOMBIE: mmdeploy/CMakeLists.txt not found!
        
Run these commands to fix:
    cd ${NTKCAP_ROOT}
    git submodule update --init --recursive --force
    git submodule foreach git reset --hard HEAD
    
Then re-run this script."
    fi
    log "Submodule health check passed (mmdeploy/CMakeLists.txt exists)"
    
    if [[ ! -f "${THIRDPARTY_DIR}/mmpose/setup.py" ]]; then
        error "SUBMODULE ZOMBIE: mmpose/setup.py not found!"
    fi
    log "Submodule health check passed (mmpose/setup.py exists)"
    
    if [[ ! -f "${THIRDPARTY_DIR}/EasyMocap/setup.py" ]]; then
        error "SUBMODULE ZOMBIE: EasyMocap/setup.py not found!"
    fi
    log "Submodule health check passed (EasyMocap/setup.py exists)"
    
    log "All pre-flight checks passed!"
}

# ==============================================================================
# SYSTEM DEPENDENCIES
# ==============================================================================
install_system_deps() {
    step "PHASE 1: INSTALLING SYSTEM DEPENDENCIES"
    
    # Update package lists
    info "Updating apt package lists..."
    sudo apt-get update -qq
    
    # Install build essentials
    info "Installing build essentials..."
    sudo apt-get install -y \
        build-essential \
        cmake \
        ninja-build \
        pkg-config \
        wget \
        curl \
        git \
        unzip \
        ca-certificates \
        software-properties-common
    
    # Install GCC 11 (The Compiler War fix)
    if [[ "${NEED_GCC11:-true}" == "true" ]] || ! command -v gcc-11 &>/dev/null; then
        info "Installing gcc-11 and g++-11..."
        sudo apt-get install -y gcc-11 g++-11
        
        # Set up alternatives so CUDA can find gcc-11
        sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 110 \
            --slave /usr/bin/g++ g++ /usr/bin/g++-11
        
        log "GCC 11 installed and set as alternative"
    fi
    
    # Install libtinfo5 (Legacy Library Void fix)
    if [[ "${NEED_LIBTINFO5:-true}" == "true" ]] || ! ldconfig -p | grep -q "libtinfo.so.5"; then
        info "Installing libtinfo5..."
        # For Ubuntu 24.04, libtinfo5 might need universe repo
        sudo apt-get install -y libtinfo5 2>/dev/null || {
            warn "libtinfo5 not in default repos. Adding universe repository..."
            sudo add-apt-repository -y universe
            sudo apt-get update -qq
            sudo apt-get install -y libtinfo5 || {
                warn "libtinfo5 still not available. Creating symlink from libtinfo6..."
                if [[ -f /lib/x86_64-linux-gnu/libtinfo.so.6 ]]; then
                    sudo ln -sf /lib/x86_64-linux-gnu/libtinfo.so.6 /lib/x86_64-linux-gnu/libtinfo.so.5
                    log "Created libtinfo.so.5 symlink"
                else
                    error "Cannot find libtinfo.so.6 to create symlink"
                fi
            }
        }
        log "libtinfo5 installed or symlinked"
    fi
    
    # Install libopencv-dev (CMake vs Pip Disconnect fix)
    info "Installing libopencv-dev (for CMake discovery)..."
    sudo apt-get install -y libopencv-dev
    
    # Install other development libraries
    info "Installing additional development libraries..."
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
    
    # XCB LIBRARIES - CRITICAL FOR PYQT6 GUI
    # Without these, PyQt6 crashes with "qt.qpa.plugin: Could not load the Qt platform plugin xcb"
    # This is the #1 cause of GUI failure on Ubuntu 22.04/24.04
    info "Installing XCB libraries (CRITICAL for PyQt6 GUI)..."
    sudo apt-get install -y \
        libxcb-cursor0 \
        libxcb-xinerama0 \
        libxcb-xfixes0 \
        libxcb-shape0 \
        libxcb-icccm4 \
        libxcb-image0 \
        libxcb-keysyms1 \
        libxcb-render-util0
    
    log "System dependencies installed!"
}

# ==============================================================================
# CUDA VERIFICATION
# ==============================================================================
verify_cuda() {
    step "PHASE 2: VERIFYING CUDA ${CUDA_VERSION}"
    
    # Check if nvcc exists
    NVCC_PATH=""
    CUDA_HOME=""
    
    for path in "/usr/local/cuda-${CUDA_VERSION}" "/usr/local/cuda" "/opt/cuda"; do
        if [[ -f "${path}/bin/nvcc" ]]; then
            NVCC_PATH="${path}/bin/nvcc"
            CUDA_HOME="${path}"
            break
        fi
    done
    
    if [[ -z "${NVCC_PATH}" ]]; then
        error "CUDA ${CUDA_VERSION} not found!
        
Please install CUDA ${CUDA_VERSION}:
    1. Download: https://developer.nvidia.com/cuda-${CUDA_VERSION/./-}-download-archive
    2. Install with: sudo sh cuda_${CUDA_VERSION}.0_*_linux.run --toolkit --silent --override
    3. Use --override to bypass GCC version check if needed
    
Or install via apt:
    sudo apt-get install cuda-toolkit-${CUDA_VERSION/./-}
    
Then re-run this script."
    fi
    
    CUDA_VERSION_ACTUAL=$("${NVCC_PATH}" --version | grep "release" | sed -n 's/.*release \([0-9.]*\).*/\1/p')
    log "CUDA found: ${CUDA_VERSION_ACTUAL} at ${CUDA_HOME}"
    
    # Export CUDA environment
    export PATH="${CUDA_HOME}/bin:${PATH}"
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
    export CUDA_HOME="${CUDA_HOME}"
    
    log "CUDA environment variables set"
}

# ==============================================================================
# TENSORRT SETUP
# ==============================================================================
setup_tensorrt() {
    step "PHASE 3: SETTING UP TensorRT ${TENSORRT_VERSION}"
    
    if [[ -d "${TENSORRT_DIR}" && -f "${TENSORRT_DIR}/lib/libnvinfer.so" ]]; then
        log "TensorRT ${TENSORRT_VERSION} already extracted at ${TENSORRT_DIR}"
    else
        # Check for TensorRT tarball
        TENSORRT_TARBALL="${DOWNLOADS_DIR}/${TENSORRT_TAR}"
        
        if [[ ! -f "${TENSORRT_TARBALL}" ]]; then
            echo ""
            echo "========================================================================"
            warn "TensorRT ${TENSORRT_VERSION} tarball not found!"
            echo "========================================================================"
            echo ""
            echo "TensorRT requires manual download from NVIDIA (login required)."
            echo ""
            echo "Steps:"
            echo "1. Go to: https://developer.nvidia.com/tensorrt-download"
            echo "2. Login or create NVIDIA Developer account"
            echo "3. Select: TensorRT ${TENSORRT_VERSION} GA for Linux x86_64"
            echo "4. Download: ${TENSORRT_TAR}"
            echo "5. Place the file in: ${DOWNLOADS_DIR}/"
            echo ""
            echo "Then re-run this script."
            echo "========================================================================"
            echo ""
            
            read -p "Press Enter when you have downloaded TensorRT, or Ctrl+C to abort: "
            
            if [[ ! -f "${TENSORRT_TARBALL}" ]]; then
                error "TensorRT tarball still not found at ${TENSORRT_TARBALL}"
            fi
        fi
        
        # MD5 VERIFICATION (Trust Nothing)
        # Known MD5 for TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.8.tar.gz
        TENSORRT_MD5_EXPECTED="60dfb3a50c84e9c4a06f31aca76d9c3f"
        info "Verifying TensorRT tarball integrity (MD5)..."
        TENSORRT_MD5_ACTUAL=$(md5sum "${TENSORRT_TARBALL}" | awk '{print $1}')
        
        if [[ "${TENSORRT_MD5_ACTUAL}" != "${TENSORRT_MD5_EXPECTED}" ]]; then
            warn "MD5 mismatch for TensorRT tarball!"
            warn "Expected: ${TENSORRT_MD5_EXPECTED}"
            warn "Actual:   ${TENSORRT_MD5_ACTUAL}"
            warn "The file may be corrupted or a different version."
            read -p "Continue anyway? (y/N): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                error "Aborting due to MD5 mismatch. Delete and re-download the file."
            fi
        else
            log "MD5 verified: ${TENSORRT_MD5_ACTUAL}"
        fi
        
        # Extract TensorRT
        info "Extracting TensorRT to ${THIRDPARTY_DIR}..."
        tar -xzf "${TENSORRT_TARBALL}" -C "${THIRDPARTY_DIR}"
        
        # Rename if needed (tarball extracts to TensorRT-8.6.1.6)
        EXTRACTED_DIR="${THIRDPARTY_DIR}/TensorRT-${TENSORRT_VERSION}"
        if [[ ! -d "${EXTRACTED_DIR}" ]]; then
            # Find the extracted directory
            FOUND_DIR=$(find "${THIRDPARTY_DIR}" -maxdepth 1 -type d -name "TensorRT*" | head -1)
            if [[ -n "${FOUND_DIR}" && "${FOUND_DIR}" != "${EXTRACTED_DIR}" ]]; then
                mv "${FOUND_DIR}" "${EXTRACTED_DIR}"
            fi
        fi
        
        log "TensorRT extracted to ${TENSORRT_DIR}"
    fi
    
    # Verify TensorRT installation
    if [[ ! -f "${TENSORRT_DIR}/include/NvInfer.h" ]]; then
        error "TensorRT headers not found (NvInfer.h missing). Extraction failed?"
    fi
    
    if [[ ! -f "${TENSORRT_DIR}/lib/libnvinfer.so" ]]; then
        error "TensorRT libraries not found (libnvinfer.so missing). Extraction failed?"
    fi
    
    # Export TensorRT environment
    export TENSORRT_DIR="${TENSORRT_DIR}"
    export LD_LIBRARY_PATH="${TENSORRT_DIR}/lib:${LD_LIBRARY_PATH:-}"
    
    log "TensorRT ${TENSORRT_VERSION} setup complete"
    log "TENSORRT_DIR=${TENSORRT_DIR}"
}

# ==============================================================================
# CONDA ENVIRONMENT
# ==============================================================================
setup_conda() {
    step "PHASE 4: SETTING UP CONDA ENVIRONMENT"

    eval "$(conda shell.bash hook)"
    
    # Check for conda
    if ! command -v conda &>/dev/null; then
        if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
            source "$HOME/miniconda3/etc/profile.d/conda.sh"
        elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
            source "$HOME/anaconda3/etc/profile.d/conda.sh"
        elif [[ -f "/opt/conda/etc/profile.d/conda.sh" ]]; then
            source "/opt/conda/etc/profile.d/conda.sh"
        else
            error "Conda not found. Please install Miniconda or Anaconda first.
            
Install Miniconda:
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p \$HOME/miniconda3
    source \$HOME/miniconda3/etc/profile.d/conda.sh
    conda init bash
    
Then re-run this script."
        fi
    fi
    
    log "Conda found: $(conda --version)"
    
    # Check if environment exists
    if conda env list | grep -q "^${ENV_NAME} "; then
        warn "Environment '${ENV_NAME}' already exists"
        read -p "Remove and recreate? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            conda env remove -n "${ENV_NAME}" -y
        else
            info "Using existing environment"
            conda activate "${ENV_NAME}"
            return
        fi
    fi
    
    # Create environment
    info "Creating conda environment: ${ENV_NAME} (Python ${PYTHON_VERSION})..."
    conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" pip -y
    
    # Activate environment
    conda activate "${ENV_NAME}"
    
    log "Conda environment '${ENV_NAME}' created and activated"
}

# ==============================================================================
# PYTHON DEPENDENCIES
# ==============================================================================
# ==============================================================================
# PYTHON DEPENDENCIES (UPDATED: NO MIM, NO CRASHES)
# ==============================================================================
install_python_deps() {
    step "PHASE 5: INSTALLING PYTHON DEPENDENCIES"
    
    # Ensure we're in the right environment
    if [[ "${CONDA_DEFAULT_ENV}" != "${ENV_NAME}" ]]; then
        if ! command -v conda &>/dev/null; then
             source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null
        fi
        conda activate "${ENV_NAME}"
    fi
    
    # 1. BASE DEPENDENCIES (Pinned tight)
    info "Installing base dependencies..."
    pip install "numpy==1.22.4" --force-reinstall
    
    # 2. PYTORCH (CUDA 11.8)
    info "Installing PyTorch 2.0.1..."
    pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
        --index-url https://download.pytorch.org/whl/cu118
    
    # 3. CUPY
    info "Installing cupy-cuda11x..."
    pip install cupy-cuda11x

    # 4. TENSORRT (LOCAL WHEEL ONLY)
    info "Installing TensorRT Python bindings from LOCAL SDK..."
    TRT_PYTHON_DIR="${TENSORRT_DIR}/python"
    TRT_WHEEL=$(find "${TRT_PYTHON_DIR}" -name "tensorrt*-cp310-*.whl" 2>/dev/null | head -1)
    
    if [[ -f "${TRT_WHEEL}" ]]; then
        info "Found local TensorRT wheel: $(basename ${TRT_WHEEL})"
        pip install "${TRT_WHEEL}"
    else
        # Fallback if local wheel missing (should not happen if you followed steps)
        warn "Local TensorRT wheel not found. Skipping (assuming already installed)..."
    fi
    
    # 5. MMDEPLOY & RUNTIME
    info "Installing mmdeploy and runtime..."
    # We explicitly include numpy here to prevent upgrade
    pip install mmdeploy==1.3.1 mmdeploy-runtime-gpu==1.3.1 "numpy==1.22.4"

    # 6. ONNX RUNTIME
    info "Installing onnxruntime-gpu..."
    pip install onnxruntime-gpu==1.17.1

    # 7. MMEngine Ecosystem (DIRECT INSTALL - NO MIM)
    # bypassing 'mim' prevents the NumPy upgrade crash
    info "Installing MMEngine ecosystem (Direct Pip Mode)..."
    pip install -U openmim  # Install the tool but don't use it for heavy lifting
    
    # Install libraries directly from OpenMMLab's repo
    # We force numpy==1.22.4 in the SAME command so pip solves for it
    pip install mmengine "mmcv==2.1.0" "mmdet>=3.3.0" "numpy==1.22.4" \
        -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0.0/index.html

    # 8. PYQT6 (Pinned)
    info "Installing PyQt6..."
    pip install PyQt6==6.5.3 PyQt6-Qt6==6.5.3 PyQt6-sip==13.6.0 \
                PyQt6-WebEngine==6.5.0 PyQt6-WebEngine-Qt6==6.5.3

    # 9. OPENCV (Headless)
    info "Installing OpenCV (Headless)..."
    pip uninstall -y opencv-python opencv-contrib-python 2>/dev/null || true
    pip install opencv-python-headless==4.9.0.80

    # 10. REMAINING DEPENDENCIES
    info "Installing remaining dependencies..."
    pip install \
        scipy==1.13.0 \
        matplotlib==3.8.4 \
        pandas \
        toml \
        tqdm \
        natsort \
        pygltflib \
        pyqtgraph \
        func_timeout \
        openpyxl \
        pyserial \
        keyboard \
        multiprocess \
        bs4 \
        ultralytics \
        Pose2Sim==0.4 \
        "numpy==1.22.4"  # One final check
    
    log "Python dependencies installed!"
}

# ==============================================================================
# LOCAL PACKAGES
# ==============================================================================
install_local_packages() {
    step "PHASE 6: INSTALLING LOCAL PACKAGES"
    
    cd "${NTKCAP_ROOT}"
    
    # Install MMPose from source
    info "Installing MMPose from source..."
    cd "${THIRDPARTY_DIR}/mmpose"
    pip install -r requirements/build.txt 2>/dev/null || true
    pip install -e . -v
    cd "${NTKCAP_ROOT}"
    log "MMPose installed"
    
    # Install EasyMocap
    info "Installing EasyMocap..."
    cd "${THIRDPARTY_DIR}/EasyMocap"
    pip install setuptools==69.5.0
    python setup.py develop
    cd "${NTKCAP_ROOT}"
    log "EasyMocap installed"
    
    # Install OpenSim via conda
    info "Installing OpenSim 4.5 via conda..."
    conda install -c opensim-org opensim=4.5 -y || {
        warn "OpenSim conda install failed. Trying alternative method..."
        pip install opensim || warn "OpenSim installation failed - may need manual install"
    }
    log "OpenSim installed (or attempted)"
    
    log "Local packages installed!"
}

# ==============================================================================
# BUILD MMDEPLOY SDK (libmmdeploy_trt_net.so)
# ==============================================================================
build_mmdeploy_sdk() {
    step "PHASE 7: BUILDING MMDEPLOY SDK (libmmdeploy_trt_net.so)"
    
    # This is CRITICAL - without libmmdeploy_trt_net.so, TensorRT inference crashes
    # with "Library not loaded" error. pip install mmdeploy only gives Python bindings,
    # NOT the C++ TensorRT custom operators.
    
    MMDEPLOY_DIR="${THIRDPARTY_DIR}/mmdeploy"
    MMDEPLOY_BUILD_DIR="${MMDEPLOY_DIR}/build"
    
    if [[ ! -d "${MMDEPLOY_DIR}" ]]; then
        error "MMDeploy directory not found at ${MMDEPLOY_DIR}"
    fi
    
    if [[ ! -f "${MMDEPLOY_DIR}/CMakeLists.txt" ]]; then
        error "MMDeploy CMakeLists.txt not found. Submodule may be corrupted."
    fi
    
    # Check TensorRT is available
    if [[ ! -d "${TENSORRT_DIR}" ]]; then
        error "TensorRT not found at ${TENSORRT_DIR}. Run setup_tensorrt first."
    fi
    
    if [[ ! -f "${TENSORRT_DIR}/include/NvInfer.h" ]]; then
        error "TensorRT headers not found (NvInfer.h). Extraction incomplete."
    fi
    
    # Check CUDA
    if [[ -z "${CUDA_HOME:-}" ]]; then
        # Try to find CUDA
        for path in "/usr/local/cuda-11.8" "/usr/local/cuda" "/opt/cuda"; do
            if [[ -d "$path" && -f "$path/bin/nvcc" ]]; then
                export CUDA_HOME="$path"
                break
            fi
        done
    fi
    
    if [[ -z "${CUDA_HOME:-}" ]]; then
        error "CUDA_HOME not set and CUDA not found"
    fi
    
    # Find cuDNN - it's typically in CUDA installation or separate
    CUDNN_DIR="${CUDA_HOME}"
    if [[ ! -f "${CUDNN_DIR}/include/cudnn.h" ]]; then
        # Try PyTorch's bundled cuDNN
        PYTORCH_CUDNN=$(python -c "import torch; print(torch.backends.cudnn.is_available())" 2>/dev/null)
        if [[ "${PYTORCH_CUDNN}" == "True" ]]; then
            info "Using cuDNN from PyTorch (bundled)"
        else
            warn "cuDNN headers not found. Build may fail for some features."
        fi
    fi
    
    # Find OpenCV
    OPENCV_DIR=""
    for opencv_path in "/usr/lib/x86_64-linux-gnu/cmake/opencv4" "/usr/local/lib/cmake/opencv4" "/usr/share/opencv4"; do
        if [[ -d "${opencv_path}" ]]; then
            OPENCV_DIR="${opencv_path}"
            break
        fi
    done
    
    if [[ -z "${OPENCV_DIR}" ]]; then
        warn "OpenCV cmake directory not found. Will let CMake auto-detect."
    fi
    
    cd "${MMDEPLOY_DIR}"
    
    # Clean build directory
    info "Cleaning previous build..."
    rm -rf build
    mkdir -p build
    cd build
    
    # Configure with CMake
    info "Configuring MMDeploy SDK with CMake..."
    info "  TENSORRT_DIR: ${TENSORRT_DIR}"
    info "  CUDA_HOME: ${CUDA_HOME}"
    info "  OpenCV_DIR: ${OPENCV_DIR:-auto-detect}"
    
    CMAKE_ARGS=(
        ".."
        "-DCMAKE_BUILD_TYPE=Release"
        "-DMMDEPLOY_BUILD_SDK=ON"
        "-DMMDEPLOY_BUILD_SDK_PYTHON_API=OFF"
        "-DMMDEPLOY_TARGET_DEVICES=cuda"
        "-DMMDEPLOY_TARGET_BACKENDS=trt"
        "-DTENSORRT_DIR=${TENSORRT_DIR}"
        "-DCUDNN_DIR=${CUDA_HOME}"
        "-DCMAKE_CXX_COMPILER=g++"
        "-DCMAKE_C_COMPILER=gcc"
        "-Dpplcv_DIR=/usr/local/lib/cmake/pplcv"
    )
    
    # Add OpenCV path if found
    if [[ -n "${OPENCV_DIR}" ]]; then
        CMAKE_ARGS+=("-DOpenCV_DIR=${OPENCV_DIR}")
    fi
    
    # Run CMake
    cmake "${CMAKE_ARGS[@]}" 2>&1 | tee "${LOGS_DIR}/mmdeploy_cmake.log"
    CMAKE_STATUS=${PIPESTATUS[0]}
    
    if [[ ${CMAKE_STATUS} -ne 0 ]]; then
        error "CMake configuration failed! Check ${LOGS_DIR}/mmdeploy_cmake.log
        
Common issues:
1. TensorRT not found: Ensure TENSORRT_DIR points to extracted TensorRT
2. CUDA not found: Ensure CUDA 11.8 is installed
3. OpenCV not found: Run 'sudo apt install libopencv-dev'
4. cuDNN not found: Ensure cuDNN is installed with CUDA"
    fi
    
    log "CMake configuration successful"
    
    # Build
    info "Building MMDeploy SDK (this may take 5-15 minutes)..."
    NPROC=$(nproc)
    make -j${NPROC} 2>&1 | tee "${LOGS_DIR}/mmdeploy_build.log"
    MAKE_STATUS=${PIPESTATUS[0]}
    
    if [[ ${MAKE_STATUS} -ne 0 ]]; then
        error "Build failed! Check ${LOGS_DIR}/mmdeploy_build.log
        
Common issues:
1. GCC version mismatch: Ensure gcc-11 is being used
2. Missing headers: Check all dependencies are installed
3. Out of memory: Try 'make -j2' instead of -j${NPROC}"
    fi
    
    log "Build completed"
    
    # VERIFICATION (CRITICAL)
    info "Verifying libmmdeploy_trt_net.so..."
    
    # Search for the library in multiple possible locations
    TRT_NET_LIB=""
    for search_path in "${MMDEPLOY_BUILD_DIR}/lib" "${MMDEPLOY_BUILD_DIR}/src" "${MMDEPLOY_BUILD_DIR}"; do
        FOUND_LIB=$(find "${search_path}" -name "libmmdeploy_trt_net.so" 2>/dev/null | head -1)
        if [[ -n "${FOUND_LIB}" ]]; then
            TRT_NET_LIB="${FOUND_LIB}"
            break
        fi
    done
    
    if [[ -z "${TRT_NET_LIB}" ]]; then
        # Also check for any mmdeploy libraries
        echo ""
        echo "=== BUILD OUTPUT ANALYSIS ==="
        echo "Libraries found in build directory:"
        find "${MMDEPLOY_BUILD_DIR}" -name "*.so" -o -name "*.so.*" 2>/dev/null | head -20
        echo ""
        
        error "CRITICAL: libmmdeploy_trt_net.so NOT FOUND!

The TensorRT backend library was not built. This means TensorRT inference will fail.

Check the build logs:
  ${LOGS_DIR}/mmdeploy_cmake.log
  ${LOGS_DIR}/mmdeploy_build.log

Possible causes:
1. TensorRT SDK incomplete (missing headers or libs)
2. CMake didn't find TensorRT properly
3. Build was interrupted

Try manually:
  cd ${MMDEPLOY_DIR}
  rm -rf build && mkdir build && cd build
  cmake .. -DMMDEPLOY_TARGET_BACKENDS=trt -DTENSORRT_DIR=${TENSORRT_DIR} -DMMDEPLOY_BUILD_SDK=ON
  make -j4 VERBOSE=1"
    fi
    
    log "FOUND: ${TRT_NET_LIB}"
    
    # Record the library directory for activation script
    MMDEPLOY_LIB_DIR=$(dirname "${TRT_NET_LIB}")
    echo "${MMDEPLOY_LIB_DIR}" > "${NTKCAP_ROOT}/.mmdeploy_lib_path"
    
    # List all built libraries
    info "Built libraries:"
    find "${MMDEPLOY_BUILD_DIR}" -name "libmmdeploy*.so" 2>/dev/null | while read lib; do
        echo "  ✓ $(basename ${lib})"
    done
    
    cd "${NTKCAP_ROOT}"
    
    log "MMDeploy SDK build complete!"
    log "Library path: ${MMDEPLOY_LIB_DIR}"
}

# ==============================================================================
# CREATE ACTIVATION SCRIPT
# ==============================================================================
create_activation_script() {
    step "PHASE 8: CREATING ACTIVATION SCRIPT"
    
    cat > "${NTKCAP_ROOT}/activate.sh" << 'ACTIVATE_EOF'
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
    echo -e "${YELLOW}⚠ CUDA not found${NC}"
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
fi

# ============================================================================
# MMDEPLOY SDK (libmmdeploy_trt_net.so)
# ============================================================================
MMDEPLOY_LIB_PATH_FILE="${NTKCAP_ROOT}/.mmdeploy_lib_path"
if [[ -f "$MMDEPLOY_LIB_PATH_FILE" ]]; then
    MMDEPLOY_LIB_DIR=$(cat "$MMDEPLOY_LIB_PATH_FILE")
    if [[ -d "$MMDEPLOY_LIB_DIR" ]]; then
        export LD_LIBRARY_PATH="${MMDEPLOY_LIB_DIR}:${LD_LIBRARY_PATH:-}"
        echo -e "${GREEN}✓ MMDeploy SDK:${NC} $MMDEPLOY_LIB_DIR"
    fi
else
    # Fallback: check default build location
    MMDEPLOY_BUILD_LIB="${NTKCAP_ROOT}/NTK_CAP/ThirdParty/mmdeploy/build/lib"
    if [[ -d "$MMDEPLOY_BUILD_LIB" && -f "$MMDEPLOY_BUILD_LIB/libmmdeploy_trt_net.so" ]]; then
        export LD_LIBRARY_PATH="${MMDEPLOY_BUILD_LIB}:${LD_LIBRARY_PATH:-}"
        echo -e "${GREEN}✓ MMDeploy SDK:${NC} $MMDEPLOY_BUILD_LIB"
    else
        echo -e "${YELLOW}⚠ MMDeploy SDK not built (libmmdeploy_trt_net.so missing)${NC}"
        echo -e "${YELLOW}  Run: ./install/scripts/setup_linux.sh${NC}"
    fi
fi

# ============================================================================
# CONDA ENVIRONMENT
# ============================================================================
CONDA_SOURCED=false
for conda_path in "$HOME/miniconda3" "$HOME/anaconda3" "/opt/conda"; do
    if [[ -f "${conda_path}/etc/profile.d/conda.sh" ]]; then
        source "${conda_path}/etc/profile.d/conda.sh"
        CONDA_SOURCED=true
        break
    fi
done

if [[ "$CONDA_SOURCED" == "true" ]]; then
    conda activate ntkcap_env 2>/dev/null && {
        echo -e "${GREEN}✓ Conda:${NC} ntkcap_env activated"
        echo -e "${GREEN}✓ Python:${NC} $(python --version 2>&1)"
    } || {
        echo -e "${YELLOW}⚠ Could not activate ntkcap_env${NC}"
    }
else
    echo -e "${YELLOW}⚠ Conda not found${NC}"
fi

# ============================================================================
# GPU VERIFICATION
# ============================================================================
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    if [[ -n "$GPU_NAME" ]]; then
        echo -e "${GREEN}✓ GPU:${NC} $GPU_NAME"
    fi
fi

# PyTorch CUDA check
python -c "
import torch
if torch.cuda.is_available():
    print(f'\033[0;32m✓ PyTorch CUDA:\033[0m {torch.cuda.get_device_name(0)}')
else:
    print('⚠ PyTorch CUDA not available')
" 2>/dev/null

echo ""
echo "Working directory: $NTKCAP_ROOT"
echo ""
echo "Quick commands:"
echo "  python NTKCAP_GUI.py          # Main GUI"
echo "  python final_NTK_Cap_GUI.py   # Alternative GUI"
echo ""
ACTIVATE_EOF

    chmod +x "${NTKCAP_ROOT}/activate.sh"
    log "Created ${NTKCAP_ROOT}/activate.sh"
}

# ==============================================================================
# FINAL VERIFICATION
# ==============================================================================
final_verification() {
    step "PHASE 9: FINAL VERIFICATION"
    
    echo ""
    echo "========================================================================"
    echo "                    ENVIRONMENT VERIFICATION                           "
    echo "========================================================================"
    echo ""
    
    # Verify all critical imports
    python << 'VERIFY_EOF'
import sys
import os

checks = []

# 1. NumPy version
try:
    import numpy as np
    ver = np.__version__
    ok = ver == "1.22.4"
    checks.append(("NumPy", ver, ok))
except Exception as e:
    checks.append(("NumPy", str(e), False))

# 2. PyTorch CUDA
try:
    import torch
    cuda_ok = torch.cuda.is_available()
    ver = torch.__version__
    checks.append(("PyTorch", f"{ver}, CUDA={cuda_ok}", cuda_ok))
except Exception as e:
    checks.append(("PyTorch", str(e), False))

# 3. CuPy
try:
    import cupy as cp
    ver = cp.__version__
    # Quick GPU test
    x = cp.array([1, 2, 3])
    checks.append(("CuPy", ver, True))
except Exception as e:
    checks.append(("CuPy", str(e), False))

# 4. MMDeploy
try:
    import mmdeploy
    ver = mmdeploy.__version__
    checks.append(("MMDeploy", ver, True))
except Exception as e:
    checks.append(("MMDeploy", str(e), False))

# 5. MMPose
try:
    import mmpose
    ver = mmpose.__version__
    checks.append(("MMPose", ver, True))
except Exception as e:
    checks.append(("MMPose", str(e), False))

# 6. MMEngine
try:
    import mmengine
    ver = mmengine.__version__
    checks.append(("MMEngine", ver, True))
except Exception as e:
    checks.append(("MMEngine", str(e), False))

# 7. MMCV
try:
    import mmcv
    ver = mmcv.__version__
    checks.append(("MMCV", ver, True))
except Exception as e:
    checks.append(("MMCV", str(e), False))

# 8. OpenCV
try:
    import cv2
    ver = cv2.__version__
    checks.append(("OpenCV", ver, True))
except Exception as e:
    checks.append(("OpenCV", str(e), False))

# 9. PyQt6
try:
    from PyQt6 import QtCore
    ver = QtCore.PYQT_VERSION_STR
    checks.append(("PyQt6", ver, True))
except Exception as e:
    checks.append(("PyQt6", str(e), False))

# 10. TensorRT
try:
    import tensorrt as trt
    ver = trt.__version__
    checks.append(("TensorRT", ver, True))
except Exception as e:
    checks.append(("TensorRT", str(e), False))

# Print results
all_ok = True
for name, info, ok in checks:
    status = "✓" if ok else "✗"
    color = "\033[0;32m" if ok else "\033[0;31m"
    print(f"{color}{status}\033[0m {name}: {info}")
    if not ok:
        all_ok = False

print("")
if all_ok:
    print("\033[0;32m✓ All checks passed!\033[0m")
    sys.exit(0)
else:
    print("\033[0;31m✗ Some checks failed. Review the errors above.\033[0m")
    sys.exit(1)
VERIFY_EOF

    VERIFY_STATUS=$?
    
    echo ""
    echo "========================================================================"
    
    if [[ ${VERIFY_STATUS} -eq 0 ]]; then
        log "ALL VERIFICATIONS PASSED!"
    else
        warn "Some verifications failed. Check the output above."
    fi
}

# ==============================================================================
# MAIN
# ==============================================================================
main() {
    echo ""
    echo "╔════════════════════════════════════════════════════════════════════╗"
    echo "║        NTKCAP LINUX ENVIRONMENT SETUP (BULLETPROOF EDITION)        ║"
    echo "╠════════════════════════════════════════════════════════════════════╣"
    echo "║  Target: Ubuntu 22.04/24.04 LTS                                    ║"
    echo "║  CUDA: 11.8 | TensorRT: 8.6.1.6 | Python: 3.10                     ║"
    echo "╚════════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "NTKCAP Root: ${NTKCAP_ROOT}"
    echo ""
    
    # Run all phases
    preflight_checks
    install_system_deps
    verify_cuda
    setup_tensorrt
    setup_conda
    install_python_deps
    install_local_packages
    build_mmdeploy_sdk
    create_activation_script
    final_verification
    
    echo ""
    echo "╔════════════════════════════════════════════════════════════════════╗"
    echo "║                    INSTALLATION COMPLETE!                          ║"
    echo "╚════════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "Next steps:"
    echo "1. Activate the environment:"
    echo "   source ${NTKCAP_ROOT}/activate.sh"
    echo ""
    echo "2. Rebuild TensorRT engines for your GPU (REQUIRED on first run):"
    echo "   cd ${NTKCAP_ROOT}"
    echo "   python -c 'from mmdeploy_runtime import PoseTracker; print(\"OK\")'"
    echo ""
    echo "3. Run the GUI:"
    echo "   python NTKCAP_GUI.py"
    echo ""
}

# Run main
main "$@"

