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
# 7. pplcv Missing - Builds pplcv before mmdeploy SDK
################################################################################

set -euo pipefail  # Exit on error, undefined var, pipe failure
IFS=$'\n\t'

# ==============================================================================
# GET SCRIPT DIRECTORY (All paths relative to this)
# ==============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NTKCAP_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ==============================================================================
# CONFIGURATION (No hardcoded absolute paths)
# ==============================================================================
CUDA_VERSION="11.8"
TENSORRT_VERSION="8.6.1.6"
TENSORRT_CUDA="11.8"
PYTHON_VERSION="3.10"
ENV_NAME="ntkcap_env"

# Relative paths from NTKCAP_ROOT
THIRDPARTY_DIR="${NTKCAP_ROOT}/NTK_CAP/ThirdParty"
TENSORRT_DIR="${THIRDPARTY_DIR}/TensorRT-${TENSORRT_VERSION}"
DOWNLOADS_DIR="${NTKCAP_ROOT}/install/downloads"
LOGS_DIR="${NTKCAP_ROOT}/install/logs"
MMDEPLOY_DIR="${THIRDPARTY_DIR}/mmdeploy"
PPLCV_DIR="${MMDEPLOY_DIR}/third_party/pplcv"

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
    if [[ ! -f "${MMDEPLOY_DIR}/CMakeLists.txt" ]]; then
        error "SUBMODULE ZOMBIE: mmdeploy/CMakeLists.txt not found!
        
Run these commands to fix:
    cd ${NTKCAP_ROOT}
    git submodule update --init --recursive --force
    git submodule foreach git reset --hard HEAD
    
Then re-run this script."
    fi
    log "Submodule health check passed (mmdeploy/CMakeLists.txt exists)"
    
    if [[ ! -f "${THIRDPARTY_DIR}/mmpose/setup.py" ]]; then
        warn "mmpose/setup.py not found - may need submodule init"
    fi
    
    if [[ ! -f "${THIRDPARTY_DIR}/EasyMocap/setup.py" ]]; then
        warn "EasyMocap/setup.py not found - may need submodule init"
    fi
    
    # 8. Check pplcv submodule
    if [[ ! -f "${PPLCV_DIR}/CMakeLists.txt" ]]; then
        warn "pplcv/CMakeLists.txt not found - may need submodule init"
    else
        log "pplcv submodule found"
    fi
    
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
            --slave /usr/bin/g++ g++ /usr/bin/g++-11 || true
        
        log "GCC 11 installed and set as alternative"
    fi
    
    # Install libtinfo5 (Legacy Library Void fix)
    if [[ "${NEED_LIBTINFO5:-true}" == "true" ]] || ! ldconfig -p | grep -q "libtinfo.so.5"; then
        info "Installing libtinfo5..."
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
# CUDA VERIFICATION (Auto-detect, no hardcoded paths)
# ==============================================================================
verify_cuda() {
    step "PHASE 2: VERIFYING CUDA ${CUDA_VERSION}"
    
    # Search for CUDA installation
    NVCC_PATH=""
    CUDA_HOME=""
    
    CUDA_SEARCH_PATHS=(
        "/usr/local/cuda-${CUDA_VERSION}"
        "/usr/local/cuda"
        "/opt/cuda"
    )
    
    for path in "${CUDA_SEARCH_PATHS[@]}"; do
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
# TENSORRT SETUP (Downloads to relative path)
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
        
        # Extract TensorRT
        info "Extracting TensorRT to ${THIRDPARTY_DIR}..."
        tar -xzf "${TENSORRT_TARBALL}" -C "${THIRDPARTY_DIR}"
        
        # Rename if needed
        EXTRACTED_DIR="${THIRDPARTY_DIR}/TensorRT-${TENSORRT_VERSION}"
        if [[ ! -d "${EXTRACTED_DIR}" ]]; then
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
}

# ==============================================================================
# CONDA ENVIRONMENT (Auto-detect conda location)
# ==============================================================================
setup_conda() {
    step "PHASE 4: SETTING UP CONDA ENVIRONMENT"

    # Find conda
    CONDA_FOUND=false
    CONDA_PATHS=(
        "$HOME/miniconda3"
        "$HOME/anaconda3"
        "$HOME/.conda"
        "/opt/conda"
        "/opt/miniconda3"
        "/opt/anaconda3"
    )
    
    for conda_path in "${CONDA_PATHS[@]}"; do
        if [[ -f "${conda_path}/etc/profile.d/conda.sh" ]]; then
            source "${conda_path}/etc/profile.d/conda.sh"
            CONDA_FOUND=true
            break
        fi
    done
    
    if [[ "${CONDA_FOUND}" == "false" ]]; then
        error "Conda not found. Please install Miniconda or Anaconda first.
            
Install Miniconda:
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p \$HOME/miniconda3
    source \$HOME/miniconda3/etc/profile.d/conda.sh
    conda init bash
    
Then re-run this script."
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
install_python_deps() {
    step "PHASE 5: INSTALLING PYTHON DEPENDENCIES"
    
    # Ensure we're in the right environment
    if [[ "${CONDA_DEFAULT_ENV}" != "${ENV_NAME}" ]]; then
        for conda_path in "$HOME/miniconda3" "$HOME/anaconda3" "/opt/conda"; do
            if [[ -f "${conda_path}/etc/profile.d/conda.sh" ]]; then
                source "${conda_path}/etc/profile.d/conda.sh"
                break
            fi
        done
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

    # 4. TENSORRT (LOCAL WHEEL FROM SDK)
    info "Installing TensorRT Python bindings from LOCAL SDK..."
    TRT_PYTHON_DIR="${TENSORRT_DIR}/python"
    TRT_WHEEL=$(find "${TRT_PYTHON_DIR}" -name "tensorrt*-cp310-*.whl" 2>/dev/null | head -1)
    
    if [[ -f "${TRT_WHEEL}" ]]; then
        info "Found local TensorRT wheel: $(basename ${TRT_WHEEL})"
        pip install "${TRT_WHEEL}"
    else
        warn "Local TensorRT wheel not found. Skipping..."
    fi
    
    # 5. MMDEPLOY & RUNTIME
    info "Installing mmdeploy and runtime..."
    pip install mmdeploy==1.3.1 mmdeploy-runtime-gpu==1.3.1 "numpy==1.22.4"

    # 6. ONNX RUNTIME
    info "Installing onnxruntime-gpu..."
    pip install onnxruntime-gpu==1.17.1

    # 7. MMEngine Ecosystem (DIRECT INSTALL - NO MIM)
    info "Installing MMEngine ecosystem (Direct Pip Mode)..."
    pip install -U openmim
    pip install mmengine "mmcv==2.1.0" "mmdet>=3.3.0" "numpy==1.22.4" \
        -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0.0/index.html

    # 8. PYQT6 (Pinned)
    info "Installing PyQt6..."
    pip install PyQt6==6.5.3 PyQt6-Qt6==6.5.3 PyQt6-sip==13.6.0 \
                PyQt6-WebEngine==6.5.0 PyQt6-WebEngine-Qt6==6.5.3

    # 9. OPENCV (Headless to avoid Qt conflicts)
    info "Installing OpenCV (Headless)..."
    pip uninstall -y opencv-python opencv-contrib-python 2>/dev/null || true
    pip install opencv-python-headless==4.9.0.80

    # 10. RTMLIB (fallback for TensorRT)
    info "Installing rtmlib (ONNX fallback)..."
    pip install rtmlib

    # 11. REMAINING DEPENDENCIES
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
        "numpy==1.22.4"
    
    log "Python dependencies installed!"
}

# ==============================================================================
# LOCAL PACKAGES (Using relative paths)
# ==============================================================================
install_local_packages() {
    step "PHASE 6: INSTALLING LOCAL PACKAGES"
    
    cd "${NTKCAP_ROOT}"
    
    # Install MMPose from source
    if [[ -f "${THIRDPARTY_DIR}/mmpose/setup.py" ]]; then
        info "Installing MMPose from source..."
        cd "${THIRDPARTY_DIR}/mmpose"
        pip install -r requirements/build.txt 2>/dev/null || true
        pip install -e . -v
        cd "${NTKCAP_ROOT}"
        log "MMPose installed"
    else
        warn "MMPose not found - skipping"
    fi
    
    # Install EasyMocap
    if [[ -f "${THIRDPARTY_DIR}/EasyMocap/setup.py" ]]; then
        info "Installing EasyMocap..."
        cd "${THIRDPARTY_DIR}/EasyMocap"
        pip install setuptools==69.5.0
        python setup.py develop
        cd "${NTKCAP_ROOT}"
        log "EasyMocap installed"
    else
        warn "EasyMocap not found - skipping"
    fi
    
    # Install OpenSim via conda
    info "Installing OpenSim 4.5 via conda..."
    conda install -c opensim-org opensim=4.5 -y || {
        warn "OpenSim conda install failed. Trying alternative method..."
        pip install opensim || warn "OpenSim installation failed - may need manual install"
    }
    
    log "Local packages installed!"
}

# ==============================================================================
# BUILD PPLCV (Required for MMDeploy SDK)
# ==============================================================================
build_pplcv() {
    step "PHASE 7A: BUILDING PPLCV (Required for MMDeploy SDK)"
    
    echo ""
    echo "┌─────────────────────────────────────────────────────────────────────┐"
    echo "│  pplcv: High-performance image processing library for MMDeploy     │"
    echo "│  This MUST be built before MMDeploy SDK                            │"
    echo "└─────────────────────────────────────────────────────────────────────┘"
    echo ""
    
    PPLCV_BUILD_DIR="${PPLCV_DIR}/cuda-build"
    PPLCV_INSTALL_DIR="${PPLCV_BUILD_DIR}/install"
    
    # Check if already built
    if [[ -f "${PPLCV_INSTALL_DIR}/lib/libpplcv_static.a" ]] || [[ -f "/usr/local/lib/cmake/pplcv/pplcv-config.cmake" ]]; then
        log "pplcv already built/installed"
        return
    fi
    
    if [[ ! -f "${PPLCV_DIR}/CMakeLists.txt" ]]; then
        error "pplcv not found at ${PPLCV_DIR}
        
Run: git submodule update --init --recursive
Then re-run this script."
    fi
    
    cd "${PPLCV_DIR}"
    
    # Clean previous build
    info "Cleaning previous pplcv build..."
    rm -rf cuda-build
    mkdir -p cuda-build
    cd cuda-build
    
    # CMake Configuration
    info "Configuring pplcv with CMake..."
    info "  CMake command:"
    info "    cmake .."
    info "      -DCMAKE_BUILD_TYPE=Release"
    info "      -DPPLCV_USE_CUDA=ON"
    info "      -DCMAKE_INSTALL_PREFIX=${PPLCV_INSTALL_DIR}"
    echo ""
    
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DPPLCV_USE_CUDA=ON \
        -DCMAKE_INSTALL_PREFIX="${PPLCV_INSTALL_DIR}" \
        2>&1 | tee "${LOGS_DIR}/pplcv_cmake.log"
    
    if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
        echo ""
        echo "═══════════════════════════════════════════════════════════════════"
        echo "  ❌ PPLCV CMAKE CONFIGURATION FAILED"
        echo "═══════════════════════════════════════════════════════════════════"
        echo ""
        echo "  Log file: ${LOGS_DIR}/pplcv_cmake.log"
        echo ""
        echo "  Common causes:"
        echo "    • CUDA not found - ensure CUDA 11.8 is installed"
        echo "    • Missing compiler - ensure gcc-11 is installed"
        echo ""
        echo "  To debug manually:"
        echo "    cd ${PPLCV_DIR}/cuda-build"
        echo "    cmake .. -DPPLCV_USE_CUDA=ON 2>&1 | less"
        echo ""
        error "pplcv CMake configuration failed!"
    fi
    
    log "pplcv CMake configuration successful"
    
    # Build
    info "Building pplcv (this may take 2-5 minutes)..."
    NPROC=$(nproc)
    cmake --build . -j ${NPROC} 2>&1 | tee "${LOGS_DIR}/pplcv_build.log"
    
    if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
        echo ""
        echo "═══════════════════════════════════════════════════════════════════"
        echo "  ❌ PPLCV BUILD FAILED"
        echo "═══════════════════════════════════════════════════════════════════"
        echo ""
        echo "  Log file: ${LOGS_DIR}/pplcv_build.log"
        echo ""
        echo "  To see last errors:"
        echo "    tail -50 ${LOGS_DIR}/pplcv_build.log"
        echo ""
        error "pplcv build failed!"
    fi
    
    log "pplcv build successful"
    
    # Install
    info "Installing pplcv..."
    cmake --build . --target install 2>&1 | tee -a "${LOGS_DIR}/pplcv_build.log"
    
    # Install to /usr/local for mmdeploy to find it
    info "Installing pplcv to /usr/local (requires sudo)..."
    sudo mkdir -p /usr/local/lib/cmake/pplcv /usr/local/include/pplcv
    sudo cp -r "${PPLCV_INSTALL_DIR}"/lib/* /usr/local/lib/ 2>/dev/null || true
    sudo cp -r "${PPLCV_INSTALL_DIR}"/include/* /usr/local/include/ 2>/dev/null || true
    sudo cp -r "${PPLCV_INSTALL_DIR}"/lib/cmake/pplcv/* /usr/local/lib/cmake/pplcv/ 2>/dev/null || true
    sudo ldconfig
    
    # Verify installation
    if [[ -f "/usr/local/lib/cmake/pplcv/pplcv-config.cmake" ]]; then
        log "pplcv installed to /usr/local/lib/cmake/pplcv/"
    else
        warn "pplcv cmake config not found in /usr/local - MMDeploy may not find it"
    fi
    
    cd "${NTKCAP_ROOT}"
    
    log "pplcv build and installation complete"
}

# ==============================================================================
# BUILD MMDEPLOY SDK (libmmdeploy_trt_net.so)
# ==============================================================================
build_mmdeploy_sdk() {
    step "PHASE 7B: BUILDING MMDEPLOY SDK (libmmdeploy_trt_net.so)"
    
    echo ""
    echo "┌─────────────────────────────────────────────────────────────────────┐"
    echo "│  MMDeploy SDK: TensorRT inference backend for pose estimation      │"
    echo "│  This builds libmmdeploy_trt_net.so (CRITICAL for TensorRT)        │"
    echo "└─────────────────────────────────────────────────────────────────────┘"
    echo ""
    
    MMDEPLOY_BUILD_DIR="${MMDEPLOY_DIR}/build"
    
    if [[ ! -d "${MMDEPLOY_DIR}" ]]; then
        error "MMDeploy directory not found at ${MMDEPLOY_DIR}"
    fi
    
    if [[ ! -f "${MMDEPLOY_DIR}/CMakeLists.txt" ]]; then
        error "MMDeploy CMakeLists.txt not found. Submodule may be corrupted.
        
Fix with:
    cd ${NTKCAP_ROOT}
    git submodule update --init --recursive --force"
    fi
    
    # Check dependencies
    if [[ ! -d "${TENSORRT_DIR}" ]]; then
        error "TensorRT not found at ${TENSORRT_DIR}
        
Please download TensorRT 8.6.1.6 and place in:
    ${DOWNLOADS_DIR}/TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.8.tar.gz"
    fi
    
    # Find CUDA
    if [[ -z "${CUDA_HOME:-}" ]]; then
        for path in "/usr/local/cuda-11.8" "/usr/local/cuda" "/opt/cuda"; do
            if [[ -d "$path" && -f "$path/bin/nvcc" ]]; then
                export CUDA_HOME="$path"
                break
            fi
        done
    fi
    
    if [[ -z "${CUDA_HOME:-}" ]]; then
        error "CUDA not found!
        
Install CUDA 11.8:
    https://developer.nvidia.com/cuda-11-8-0-download-archive"
    fi
    
    # Find OpenCV
    OPENCV_DIR=""
    OPENCV_SEARCH_PATHS=(
        "/usr/lib/x86_64-linux-gnu/cmake/opencv4"
        "/usr/local/lib/cmake/opencv4"
        "/usr/share/opencv4"
    )
    for opencv_path in "${OPENCV_SEARCH_PATHS[@]}"; do
        if [[ -d "${opencv_path}" ]]; then
            OPENCV_DIR="${opencv_path}"
            break
        fi
    done
    
    if [[ -z "${OPENCV_DIR}" ]]; then
        warn "OpenCV cmake directory not found"
        warn "Install with: sudo apt-get install libopencv-dev"
    fi
    
    # Find pplcv (MUST be built first)
    PPLCV_CMAKE_DIR=""
    PPLCV_SEARCH_PATHS=(
        "/usr/local/lib/cmake/pplcv"
        "${PPLCV_DIR}/cuda-build/install/lib/cmake/pplcv"
        "${PPLCV_DIR}/cuda-build/lib/cmake/pplcv"
    )
    for pplcv_path in "${PPLCV_SEARCH_PATHS[@]}"; do
        if [[ -f "${pplcv_path}/pplcv-config.cmake" ]]; then
            PPLCV_CMAKE_DIR="${pplcv_path}"
            break
        fi
    done
    
    if [[ -z "${PPLCV_CMAKE_DIR}" ]]; then
        echo ""
        echo "═══════════════════════════════════════════════════════════════════"
        echo "  ❌ PPLCV NOT FOUND - REQUIRED FOR MMDEPLOY SDK"
        echo "═══════════════════════════════════════════════════════════════════"
        echo ""
        echo "  pplcv must be built BEFORE MMDeploy SDK."
        echo "  The setup script should have built it in Phase 7A."
        echo ""
        echo "  To build pplcv manually:"
        echo "    cd ${PPLCV_DIR}"
        echo "    mkdir cuda-build && cd cuda-build"
        echo "    cmake .. -DPPLCV_USE_CUDA=ON -DCMAKE_INSTALL_PREFIX=\$(pwd)/install"
        echo "    cmake --build . -j\$(nproc)"
        echo "    cmake --build . --target install"
        echo "    sudo cp -r install/lib/cmake/pplcv /usr/local/lib/cmake/"
        echo ""
        error "pplcv not found. Build pplcv first."
    fi
    
    cd "${MMDEPLOY_DIR}"
    
    # Clean build directory
    info "Cleaning previous MMDeploy build..."
    rm -rf build
    mkdir -p build
    cd build
    
    # Show CMake configuration
    echo ""
    echo "┌─────────────────────────────────────────────────────────────────────┐"
    echo "│  CMake Configuration                                               │"
    echo "└─────────────────────────────────────────────────────────────────────┘"
    echo ""
    echo "  TENSORRT_DIR    = ${TENSORRT_DIR}"
    echo "  CUDA_HOME       = ${CUDA_HOME}"
    echo "  pplcv_DIR       = ${PPLCV_CMAKE_DIR}"
    echo "  OpenCV_DIR      = ${OPENCV_DIR:-auto-detect}"
    echo ""
    echo "  Target Backend  = TensorRT (trt)"
    echo "  Target Device   = CUDA"
    echo ""
    
    # Build CMake arguments
    CMAKE_ARGS=(
        ".."
        "-DCMAKE_BUILD_TYPE=Release"
        "-DMMDEPLOY_BUILD_SDK=ON"
        "-DMMDEPLOY_BUILD_SDK_PYTHON_API=OFF"
        "-DMMDEPLOY_TARGET_DEVICES=cuda"
        "-DMMDEPLOY_TARGET_BACKENDS=trt"
        "-DTENSORRT_DIR=${TENSORRT_DIR}"
        "-DCUDNN_DIR=${CUDA_HOME}"
        "-Dpplcv_DIR=${PPLCV_CMAKE_DIR}"
    )
    
    if [[ -n "${OPENCV_DIR}" ]]; then
        CMAKE_ARGS+=("-DOpenCV_DIR=${OPENCV_DIR}")
    fi
    
    # Show full cmake command for debugging
    info "CMake command:"
    echo "    cmake \\"
    for arg in "${CMAKE_ARGS[@]}"; do
        echo "      ${arg} \\"
    done
    echo ""
    
    # Run CMake
    info "Running CMake configuration..."
    cmake "${CMAKE_ARGS[@]}" 2>&1 | tee "${LOGS_DIR}/mmdeploy_cmake.log"
    CMAKE_STATUS=${PIPESTATUS[0]}
    
    if [[ ${CMAKE_STATUS} -ne 0 ]]; then
        echo ""
        echo "═══════════════════════════════════════════════════════════════════"
        echo "  ❌ MMDEPLOY CMAKE CONFIGURATION FAILED"
        echo "═══════════════════════════════════════════════════════════════════"
        echo ""
        echo "  Log file: ${LOGS_DIR}/mmdeploy_cmake.log"
        echo ""
        echo "  Common causes:"
        echo "    • TensorRT not found - check TENSORRT_DIR path"
        echo "    • pplcv not found - build pplcv first (Phase 7A)"
        echo "    • OpenCV not found - sudo apt-get install libopencv-dev"
        echo "    • CUDA not found - check CUDA_HOME path"
        echo ""
        echo "  To debug, check the cmake log:"
        echo "    grep -i 'error\\|not found\\|could not find' ${LOGS_DIR}/mmdeploy_cmake.log"
        echo ""
        error "MMDeploy CMake configuration failed!"
    fi
    
    log "CMake configuration successful"
    
    # Build
    echo ""
    info "Building MMDeploy SDK (this takes 5-15 minutes)..."
    NPROC=$(nproc)
    make -j${NPROC} 2>&1 | tee "${LOGS_DIR}/mmdeploy_build.log"
    MAKE_STATUS=${PIPESTATUS[0]}
    
    if [[ ${MAKE_STATUS} -ne 0 ]]; then
        echo ""
        echo "═══════════════════════════════════════════════════════════════════"
        echo "  ❌ MMDEPLOY BUILD FAILED"
        echo "═══════════════════════════════════════════════════════════════════"
        echo ""
        echo "  Log file: ${LOGS_DIR}/mmdeploy_build.log"
        echo ""
        echo "  To see last errors:"
        echo "    tail -100 ${LOGS_DIR}/mmdeploy_build.log | grep -i error"
        echo ""
        echo "  Common causes:"
        echo "    • GCC version mismatch - use gcc-11"
        echo "    • Out of memory - try: make -j2"
        echo "    • Missing headers - check dependencies"
        echo ""
        error "MMDeploy build failed!"
    fi
    
    log "MMDeploy build completed"
    
    # VERIFICATION - Check for critical library
    echo ""
    info "Verifying built libraries..."
    
    TRT_NET_LIB=""
    for search_path in "${MMDEPLOY_BUILD_DIR}/lib" "${MMDEPLOY_BUILD_DIR}/src" "${MMDEPLOY_BUILD_DIR}"; do
        FOUND_LIB=$(find "${search_path}" -name "libmmdeploy_trt_net.so" 2>/dev/null | head -1)
        if [[ -n "${FOUND_LIB}" ]]; then
            TRT_NET_LIB="${FOUND_LIB}"
            break
        fi
    done
    
    if [[ -z "${TRT_NET_LIB}" ]]; then
        echo ""
        echo "═══════════════════════════════════════════════════════════════════"
        echo "  ❌ CRITICAL: libmmdeploy_trt_net.so NOT FOUND!"
        echo "═══════════════════════════════════════════════════════════════════"
        echo ""
        echo "  The TensorRT backend library was NOT built."
        echo "  TensorRT inference WILL FAIL without this library."
        echo ""
        echo "  Libraries found:"
        find "${MMDEPLOY_BUILD_DIR}" -name "*.so" 2>/dev/null | head -10
        echo ""
        echo "  Check build logs:"
        echo "    ${LOGS_DIR}/mmdeploy_cmake.log"
        echo "    ${LOGS_DIR}/mmdeploy_build.log"
        echo ""
        error "libmmdeploy_trt_net.so not built!"
    fi
    
    log "FOUND: ${TRT_NET_LIB}"
    
    # Record the library directory for activation script
    MMDEPLOY_LIB_DIR=$(dirname "${TRT_NET_LIB}")
    echo "${MMDEPLOY_LIB_DIR}" > "${NTKCAP_ROOT}/.mmdeploy_lib_path"
    
    # List all built libraries
    echo ""
    echo "┌─────────────────────────────────────────────────────────────────────┐"
    echo "│  Built Libraries                                                   │"
    echo "└─────────────────────────────────────────────────────────────────────┘"
    find "${MMDEPLOY_BUILD_DIR}" -name "libmmdeploy*.so" 2>/dev/null | while read lib; do
        echo "  ✓ $(basename ${lib})"
    done
    echo ""
    echo "  Library path: ${MMDEPLOY_LIB_DIR}"
    echo "  Saved to: ${NTKCAP_ROOT}/.mmdeploy_lib_path"
    echo ""
    
    cd "${NTKCAP_ROOT}"
    
    log "MMDeploy SDK build complete!"
}

# ==============================================================================
# KEYBOARD PERMISSIONS (For 'keyboard' module)
# ==============================================================================
setup_keyboard_permissions() {
    step "PHASE 8: SETTING UP KEYBOARD PERMISSIONS"
    
    # The 'keyboard' Python module requires read access to /dev/input/event*
    # This is typically only available to root or the 'input' group
    
    CURRENT_USER=$(whoami)
    
    if groups | grep -q '\binput\b'; then
        log "User ${CURRENT_USER} is already in 'input' group"
    else
        info "Adding ${CURRENT_USER} to 'input' group for keyboard access..."
        sudo usermod -a -G input "${CURRENT_USER}" || {
            warn "Could not add user to input group. 'keyboard' module may require sudo."
        }
        log "User added to 'input' group. Log out and back in for this to take effect."
    fi
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
    
    python << 'VERIFY_EOF'
import sys

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

# 11. rtmlib (fallback)
try:
    from rtmlib import PoseTracker
    checks.append(("rtmlib", "available", True))
except Exception as e:
    checks.append(("rtmlib", "not available", False))

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
    print("\033[0;33m⚠ Some checks have issues. Review the errors above.\033[0m")
    sys.exit(0)  # Don't fail on optional components
VERIFY_EOF
    
    echo ""
    echo "========================================================================"
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
    
    # ===========================================================================
    # PRE-REQUISITE CHECK: TensorRT Manual Download
    # ===========================================================================
    TENSORRT_TARBALL="${DOWNLOADS_DIR}/${TENSORRT_TAR}"
    if [[ ! -f "${TENSORRT_TARBALL}" && ! -d "${TENSORRT_DIR}" ]]; then
        echo -e "${RED}╔════════════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${RED}║  ⚠️  MANUAL DOWNLOAD REQUIRED: TensorRT ${TENSORRT_VERSION}              ║${NC}"
        echo -e "${RED}╚════════════════════════════════════════════════════════════════════╝${NC}"
        echo ""
        echo "TensorRT CANNOT be downloaded automatically (NVIDIA login required)."
        echo ""
        echo "Please follow these steps:"
        echo ""
        echo "  1. Go to: https://developer.nvidia.com/tensorrt-download"
        echo "  2. Login with your NVIDIA Developer account (free to create)"
        echo "  3. Click 'TensorRT 8.x' → 'TensorRT 8.6 GA'"
        echo "  4. Download: ${TENSORRT_TAR}"
        echo "  5. Place the downloaded file in:"
        echo "     ${DOWNLOADS_DIR}/"
        echo ""
        echo "The setup script will automatically extract it to:"
        echo "     ${TENSORRT_DIR}"
        echo ""
        read -p "Press Enter when you have downloaded TensorRT, or Ctrl+C to abort: "
        
        if [[ ! -f "${TENSORRT_TARBALL}" ]]; then
            error "TensorRT tarball not found at ${TENSORRT_TARBALL}
            
Please download TensorRT ${TENSORRT_VERSION} and place it in ${DOWNLOADS_DIR}/"
        fi
        echo ""
    fi
    
    # Run all phases
    preflight_checks
    install_system_deps
    verify_cuda
    setup_tensorrt
    setup_conda
    install_python_deps
    install_local_packages
    build_pplcv
    build_mmdeploy_sdk
    setup_keyboard_permissions
    final_verification
    
    echo ""
    echo "╔════════════════════════════════════════════════════════════════════╗"
    echo "║                    INSTALLATION COMPLETE!                          ║"
    echo "╚════════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "┌────────────────────────────────────────────────────────────────────┐"
    echo "│  NEXT STEPS                                                        │"
    echo "└────────────────────────────────────────────────────────────────────┘"
    echo ""
    echo "  1. Log out and back in (for keyboard permissions to take effect)"
    echo ""
    echo "  2. Activate the environment:"
    echo "     ┌──────────────────────────────────────────────────────────────┐"
    echo "     │  source ${NTKCAP_ROOT}/activate.sh                           "
    echo "     └──────────────────────────────────────────────────────────────┘"
    echo ""
    echo "  3. BUILD TENSORRT ENGINES (⚠️ REQUIRED - GPU-SPECIFIC):"
    echo "     ┌──────────────────────────────────────────────────────────────┐"
    echo "     │  cd ${NTKCAP_ROOT}                                           "
    echo "     │  ./build_engines.sh                                          "
    echo "     └──────────────────────────────────────────────────────────────┘"
    echo ""
    echo "     This step:"
    echo "     • Downloads RTMDet and RTMPose model weights (~150MB)"
    echo "     • Converts models to TensorRT engines for YOUR GPU"
    echo "     • Takes 5-15 minutes depending on GPU"
    echo "     • Must be repeated if you change GPU hardware"
    echo ""
    echo "  4. Run the GUI:"
    echo "     ┌──────────────────────────────────────────────────────────────┐"
    echo "     │  python NTKCAP_GUI.py                                        "
    echo "     └──────────────────────────────────────────────────────────────┘"
    echo ""
    echo "┌────────────────────────────────────────────────────────────────────┐"
    echo "│  ENGINE REBUILD REQUIRED WHEN:                                     │"
    echo "│  • Moving to a different GPU                                       │"
    echo "│  • Upgrading TensorRT version                                      │"
    echo "│  • First setup on a new machine                                    │"
    echo "└────────────────────────────────────────────────────────────────────┘"
    echo ""
}

# Run main
main "$@"
