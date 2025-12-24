#!/bin/bash
################################################################################
# NTKCAP TensorRT Engine Builder
# ================================================================================
# This script builds TensorRT engines for RTMDet and RTMPose models.
#
# WHAT THIS DOES:
#   1. Downloads model weights from OpenMMLab (~150MB total)
#   2. Converts PyTorch models to TensorRT engines
#   3. Optimizes engines for YOUR specific GPU
#
# PREREQUISITES:
#   - Run install/scripts/setup_linux.sh first
#   - TensorRT 8.6.1.6 must be installed
#   - NVIDIA GPU with CUDA support
#
# WHEN TO RUN:
#   ✓ First time setup
#   ✓ After changing GPU hardware
#   ✓ After upgrading TensorRT
#   ✗ NOT needed when just updating code
#
# USAGE:
#   source activate.sh     # Activate environment first!
#   ./build_engines.sh     # Build engines
#
# OUTPUT:
#   NTK_CAP/ThirdParty/mmdeploy/rtmpose-trt/
#   ├── rtmdet-m/end2end.engine    # Person detection (320x320)
#   └── rtmpose-m/end2end.engine   # Pose estimation (256x192)
#
# TIME: 5-15 minutes depending on GPU
################################################################################

set -e

# ==============================================================================
# GET SCRIPT DIRECTORY (All paths relative to this)
# ==============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log() { echo -e "${GREEN}[$(date +'%H:%M:%S')] ✓ $1${NC}"; }
warn() { echo -e "${YELLOW}[$(date +'%H:%M:%S')] ⚠ $1${NC}"; }
error() { echo -e "${RED}[$(date +'%H:%M:%S')] ✗ ERROR: $1${NC}"; exit 1; }
info() { echo -e "${CYAN}[$(date +'%H:%M:%S')] ℹ $1${NC}"; }

echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║           NTKCAP TensorRT Engine Builder                           ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

# ==============================================================================
# RELATIVE PATHS (from project root)
# ==============================================================================
WEIGHTS_DIR="${SCRIPT_DIR}/models/weights"
THIRDPARTY_DIR="${SCRIPT_DIR}/NTK_CAP/ThirdParty"
MMDEPLOY_DIR="${THIRDPARTY_DIR}/mmdeploy"

# TensorRT engine output directories
RTMPOSE_ENGINE_DIR="${MMDEPLOY_DIR}/rtmpose-trt/rtmpose-m"
RTMDET_ENGINE_DIR="${MMDEPLOY_DIR}/rtmpose-trt/rtmdet-m"

# Deploy script
DEPLOY_SCRIPT="${MMDEPLOY_DIR}/tools/deploy.py"

# Config files (mmdeploy configs)
RTMDET_DEPLOY_CFG="${MMDEPLOY_DIR}/configs/mmdet/detection/detection_tensorrt_static-320x320.py"
RTMPOSE_DEPLOY_CFG="${MMDEPLOY_DIR}/configs/mmpose/pose-detection_simcc_tensorrt_dynamic-256x192.py"

# Model configs (from build_engines subdirectory)
RTMDET_MODEL_CFG="${MMDEPLOY_DIR}/build_engines/mmdetection/mmdet/configs/rtmdet/rtmdet_m_8xb32_300e_coco.py"
RTMPOSE_MODEL_CFG="${MMDEPLOY_DIR}/build_engines/mmpose/projects/rtmpose/rtmpose/body_2d_keypoint/rtmpose-m_8xb256-420e_coco-256x192.py"

# Demo images for calibration
DET_IMAGE="${MMDEPLOY_DIR}/demo/resources/det.jpg"
POSE_IMAGE="${MMDEPLOY_DIR}/demo/resources/human-pose.jpg"

# Weight URLs (verified working as of Dec 2024)
RTMDET_WEIGHT_URL="https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_m_8xb32-300e_coco/rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth"
RTMPOSE_WEIGHT_URL="https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192-d8dd5ca4_20230127.pth"

# ==============================================================================
# VERIFICATION
# ==============================================================================
info "Verifying paths..."

# Check deploy script
if [[ ! -f "${DEPLOY_SCRIPT}" ]]; then
    error "Deploy script not found: ${DEPLOY_SCRIPT}
    
Make sure mmdeploy submodule is initialized:
    git submodule update --init --recursive"
fi
log "Deploy script found"

# Check config files
if [[ ! -f "${RTMDET_DEPLOY_CFG}" ]]; then
    error "RTMDet deploy config not found: ${RTMDET_DEPLOY_CFG}"
fi

if [[ ! -f "${RTMPOSE_DEPLOY_CFG}" ]]; then
    error "RTMPose deploy config not found: ${RTMPOSE_DEPLOY_CFG}"
fi

if [[ ! -f "${RTMDET_MODEL_CFG}" ]]; then
    error "RTMDet model config not found: ${RTMDET_MODEL_CFG}"
fi

if [[ ! -f "${RTMPOSE_MODEL_CFG}" ]]; then
    error "RTMPose model config not found: ${RTMPOSE_MODEL_CFG}"
fi
log "Config files verified"

# Check demo images
if [[ ! -f "${DET_IMAGE}" ]]; then
    error "Detection demo image not found: ${DET_IMAGE}"
fi

if [[ ! -f "${POSE_IMAGE}" ]]; then
    error "Pose demo image not found: ${POSE_IMAGE}"
fi
log "Demo images verified"

# ==============================================================================
# CREATE DIRECTORIES
# ==============================================================================
info "Creating directories..."
mkdir -p "${WEIGHTS_DIR}"
mkdir -p "${RTMPOSE_ENGINE_DIR}"
mkdir -p "${RTMDET_ENGINE_DIR}"
log "Directories created"

# ==============================================================================
# DOWNLOAD WEIGHTS
# ==============================================================================
info "Downloading model weights..."

RTMDET_WEIGHT="${WEIGHTS_DIR}/rtmdet_m_coco.pth"
RTMPOSE_WEIGHT="${WEIGHTS_DIR}/rtmpose_m_coco.pth"

if [[ ! -f "${RTMDET_WEIGHT}" ]]; then
    info "Downloading RTMDet-m weights..."
    wget -q --show-progress -O "${RTMDET_WEIGHT}" "${RTMDET_WEIGHT_URL}" || error "Failed to download RTMDet weights"
    log "RTMDet weights downloaded"
else
    log "RTMDet weights already exist"
fi

if [[ ! -f "${RTMPOSE_WEIGHT}" ]]; then
    info "Downloading RTMPose-m weights..."
    wget -q --show-progress -O "${RTMPOSE_WEIGHT}" "${RTMPOSE_WEIGHT_URL}" || error "Failed to download RTMPose weights"
    log "RTMPose weights downloaded"
else
    log "RTMPose weights already exist"
fi

# ==============================================================================
# ENVIRONMENT CHECK
# ==============================================================================
info "Checking environment..."

# Check if conda environment is active
if [[ -z "${CONDA_DEFAULT_ENV}" ]] || [[ "${CONDA_DEFAULT_ENV}" != "ntkcap_env" ]]; then
    warn "ntkcap_env not activated. Attempting to activate..."
    
    # Find and source conda
    for conda_path in "$HOME/miniconda3" "$HOME/anaconda3" "/opt/conda"; do
        if [[ -f "${conda_path}/etc/profile.d/conda.sh" ]]; then
            source "${conda_path}/etc/profile.d/conda.sh"
            break
        fi
    done
    
    conda activate ntkcap_env 2>/dev/null || error "Cannot activate ntkcap_env. Run setup_linux.sh first."
fi
log "Conda environment: ${CONDA_DEFAULT_ENV}"

# Check TensorRT
TENSORRT_DIR="${THIRDPARTY_DIR}/TensorRT-8.6.1.6"
if [[ ! -d "${TENSORRT_DIR}" ]]; then
    error "TensorRT not found at ${TENSORRT_DIR}
    
Run setup_linux.sh first to install TensorRT."
fi

# Set environment variables
export LD_LIBRARY_PATH="${TENSORRT_DIR}/lib:${LD_LIBRARY_PATH:-}"
export TENSORRT_DIR="${TENSORRT_DIR}"
log "TensorRT: ${TENSORRT_DIR}"

# Check CUDA
if ! command -v nvcc &>/dev/null; then
    for cuda_path in "/usr/local/cuda-11.8" "/usr/local/cuda"; do
        if [[ -f "${cuda_path}/bin/nvcc" ]]; then
            export PATH="${cuda_path}/bin:${PATH}"
            export LD_LIBRARY_PATH="${cuda_path}/lib64:${LD_LIBRARY_PATH:-}"
            export CUDA_HOME="${cuda_path}"
            break
        fi
    done
fi
log "CUDA: $(nvcc --version | grep release | sed 's/.*release //' | sed 's/,.*//')"

# Check GPU
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
if [[ -z "${GPU_NAME}" ]]; then
    error "No GPU detected. TensorRT requires NVIDIA GPU."
fi
log "GPU: ${GPU_NAME}"

# ==============================================================================
# BUILD RTMDET ENGINE
# ==============================================================================
echo ""
info "Building RTMDet TensorRT engine..."
info "  Config: detection_tensorrt_static-320x320"
info "  Output: ${RTMDET_ENGINE_DIR}"
echo ""

python "${DEPLOY_SCRIPT}" \
    "${RTMDET_DEPLOY_CFG}" \
    "${RTMDET_MODEL_CFG}" \
    "${RTMDET_WEIGHT}" \
    "${DET_IMAGE}" \
    --work-dir "${RTMDET_ENGINE_DIR}" \
    --device cuda:0 \
    --dump-info

if [[ $? -ne 0 ]]; then
    error "RTMDet engine build failed!"
fi
log "RTMDet engine built successfully"

# ==============================================================================
# BUILD RTMPOSE ENGINE  
# ==============================================================================
echo ""
info "Building RTMPose TensorRT engine..."
info "  Config: pose-detection_simcc_tensorrt_dynamic-256x192"
info "  Output: ${RTMPOSE_ENGINE_DIR}"
echo ""

python "${DEPLOY_SCRIPT}" \
    "${RTMPOSE_DEPLOY_CFG}" \
    "${RTMPOSE_MODEL_CFG}" \
    "${RTMPOSE_WEIGHT}" \
    "${POSE_IMAGE}" \
    --work-dir "${RTMPOSE_ENGINE_DIR}" \
    --device cuda:0 \
    --dump-info

if [[ $? -ne 0 ]]; then
    error "RTMPose engine build failed!"
fi
log "RTMPose engine built successfully"

# ==============================================================================
# VERIFY ENGINES
# ==============================================================================
echo ""
info "Verifying built engines..."

RTMDET_ENGINE=$(find "${RTMDET_ENGINE_DIR}" -name "*.engine" 2>/dev/null | head -1)
RTMPOSE_ENGINE=$(find "${RTMPOSE_ENGINE_DIR}" -name "*.engine" 2>/dev/null | head -1)

if [[ -z "${RTMDET_ENGINE}" ]]; then
    error "RTMDet .engine file not found in ${RTMDET_ENGINE_DIR}"
fi
log "RTMDet engine: $(basename ${RTMDET_ENGINE})"

if [[ -z "${RTMPOSE_ENGINE}" ]]; then
    error "RTMPose .engine file not found in ${RTMPOSE_ENGINE_DIR}"
fi
log "RTMPose engine: $(basename ${RTMPOSE_ENGINE})"

# ==============================================================================
# DONE
# ==============================================================================
echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║           TensorRT Engines Built Successfully!                     ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Engine locations:"
echo "  RTMDet:  ${RTMDET_ENGINE_DIR}"
echo "  RTMPose: ${RTMPOSE_ENGINE_DIR}"
echo ""
echo "To use these engines, ensure you run:"
echo "  source ${SCRIPT_DIR}/activate.sh"
echo ""
