#!/bin/bash
################################################################################
# NTKCAP Master Installation Script
# ================================================================================
# This script clones the NTKCAP repository and runs the complete installation.
#
# PREREQUISITES:
#   - Ubuntu 20.04/22.04 with NVIDIA GPU
#   - CUDA 11.8 installed
#   - cuDNN 8.x installed
#   - Git installed
#
# USAGE:
#   curl -sSL https://raw.githubusercontent.com/adigokul/NTKCAP/ntkcaptensor/install/master_install.sh | bash
#
#   OR
#
#   wget -qO- https://raw.githubusercontent.com/adigokul/NTKCAP/ntkcaptensor/install/master_install.sh | bash
#
################################################################################

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}"
echo "╔═══════════════════════════════════════════════════════════════════════════╗"
echo "║                    NTKCAP Master Installation Script                       ║"
echo "║                  Motion Capture with TensorRT Acceleration                 ║"
echo "╚═══════════════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Default installation directory
DEFAULT_INSTALL_DIR="${HOME}/ntkcaptensor"
REPO_URL="https://github.com/adigokul/NTKCAP-public.git"
BRANCH="main"

# Ask for installation directory
echo -e "${YELLOW}Where do you want to install NTKCAP?${NC}"
echo -e "Default: ${DEFAULT_INSTALL_DIR}"
read -p "Press Enter to use default or type a new path: " INSTALL_DIR
INSTALL_DIR="${INSTALL_DIR:-$DEFAULT_INSTALL_DIR}"

# Expand ~ if used
INSTALL_DIR="${INSTALL_DIR/#\~/$HOME}"

echo ""
echo -e "${GREEN}Installation directory: ${INSTALL_DIR}${NC}"
echo ""

# Check prerequisites
echo -e "${BLUE}Checking prerequisites...${NC}"

# Check git
if ! command -v git &>/dev/null; then
    echo -e "${RED}Error: Git is not installed.${NC}"
    echo "Please install git first: sudo apt-get install git"
    exit 1
fi
echo -e "  ${GREEN}✓${NC} Git found"

# Check CUDA
if ! command -v nvcc &>/dev/null; then
    # Try common CUDA paths
    for cuda_path in "/usr/local/cuda-11.8" "/usr/local/cuda"; do
        if [[ -f "${cuda_path}/bin/nvcc" ]]; then
            export PATH="${cuda_path}/bin:${PATH}"
            break
        fi
    done
fi

if ! command -v nvcc &>/dev/null; then
    echo -e "${RED}Error: CUDA is not installed or nvcc not in PATH.${NC}"
    echo "Please install CUDA 11.8 first."
    exit 1
fi
echo -e "  ${GREEN}✓${NC} CUDA found: $(nvcc --version | grep release | sed 's/.*release //' | sed 's/,.*//')"

# Check nvidia-smi
if ! command -v nvidia-smi &>/dev/null; then
    echo -e "${RED}Error: NVIDIA drivers not installed.${NC}"
    exit 1
fi
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo -e "  ${GREEN}✓${NC} GPU found: ${GPU_NAME}"

# Check conda
if ! command -v conda &>/dev/null; then
    echo -e "${RED}Error: Conda is not installed.${NC}"
    echo "Please install Miniconda or Anaconda first."
    echo "Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi
echo -e "  ${GREEN}✓${NC} Conda found: $(conda --version)"

echo ""
echo -e "${GREEN}All prerequisites satisfied!${NC}"
echo ""

# Confirm installation
echo -e "${YELLOW}Ready to install NTKCAP.${NC}"
echo "This will:"
echo "  1. Clone the repository to ${INSTALL_DIR}"
echo "  2. Run the complete installation (~30-60 minutes)"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Installation cancelled."
    exit 0
fi

# Clone repository
echo ""
echo -e "${BLUE}Cloning NTKCAP repository...${NC}"

if [[ -d "${INSTALL_DIR}" ]]; then
    echo -e "${YELLOW}Directory already exists: ${INSTALL_DIR}${NC}"
    read -p "Delete and re-clone? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "${INSTALL_DIR}"
    else
        echo "Using existing directory."
    fi
fi

if [[ ! -d "${INSTALL_DIR}" ]]; then
    git clone --branch "${BRANCH}" "${REPO_URL}" "${INSTALL_DIR}"
    echo -e "${GREEN}Repository cloned successfully!${NC}"
else
    cd "${INSTALL_DIR}"
    git fetch origin
    git checkout "${BRANCH}"
    git pull origin "${BRANCH}"
    echo -e "${GREEN}Repository updated!${NC}"
fi

# Run installation script
echo ""
echo -e "${BLUE}Starting NTKCAP installation...${NC}"
echo ""

cd "${INSTALL_DIR}/install"
chmod +x install_ntkcap.sh
./install_ntkcap.sh

echo ""
echo -e "${GREEN}╔═══════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                    NTKCAP Installation Complete!                          ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "To activate the environment:"
echo "  source ${INSTALL_DIR}/install/activate_ntkcap.sh"
echo ""
echo "Or manually:"
echo "  conda activate NTKCAP"
echo ""
