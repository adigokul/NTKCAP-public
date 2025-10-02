#!/bin/bash
# NTKCAP Complete Environment Setup Script
# This script sets up the complete NTKCAP environment with Poetry, CUDA, and all dependencies
# Requirements: Ubuntu 22.04 LTS

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}"
    exit 1
}

info() {
    echo -e "${BLUE}[INFO] $1${NC}"
}

# System requirements check
check_system_requirements() {
    log "Checking system requirements..."
    
    # Check Ubuntu version
    if ! grep -q "Ubuntu 22.04" /etc/os-release; then
        warn "This script is designed for Ubuntu 22.04 LTS"
        warn "Current system: $(lsb_release -d | cut -f2)"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            error "Please use Ubuntu 22.04 LTS for best compatibility"
        fi
    else
        info "✅ Ubuntu 22.04 LTS detected"
    fi
    
    # Check if running as root
    if [[ $EUID -eq 0 ]]; then
        error "This script should not be run as root. Please run as a regular user with sudo privileges."
    fi
    
    # Check sudo privileges
    if ! sudo -n true 2>/dev/null; then
        warn "This script requires sudo privileges. You may be prompted for your password."
    fi
    
    # Check internet connection
    if ! ping -c 1 google.com &> /dev/null; then
        error "Internet connection required for downloading dependencies"
    fi
    
    info "✅ System requirements check passed"
}

# Setup CUDA environment variables
setup_cuda_environment() {
    log "Setting up CUDA environment variables..."
    
    # Find CUDA installation
    CUDA_PATHS=("/usr/local/cuda-11.8" "/usr/local/cuda" "/opt/cuda")
    CUDA_PATH=""
    
    for path in "${CUDA_PATHS[@]}"; do
        if [[ -d "$path" && -f "$path/bin/nvcc" ]]; then
            CUDA_PATH="$path"
            break
        fi
    done
    
    if [[ -z "$CUDA_PATH" ]]; then
        warn "CUDA installation directory not found"
        return 1
    fi
    
    info "Found CUDA at: $CUDA_PATH"
    
    # Set environment variables for current session
    export PATH="$CUDA_PATH/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_PATH/lib64:$LD_LIBRARY_PATH"
    export CUDA_HOME="$CUDA_PATH"
    
    # Add to ~/.bashrc if not already present
    if ! grep -q "$CUDA_PATH/bin" ~/.bashrc; then
        echo "export PATH=\"$CUDA_PATH/bin:\$PATH\"" >> ~/.bashrc
        echo "export LD_LIBRARY_PATH=\"$CUDA_PATH/lib64:\$LD_LIBRARY_PATH\"" >> ~/.bashrc
        echo "export CUDA_HOME=\"$CUDA_PATH\"" >> ~/.bashrc
        info "✅ CUDA environment variables added to ~/.bashrc"
    else
        info "✅ CUDA environment variables already in ~/.bashrc"
    fi
}

# Setup CUDA environment variables
setup_cuda_environment() {
    log "Setting up CUDA environment variables..."
    
    # Find CUDA installation
    CUDA_PATHS=("/usr/local/cuda-11.8" "/usr/local/cuda" "/opt/cuda")
    CUDA_PATH=""
    
    for path in "${CUDA_PATHS[@]}"; do
        if [[ -d "$path" && -f "$path/bin/nvcc" ]]; then
            CUDA_PATH="$path"
            break
        fi
    done
    
    if [[ -z "$CUDA_PATH" ]]; then
        warn "CUDA installation directory not found"
        return 1
    fi
    
    info "Found CUDA at: $CUDA_PATH"
    
    # Set environment variables for current session
    export PATH="$CUDA_PATH/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_PATH/lib64:$LD_LIBRARY_PATH"
    export CUDA_HOME="$CUDA_PATH"
    
    # Add to ~/.bashrc if not already present
    if ! grep -q "$CUDA_PATH/bin" ~/.bashrc; then
        echo "export PATH=\"$CUDA_PATH/bin:\$PATH\"" >> ~/.bashrc
        echo "export LD_LIBRARY_PATH=\"$CUDA_PATH/lib64:\$LD_LIBRARY_PATH\"" >> ~/.bashrc
        echo "export CUDA_HOME=\"$CUDA_PATH\"" >> ~/.bashrc
        info "✅ CUDA environment variables added to ~/.bashrc"
    else
        info "✅ CUDA environment variables already in ~/.bashrc"
    fi
}

# Update system packages
update_system() {
    log "Updating system packages..."
    sudo apt update
    sudo apt upgrade -y
    
    # Install essential development tools and dependencies
    sudo apt install -y wget curl git build-essential software-properties-common \
                       python3-dev python3-pip lsb-release ca-certificates \
                       pkg-config cmake ninja-build
    
    info "✅ System packages updated"
}

# Install Miniconda
install_miniconda() {
    log "Checking Miniconda installation..."
    
    # Check if conda is already available
    if command -v conda &> /dev/null; then
        info "✅ Conda already installed: $(conda --version)"
        # Ensure conda is initialized
        if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
            source "$HOME/miniconda3/etc/profile.d/conda.sh"
        fi
        return
    fi
    
    # Check if miniconda directory exists but not in PATH
    if [[ -d "$HOME/miniconda3" ]] && [[ -f "$HOME/miniconda3/bin/conda" ]]; then
        info "✅ Miniconda found but not in PATH. Adding to environment..."
        export PATH="$HOME/miniconda3/bin:$PATH"
        # Add to bashrc if not already there
        if ! grep -q 'miniconda3/bin' ~/.bashrc; then
            echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc
        fi
        $HOME/miniconda3/bin/conda init bash
        source ~/.bashrc
        return
    fi
    
    log "Installing Miniconda..."
    
    # Create miniconda3 directory
    mkdir -p ~/miniconda3
    
    # Download if not already present
    if [[ ! -f "~/miniconda3/miniconda.sh" ]]; then
        log "Downloading Miniconda installer..."
        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
    else
        info "Miniconda installer already downloaded"
    fi
    
    # Install using standard procedure
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    
    # Clean up installer
    rm ~/miniconda3/miniconda.sh
    
    # Add conda to PATH
    export PATH="$HOME/miniconda3/bin:$PATH"
    if ! grep -q 'miniconda3/bin' ~/.bashrc; then
        echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc
    fi
    
    # Initialize conda
    $HOME/miniconda3/bin/conda init bash
    
    # Source the conda setup
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    
    info "✅ Miniconda installed successfully"
}

# Install Poetry
install_poetry() {
    log "Checking Poetry installation..."
    
    # Add Poetry to PATH for this session
    export PATH="$HOME/.local/bin:$PATH"
    
    if command -v poetry &> /dev/null; then
        info "✅ Poetry already installed: $(poetry --version)"
        # Configure Poetry if not already configured
        poetry config virtualenvs.create false 2>/dev/null || true
        poetry config installer.max-workers 10 2>/dev/null || true
        return
    fi
    
    log "Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
    
    # Add Poetry to PATH permanently
    if ! grep -q '.local/bin' ~/.bashrc; then
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
    fi
    
    # Verify installation
    if ! command -v poetry &> /dev/null; then
        error "Poetry installation failed"
    fi
    
    # Configure Poetry
    poetry config virtualenvs.create false
    poetry config installer.max-workers 10
    
    info "✅ Poetry installed successfully: $(poetry --version)"
}

# Install NVIDIA drivers and CUDA
install_nvidia_cuda() {
    log "Checking NVIDIA GPU and drivers..."
    
    # Check if NVIDIA GPU exists
    if ! lspci | grep -i nvidia &> /dev/null; then
        warn "No NVIDIA GPU detected. Skipping NVIDIA driver installation."
        return
    fi
    
    info "NVIDIA GPU detected: $(lspci | grep -i nvidia | head -1)"
    
    # Check if nvidia-smi works
    if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
        info "✅ NVIDIA drivers already working: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -1)"
        
        # Check CUDA
        if command -v nvcc &> /dev/null; then
            info "✅ CUDA already installed: $(nvcc --version | grep release | awk '{print $6}' | cut -c2-)"
            return
        fi
    fi
    
    log "Installing NVIDIA drivers and CUDA 11.8..."
    
    # Install required packages for compilation
    sudo apt install -y gcc-12 g++-12 make dkms
    
    # Set GCC 12 as default (required for recent NVIDIA drivers)
    sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 120 --slave /usr/bin/g++ g++ /usr/bin/g++-12
    
    # Add NVIDIA package repository if not already added
    if ! apt-cache policy | grep -q "developer.download.nvidia.com"; then
        log "Adding NVIDIA repository..."
        wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
        sudo dpkg -i cuda-keyring_1.0-1_all.deb
        sudo apt-get update
        rm -f cuda-keyring_1.0-1_all.deb
    else
        info "NVIDIA repository already configured"
    fi
    
    # Install CUDA using standard procedure
    sudo apt-get -y install cuda
    
    # Set up CUDA environment variables
    setup_cuda_environment
    
    # Verify CUDA installation
    if [[ -f "/usr/local/cuda-11.8/bin/nvcc" ]]; then
        info "✅ CUDA 11.8 nvcc compiler found"
    else
        warn "CUDA compiler not found, checking for any CUDA version..."
        if command -v nvcc &> /dev/null; then
            info "✅ CUDA compiler found: $(nvcc --version | grep release || echo 'Version unknown')"
        fi
    fi
    
    warn "⚠️  NVIDIA drivers installed. A system reboot is required to activate the drivers."
    info "✅ CUDA 11.8 installation completed"
}

# Create and setup conda environment
setup_conda_environment() {
    log "Setting up conda environment..."
    
    # Source conda
    source $HOME/miniconda3/etc/profile.d/conda.sh
    
    # Check if environment exists
    if conda env list | grep -q "ntkcap"; then
        info "✅ ntkcap environment already exists"
        conda activate ntkcap
    else
        log "Creating ntkcap conda environment from environment.yml..."
        cd /home/ntk/NTKCAP
        
        # Use environment.yml if it exists, otherwise create manually
        if [[ -f "install/environment.yml" ]]; then
            conda env create -f install/environment.yml
        else
            # Fallback to manual creation
            conda create -n ntkcap python=3.10 -y
        fi
        
        conda activate ntkcap
        info "✅ ntkcap environment created"
    fi
    
    # Install additional conda packages (avoid duplicates with Poetry)
    log "Installing conda-specific packages..."
    
    # Check and install opensim if not already present  
    if ! conda list | grep -q "^opensim "; then
        conda install -c opensim-org opensim=4.5 -y
    else
        info "✅ opensim already installed via conda"
    fi
    
    # OpenCV will be installed via Poetry to avoid conflicts
    
    info "✅ Conda environment setup completed"
}

# Install Python dependencies with Poetry
install_python_dependencies() {
    log "Installing Python dependencies with Poetry..."
    
    # Make sure we're in the project directory
    cd /home/ntk/NTKCAP
    
    # Source conda environment and set up paths
    source $HOME/miniconda3/etc/profile.d/conda.sh
    conda activate ntkcap
    
    # Ensure Poetry is in PATH
    export PATH="$HOME/.local/bin:$PATH"
    
    # Set up CUDA environment for this session
    setup_cuda_environment
    
    # Check if pyproject.toml exists
    if [[ ! -f "pyproject.toml" ]]; then
        error "pyproject.toml not found. Please ensure you're in the correct NTKCAP directory."
    fi
    
    # Install dependencies
    log "Installing Poetry dependencies..."
    poetry install --only=main
    
    # Install MMEngine ecosystem with better error handling
    log "Installing MMEngine ecosystem..."
    
    # Install openmim
    if ! python -c "import mim" &> /dev/null; then
        log "Installing openmim..."
        poetry run pip install -U openmim
    else
        info "✅ openmim already installed"
    fi
    
    # Install mmengine
    if ! python -c "import mmengine" &> /dev/null; then
        log "Installing mmengine..."
        poetry run mim install mmengine
    else
        info "✅ mmengine already installed"
    fi
    
    # Install mmcv
    if ! python -c "import mmcv" &> /dev/null; then
        log "Installing mmcv..."
        poetry run mim install "mmcv==2.1.0"
    else
        info "✅ mmcv already installed"
    fi
    
    # Install mmdet
    if ! python -c "import mmdet" &> /dev/null; then
        log "Installing mmdet..."
        poetry run mim install "mmdet>=3.3.0"
    else
        info "✅ mmdet already installed"
    fi
    
    # Install local packages
    log "Installing local packages..."
    
    # mmpose
    if [[ -d "NTK_CAP/ThirdParty/mmpose" ]]; then
        cd NTK_CAP/ThirdParty/mmpose
        if [[ -f "requirements.txt" ]]; then
            poetry run pip install -r requirements.txt
        fi
        poetry run pip install -v -e .
        cd ../../..
        info "✅ mmpose installed"
    else
        warn "mmpose directory not found"
    fi
    
    # EasyMocap
    if [[ -d "NTK_CAP/ThirdParty/EasyMocap" ]]; then
        cd NTK_CAP/ThirdParty/EasyMocap
        poetry run python setup.py develop --user
        cd ../../..
        info "✅ EasyMocap installed"
    else
        warn "EasyMocap directory not found"
    fi
    
    info "✅ Python dependencies installed successfully"
}

# Create activation script
create_activation_script() {
    log "Creating environment activation script..."
    
    cat > /home/ntk/NTKCAP/install/activate_ntkcap.sh << 'EOF'
#!/bin/bash
# NTKCAP Environment Activation Script

# Find and set CUDA environment variables
CUDA_PATHS=("/usr/local/cuda-11.8" "/usr/local/cuda" "/opt/cuda")
for path in "${CUDA_PATHS[@]}"; do
    if [[ -d "$path" && -f "$path/bin/nvcc" ]]; then
        export PATH="$path/bin:$PATH"
        export LD_LIBRARY_PATH="$path/lib64:$LD_LIBRARY_PATH"
        export CUDA_HOME="$path"
        echo "🔧 CUDA found at: $path"
        break
    fi
done

# Add Poetry to PATH
export PATH="$HOME/.local/bin:$PATH"

# Activate conda environment
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate ntkcap
    echo "🚀 NTKCAP environment activated!"
    echo "Current Python: $(which python)"
    echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'PyTorch not loaded')"
    echo ""
    echo "Available commands:"
    echo "  python final_NTK_Cap_GUI.py   # Run main GUI"
    echo "  python NTKCAP_GUI.py          # Run alternative GUI"
    echo "  poetry run <command>          # Run with Poetry"
else
    echo "❌ Conda not found. Please install Miniconda first."
    exit 1
fi

# Start a new shell with the environment
exec bash
EOF
    
    chmod +x /home/ntk/NTKCAP/install/activate_ntkcap.sh
    
    # Also create a symlink in the main directory for convenience
    ln -sf install/activate_ntkcap.sh /home/ntk/NTKCAP/activate_ntkcap.sh
    
    info "✅ Activation script created: install/activate_ntkcap.sh"
    info "✅ Symlink created: activate_ntkcap.sh -> install/activate_ntkcap.sh"
}

# Main installation function
main() {
    echo "======================================"
    echo "  NTKCAP Complete Environment Setup  "
    echo "======================================"
    echo ""
    
    check_system_requirements
    update_system
    install_miniconda
    install_poetry
    install_nvidia_cuda
    setup_conda_environment
    install_python_dependencies
    create_activation_script
    
    echo ""
    echo "======================================"
    echo "        Installation Complete!       "
    echo "======================================"
    echo ""
    
    if lspci | grep -i nvidia &> /dev/null; then
        warn "⚠️  Important: If NVIDIA drivers were installed, please REBOOT your system now!"
        echo ""
        echo "After reboot, test your installation:"
        echo "  cd /home/ntk/NTKCAP/install/scripts"
        echo "  ./test_installation.sh"
    else
        echo "Test your installation:"
        echo "  cd /home/ntk/NTKCAP/install/scripts"
        echo "  ./test_installation.sh"
    fi
    
    echo ""
    echo "To activate the environment manually:"
    echo "  cd /home/ntk/NTKCAP"
    echo "  ./activate_ntkcap.sh"
    echo "  # OR from install directory:"
    echo "  ./install/activate_ntkcap.sh"
    echo ""
    echo "🎉 NTKCAP environment setup completed successfully!"
}

# Run main function
main "$@"