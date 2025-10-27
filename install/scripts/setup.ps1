# NTKCAP Complete Environment Setup Script for Windows
# This script sets up the complete NTKCAP environment with Poetry, CUDA, and all dependencies
# Requirements: Windows 10/11 with Anaconda/Miniconda

param(
    [switch]$SkipCudaCheck = $false
)

# Set error action preference
$ErrorActionPreference = "Stop"

# Colors for output
function Write-Log {
    param([string]$Message)
    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $Message" -ForegroundColor Green
}

function Write-Warning-Custom {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

function Write-Error-Custom {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
    exit 1
}

function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Cyan
}

# Check system requirements
function Test-SystemRequirements {
    Write-Log "Checking system requirements..."
    
    # Check Windows version
    $osVersion = [System.Environment]::OSVersion.Version
    if ($osVersion.Major -lt 10) {
        Write-Error-Custom "Windows 10 or later is required"
    }
    Write-Info "✅ Windows $($osVersion.Major).$($osVersion.Minor) detected"
    
    # Check if running as Administrator
    $currentPrincipal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
    if (-not $currentPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        Write-Warning-Custom "Running without administrator privileges. Some operations may fail."
    }
    
    # Check conda installation
    if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
        Write-Error-Custom "Conda not found. Please install Anaconda or Miniconda first."
    }
    Write-Info "✅ Conda found: $(conda --version)"
    
    Write-Info "✅ System requirements check passed"
}

# Check CUDA installation
function Test-CudaInstallation {
    Write-Log "Checking CUDA installation..."
    
    # Check for NVIDIA GPU
    try {
        $gpu = Get-WmiObject Win32_VideoController | Where-Object { $_.Name -like "*NVIDIA*" }
        if ($null -eq $gpu) {
            Write-Warning-Custom "No NVIDIA GPU detected"
            return $false
        }
        Write-Info "✅ NVIDIA GPU detected: $($gpu.Name)"
    }
    catch {
        Write-Warning-Custom "Could not check for NVIDIA GPU"
        return $false
    }
    
    # Check nvidia-smi
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        $nvidiaVersion = nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits 2>$null | Select-Object -First 1
        Write-Info "✅ NVIDIA drivers installed: Version $nvidiaVersion"
    }
    else {
        Write-Warning-Custom "nvidia-smi not found. NVIDIA drivers may not be installed."
        return $false
    }
    
    # Check CUDA
    if (Get-Command nvcc -ErrorAction SilentlyContinue) {
        $cudaVersion = nvcc --version 2>$null | Select-String "release" | ForEach-Object { $_.Line }
        Write-Info "✅ CUDA installed: $cudaVersion"
        return $true
    }
    else {
        Write-Warning-Custom "CUDA (nvcc) not found in PATH"
        
        # Check common CUDA installation paths
        $cudaPaths = @(
            "$env:ProgramFiles\NVIDIA GPU Computing Toolkit\CUDA\v11.8",
            "$env:ProgramFiles\NVIDIA GPU Computing Toolkit\CUDA\v12.0",
            "$env:ProgramFiles\NVIDIA GPU Computing Toolkit\CUDA\v12.1"
        )
        
        foreach ($path in $cudaPaths) {
            if (Test-Path "$path\bin\nvcc.exe") {
                Write-Info "✅ CUDA found at: $path"
                Write-Info "Adding CUDA to PATH for this session..."
                $env:Path = "$path\bin;$env:Path"
                $env:CUDA_PATH = $path
                return $true
            }
        }
        
        Write-Warning-Custom "CUDA installation not found. PyTorch will use CPU only."
        Write-Info "To install CUDA, visit: https://developer.nvidia.com/cuda-downloads"
        return $false
    }
}

# Install Poetry
function Install-Poetry {
    Write-Log "Checking Poetry installation..."
    
    if (Get-Command poetry -ErrorAction SilentlyContinue) {
        Write-Info "✅ Poetry already installed: $(poetry --version)"
        # Configure Poetry
        poetry config virtualenvs.create false 2>$null
        poetry config keyring.enabled false 2>$null
        return
    }
    
    Write-Log "Installing Poetry..."
    
    try {
        # Install using official installer
        (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
        
        # Add Poetry to PATH for current session
        $poetryPath = "$env:APPDATA\Python\Scripts"
        if (Test-Path $poetryPath) {
            $env:Path = "$poetryPath;$env:Path"
        }
        
        # Verify installation
        if (Get-Command poetry -ErrorAction SilentlyContinue) {
            Write-Info "✅ Poetry installed successfully: $(poetry --version)"
            
            # Configure Poetry
            poetry config virtualenvs.create false
            poetry config keyring.enabled false
            
            Write-Info "Poetry has been added to PATH for this session."
            Write-Warning-Custom "You may need to restart PowerShell for Poetry to be available in new sessions."
        }
        else {
            Write-Error-Custom "Poetry installation failed. Please install manually."
        }
    }
    catch {
        Write-Error-Custom "Failed to install Poetry: $_"
    }
}

# Create conda environment from environment.yml
function New-CondaEnvironment {
    Write-Log "Setting up conda environment..."
    
    # Change to NTKCAP directory
    $ntkcapPath = "D:\NTKCAP"
    if (-not (Test-Path $ntkcapPath)) {
        Write-Error-Custom "NTKCAP directory not found at $ntkcapPath"
    }
    
    Set-Location $ntkcapPath
    
    # Check if environment already exists
    $envExists = conda env list | Select-String "ntkcap"
    if ($envExists) {
        Write-Info "ntkcap environment already exists. Removing it..."
        conda env remove -n ntkcap -y
    }
    
    # Create environment from YAML
    Write-Log "Creating ntkcap conda environment from environment.yml..."
    
    if (Test-Path "install\environment.yml") {
        conda env create -f install\environment.yml
    }
    else {
        Write-Error-Custom "install\environment.yml not found"
    }
    
    Write-Info "✅ ntkcap environment created"
}

# Install Python dependencies with Poetry
function Install-PythonDependencies {
    Write-Log "Installing Python dependencies with Poetry..."
    
    # Activate conda environment
    conda activate ntkcap
    
    # Check if pyproject.toml exists
    if (-not (Test-Path "pyproject.toml")) {
        Write-Error-Custom "pyproject.toml not found in current directory"
    }
    
    # Install main dependencies
    Write-Log "Installing Poetry dependencies (this may take a while)..."
    try {
        poetry install --only=main
        Write-Info "✅ Main dependencies installed"
    }
    catch {
        Write-Warning-Custom "Poetry install encountered issues. Continuing with manual installation..."
    }
    
    # Install MMEngine ecosystem
    Write-Log "Installing MMEngine ecosystem..."
    
    # Install openmim
    Write-Log "Installing openmim..."
    pip install -U openmim
    
    # Install mmengine
    Write-Log "Installing mmengine..."
    mim install mmengine
    
    # Install mmcv
    Write-Log "Installing mmcv..."
    mim install "mmcv==2.1.0"
    
    # Install mmdet
    Write-Log "Installing mmdet..."
    mim install "mmdet>=3.3.0"
    
    Write-Info "✅ MMEngine ecosystem installed"
    
    # Install local packages
    Write-Log "Installing local packages..."
    
    # mmpose
    if (Test-Path "NTK_CAP\ThirdParty\mmpose") {
        Write-Log "Installing mmpose..."
        Push-Location "NTK_CAP\ThirdParty\mmpose"
        
        if (Test-Path "requirements.txt") {
            pip install -r requirements.txt
        }
        pip install -v -e .
        
        Pop-Location
        Write-Info "✅ mmpose installed"
    }
    else {
        Write-Warning-Custom "mmpose directory not found"
    }
    
    # EasyMocap
    if (Test-Path "NTK_CAP\ThirdParty\EasyMocap") {
        Write-Log "Installing EasyMocap..."
        Push-Location "NTK_CAP\ThirdParty\EasyMocap"
        
        python setup.py develop --user
        
        Pop-Location
        Write-Info "✅ EasyMocap installed"
    }
    else {
        Write-Warning-Custom "EasyMocap directory not found"
    }
    
    Write-Info "✅ Python dependencies installed successfully"
}

# Create activation script
function New-ActivationScript {
    Write-Log "Creating environment activation script..."
    
    $activationScript = @'
# NTKCAP Environment Activation Script for Windows
# Run this script to activate the NTKCAP environment

Write-Host "🚀 Activating NTKCAP environment..." -ForegroundColor Green

# Activate conda environment
conda activate ntkcap

# Check CUDA availability
if (Get-Command nvcc -ErrorAction SilentlyContinue) {
    $cudaVersion = nvcc --version 2>$null | Select-String "release"
    Write-Host "✅ CUDA available: $cudaVersion" -ForegroundColor Green
}
else {
    Write-Host "⚠️  CUDA not found in PATH" -ForegroundColor Yellow
}

# Show Python version
Write-Host "✅ Python: $(python --version)" -ForegroundColor Green

# Check PyTorch CUDA
try {
    $torchCuda = python -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>$null
    Write-Host "✅ PyTorch: $torchCuda" -ForegroundColor Green
}
catch {
    Write-Host "⚠️  PyTorch not installed or error checking CUDA" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "🎉 NTKCAP environment activated!" -ForegroundColor Cyan
Write-Host ""
Write-Host "Available commands:" -ForegroundColor Cyan
Write-Host "  python NTKCAP_GUI.py          # Run main GUI" -ForegroundColor White
Write-Host "  python final_NTK_Cap_GUI.py   # Run alternative GUI" -ForegroundColor White
Write-Host "  poetry run <command>          # Run with Poetry" -ForegroundColor White
Write-Host ""
'@
    
    $activationScript | Out-File -FilePath "install\activate_ntkcap.ps1" -Encoding UTF8
    
    # Create symlink in main directory
    if (Test-Path "activate_ntkcap.ps1") {
        Remove-Item "activate_ntkcap.ps1" -Force
    }
    Copy-Item "install\activate_ntkcap.ps1" "activate_ntkcap.ps1"
    
    Write-Info "✅ Activation script created: install\activate_ntkcap.ps1"
    Write-Info "✅ Copy created: activate_ntkcap.ps1"
}

# Main installation function
function Start-Installation {
    Write-Host "======================================" -ForegroundColor Cyan
    Write-Host "  NTKCAP Complete Environment Setup  " -ForegroundColor Cyan
    Write-Host "  for Windows                        " -ForegroundColor Cyan
    Write-Host "======================================" -ForegroundColor Cyan
    Write-Host ""
    
    Test-SystemRequirements
    
    if (-not $SkipCudaCheck) {
        $cudaAvailable = Test-CudaInstallation
        if (-not $cudaAvailable) {
            Write-Warning-Custom "Continuing without CUDA support"
        }
    }
    
    Install-Poetry
    New-CondaEnvironment
    Install-PythonDependencies
    New-ActivationScript
    
    Write-Host ""
    Write-Host "======================================" -ForegroundColor Green
    Write-Host "        Installation Complete!       " -ForegroundColor Green
    Write-Host "======================================" -ForegroundColor Green
    Write-Host ""
    
    Write-Host "To activate the environment:" -ForegroundColor Cyan
    Write-Host "  cd D:\NTKCAP" -ForegroundColor White
    Write-Host "  .\activate_ntkcap.ps1" -ForegroundColor White
    Write-Host "  # OR from install directory:" -ForegroundColor Gray
    Write-Host "  .\install\activate_ntkcap.ps1" -ForegroundColor White
    Write-Host ""
    Write-Host "🎉 NTKCAP environment setup completed successfully!" -ForegroundColor Green
}

# Run main installation
Start-Installation
