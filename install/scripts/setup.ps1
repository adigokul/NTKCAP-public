# NTKCAP Complete Environment Setup Script for Windows
# This script sets up the complete NTKCAP environment with Poetry or direct pip, CUDA, and all dependencies
# Requirements: Windows 10/11 with Anaconda/Miniconda
#
# Usage:
#   .\setup.ps1                              # Interactive installation (prompts for environment name)
#   .\setup.ps1 -UseDirectPip                # Use direct pip instead of Poetry  
#   .\setup.ps1 -CondaEnvName "my_env"       # Use specific environment name (skip prompt)
#   .\setup.ps1 -SkipCudaCheck               # Skip CUDA installation check
#   .\setup.ps1 -SkipPoetry                  # Skip Poetry dependencies
#   .\setup.ps1 -ForceRecreateEnv            # Force recreate existing environment
#   .\setup.ps1 -SkipTensorRTDeploy          # Skip TensorRT model deployment
#   .\setup.ps1 -AutoYes                     # Auto-answer yes to all prompts (uses default env name)
#   .\setup.ps1 -CondaEnvName "ntkcap_fast" -UseDirectPip -AutoYes  # Fully automated
#
# Functions:
# Test-SystemRequirements - 系統需求檢查
# Test-CudaInstallation - CUDA 檢查
# New-CondaEnvironment - 建立 conda 環境 (支援自訂名稱)
# Ensure-CondaEnvironment - 確保在正確環境
# pip install torch - 安裝 PyTorch (在 Check-Poetry 之前)
# Check-Poetry - 檢查並安裝 Poetry
# Install-MMComponents - 安裝 MM 生態系、MMPose、mmdeploy、TensorRT wheels、部署模型、EasyMocap
# Install-PoetryDependencies - Poetry 安裝依賴
# Install-DirectPipDependencies - 直接 pip 安裝依賴 (替代 Poetry)
# Apply-CompatibilityFixes - 最後的 pip 補正（Pose2Sim, PyQt6, numpy==1.22.4）
# New-ActivationScript - 建立啟動腳本

param(
    [switch]$SkipCudaCheck = $false,
    [switch]$SkipPoetry = $false,
    [switch]$UseDirectPip = $false,
    [switch]$ForceRecreateEnv = $false,
    [switch]$SkipTensorRTDeploy = $false,
    [switch]$AutoYes = $false,
    [string]$CondaEnvName = ""
)

# Set error action preference
$ErrorActionPreference = "Stop"

# Colors for output functions - Define early for use throughout script
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

# ==================== Global Configuration ====================
# Root directory - Change this if NTKCAP is installed elsewhere
$NTKCAP_ROOT = "D:\NTKCAP"
# Conda environment name - Handle interactive input or parameter
if ([string]::IsNullOrWhiteSpace($CondaEnvName)) {
    if ($AutoYes) {
        $ENV_NAME = "ntkcap_env"
        Write-Info "Using default environment name: ntkcap_env (--AutoYes flag enabled)"
    } else {
        Write-Host ""
        Write-Host "Environment Name Configuration" -ForegroundColor Cyan
        Write-Host "==============================" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "Please specify a name for your conda environment." -ForegroundColor White
        Write-Host "This allows you to have multiple NTKCAP installations." -ForegroundColor White
        Write-Host ""
        Write-Host "Examples:" -ForegroundColor Gray
        Write-Host "  ntkcap_env      (default)" -ForegroundColor Gray
        Write-Host "  ntkcap_dev      (for development)" -ForegroundColor Gray
        Write-Host "  ntkcap_prod     (for production)" -ForegroundColor Gray
        Write-Host "  ntkcap_v2       (version 2)" -ForegroundColor Gray
        Write-Host ""
        
        do {
            $userInput = Read-Host "Enter environment name (press Enter for default 'ntkcap_env')"
            if ([string]::IsNullOrWhiteSpace($userInput)) {
                $ENV_NAME = "ntkcap_env"
                Write-Info "Using default environment name: ntkcap_env"
                break
            } else {
                $ENV_NAME = $userInput.Trim()
                # Validate environment name
                if ($ENV_NAME -match "^[a-zA-Z0-9_-]+$" -and $ENV_NAME.Length -ge 3 -and $ENV_NAME.Length -le 50) {
                    Write-Info "Using environment name: $ENV_NAME"
                    break
                } else {
                    Write-Warning-Custom "Invalid environment name. Please use only letters, numbers, underscores, and hyphens (3-50 characters)."
                }
            }
        } while ($true)
        Write-Host ""
    }
} else {
    $ENV_NAME = $CondaEnvName.Trim()
    Write-Info "Using environment name from parameter: $ENV_NAME"
}
# ================================================================

# Download TensorRT wheels from Google Drive
function Download-TensorRTWheels {
    Write-Log "Downloading TensorRT wheels from Google Drive..."
    
    $wheelsDir = "install\wheels"
    $googleDriveUrl = "https://drive.google.com/file/d/1-KLPSxea260P0CWfWUjDeZnE44HqPMys/view?usp=sharing"
    $fileId = "1-KLPSxea260P0CWfWUjDeZnE44HqPMys"
    $downloadUrl = "https://drive.google.com/uc?export=download&id=$fileId"
    $outputFile = "$wheelsDir\wheels.tar"
    
    # Create wheels directory if it doesn't exist
    if (-not (Test-Path $wheelsDir)) {
        New-Item -ItemType Directory -Path $wheelsDir -Force
        Write-Info "Created directory: $wheelsDir"
    }
    
    # Check if wheels already exist
    $wheelFiles = @(
        "$wheelsDir\tensorrt-8.6.1-cp310-none-win_amd64.whl",
        "$wheelsDir\tensorrt_lean-8.6.1-cp310-none-win_amd64.whl",
        "$wheelsDir\tensorrt_dispatch-8.6.1-cp310-none-win_amd64.whl"
    )
    
    $allWheelsExist = $true
    foreach ($wheel in $wheelFiles) {
        if (-not (Test-Path $wheel)) {
            $allWheelsExist = $false
            break
        }
    }
    
    if ($allWheelsExist) {
        Write-Info "✅ TensorRT wheels already exist, skipping download"
        return
    }
    
    Write-Info "Downloading TensorRT wheels archive..."
    Write-Host "Source: $googleDriveUrl" -ForegroundColor Gray
    Write-Host "Target: $outputFile" -ForegroundColor Gray
    
    try {
        # Install gdown if not available
        if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
            Write-Error-Custom "Python not found. Please ensure conda environment is activated."
        }
        
        Write-Info "Installing gdown for Google Drive downloads..."
        pip install gdown -q
        if ($LASTEXITCODE -ne 0) { throw "gdown installation failed" }
        
        # Use gdown to download from Google Drive (handles confirmation tokens)
        Write-Info "Downloading with gdown..."
        python -m gdown $downloadUrl -O $outputFile
        if ($LASTEXITCODE -ne 0) { throw "gdown download failed" }
        Write-Info "✅ Download completed"
        
        # Extract the tar file
        if (Test-Path $outputFile) {
            Write-Log "Extracting TensorRT wheels..."
            
            # Create temporary extraction directory
            $tempExtractDir = Join-Path $wheelsDir "temp_extract"
            if (Test-Path $tempExtractDir) {
                Remove-Item $tempExtractDir -Recurse -Force
            }
            New-Item -ItemType Directory -Path $tempExtractDir -Force | Out-Null
            
            # Check if tar is available (Windows 10 1903+ has built-in tar)
            if (Get-Command tar -ErrorAction SilentlyContinue) {
                tar -xf $outputFile -C $tempExtractDir
                Write-Info "✅ Extraction completed using built-in tar"
            }
            # Fallback: try to use 7-Zip if available
            elseif (Test-Path "${env:ProgramFiles}\7-Zip\7z.exe") {
                & "${env:ProgramFiles}\7-Zip\7z.exe" x $outputFile -o"$tempExtractDir" -y
                Write-Info "✅ Extraction completed using 7-Zip"
            }
            # Fallback: use PowerShell's Expand-Archive if it's a zip-compatible format
            else {
                Write-Warning-Custom "No extraction tool found (tar, 7-Zip). Please extract manually:"
                Write-Host "  Extract: $outputFile" -ForegroundColor White
                Write-Host "  To: $wheelsDir" -ForegroundColor White
                Remove-Item $tempExtractDir -Recurse -Force
                return
            }
            
            # Move wheel files from nested directory to wheelsDir
            $wheelFiles = Get-ChildItem -Path $tempExtractDir -Filter "*.whl" -Recurse
            if ($wheelFiles.Count -gt 0) {
                foreach ($wheel in $wheelFiles) {
                    $destPath = Join-Path $wheelsDir $wheel.Name
                    Copy-Item -Path $wheel.FullName -Destination $destPath -Force
                    Write-Info "  Moved: $($wheel.Name)"
                }
            }
            
            # Clean up temporary directory and tar file
            Remove-Item $tempExtractDir -Recurse -Force
            Remove-Item $outputFile -Force
            Write-Info "Cleaned up temporary files"
            
            # Verify extraction
            $extractedWheels = Get-ChildItem -Path $wheelsDir -Filter "*.whl" | Measure-Object
            if ($extractedWheels.Count -gt 0) {
                Write-Info "✅ TensorRT wheels extracted successfully ($($extractedWheels.Count) files)"
            }
            else {
                Write-Warning-Custom "No wheel files found after extraction"
            }
        }
        else {
            Write-Error-Custom "Download failed: File not found at $outputFile"
        }
    }
    catch {
        Write-Error-Custom "Failed to download TensorRT wheels: $($_.Exception.Message)"
        Write-Info "Please download manually from: $googleDriveUrl"
        Write-Info "Extract to: $wheelsDir"
    }
    finally {
        $ProgressPreference = 'Continue'  # Re-enable progress bar
    }
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
        Write-Host ""
        Write-Host "⚠️  CUDA 11.8 Installation Required" -ForegroundColor Yellow
        Write-Host "PyTorch will use PyTorch-bundled CUDA for inference," -ForegroundColor White
        Write-Host "but system CUDA is recommended for optimal performance." -ForegroundColor White
        Write-Host ""
        Write-Host "To install CUDA 11.8:" -ForegroundColor Cyan
        Write-Host "  1. Download: https://developer.nvidia.com/cuda-11-8-0-download-archive" -ForegroundColor White
        Write-Host "     Direct link (Windows):" -ForegroundColor White
        Write-Host "     https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_522.06_windows.exe" -ForegroundColor White
        Write-Host "  2. Run installer and select: Cuda Toolkit 11.8" -ForegroundColor White
        Write-Host "  3. Restart this script after installation" -ForegroundColor White
        Write-Host ""
        
        if ($AutoYes) {
            Write-Warning-Custom "Auto-continuing without CUDA (--AutoYes flag enabled)"
            return $false
        }
        
        $response = Read-Host "Do you want to continue without CUDA? (y/N)"
        if ($response -ne 'y' -and $response -ne 'Y') {
            Write-Error-Custom "CUDA installation is required. Please install and retry."
        }
        
        return $false
    }
}

# Check Poetry installation
function Check-Poetry {
    Write-Log "Checking Poetry installation..."
    
    if (Get-Command poetry -ErrorAction SilentlyContinue) {
        Write-Info "✅ Poetry found: $(poetry --version)"
    } else {
        Write-Warning-Custom "Poetry not found in PATH. Installing Poetry via pip..."
        try {
            pip install poetry
            if ($LASTEXITCODE -ne 0) { throw "pip install failed" }
            Write-Info "✅ Poetry installed via pip"
        } catch {
            Write-Error-Custom "Poetry installation via pip failed"
        }
    }
    # Configure Poetry to use conda environment
    Write-Log "Configuring Poetry to use conda environment..."
    try {
        poetry config virtualenvs.create false
        if ($LASTEXITCODE -ne 0) { throw "poetry config failed" }
        poetry config virtualenvs.in-project false
        if ($LASTEXITCODE -ne 0) { throw "poetry config failed" }
        poetry config keyring.enabled false
        if ($LASTEXITCODE -ne 0) { throw "poetry config failed" }
        
        # Configure Poetry to use Aliyun mirror (more reliable in China)
        Write-Info "Configuring Poetry to use Aliyun PyPI mirror..."
        poetry source add --priority primary aliyun https://mirrors.aliyun.com/pypi/simple/
        if ($LASTEXITCODE -ne 0) { Write-Warning-Custom "Could not add Aliyun mirror, continuing..." }
        
        # Keep PyTorch source explicit
        poetry source add --priority explicit pytorch https://download.pytorch.org/whl/cu118
        if ($LASTEXITCODE -ne 0) { Write-Warning-Custom "Could not add PyTorch source, continuing..." }
        
        Write-Info "✅ Poetry configured to use conda environment and mirror sources."
        return $true
    }
    catch {
        Write-Error-Custom "Poetry configuration failed"
    }
}

# Create conda environment from environment.yml
function New-CondaEnvironment {
    Write-Log "Setting up conda/mamba environment ($ENV_NAME)..."

    # Change to NTKCAP directory
    if (-not (Test-Path $NTKCAP_ROOT)) {
        Write-Error-Custom "NTKCAP directory not found at $NTKCAP_ROOT"
    }

    Set-Location $NTKCAP_ROOT

    # Check if environment already exists
    try {
        $envList = conda env list 2>$null
        $envExists = $envList | Select-String $ENV_NAME
        if ($envExists) {
            Write-Warning-Custom "$ENV_NAME environment already exists."
            if ($ForceRecreateEnv -or $AutoYes) {
                Write-Info "Auto-removing existing $ENV_NAME environment (--ForceRecreateEnv or --AutoYes flag enabled)..."
                conda env remove -n $ENV_NAME -y
                if ($LASTEXITCODE -ne 0) {
                    Write-Error-Custom "Failed to remove existing environment"
                }
            } else {
                $response = Read-Host "Do you want to remove the existing environment and create a new one? (y/N)"
                if ($response -eq 'y' -or $response -eq 'Y' -or $response -eq 'yes' -or $response -eq 'Yes') {
                    Write-Info "Removing existing $ENV_NAME environment..."
                    conda env remove -n $ENV_NAME -y
                    if ($LASTEXITCODE -ne 0) {
                        Write-Error-Custom "Failed to remove existing environment"
                    }
                }
                else {
                    Write-Info "Keeping existing environment. Skipping environment creation."
                    Write-Info "✅ Using existing $ENV_NAME environment"
                    return
                }
            }
        }
        
        # Get conda info to find envs directory
        $condaInfo = conda info --json 2>$null | ConvertFrom-Json
        $envsDir = $condaInfo.envs_dirs[0]  # First envs directory
        $envPath = Join-Path $envsDir $ENV_NAME
        
        # Check for non-conda folder at the envs location
        if (Test-Path $envPath) {
            Write-Warning-Custom "A folder exists at the $ENV_NAME location but it's not a valid conda environment."
            Write-Host "Path: $envPath" -ForegroundColor Gray
            if ($ForceRecreateEnv -or $AutoYes) {
                Write-Info "Auto-removing invalid folder (--ForceRecreateEnv or --AutoYes flag enabled)..."
            } else {
                $response = Read-Host "Do you want to remove this folder and create a new conda environment? (y/N)"
                if ($response -ne 'y' -and $response -ne 'Y') {
                    Write-Warning-Custom "Cannot proceed with environment creation while invalid folder exists."
                    Write-Error-Custom "Please manually remove the folder: $envPath"
                }
            }
            
            Write-Info "Removing invalid folder..."
            try {
                # Force remove with all attributes
                if (Test-Path $envPath) {
                    Remove-Item -Path $envPath -Recurse -Force -ErrorAction Stop
                    Start-Sleep -Seconds 2  # Wait a moment for filesystem
                }
                
                # Verify removal
                if (Test-Path $envPath) {
                    Write-Warning-Custom "Folder still exists. Trying alternative removal method..."
                    # Use takeown and icacls for stubborn folders
                    takeown /F $envPath /R /D Y 2>$null
                    icacls $envPath /grant administrators:F /T 2>$null
                    Remove-Item -Path $envPath -Recurse -Force -ErrorAction Stop
                }
                
                if (-not (Test-Path $envPath)) {
                    Write-Info "✅ Invalid folder removed successfully."
                }
                else {
                    Write-Error-Custom "Failed to remove folder. Please manually delete: $envPath"
                }
            }
            catch {
                Write-Error-Custom "Failed to remove invalid folder: $($_.Exception.Message)"
            }
        }
    }
    catch {
        Write-Warning-Custom "Could not check existing environments due to conda error. Proceeding with creation..."
    }

    # Prefer mamba if available
    $useMamba = $false
    if (Get-Command mamba -ErrorAction SilentlyContinue) {
        Write-Info "✅ mamba found, will use mamba for environment creation."
        $useMamba = $true
    } else {
        Write-Info "mamba not found, will use conda for environment creation."
    }

    # Create environment from YAML
    Write-Log "Creating $ENV_NAME environment from environment.yml..."
    if (Test-Path "install\environment.yml") {
        # Modify environment.yml to use custom name if needed
        if ($ENV_NAME -ne "ntkcap_env") {
            Write-Log "Modifying environment.yml to use custom environment name: $ENV_NAME"
            $yamlContent = Get-Content "install\environment.yml" -Raw
            $yamlContent = $yamlContent -replace "name: ntkcap_env", "name: $ENV_NAME"
            $tempYamlPath = "install\environment_temp.yml"
            $yamlContent | Out-File -FilePath $tempYamlPath -Encoding UTF8
            $yamlFile = $tempYamlPath
        } else {
            $yamlFile = "install\environment.yml"
        }
        
        try {
            if ($useMamba) {
                mamba env create -f $yamlFile
            } else {
                conda env create -f $yamlFile
            }
            
            # Clean up temporary file if created
            if ($ENV_NAME -ne "ntkcap_env" -and (Test-Path $tempYamlPath)) {
                Remove-Item $tempYamlPath -Force
            }
            
            # Verify environment was created successfully
            $envList = conda env list 2>$null
            $envCreated = $envList | Select-String $ENV_NAME
            if ($envCreated) {
                Write-Info "✅ $ENV_NAME environment created successfully"
            }
            else {
                Write-Error-Custom "Environment creation failed - $ENV_NAME not found in environment list"
            }
        }
        catch {
            Write-Warning-Custom "Environment creation failed. This may be due to:"
            Write-Host "  1. Environment already exists (non-conda folder)" -ForegroundColor Yellow
            Write-Host "  2. libmamba solver issues" -ForegroundColor Yellow
            Write-Host "  3. Permission problems" -ForegroundColor Yellow
            Write-Host ""
            Write-Error-Custom "Cannot proceed without a valid $ENV_NAME environment"
        }
    } else {
        Write-Error-Custom "install\environment.yml not found"
    }
}

# Ensure we're in the correct conda environment
function Ensure-CondaEnvironment {
    param([string]$EnvName = $ENV_NAME)
    
    Write-Log "Ensuring we're in $EnvName environment..."
    
    # Check current environment
    $currentEnv = $env:CONDA_DEFAULT_ENV
    if ($currentEnv -eq $EnvName) {
        Write-Info "✅ Already in $EnvName environment"
        return $true
    }
    
    # Try to activate the environment
    try {
        conda activate $EnvName
        $newEnv = $env:CONDA_DEFAULT_ENV
        if ($newEnv -eq $EnvName) {
            Write-Info "✅ Successfully activated $EnvName environment"
            return $true
        }
        else {
            Write-Warning-Custom "Failed to activate $EnvName - current environment: $newEnv"
            return $false
        }
    }
    catch {
        Write-Warning-Custom "Failed to activate $EnvName environment: $($_.Exception.Message)"
        return $false
    }
}

# Install Python dependencies with Poetry
function Install-PythonDependencies {
    Write-Log "Installing Python dependencies..."
    
    # Activate conda environment
    conda activate $ENV_NAME

    # Ensure opensim 4.5 is installed (from conda)
    Write-Log "Ensuring opensim 4.5 is installed..."
    try {
        conda install -c opensim-org opensim=4.5=py310np121 -y
        if ($LASTEXITCODE -ne 0) { throw "conda install failed" }
        Write-Info "✅ OpenSim installed successfully"
    }
    catch {
        Write-Error-Custom "OpenSim installation failed"
    }

    # Install PyTorch CUDA 11.8 (always install)
    Write-Log "Installing PyTorch 2.0.1, torchvision 0.15.2, torchaudio 2.0.2 with CUDA 11.8..."
    try {
        pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
        if ($LASTEXITCODE -ne 0) { throw "pip install failed" }
        Write-Info "✅ PyTorch and related packages installed"
    }
    catch {
        Write-Error-Custom "PyTorch installation failed"
    }
    
    Write-Info "✅ Basic Python dependencies installation completed"
}

# Install dependencies directly via pip (alternative to Poetry)
function Install-DirectPipDependencies {
    Write-Log "Installing dependencies directly via pip..."
    
    # Activate conda environment
    conda activate $ENV_NAME
    
    # Ensure we're in the correct environment
    if (-not (Ensure-CondaEnvironment $ENV_NAME)) {
        Write-Error-Custom "Cannot proceed without $ENV_NAME environment"
    }
    
    # Install OpenSim first (from conda)
    Write-Log "Ensuring opensim 4.5 is installed..."
    try {
        if (Get-Command mamba -ErrorAction SilentlyContinue) {
            mamba install -c opensim-org opensim=4.5=py310np121 -y
        } else {
            conda install -c opensim-org opensim=4.5=py310np121 -y
        }
        if ($LASTEXITCODE -ne 0) { throw "conda install failed" }
        Write-Info "✅ OpenSim installed successfully"
    }
    catch {
        Write-Error-Custom "OpenSim installation failed"
    }
    
    # Install packages in the specified order
    $packages = @(
        "bs4",
        "multiprocess", 
        "keyboard",
        "import_ipynb",
        "kivy",
        "Pose2Sim==0.4",
        "numpy==1.21.6",
        "scipy==1.13.0",
        "ultralytics",
        "tkfilebrowser",
        "matplotlib==3.8.4",
        "pyserial",
        "func_timeout",
        "pygltflib",
        "natsort",
        "openpyxl",
        "pyqtgraph"
    )
    
    foreach ($package in $packages) {
        Write-Log "Installing $package..."
        try {
            pip install $package
            if ($LASTEXITCODE -ne 0) { throw "pip install failed" }
            Write-Info "✅ $package installed successfully"
        }
        catch {
            Write-Warning-Custom "Failed to install $package, continuing..."
        }
    }
    
    # Install numpy 1.21.6 again to ensure version
    Write-Log "Ensuring numpy==1.21.6..."
    try {
        pip install numpy==1.21.6 --force-reinstall
        if ($LASTEXITCODE -ne 0) { throw "pip install failed" }
        Write-Info "✅ numpy 1.21.6 installed"
    }
    catch {
        Write-Warning-Custom "Failed to install numpy 1.21.6"
    }
    
    # Install PyQt6 packages
    Write-Log "Installing PyQt6 packages..."
    try {
        pip install PyQt6==6.7.0
        if ($LASTEXITCODE -ne 0) { throw "pip install failed" }
        Write-Info "✅ PyQt6==6.7.0 installed"
    }
    catch {
        Write-Warning-Custom "Failed to install PyQt6==6.7.0"
    }
    
    try {
        pip install PyQt6-WebEngine==6.7.0
        if ($LASTEXITCODE -ne 0) { throw "pip install failed" }
        Write-Info "✅ PyQt6-WebEngine==6.7.0 installed"
    }
    catch {
        Write-Warning-Custom "Failed to install PyQt6-WebEngine==6.7.0"
    }
    
    # Install cupy-cuda11x
    Write-Log "Installing cupy-cuda11x..."
    try {
        pip install cupy-cuda11x
        if ($LASTEXITCODE -ne 0) { throw "pip install failed" }
        Write-Info "✅ cupy-cuda11x installed"
    }
    catch {
        Write-Warning-Custom "Failed to install cupy-cuda11x"
    }
    
    # Final numpy version fix
    Write-Log "Final numpy version correction to 1.22.4..."
    try {
        pip install numpy==1.22.4 --force-reinstall
        if ($LASTEXITCODE -ne 0) { throw "pip install failed" }
        Write-Info "✅ numpy==1.22.4 installed (final version)"
    }
    catch {
        Write-Warning-Custom "Failed to install final numpy version"
    }
    
    Write-Info "✅ Direct pip dependencies installation completed"
}

# Install Poetry dependencies (after MMEngine ecosystem)
function Install-PoetryDependencies {
    # Check if Poetry installation should be skipped
    if ($SkipPoetry) {
        Write-Warning-Custom "Poetry installation SKIPPED (--SkipPoetry flag enabled)"
        Write-Info "You can run Poetry manually later with:"
        Write-Host "  cd `$NTKCAP_ROOT  # e.g., D:\NTKCAP" -ForegroundColor White
        Write-Host "  conda activate $ENV_NAME" -ForegroundColor White
        Write-Host "  poetry lock --no-cache" -ForegroundColor White
        Write-Host "  poetry install" -ForegroundColor White
        return
    }
    
    # Ask user if they want to install Poetry dependencies
    Write-Host ""
    Write-Host "================================" -ForegroundColor Cyan
    Write-Host "Poetry Dependency Installation" -ForegroundColor Cyan
    Write-Host "================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Poetry will now generate a lockfile and install dependencies." -ForegroundColor White
    Write-Host "This may take some time (5-10 minutes) depending on network speed." -ForegroundColor White
    Write-Host ""
    $response = Read-Host "Do you want to proceed with Poetry dependency installation? (y/N)"
    if ($response -ne 'y' -and $response -ne 'Y') {
        Write-Warning-Custom "Poetry dependency installation SKIPPED by user"

        # If user explicitly requested direct pip install via flag, or AutoYes, run direct pip path
        if ($UseDirectPip -or $AutoYes) {
            Write-Info "Proceeding with direct pip installation (--UseDirectPip or --AutoYes detected)..."
            Install-DirectPipDependencies
            return
        }

        # Ask whether to fallback to direct pip installation
        $pipResponse = Read-Host "Do you want to install dependencies via direct pip instead? (y/N)"
        if ($pipResponse -eq 'y' -or $pipResponse -eq 'Y') {
            Write-Info "User selected direct pip installation. Running direct pip installer..."
            Install-DirectPipDependencies
            return
        }

        Write-Info "You can run Poetry manually later with:"
        Write-Host "  cd `$NTKCAP_ROOT  # e.g., D:\NTKCAP" -ForegroundColor White
        Write-Host "  conda activate $ENV_NAME" -ForegroundColor White
        Write-Host "  poetry lock --no-cache" -ForegroundColor White
        Write-Host "  poetry install" -ForegroundColor White
        return
    }
    
    Write-Log "Installing Poetry dependencies..."
    
    # Activate conda environment
    conda activate $ENV_NAME

    # Check if Poetry is available - only proceed if Poetry is installed
    if (-not (Get-Command poetry -ErrorAction SilentlyContinue)) {
        Write-Warning-Custom "Poetry is not installed. Skipping Poetry-based dependency installation."
        Write-Info "To install other dependencies, please install Poetry manually:"
        Write-Host "  pip install poetry" -ForegroundColor White
        return
    }

    # Check if pyproject.toml exists for Poetry usage
    if (-not (Test-Path "pyproject.toml")) {
        Write-Warning-Custom "pyproject.toml not found in current directory"
        Write-Info "Cannot proceed with Poetry installation without pyproject.toml"
        return
    }

    # Install Poetry dependencies
    Write-Log "Installing Poetry dependencies from pyproject.toml..."
    Write-Info "Note: Installing dependencies via Poetry after MMEngine ecosystem"
    try {
        # Ensure we're in conda environment and clear any existing lockfile conflicts
        conda activate $ENV_NAME
        Write-Log "Regenerating Poetry lockfile to resolve dependency conflicts..."
        poetry lock --no-cache
        if ($LASTEXITCODE -ne 0) { throw "poetry lock failed" }
        Write-Log "Installing Poetry dependencies..."
        poetry install
        if ($LASTEXITCODE -ne 0) { throw "poetry install failed" }
        Write-Info "✅ Poetry install completed"
    }
    catch {
        Write-Error-Custom "Poetry install failed"
    }
    
    Write-Info "✅ Poetry dependencies installation completed"
}

# Install MMEngine ecosystem and MMPose
function Install-MMComponents {
    Write-Log "Installing MMEngine ecosystem and MMPose..."
    
    # Ensure we're in the correct environment
    if (-not (Ensure-CondaEnvironment $ENV_NAME)) {
        Write-Error-Custom "Cannot proceed without $ENV_NAME environment"
    }
    
    # Install openmim first (required for mim commands)
    Write-Log "Installing openmim..."
    try {
        pip install -U openmim
        if ($LASTEXITCODE -ne 0) { throw "pip install failed" }
        Write-Info "✅ openmim installed successfully"
    }
    catch {
        Write-Error-Custom "openmim installation failed"
    }
    
    # Install MMEngine ecosystem using mim
    Write-Log "Installing mmengine via mim..."
    try {
        mim install mmengine
        if ($LASTEXITCODE -ne 0) { throw "mim install failed" }
        Write-Info "✅ mmengine installed"
    }
    catch {
        Write-Error-Custom "mmengine installation failed"
    }
    
    Write-Log "Installing mmcv (2.1.0) via mim..."
    try {
        mim install "mmcv==2.1.0"
        if ($LASTEXITCODE -ne 0) { throw "mim install failed" }
        Write-Info "✅ mmcv 2.1.0 installed"
    }
    catch {
        Write-Error-Custom "mmcv installation failed"
    }
    
    Write-Log "Installing mmdet via mim..."
    try {
        mim install "mmdet>=3.3.0"
        if ($LASTEXITCODE -ne 0) { throw "mim install failed" }
        Write-Info "✅ mmdet installed"
    }
    catch {
        Write-Error-Custom "mmdet installation failed"
    }
    
    Write-Info "✅ MMEngine ecosystem installed"
    
    # Install MMPose from source (before mmdeploy)
    Write-Log "Installing MMPose from source..."
    $mmposePath = Join-Path $NTKCAP_ROOT "NTK_CAP\ThirdParty\mmpose"
    
    if (Test-Path $mmposePath) {
        Set-Location $mmposePath
        Write-Log "Installing MMPose requirements..."
        try {
            pip install -r requirements.txt
            if ($LASTEXITCODE -ne 0) { throw "pip install failed" }
            Write-Info "✅ MMPose requirements installed"
        }
        catch {
            Write-Error-Custom "MMPose requirements installation failed"
        }
        
        Write-Log "Installing MMPose in editable mode..."
        try {
            pip install -v -e .
            if ($LASTEXITCODE -ne 0) { throw "pip install failed" }
            Write-Info "✅ MMPose installed successfully"
        }
        catch {
            Write-Error-Custom "MMPose installation failed"
        }
        
        Set-Location $NTKCAP_ROOT
    }
    else {
        Write-Error-Custom "MMPose directory not found at: $mmposePath"
    }
    
    # Install mmdeploy and related packages (after MMPose)
    Write-Log "Installing mmdeploy and related packages..."
    try {
        pip install mmdeploy==1.3.1
        if ($LASTEXITCODE -ne 0) { throw "pip install failed" }
        Write-Info "✅ mmdeploy 1.3.1 installed"
    }
    catch {
        Write-Error-Custom "mmdeploy installation failed"
    }
    
    try {
        pip install mmdeploy-runtime-gpu==1.3.1
        if ($LASTEXITCODE -ne 0) { throw "pip install failed" }
        Write-Info "✅ mmdeploy-runtime-gpu 1.3.1 installed"
    }
    catch {
        Write-Error-Custom "mmdeploy-runtime-gpu installation failed"
    }
    
    try {
        pip install onnxruntime-gpu==1.17.1
        if ($LASTEXITCODE -ne 0) { throw "pip install failed" }
        Write-Info "✅ onnxruntime-gpu 1.17.1 installed"
    }
    catch {
        Write-Error-Custom "onnxruntime-gpu installation failed"
    }
    
    try {
        pip install pycuda
        if ($LASTEXITCODE -ne 0) { throw "pip install failed" }
        Write-Info "✅ pycuda installed"
    }
    catch {
        Write-Error-Custom "pycuda installation failed"
    }
    
    # try {
    #     pip install numpy==1.21.6
    #     Write-Info "✅ numpy 1.21.6 installed"
    # }
    # catch {
    #     Write-Warning-Custom "numpy 1.21.6 installation failed"
    # }
    
    # Install TensorRT wheels (after mmdeploy packages)
    Write-Log "Installing TensorRT wheels..."
    $wheelsDir = "install\wheels"
    
    if (Test-Path $wheelsDir) {
        $tensorrtWheels = @(
            "$wheelsDir\tensorrt_dispatch-8.6.1-cp310-none-win_amd64.whl",
            "$wheelsDir\tensorrt_lean-8.6.1-cp310-none-win_amd64.whl",
            "$wheelsDir\tensorrt-8.6.1-cp310-none-win_amd64.whl"
        )
        
            foreach ($wheel in $tensorrtWheels) {
                if (Test-Path $wheel) {
                    try {
                        pip install $wheel
                        if ($LASTEXITCODE -ne 0) { throw "pip install failed" }
                        Write-Info "✅ Installed $(Split-Path $wheel -Leaf)"
                    }
                    catch {
                        Write-Error-Custom "Failed to install $(Split-Path $wheel -Leaf)"
                    }
                }
                else {
                    Write-Error-Custom "TensorRT wheel not found: $wheel"
                }
            }
            
            # Apply numpy version correction after TensorRT installation
            Write-Log "Applying numpy version correction for TensorRT and MM compatibility..."
            try {
                pip uninstall numpy -y
                if ($LASTEXITCODE -ne 0) { throw "pip uninstall failed" }
                pip install numpy==1.23.5
                if ($LASTEXITCODE -ne 0) { throw "pip install failed" }
                Write-Info "✅ numpy 1.23.5 installed (cleaned and reinstalled for compatibility)"
                
                # Verify numpy version
                $numpyVersion = python -c "import numpy; print(numpy.__version__)" 2>$null
                if ($LASTEXITCODE -ne 0) { throw "python check failed" }
                if ($numpyVersion -eq "1.23.5") {
                    Write-Info "✅ numpy version verified: $numpyVersion"
                }
                else {
                    Write-Error-Custom "numpy version mismatch: expected 1.23.5, got $numpyVersion. Installation cannot continue."
                }
            }
            catch {
                Write-Error-Custom "numpy version correction failed. Installation cannot continue."
            }
        }
        else {
            Write-Error-Custom "TensorRT wheels directory not found at: $wheelsDir. Please ensure TensorRT wheels are available."
        }    # Deploy models to TensorRT format (after all MM components are installed)
    Write-Log "Deploying models to TensorRT format..."
    if ($SkipTensorRTDeploy) {
        Write-Info "⏭️  Skipping TensorRT model deployment (--SkipTensorRTDeploy flag enabled)"
    } elseif ($AutoYes) {
        Write-Info "⏭️  Skipping TensorRT model deployment (--AutoYes flag enabled, use --SkipTensorRTDeploy=false to force deploy)"
    } else {
        $response = Read-Host "Do you want to deploy models to TensorRT format? This may take some time (y/N)"
        if ($response -eq 'y' -or $response -eq 'Y') {
        $mmdeployPath = Join-Path $NTKCAP_ROOT "NTK_CAP\ThirdParty\mmdeploy"
        
        if (Test-Path $mmdeployPath) {
            Set-Location $mmdeployPath
            
            # Deploy RTMDet model
            Write-Log "Deploying RTMDet model to TensorRT..."
            try {
                python tools/deploy.py configs/mmdet/detection/detection_tensorrt_static-320x320.py ../mmpose/projects/rtmpose/rtmdet/person/rtmdet_nano_320-8xb32_coco-person.py https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth demo/resources/human-pose.jpg --work-dir rtmpose-trt/rtmdet-nano --device cuda:0 --show --dump-info
                if ($LASTEXITCODE -ne 0) { throw "python deploy failed" }
                Write-Info "✅ RTMDet model deployed to TensorRT"
            }
            catch {
                Write-Error-Custom "RTMDet TensorRT deployment failed"
            }
            
            # Deploy RTMPose model
            Write-Log "Deploying RTMPose model to TensorRT..."
            try {
                python tools/deploy.py configs/mmpose/pose-detection_tensorrt_static-384x288.py ../mmpose/projects/rtmpose/rtmpose/body_2d_keypoint/rtmpose-m_8xb512-700e_body8-halpe26-384x288.py https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7-halpe26_700e-384x288-89e6428b_20230605.pth demo/resources/human-pose.jpg --work-dir rtmpose-trt/rtmpose-m --device cuda:0 --show --dump-info
                if ($LASTEXITCODE -ne 0) { throw "python deploy failed" }
                Write-Info "✅ RTMPose model deployed to TensorRT"
            }
            catch {
                Write-Error-Custom "RTMPose TensorRT deployment failed"
            }
            
            Set-Location $NTKCAP_ROOT
            Write-Info "✅ TensorRT model deployment completed"
        }
        else {
            Write-Error-Custom "MMDeploy directory not found at: $mmdeployPath"
        }
        } else {
            Write-Info "⏭️  Skipping TensorRT model deployment"
        }
    }
    
    # Install EasyMocap
    Write-Log "Installing EasyMocap..."
    $easymocapPath = Join-Path $NTKCAP_ROOT "NTK_CAP\ThirdParty\EasyMocap"
    
    if (Test-Path $easymocapPath) {
        Set-Location $easymocapPath
        Write-Log "Installing setuptools 69.5.0..."
        try {
            pip install setuptools==69.5.0
            if ($LASTEXITCODE -ne 0) { throw "pip install failed" }
            Write-Info "✅ setuptools 69.5.0 installed"
        }
        catch {
            Write-Error-Custom "setuptools installation failed"
        }
        
        Write-Log "Installing EasyMocap in development mode..."
        try {
            python setup.py develop --user
            if ($LASTEXITCODE -ne 0) { throw "python setup failed" }
            Write-Info "✅ EasyMocap installed successfully"
        }
        catch {
            Write-Error-Custom "EasyMocap installation failed"
        }
        
        # Return to root directory - use parent of parent path to be safe
        $rootPath = Split-Path (Split-Path (Split-Path $easymocapPath -Parent) -Parent) -Parent
        if (Test-Path $rootPath) {
            Set-Location $rootPath
            Write-Info "✅ Returned to: $rootPath"
        }
        else {
            # Fallback to NTKCAP_ROOT if calculation failed
            Set-Location $NTKCAP_ROOT
            Write-Info "✅ Returned to: $NTKCAP_ROOT"
        }
        Write-Info "✅ EasyMocap installation completed"
    }
    else {
        Write-Error-Custom "EasyMocap directory not found at: $easymocapPath"
    }
}

# Create activation script
function New-ActivationScript {
    Write-Log "Creating environment activation script..."
    $activationScript = @"
# NTKCAP Environment Activation Script for Windows
# Run this script to activate the NTKCAP environment

Write-Host "🚀 Activating NTKCAP environment ($ENV_NAME)..." -ForegroundColor Green

# Activate conda environment
conda activate $ENV_NAME

# Check CUDA availability
if (Get-Command nvcc -ErrorAction SilentlyContinue) {
    `$cudaVersion = nvcc --version 2>`$null | Select-String "release"
    Write-Host "✅ CUDA available: `$cudaVersion" -ForegroundColor Green
}
else {
    Write-Host "⚠️  CUDA not found in PATH" -ForegroundColor Yellow
}

# Show Python version
Write-Host "✅ Python: `$(python --version)" -ForegroundColor Green

# Check PyTorch CUDA
try {
    `$torchCuda = python -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>`$null
    Write-Host "✅ PyTorch: `$torchCuda" -ForegroundColor Green
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
"@
    $activationScript | Out-File -FilePath "install\activate_ntkcap.ps1" -Encoding UTF8
    # Create symlink in main directory
    if (Test-Path "activate_ntkcap.ps1") {
        Remove-Item "activate_ntkcap.ps1" -Force
    }
    Copy-Item "install\activate_ntkcap.ps1" "activate_ntkcap.ps1"
    Write-Info "✅ Activation script created: install\activate_ntkcap.ps1"
    Write-Info "✅ Copy created: activate_ntkcap.ps1"
}

# Apply final compatibility fixes
function Apply-CompatibilityFixes {
    Write-Log "Applying final compatibility fixes..."
    
    # Activate conda environment
    conda activate $ENV_NAME
    
    # Fix PyQt version conflicts and ensure compatibility
    Write-Log "Applying PyQt compatibility fixes..."
    try {
        Write-Info "Installing specific versions to resolve PyQt5/PyQt6 conflicts..."
        
        # Only install these if not using direct pip (which already handles them)
        if (-not $UseDirectPip) {
            pip install "Pose2Sim==0.4"
            if ($LASTEXITCODE -ne 0) { throw "pip install Pose2Sim failed" }
        }
        
        # Always clean up and reinstall PyQt6 for compatibility
        pip uninstall PyQt6 PyQt6-WebEngine PyQt6-Qt6 PyQt6-WebEngine-Qt6 PyQt6-sip -y
        if ($LASTEXITCODE -ne 0) { throw "pip uninstall PyQt6 failed" }
        
        # Use stable 6.7.0 version for all installation methods (避免 DLL 問題)
        pip install PyQt6==6.7.0
        if ($LASTEXITCODE -ne 0) { throw "pip install PyQt6 failed" }
        pip install PyQt6-WebEngine==6.7.0
        if ($LASTEXITCODE -ne 0) { throw "pip install PyQt6-WebEngine failed" }
        
        # Final numpy version - only if not using direct pip (which handles this)
        if (-not $UseDirectPip) {
            pip install "numpy==1.22.4"
            if ($LASTEXITCODE -ne 0) { throw "pip install numpy failed" }
        }
        
        Write-Info "✅ PyQt compatibility fixes applied"
    }
    catch {
        Write-Error-Custom "PyQt compatibility fixes failed"
    }
    
    Write-Info "✅ Final compatibility fixes completed"
}

# Cleanup function to restore state on failure
function Cleanup-OnFailure {
    Write-Host ""
    Write-Warning-Custom "Installation failed. Cleaning up..."
    
    # Return to project root
    Write-Info "Returning to project root directory..."
    if (Test-Path $NTKCAP_ROOT) {
        Set-Location $NTKCAP_ROOT
        Write-Info "✅ Returned to: $NTKCAP_ROOT"
    }
    
    # Deactivate conda environment
    Write-Info "Deactivating conda environment..."
    try {
        conda deactivate
        Write-Info "✅ Conda environment deactivated"
    }
    catch {
        Write-Warning-Custom "Could not deactivate conda environment: $($_.Exception.Message)"
    }
    
    Write-Host ""
    Write-Host "❌ Installation failed and rolled back" -ForegroundColor Red
    Write-Host "Please review the errors above and try again." -ForegroundColor Red
    Write-Host ""
}

# Main installation function
function Start-Installation {
    Write-Host "======================================" -ForegroundColor Cyan
    Write-Host "  NTKCAP Complete Environment Setup  " -ForegroundColor Cyan
    Write-Host "  for Windows                        " -ForegroundColor Cyan
    Write-Host "======================================" -ForegroundColor Cyan
    Write-Host ""
    
    # Display configuration
    Write-Host "Configuration:" -ForegroundColor Cyan
    Write-Host "  Environment name: $ENV_NAME" -ForegroundColor White
    Write-Host "  Installation method: $(if ($UseDirectPip) { 'Direct pip' } else { 'Poetry' })" -ForegroundColor White
    Write-Host ""
    
    # Display active options
    if ($SkipCudaCheck) {
        Write-Warning-Custom "CUDA check SKIPPED (--SkipCudaCheck flag enabled)"
    }
    if ($SkipPoetry) {
        Write-Warning-Custom "Poetry installation SKIPPED (--SkipPoetry flag enabled)"
    }
    if ($UseDirectPip) {
        Write-Info "Using DIRECT PIP installation method (--UseDirectPip flag enabled)"
    }
    if ($ForceRecreateEnv) {
        Write-Info "Force recreate environment enabled (--ForceRecreateEnv flag enabled)"
    }
    if ($SkipTensorRTDeploy) {
        Write-Info "TensorRT deployment SKIPPED (--SkipTensorRTDeploy flag enabled)"
    }
    if ($AutoYes) {
        Write-Info "Auto-yes mode enabled (--AutoYes flag enabled) - will skip interactive prompts"
    }
    Write-Host ""
    
    try {
        Test-SystemRequirements
        
        if (-not $SkipCudaCheck) {
            $cudaAvailable = Test-CudaInstallation
            if (-not $cudaAvailable) {
                Write-Warning-Custom "Continuing without CUDA support"
            }
        }
        
        New-CondaEnvironment
        
        # Verify environment was created and we can activate it
        if (-not (Ensure-CondaEnvironment $ENV_NAME)) {
            throw "Failed to create or activate $ENV_NAME environment. Cannot continue."
        }

        # Install PyTorch and related packages immediately after environment creation
        Write-Log "Installing PyTorch 2.0.1, torchvision 0.15.2, torchaudio 2.0.2 with CUDA 11.8..."
        try {
            pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
            if ($LASTEXITCODE -ne 0) { throw "pip install failed" }
            Write-Info "✅ PyTorch and related packages installed"
        }
        catch {
            throw "PyTorch installation failed: $($_.Exception.Message)"
        }

        Check-Poetry
        
        # Download TensorRT wheels before MM components are installed
        Download-TensorRTWheels
        
        Install-MMComponents
        
        # Choose installation method based on flags
        if ($UseDirectPip) {
            Write-Log "Using direct pip installation method(faster installation for not developement environment) (--UseDirectPip flag enabled)..."
            Install-DirectPipDependencies
        } else {
            Install-PoetryDependencies
        }
        
        Apply-CompatibilityFixes
        New-ActivationScript
        
        Write-Host ""
        Write-Host "======================================" -ForegroundColor Green
        Write-Host "        Installation Complete!       " -ForegroundColor Green
        Write-Host "======================================" -ForegroundColor Green
        Write-Host ""
        
        Write-Host "To activate the environment:" -ForegroundColor Cyan
        Write-Host "  cd `$NTKCAP_ROOT  # e.g., D:\NTKCAP" -ForegroundColor White
        Write-Host "  .\activate_ntkcap.ps1" -ForegroundColor White
        Write-Host "  # OR from install directory:" -ForegroundColor Gray
        Write-Host "  .\install\activate_ntkcap.ps1" -ForegroundColor White
        Write-Host ""
        Write-Host "🎉 NTKCAP environment setup completed successfully!" -ForegroundColor Green
    }
    catch {
        Write-Error-Custom "Fatal error: $($_.Exception.Message)"
        Cleanup-OnFailure
    }
}

# Run main installation
Start-Installation
