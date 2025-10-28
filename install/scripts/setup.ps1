# NTKCAP Complete Environment Setup Script for Windows
# This script sets up the complete NTKCAP environment with Poetry, CUDA, and all dependencies
# Requirements: Windows 10/11 with Anaconda/Miniconda
# Test-SystemRequirements - 系統需求檢查
# Test-CudaInstallation - CUDA 檢查
# New-CondaEnvironment - 建立 conda 環境
# Ensure-CondaEnvironment - 確保在正確環境
# pip install torch - 安裝 PyTorch (在 Check-Poetry 之前)
# Check-Poetry - 檢查並安裝 Poetry
# Install-MMComponents - 安裝 MM 生態系、MMPose、mmdeploy、TensorRT wheels、部署模型、EasyMocap
# Install-PoetryDependencies - Poetry 安裝依賴
# Apply-CompatibilityFixes - 最後的 pip 補正（Pose2Sim, PyQt6, numpy==1.22.4）
# New-ActivationScript - 建立啟動腳本

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
        # Use PowerShell's Invoke-WebRequest to download
        $ProgressPreference = 'SilentlyContinue'  # Disable progress bar for faster download
        Invoke-WebRequest -Uri $downloadUrl -OutFile $outputFile -UseBasicParsing
        Write-Info "✅ Download completed"
        
        # Extract the tar file
        if (Test-Path $outputFile) {
            Write-Log "Extracting TensorRT wheels..."
            
            # Check if tar is available (Windows 10 1903+ has built-in tar)
            if (Get-Command tar -ErrorAction SilentlyContinue) {
                tar -xf $outputFile -C $wheelsDir
                Write-Info "✅ Extraction completed using built-in tar"
            }
            # Fallback: try to use 7-Zip if available
            elseif (Test-Path "${env:ProgramFiles}\7-Zip\7z.exe") {
                & "${env:ProgramFiles}\7-Zip\7z.exe" x $outputFile -o"$wheelsDir" -y
                Write-Info "✅ Extraction completed using 7-Zip"
            }
            # Fallback: use PowerShell's Expand-Archive if it's a zip-compatible format
            else {
                Write-Warning-Custom "No extraction tool found (tar, 7-Zip). Please extract manually:"
                Write-Host "  Extract: $outputFile" -ForegroundColor White
                Write-Host "  To: $wheelsDir" -ForegroundColor White
                return
            }
            
            # Clean up tar file after extraction
            Remove-Item $outputFile -Force
            Write-Info "Cleaned up archive file"
            
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
        Write-Info "✅ Poetry configured to use conda environment."
        return $true
    }
    catch {
        Write-Error-Custom "Poetry configuration failed"
    }
}

# Create conda environment from environment.yml
function New-CondaEnvironment {
    Write-Log "Setting up conda/mamba environment..."

    # Change to NTKCAP directory
    $ntkcapPath = "D:\NTKCAP"
    if (-not (Test-Path $ntkcapPath)) {
        Write-Error-Custom "NTKCAP directory not found at $ntkcapPath"
    }

    Set-Location $ntkcapPath

    # Check if environment already exists
    try {
        $envList = conda env list 2>$null
        $envExists = $envList | Select-String "ntkcap_env"
        if ($envExists) {
            Write-Warning-Custom "ntkcap_env environment already exists."
            $response = Read-Host "Do you want to remove the existing environment and create a new one? (y/N)"
            if ($response -eq 'y' -or $response -eq 'Y' -or $response -eq 'yes' -or $response -eq 'Yes') {
                Write-Info "Removing existing ntkcap_env environment..."
                conda env remove -n ntkcap_env -y
                if ($LASTEXITCODE -ne 0) {
                    Write-Error-Custom "Failed to remove existing environment"
                }
            }
            else {
                Write-Info "Keeping existing environment. Skipping environment creation."
                Write-Info "✅ Using existing ntkcap_env environment"
                return
            }
        }
        
        # Get conda info to find envs directory
        $condaInfo = conda info --json 2>$null | ConvertFrom-Json
        $envsDir = $condaInfo.envs_dirs[0]  # First envs directory
        $envPath = Join-Path $envsDir "ntkcap_env"
        
        # Check for non-conda folder at the envs location
        if (Test-Path $envPath) {
            Write-Warning-Custom "A folder exists at the ntkcap_env location but it's not a valid conda environment."
            Write-Host "Path: $envPath" -ForegroundColor Gray
            $response = Read-Host "Do you want to remove this folder and create a new conda environment? (y/N)"
            if ($response -eq 'y' -or $response -eq 'Y') {
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
            else {
                Write-Warning-Custom "Cannot proceed with environment creation while invalid folder exists."
                Write-Error-Custom "Please manually remove the folder: $envPath"
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
    Write-Log "Creating ntkcap_env environment from environment.yml..."
    if (Test-Path "install\environment.yml") {
        try {
            if ($useMamba) {
                mamba env create -f install\environment.yml
            } else {
                conda env create -f install\environment.yml
            }
            
            # Verify environment was created successfully
            $envList = conda env list 2>$null
            $envCreated = $envList | Select-String "ntkcap_env"
            if ($envCreated) {
                Write-Info "✅ ntkcap_env environment created successfully"
            }
            else {
                Write-Error-Custom "Environment creation failed - ntkcap_env not found in environment list"
            }
        }
        catch {
            Write-Warning-Custom "Environment creation failed. This may be due to:"
            Write-Host "  1. Environment already exists (non-conda folder)" -ForegroundColor Yellow
            Write-Host "  2. libmamba solver issues" -ForegroundColor Yellow
            Write-Host "  3. Permission problems" -ForegroundColor Yellow
            Write-Host ""
            Write-Error-Custom "Cannot proceed without a valid ntkcap_env environment"
        }
    } else {
        Write-Error-Custom "install\environment.yml not found"
    }
}

# Ensure we're in the correct conda environment
function Ensure-CondaEnvironment {
    param([string]$EnvName = "ntkcap_env")
    
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
    conda activate ntkcap_env

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

# Install Poetry dependencies (after MMEngine ecosystem)
function Install-PoetryDependencies {
    Write-Log "Installing Poetry dependencies..."
    
    # Activate conda environment
    conda activate ntkcap_env

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
        conda activate ntkcap_env
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
    if (-not (Ensure-CondaEnvironment "ntkcap_env")) {
        Write-Error-Custom "Cannot proceed without ntkcap_env environment"
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
    $ntkcapPath = "D:\NTKCAP"
    $mmposePath = Join-Path $ntkcapPath "NTK_CAP\ThirdParty\mmpose"
    
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
        
        Set-Location $ntkcapPath
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
    $response = Read-Host "Do you want to deploy models to TensorRT format? This may take some time (y/N)"
    if ($response -eq 'y' -or $response -eq 'Y') {
        $mmdeployPath = Join-Path $ntkcapPath "NTK_CAP\ThirdParty\mmdeploy"
        
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
            
            Set-Location $ntkcapPath
            Write-Info "✅ TensorRT model deployment completed"
        }
        else {
            Write-Error-Custom "MMDeploy directory not found at: $mmdeployPath"
        }
    }
    else {
        Write-Info "⏭️  Skipping TensorRT model deployment"
    }
    
    # Install EasyMocap
    Write-Log "Installing EasyMocap..."
    $easymocapPath = "NTK_CAP\ThirdParty\EasyMocap"
    
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
        
        Set-Location $ntkcapPath
        Write-Info "✅ EasyMocap installation completed"
    }
    else {
        Write-Error-Custom "EasyMocap directory not found at: $easymocapPath"
    }
}

# Create activation script
function New-ActivationScript {
    Write-Log "Creating environment activation script..."
    $activationScript = @'
# NTKCAP Environment Activation Script for Windows
# Run this script to activate the NTKCAP environment

Write-Host "🚀 Activating NTKCAP environment..." -ForegroundColor Green

# Activate conda environment
conda activate ntkcap_env

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

# Apply final compatibility fixes
function Apply-CompatibilityFixes {
    Write-Log "Applying final compatibility fixes..."
    
    # Activate conda environment
    conda activate ntkcap_env
    
    # Fix PyQt version conflicts and ensure compatibility
    Write-Log "Applying PyQt compatibility fixes..."
    try {
        Write-Info "Installing specific versions to resolve PyQt5/PyQt6 conflicts..."
        pip install "Pose2Sim==0.4"
        if ($LASTEXITCODE -ne 0) { throw "pip install Pose2Sim failed" }
        pip uninstall PyQt6 PyQt6-WebEngine PyQt6-Qt6 PyQt6-WebEngine-Qt6 PyQt6-sip -y
        if ($LASTEXITCODE -ne 0) { throw "pip uninstall PyQt6 failed" }
        # 使用穩定的 6.7.0 版本（避免 DLL 問題）
        pip install PyQt6==6.7.0
        if ($LASTEXITCODE -ne 0) { throw "pip install PyQt6 failed" }
        pip install PyQt6-WebEngine==6.7.0
        if ($LASTEXITCODE -ne 0) { throw "pip install PyQt6-WebEngine failed" }
        pip install "numpy==1.22.4"
        if ($LASTEXITCODE -ne 0) { throw "pip install numpy failed" }
        Write-Info "✅ PyQt compatibility fixes applied"
    }
    catch {
        Write-Error-Custom "PyQt compatibility fixes failed"
    }
    
    Write-Info "✅ Final compatibility fixes completed"
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
    
    New-CondaEnvironment
    
    # Verify environment was created and we can activate it
    if (-not (Ensure-CondaEnvironment "ntkcap_env")) {
        Write-Error-Custom "Failed to create or activate ntkcap_env environment. Cannot continue."
    }

    # Install PyTorch and related packages immediately after environment creation
    Write-Log "Installing PyTorch 2.0.1, torchvision 0.15.2, torchaudio 2.0.2 with CUDA 11.8..."
    try {
        pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
        if ($LASTEXITCODE -ne 0) { throw "pip install failed" }
        Write-Info "✅ PyTorch and related packages installed"
    }
    catch {
        Write-Error-Custom "PyTorch installation failed"
    }

    Check-Poetry
    Install-MMComponents
    
    # Download TensorRT wheels after MM components are installed
    Download-TensorRTWheels
    
    Install-PoetryDependencies
    Apply-CompatibilityFixes
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
