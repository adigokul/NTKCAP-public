@echo off
REM --------------------------------------------------
REM Activate Conda Environment and Run NTKCAP GUI
REM --------------------------------------------------

:: Set encoding to UTF-8
chcp 65001 >nul 2>&1

:: Detect Conda installation path
echo Detecting Conda installation...
set "CONDA_BAT="

:: Check common Conda installation locations
if exist "%UserProfile%\anaconda3\Scripts\activate.bat" (
    set "CONDA_BAT=%UserProfile%\anaconda3\Scripts\activate.bat"
) else if exist "%UserProfile%\miniconda3\Scripts\activate.bat" (
    set "CONDA_BAT=%UserProfile%\miniconda3\Scripts\activate.bat"
) else if exist "%ProgramData%\anaconda3\Scripts\activate.bat" (
    set "CONDA_BAT=%ProgramData%\anaconda3\Scripts\activate.bat"
) else if exist "%ProgramData%\miniconda3\Scripts\activate.bat" (
    set "CONDA_BAT=%ProgramData%\miniconda3\Scripts\activate.bat"
) else if exist "C:\Anaconda3\Scripts\activate.bat" (
    set "CONDA_BAT=C:\Anaconda3\Scripts\activate.bat"
) else if exist "C:\Miniconda3\Scripts\activate.bat" (
    set "CONDA_BAT=C:\Miniconda3\Scripts\activate.bat"
)

:: Initialize Conda
if defined CONDA_BAT (
    echo Found Conda at: %CONDA_BAT%
    CALL "%CONDA_BAT%"
) else (
    echo WARNING: Conda installation not found in common locations
    echo Attempting to use conda from PATH...
)

:: Get batch file directory (project root)
set "PROJECT_ROOT=%~dp0"
set "PROJECT_ROOT=%PROJECT_ROOT:~0,-1%"

:: Change to project directory
cd /d "%PROJECT_ROOT%"
echo Project directory: %PROJECT_ROOT%

:: Dynamically read environment name from activate_ntkcap.ps1
echo Detecting environment name...

:: Read from install directory using PowerShell
set ENV_NAME=ntkcap_env
if exist "install\activate_ntkcap.ps1" (
    for /f "delims=" %%i in ('powershell -Command "(Select-String -Path 'install\activate_ntkcap.ps1' -Pattern 'conda activate').Line -replace '.*conda activate\s+(\S+).*','$1'"') do set ENV_NAME=%%i
    echo Detected environment name: %ENV_NAME%
    goto env_detected
)

:: Final fallback to default
echo Warning: Cannot read environment name from install\activate_ntkcap.ps1, using default ntkcap_env

:env_detected

:: Activate environment
echo Activating Conda environment: %ENV_NAME%

:: Use --no-plugins to avoid encoding issues
set CONDA_NO_PLUGINS=true
CALL conda activate %ENV_NAME%

:: If failed, try with full path
if errorlevel 1 (
    echo Retrying activation with full path...
    if defined CONDA_BAT (
        for %%i in ("%CONDA_BAT%") do set "CONDA_ROOT=%%~dpi"
        CALL "%CONDA_ROOT%..\..\Scripts\activate.bat" %ENV_NAME%
    )
)

:: Detect and set CUDA_PATH
echo Detecting CUDA installation...
set "CUDA_PATH="

:: Check common CUDA installation paths
if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.7" (
    set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.7"
    echo Found CUDA v12.7
) else if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1" (
    set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1"
    echo Found CUDA v12.1
) else if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0" (
    set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0"
    echo Found CUDA v12.0
) else if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8" (
    set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
    echo Found CUDA v11.8
) else if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7" (
    set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7"
    echo Found CUDA v11.7
) else (
    echo WARNING: CUDA installation not found in standard locations
    echo Some features may not work properly
)

:: Set CUDA environment variables if found
if defined CUDA_PATH (
    echo Setting CUDA_PATH: %CUDA_PATH%
    set "PATH=%CUDA_PATH%\bin;%CUDA_PATH%\lib;%CUDA_PATH%\libnvvp;%PATH%"
) else (
    echo WARNING: CUDA_PATH not set - mmdeploy may fail to load
)

:: Add TensorRT to PATH
echo Adding TensorRT to PATH...
set "TENSORRT_DIR=%PROJECT_ROOT%\NTK_CAP\ThirdParty\TensorRT-8.6.1.6"
set "PATH=%TENSORRT_DIR%\lib;%TENSORRT_DIR%\bin;%PATH%"
set "TRT_LIBPATH=%TENSORRT_DIR%\lib"
set "TENSORRT_ROOT=%TENSORRT_DIR%"
echo TensorRT directory: %TENSORRT_DIR%

:: Set environment variables to suppress NumPy version compatibility warnings
set PYTHONWARNINGS=ignore::UserWarning:torch.distributed.optim.zero_redundancy_optimizer

:: Set Python to use UTF-8 encoding
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

:: Execute Python GUI with warning suppression
echo Starting NTKCAP GUI...
python -W ignore::UserWarning:torch.distributed.optim.zero_redundancy_optimizer .\NTKCAP_GUI.py

:: Keep window open to display errors or output
echo.
echo ==============================================
echo Program finished. Press CTRL+C to close...
pause >nul
