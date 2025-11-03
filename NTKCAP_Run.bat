@echo off
REM --------------------------------------------------
REM Activate Conda Environment and Run NTKCAP GUI
REM --------------------------------------------------

:: 設置編碼為 UTF-8 來處理中文路徑
chcp 65001 >nul 2>&1

:: 動態檢測 Conda 安裝路徑
echo Detecting Conda installation...
set "CONDA_BAT="

:: 檢查常見的 Conda 安裝位置
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

:: 初始化 Conda
if defined CONDA_BAT (
    echo Found Conda at: %CONDA_BAT%
    CALL "%CONDA_BAT%"
) else (
    echo WARNING: Conda installation not found in common locations
    echo Attempting to use conda from PATH...
)

:: 取得批次檔所在目錄（專案根目錄）
set "PROJECT_ROOT=%~dp0"
set "PROJECT_ROOT=%PROJECT_ROOT:~0,-1%"

:: 切換到專案目錄
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

:: 使用 --no-plugins 選項來避免編碼問題
set CONDA_NO_PLUGINS=true
CALL conda activate %ENV_NAME%

:: 如果上面失敗，嘗試使用完整路徑
if errorlevel 1 (
    echo Retrying activation with full path...
    if defined CONDA_BAT (
        for %%i in ("%CONDA_BAT%") do set "CONDA_ROOT=%%~dpi"
        CALL "%CONDA_ROOT%..\..\Scripts\activate.bat" %ENV_NAME%
    )
)

:: Add TensorRT to PATH (使用相對路徑)
echo Adding TensorRT to PATH...
set "TENSORRT_DIR=%PROJECT_ROOT%\NTK_CAP\ThirdParty\TensorRT-8.6.1.6"
set "PATH=%TENSORRT_DIR%\lib;%TENSORRT_DIR%\bin;%PATH%"
set "TRT_LIBPATH=%TENSORRT_DIR%\lib"
set "TENSORRT_ROOT=%TENSORRT_DIR%"
echo TensorRT directory: %TENSORRT_DIR%

:: 設置環境變數來抑制 NumPy 版本兼容性警告
set PYTHONWARNINGS=ignore::UserWarning:torch.distributed.optim.zero_redundancy_optimizer

:: 設置 Python 使用 UTF-8 編碼
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
