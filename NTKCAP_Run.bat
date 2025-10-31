@echo off
REM --------------------------------------------------
REM Activate Conda Environment and Run NTKCAP GUI
REM --------------------------------------------------

:: 初始化 Conda（這一行很重要，確保能使用 conda activate）
CALL "%UserProfile%\anaconda3\Scripts\activate.bat"

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
CALL conda activate %ENV_NAME%

:: Add TensorRT to PATH (使用相對路徑)
echo Adding TensorRT to PATH...
set "TENSORRT_DIR=%PROJECT_ROOT%\NTK_CAP\ThirdParty\TensorRT-8.6.1.6"
set "PATH=%TENSORRT_DIR%\lib;%TENSORRT_DIR%\bin;%PATH%"
set "TRT_LIBPATH=%TENSORRT_DIR%\lib"
set "TENSORRT_ROOT=%TENSORRT_DIR%"
echo TensorRT directory: %TENSORRT_DIR%

:: 設置環境變數來抑制 NumPy 版本兼容性警告
set PYTHONWARNINGS=ignore::UserWarning:torch.distributed.optim.zero_redundancy_optimizer

:: Execute Python GUI with warning suppression
echo Starting NTKCAP GUI...
python -W ignore::UserWarning:torch.distributed.optim.zero_redundancy_optimizer .\NTKCAP_GUI.py

:: Keep window open to display errors or output
echo.
echo ==============================================
echo Program finished. Press any key to close...
pause >nul
