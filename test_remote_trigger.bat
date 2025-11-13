@echo off
REM Test Remote Calculation Trigger
REM This script tests the remote calculation interface

echo ========================================
echo NTKCAP Remote Calculation Test
echo ========================================

REM Activate conda environment
echo.
echo [1/5] Activating conda environment: ntkcap_fast
call conda activate ntkcap_fast
if errorlevel 1 (
    echo ❌ Failed to activate conda environment
    pause
    exit /b 1
)

REM Change to NTKCAP directory
cd /d D:\NTKCAP
echo ✅ Working directory: %CD%

REM Install required packages (if not already installed)
echo.
echo [2/5] Checking required packages...
pip show paramiko >nul 2>&1
if errorlevel 1 (
    echo Installing paramiko and scp...
    pip install paramiko scp
)
echo ✅ Required packages ready

REM Test connection only
echo.
echo [3/5] Testing connection to remote server...
python trigger_remote_calculation.py --server-config config/remote_server_example.json --test-connection
if errorlevel 1 (
    echo ❌ Connection test failed
    echo Please check config/remote_server_example.json settings
    pause
    exit /b 1
)

REM Show calculation config
echo.
echo [4/5] Calculation configuration:
type config\calculation_example.json
echo.

REM Ask before running
echo [5/5] Ready to trigger remote calculation
echo.
echo Configuration files:
echo   Server: config/remote_server_example.json
echo   Calculation: config/calculation_example.json
echo.
set /p CONFIRM="Do you want to proceed with calculation? (Y/N): "
if /i not "%CONFIRM%"=="Y" (
    echo Cancelled by user
    pause
    exit /b 0
)

REM Trigger remote calculation
echo.
echo Starting remote calculation...
python trigger_remote_calculation.py ^
    --server-config config/remote_server_example.json ^
    --calc-config config/calculation_example.json ^
    --verbose

if errorlevel 1 (
    echo.
    echo ❌ Remote calculation failed
    pause
    exit /b 1
) else (
    echo.
    echo ✅ Remote calculation completed successfully
)

echo.
echo ========================================
echo Test completed
echo ========================================
pause
