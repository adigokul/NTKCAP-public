# ============================================================================
# Windows CUDA Visual Studio Version Handler
# ============================================================================
# This script handles the compatibility issue between CUDA 11.8 and 
# Visual Studio 2022 version 14.40.33807 when compiling CUDA extensions.
#
# Problem: CUDA 11.8 host_config.h rejects VS 2022 version 14.40+ (>= 1940)
# Solution: Modify host_config.h to accept VS 2022 versions up to 1950
#
# Changes made:
# - Backup original host_config.h
# - Modify version check from "_MSC_VER >= 1940" to "_MSC_VER >= 1950"
# ============================================================================

#Requires -RunAsAdministrator

param(
    [string]$Package = "",
    [switch]$UseMim = $false,
    [switch]$RestoreBackup = $false,
    [switch]$SkipInstall = $false
)

Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host " Windows CUDA Visual Studio Version Handler" -ForegroundColor Cyan
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "注意: 此腳本需要管理員權限來修改 CUDA 系統檔案" -ForegroundColor Yellow
Write-Host ""

# Define CUDA paths
$cudaBasePath = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
$hostConfigPath = "$cudaBasePath\include\crt\host_config.h"
$backupPath = "$cudaBasePath\include\crt\host_config.h.backup"

# Check CUDA installation
Write-Host "[1/6] 檢查 CUDA 11.8 安裝..." -ForegroundColor Cyan
if (-not (Test-Path $cudaBasePath)) {
    Write-Host "✗ 錯誤: 未找到 CUDA 11.8 安裝於 $cudaBasePath" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $hostConfigPath)) {
    Write-Host "✗ 錯誤: 未找到 host_config.h 於 $hostConfigPath" -ForegroundColor Red
    exit 1
}

Write-Host "✓ 找到 CUDA 11.8 安裝" -ForegroundColor Green
Write-Host "  路徑: $cudaBasePath" -ForegroundColor Gray

# Handle restore backup option
Write-Host ""
if ($RestoreBackup) {
    Write-Host "[2/6] 還原備份檔案..." -ForegroundColor Cyan
    
    if (-not (Test-Path $backupPath)) {
        Write-Host "✗ 錯誤: 找不到備份檔案 $backupPath" -ForegroundColor Red
        exit 1
    }
    
    try {
        Copy-Item -Path $backupPath -Destination $hostConfigPath -Force
        Write-Host "✓ 成功還原 host_config.h 從備份" -ForegroundColor Green
        Write-Host ""
        Write-Host "原始檔案已還原。" -ForegroundColor Green
        exit 0
    }
    catch {
        Write-Host "✗ 還原失敗: $_" -ForegroundColor Red
        exit 1
    }
}

# Detect Visual Studio version
Write-Host "[2/6] 檢測 Visual Studio 版本..." -ForegroundColor Cyan
$vsPath = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC"
if (Test-Path $vsPath) {
    $vsVersions = Get-ChildItem $vsPath | Sort-Object Name -Descending | Select-Object -First 1
    Write-Host "✓ 找到 Visual Studio 2022: $($vsVersions.Name)" -ForegroundColor Green
    
    # Check if version is too new (>= 14.40 = _MSC_VER 1940)
    $versionParts = $vsVersions.Name -split '\.'
    $major = [int]$versionParts[0]
    $minor = [int]$versionParts[1]
    
    if ($major -eq 14 -and $minor -ge 40) {
        Write-Host "  警告: VS 2022 版本 $($vsVersions.Name) 對 CUDA 11.8 預設不支援" -ForegroundColor Yellow
        Write-Host "  _MSC_VER: 19$minor (_MSC_VER >= 1940 會被 CUDA 拒絕)" -ForegroundColor Yellow
        $needsPatch = $true
    }
    else {
        Write-Host "  VS 2022 版本 $($vsVersions.Name) 與 CUDA 11.8 相容" -ForegroundColor Green
        $needsPatch = $false
    }
} else {
    Write-Host "✗ 未找到 Visual Studio 2022" -ForegroundColor Red
    exit 1
}

# Backup original host_config.h
Write-Host ""
Write-Host "[3/6] 備份原始 host_config.h..." -ForegroundColor Cyan

if (Test-Path $backupPath) {
    Write-Host "✓ 備份檔案已存在: $backupPath" -ForegroundColor Green
    Write-Host "  (跳過備份，使用現有備份)" -ForegroundColor Gray
} else {
    try {
        Copy-Item -Path $hostConfigPath -Destination $backupPath -Force
        Write-Host "✓ 成功備份至: $backupPath" -ForegroundColor Green
    }
    catch {
        Write-Host "✗ 備份失敗: $_" -ForegroundColor Red
        Write-Host "  請確認您有管理員權限" -ForegroundColor Yellow
        exit 1
    }
}

# Read and check current host_config.h content
Write-Host ""
Write-Host "[4/6] 檢查並修改 host_config.h..." -ForegroundColor Cyan

try {
    $content = Get-Content $hostConfigPath -Raw -Encoding UTF8
    
    # Check current version constraint
    if ($content -match '#if _MSC_VER < 1910 \|\| _MSC_VER >= (\d+)') {
        $currentLimit = $matches[1]
        Write-Host "  當前版本限制: _MSC_VER >= $currentLimit" -ForegroundColor Gray
        
        if ($currentLimit -eq "1940") {
            Write-Host "  需要修改: 1940 -> 1950 (支援 VS 2022 14.40-14.49)" -ForegroundColor Yellow
            
            # Modify the version check
            $newContent = $content -replace '#if _MSC_VER < 1910 \|\| _MSC_VER >= 1940', '#if _MSC_VER < 1910 || _MSC_VER >= 1950'
            
            # Write back to file
            Set-Content -Path $hostConfigPath -Value $newContent -NoNewline -Encoding UTF8
            
            Write-Host "✓ 成功修改 host_config.h" -ForegroundColor Green
            Write-Host "  新版本限制: _MSC_VER >= 1950 (VS 2022 14.40-14.49 現在被支援)" -ForegroundColor Green
        }
        elseif ($currentLimit -eq "1950") {
            Write-Host "✓ host_config.h 已經修改過，無需再次修改" -ForegroundColor Green
        }
        else {
            Write-Host "  警告: 未預期的版本限制 $currentLimit" -ForegroundColor Yellow
            Write-Host "  手動檢查可能需要" -ForegroundColor Yellow
        }
    }
    else {
        Write-Host "✗ 無法找到版本檢查行" -ForegroundColor Red
        Write-Host "  host_config.h 格式可能已改變" -ForegroundColor Yellow
        exit 1
    }
}
catch {
    Write-Host "✗ 修改失敗: $_" -ForegroundColor Red
    Write-Host "  嘗試還原備份..." -ForegroundColor Yellow
    
    if (Test-Path $backupPath) {
        Copy-Item -Path $backupPath -Destination $hostConfigPath -Force
        Write-Host "✓ 已還原備份" -ForegroundColor Green
    }
    
    exit 1
}

# Verify modification
Write-Host ""
Write-Host "[5/6] 驗證修改..." -ForegroundColor Cyan

try {
    $verifyContent = Get-Content $hostConfigPath -Raw -Encoding UTF8
    
    if ($verifyContent -match '_MSC_VER >= 1950') {
        Write-Host "✓ 驗證成功: host_config.h 已正確修改" -ForegroundColor Green
    }
    else {
        Write-Host "✗ 驗證失敗: 修改未生效" -ForegroundColor Red
        exit 1
    }
}
catch {
    Write-Host "✗ 驗證失敗: $_" -ForegroundColor Red
    exit 1
}

# Install package if requested
Write-Host ""
if ($SkipInstall) {
    Write-Host "[6/6] 跳過套件安裝 (--SkipInstall 已指定)" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "============================================================================" -ForegroundColor Green
    Write-Host " ✓ host_config.h 修改完成!" -ForegroundColor Green
    Write-Host "============================================================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "現在您可以正常編譯需要 CUDA 的 Python 套件了。" -ForegroundColor Green
    Write-Host ""
    Write-Host "安裝 mmcv 範例:" -ForegroundColor Cyan
    Write-Host "  mim install mmcv==2.1.0" -ForegroundColor Gray
    Write-Host "或" -ForegroundColor Cyan
    Write-Host "  pip install mmcv==2.1.0" -ForegroundColor Gray
    Write-Host ""
    Write-Host "如需還原原始設定:" -ForegroundColor Cyan
    Write-Host "  .\install\scripts\windows_cuda_vs_version_handle.ps1 -RestoreBackup" -ForegroundColor Gray
    Write-Host ""
    exit 0
}

if ([string]::IsNullOrWhiteSpace($Package)) {
    Write-Host "[6/6] 未指定套件，跳過安裝" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "============================================================================" -ForegroundColor Green
    Write-Host " ✓ host_config.h 修改完成!" -ForegroundColor Green
    Write-Host "============================================================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "現在您可以正常編譯需要 CUDA 的 Python 套件了。" -ForegroundColor Green
    Write-Host ""
    Write-Host "安裝 mmcv 範例:" -ForegroundColor Cyan
    Write-Host "  .\install\scripts\windows_cuda_vs_version_handle.ps1 -Package 'mmcv==2.1.0' -UseMim" -ForegroundColor Gray
    Write-Host ""
    exit 0
}

# Check if conda environment is activated
if (-not $env:CONDA_DEFAULT_ENV) {
    Write-Host "✗ 錯誤: 請先啟動 conda 環境" -ForegroundColor Red
    Write-Host "  執行: conda activate ntkcap" -ForegroundColor Yellow
    exit 1
}

Write-Host "[6/6] 安裝套件: $Package" -ForegroundColor Cyan
Write-Host "  當前環境: $env:CONDA_DEFAULT_ENV" -ForegroundColor Gray
Write-Host "------------------------------------------------------------" -ForegroundColor Gray

if ($UseMim) {
    Write-Host "使用 mim 安裝..." -ForegroundColor Yellow
    $installCmd = "mim install $Package"
} else {
    Write-Host "使用 pip 安裝..." -ForegroundColor Yellow
    $installCmd = "pip install $Package --no-cache-dir"
}

Write-Host "執行命令: $installCmd" -ForegroundColor Gray
Write-Host "------------------------------------------------------------" -ForegroundColor Gray
Write-Host ""

# Execute installation
$ErrorActionPreference = 'Continue'
Invoke-Expression $installCmd
$exitCode = $LASTEXITCODE

Write-Host ""
Write-Host "------------------------------------------------------------" -ForegroundColor Gray

# Report result
Write-Host ""
if ($exitCode -eq 0) {
    Write-Host "============================================================================" -ForegroundColor Green
    Write-Host " ✓ 安裝成功!" -ForegroundColor Green
    Write-Host "============================================================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "套件 $Package 已成功編譯並安裝。" -ForegroundColor Green
    Write-Host ""
} else {
    Write-Host "============================================================================" -ForegroundColor Red
    Write-Host " ✗ 安裝失敗 (Exit Code: $exitCode)" -ForegroundColor Red
    Write-Host "============================================================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "雖然 host_config.h 已修改，但安裝仍然失敗。" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "其他可能的解決方案:" -ForegroundColor Yellow
    Write-Host "1. 檢查磁碟空間 (C: 磁碟需要足夠空間用於編譯)" -ForegroundColor Yellow
    Write-Host "2. 降級 Visual Studio 2022 到 14.29.x" -ForegroundColor Yellow
    Write-Host "3. 升級到 CUDA 12.x (原生支援較新的 VS 2022)" -ForegroundColor Yellow
    Write-Host "4. 使用預編譯的 wheel 檔案" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "更多資訊:" -ForegroundColor Yellow
    Write-Host "  - CUDA Compatibility: https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/" -ForegroundColor Cyan
    Write-Host "  - OpenMMLab mmcv: https://mmcv.readthedocs.io/en/latest/get_started/installation.html" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "如需還原原始設定:" -ForegroundColor Cyan
    Write-Host "  .\install\scripts\windows_cuda_vs_version_handle.ps1 -RestoreBackup" -ForegroundColor Gray
    Write-Host ""
}

exit $exitCode
