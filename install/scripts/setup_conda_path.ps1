# Setup Conda as Global Environment Variable
# This script adds Conda to system PATH so it's available globally
# Run as Administrator

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Conda Global PATH Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if running as administrator
$currentPrincipal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
$isAdmin = $currentPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Host "ERROR: This script must be run as Administrator!" -ForegroundColor Red
    Write-Host "Right-click PowerShell and select 'Run as Administrator'" -ForegroundColor Yellow
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

# Detect conda installation
$condaPath = $null
$possiblePaths = @(
    "C:\ProgramData\Miniconda3",
    "C:\ProgramData\Anaconda3",
    "C:\Users\$env:USERNAME\Miniconda3",
    "C:\Users\$env:USERNAME\Anaconda3"
)

foreach ($path in $possiblePaths) {
    if (Test-Path $path) {
        $condaPath = $path
        break
    }
}

if ($null -eq $condaPath) {
    Write-Host "ERROR: Conda installation not found!" -ForegroundColor Red
    Write-Host "Searched locations:" -ForegroundColor Yellow
    foreach ($path in $possiblePaths) {
        Write-Host "  - $path" -ForegroundColor Gray
    }
    exit 1
}

Write-Host "Found Conda at: $condaPath" -ForegroundColor Green
Write-Host ""

# Paths to add
$pathsToAdd = @(
    "$condaPath",
    "$condaPath\Scripts",
    "$condaPath\Library\bin",
    "$condaPath\Library\mingw-w64\bin",
    "$condaPath\Library\usr\bin"
)

Write-Host "Paths to be added to system PATH:" -ForegroundColor Cyan
foreach ($path in $pathsToAdd) {
    Write-Host "  - $path" -ForegroundColor White
}
Write-Host ""

# Get current system PATH
$currentPath = [Environment]::GetEnvironmentVariable("Path", "Machine")

# Check which paths are already in PATH
$pathsAlreadyAdded = @()
$pathsToAddNew = @()

foreach ($path in $pathsToAdd) {
    if ($currentPath -like "*$path*") {
        $pathsAlreadyAdded += $path
    } else {
        $pathsToAddNew += $path
    }
}

if ($pathsAlreadyAdded.Count -gt 0) {
    Write-Host "Already in PATH (skipping):" -ForegroundColor Yellow
    foreach ($path in $pathsAlreadyAdded) {
        Write-Host "  - $path" -ForegroundColor Gray
    }
    Write-Host ""
}

if ($pathsToAddNew.Count -eq 0) {
    Write-Host "All Conda paths are already in system PATH!" -ForegroundColor Green
    Write-Host "No changes needed." -ForegroundColor Green
    Write-Host ""
} else {
    Write-Host "Will add to system PATH:" -ForegroundColor Green
    foreach ($path in $pathsToAddNew) {
        Write-Host "  - $path" -ForegroundColor White
    }
    Write-Host ""

    $confirm = Read-Host "Add these paths to system PATH? (y/N)"

    if ($confirm -eq "y" -or $confirm -eq "Y") {
        try {
            # Build new PATH
            $newPath = $currentPath
            foreach ($path in $pathsToAddNew) {
                $newPath = "$path;$newPath"
            }

            # Set system PATH
            [Environment]::SetEnvironmentVariable("Path", $newPath, "Machine")

            Write-Host ""
            Write-Host "SUCCESS: Conda paths added to system PATH!" -ForegroundColor Green
            Write-Host ""
            Write-Host "IMPORTANT: You need to restart your terminal/PowerShell for changes to take effect." -ForegroundColor Yellow
            Write-Host ""

            # Also add to current session
            $env:Path = "$($pathsToAddNew -join ';');$env:Path"
            Write-Host "Paths also added to current session (temporary)." -ForegroundColor Cyan
            Write-Host ""
        }
        catch {
            Write-Host ""
            Write-Host "ERROR: Failed to update system PATH: $($_.Exception.Message)" -ForegroundColor Red
            Write-Host ""
            exit 1
        }
    } else {
        Write-Host "Operation cancelled by user." -ForegroundColor Yellow
        exit 0
    }
}

# Run conda init for PowerShell
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Initialize Conda for PowerShell?" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "This will enable 'conda activate' in PowerShell." -ForegroundColor White
Write-Host "Your PowerShell profile will be modified." -ForegroundColor Yellow
Write-Host ""

$initConda = Read-Host "Run 'conda init powershell'? (y/N)"

if ($initConda -eq "y" -or $initConda -eq "Y") {
    try {
        & "$condaPath\Scripts\conda.exe" init powershell
        Write-Host ""
        Write-Host "SUCCESS: Conda initialized for PowerShell!" -ForegroundColor Green
        Write-Host ""
        Write-Host "IMPORTANT: Restart PowerShell to use 'conda activate' command." -ForegroundColor Yellow
        Write-Host ""
    }
    catch {
        Write-Host ""
        Write-Host "ERROR: Failed to initialize conda: $($_.Exception.Message)" -ForegroundColor Red
        Write-Host ""
    }
} else {
    Write-Host "Skipped conda init." -ForegroundColor Gray
    Write-Host ""
}

# Verify conda is accessible
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Verification" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

try {
    $condaVersion = & "$condaPath\Scripts\conda.exe" --version
    Write-Host "Conda version: $condaVersion" -ForegroundColor Green
    Write-Host ""
    Write-Host "Setup complete!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Cyan
    Write-Host "1. Restart your PowerShell/Command Prompt" -ForegroundColor White
    Write-Host "2. Test with: conda --version" -ForegroundColor White
    Write-Host "3. Test with: conda env list" -ForegroundColor White
    Write-Host ""
}
catch {
    Write-Host "WARNING: Could not verify conda installation" -ForegroundColor Yellow
    Write-Host ""
}

Read-Host "Press Enter to exit"
