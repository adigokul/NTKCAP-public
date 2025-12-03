# NTKCAP Environment Activation Script for Windows
# Run this script to activate the NTKCAP environment

Write-Host "🚀 Activating NTKCAP environment (ntkcap_env)..." -ForegroundColor Green

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
