#!/bin/bash
# NTKCAP Environment Activation Script

# Find and set CUDA environment variables
CUDA_PATHS=("/usr/local/cuda-11.8" "/usr/local/cuda" "/opt/cuda")
for path in "${CUDA_PATHS[@]}"; do
    if [[ -d "$path" && -f "$path/bin/nvcc" ]]; then
        export PATH="$path/bin:$PATH"
        export LD_LIBRARY_PATH="$path/lib64:$LD_LIBRARY_PATH"
        export CUDA_HOME="$path"
        echo "🔧 CUDA found at: $path"
        break
    fi
done

# Add Poetry to PATH
export PATH="$HOME/.local/bin:$PATH"

# Activate conda environment
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate ntkcap
    echo "🚀 NTKCAP environment activated!"
    echo "Current Python: $(which python)"
    echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'PyTorch not loaded')"
    echo ""
    echo "Available commands:"
    echo "  python final_NTK_Cap_GUI.py   # Run main GUI"
    echo "  python NTKCAP_GUI.py          # Run alternative GUI"
    echo "  poetry run <command>          # Run with Poetry"
else
    echo "❌ Conda not found. Please install Miniconda first."
    exit 1
fi

# Start a new shell with the environment
exec bash
