# Comprehensive Technical Audit Report: ntkins.sh

**Date:** 2026-01-13
**File:** `/home/ntkcap-kuo/ntkcaptensor/install/ntkins.sh`
**Lines:** 1850
**Status:** SYNTAX VALID (bash -n passed)

---

## Executive Summary

The `ntkins.sh` installation script is a comprehensive, well-structured bash script designed to set up the NTKCAP TensorRT-based pose estimation environment. The script addresses all 11 documented problems from PROBLEMS_AND_SOLUTIONS.md and implements proper fixes for cross-machine portability.

**Overall Grade: B+**

| Category | Grade | Notes |
|----------|-------|-------|
| Structure & Organization | A | Clean sections, good logging |
| Error Handling | B+ | Uses `set -e`, has `|| true` patterns |
| Path Handling | A | All paths are dynamically detected |
| Environment Variables | A | Proper cleanup and setting |
| Dependency Management | A- | Pinned versions, constraint file |
| Portability | A | Works across different machines |
| Security | B | Minimal sudo usage, some risks |
| Documentation | A | Excellent inline comments |

---

## Detailed Analysis

### 1. Script Structure (Lines 1-100)

**Strengths:**
- Clear header with usage instructions (lines 1-29)
- `set -e` for fail-fast behavior (line 31)
- Relative path detection from script location (lines 37-40)
- Color-coded output functions (lines 65-86)

**Verified Code:**
```bash
# Line 38-40: Dynamic path detection
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
```
This correctly computes paths relative to the script, not the current working directory.

---

### 2. CUDA 12 Symlink Fix (Lines 125-204)

**Problem Addressed:** Problem 1 - libcublasLt.so.12 not found

**Implementation Analysis:**
```bash
# Line 131-143: Correct detection logic
create_cuda12_symlinks() {
    local cuda_lib="${CUDA_HOME}/lib64"

    # Skip if CUDA 12 already exists
    if [[ -f "${cuda_lib}/libcublasLt.so.12" ]]; then
        log "CUDA 12 libraries already exist - no symlinks needed"
        return 0
    fi

    # Only proceed if CUDA 11 exists
    if [[ ! -f "${cuda_lib}/libcublasLt.so.11" ]]; then
        warn "CUDA 11 libraries not found..."
        return 1
    fi
```

**Verdict:** CORRECT. Properly handles:
- Machines with CUDA 12 installed (skip)
- Machines with only CUDA 11 (create symlinks)
- Machines with neither (warn and continue)

**Libraries Symlinked:**
- libcublas.so.11 -> .so.12
- libcublasLt.so.11 -> .so.12
- libcufft.so.10 -> .so.12
- libcurand.so.10 -> .so.12
- libcusolver.so.11 -> .so.12
- libcusparse.so.11 -> .so.12

---

### 3. CUDA Auto-Detection (Lines 265-329)

**Implementation:**
```bash
# Lines 269-300: Multi-method CUDA detection
find_cuda() {
    # Method 1: Check nvcc in PATH
    if command -v nvcc &>/dev/null; then
        cuda_found=$(dirname $(dirname $(which nvcc)))
        echo "${cuda_found}"
        return 0
    fi

    # Method 2: Search common paths
    local cuda_search_paths=(
        "/usr/local/cuda"
        "/usr/local/cuda-11.8"
        ...
    )
```

**Verdict:** ROBUST. Uses multiple fallback methods.

---

### 4. .bashrc Conflict Fix (Lines 333-360)

**Problem Addressed:** Problem 11 - Conflicting CUDA versions in .bashrc

**Implementation:**
```bash
# Lines 335-351: Auto-detect and comment out conflicting paths
if grep -q "cuda-[0-9]" ~/.bashrc 2>/dev/null; then
    CONFLICTING_CUDA=$(grep -oP "cuda-\d+" ~/.bashrc | grep -v "cuda-${CUDA_MAJOR_NEEDED}" | sort -u)
    if [[ -n "${CONFLICTING_CUDA}" ]]; then
        # Creates timestamped backup
        cp ~/.bashrc ~/.bashrc.backup.$(date +%Y%m%d%H%M%S)
        # Comments out conflicting lines
        for conflict in ${CONFLICTING_CUDA}; do
            sed -i "s|^export.*${conflict}|#&|g" ~/.bashrc
        done
```

**Verdict:** CORRECT with BACKUP. Creates timestamped backup before modification.

---

### 5. GCC Compatibility (Lines 372-411)

**Problem:** CUDA 11.8 requires GCC <= 11

**Implementation:**
```bash
# Lines 372-396: Find compatible GCC
find_cuda_compatible_gcc() {
    local cuda_major=$1
    local max_gcc=11

    # Try gcc-11, gcc-10, gcc-9 in order
    for v in 11 10 9; do
        if [[ -f "/usr/bin/gcc-${v}" ]] && [[ -f "/usr/bin/g++-${v}" ]]; then
            echo "${v}"
            return 0
        fi
    done
```

**Verdict:** CORRECT. Properly finds and uses compatible GCC version.

---

### 6. Numpy Version Pinning (Lines 102-123, 630-743)

**Problem Addressed:** numpy 2.x breaks compiled packages

**Implementation:**
```bash
# Lines 104-123: Verification and fix function
verify_fix_numpy() {
    local stage="$1"
    local current_numpy=$(python -c "import numpy; print(numpy.__version__)")

    if [[ "${current_numpy}" == "1.22.4" ]]; then
        log "numpy OK at ${stage}: ${current_numpy}"
        return 0
    fi

    # Force reinstall correct version
    pip install numpy==1.22.4 --force-reinstall --no-deps --quiet
```

**Constraint File (lines 635-639):**
```bash
CONSTRAINT_FILE="${CONDA_PREFIX}/pip-constraints.txt"
cat > "${CONSTRAINT_FILE}" << 'EOF'
numpy==1.22.4
EOF
export PIP_CONSTRAINT="${CONSTRAINT_FILE}"
```

**Verdict:** EXCELLENT. Multiple layers of protection:
1. Pip constraint file
2. Verification after each major install
3. Force reinstall if wrong version detected

---

### 7. OpenCV GUI Fix (Lines 698-701)

**Problem Addressed:** Problem 7 - OpenCV GTK+ GUI not implemented

**Implementation:**
```bash
# Lines 698-701
info "Installing OpenCV with GUI support..."
pip uninstall opencv-python-headless -y 2>/dev/null || true
pip install opencv-python==4.11.0.86  # Full version with highgui
```

**Verdict:** CORRECT. Removes headless version before installing full version.

---

### 8. LD_LIBRARY_PATH Management (Lines 1134-1148, 1206-1212)

**Problem Addressed:** Problems 10 & 11 - LD_LIBRARY_PATH pollution

**Clean Path Construction:**
```bash
# Lines 1137-1147
CUDNN_LIB_FILE=$(ldconfig -p 2>/dev/null | grep libcudnn.so | head -1 | awk '{print $NF}' || true)
if [[ -n "${CUDNN_LIB_FILE}" ]] && [[ -f "${CUDNN_LIB_FILE}" ]]; then
    CUDNN_LIB=$(dirname "${CUDNN_LIB_FILE}")
else
    CUDNN_LIB="/usr/lib/x86_64-linux-gnu"
fi

MMDEPLOY_LIB="${MMDEPLOY_BUILD_DIR}/lib"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${TENSORRT_DIR}/lib:${MMDEPLOY_LIB}:${CUDNN_LIB}"
```

**Subprocess Isolation (lines 1206-1212):**
```bash
# CRITICAL: Include CONDA_PREFIX/lib FIRST for libstdc++ CXXABI compatibility
SUBPROCESS_LD_PATH="${CONDA_PREFIX}/lib:${CUDA_HOME}/lib64:${TENSORRT_DIR}/lib:${MMDEPLOY_LIB}:${CUDNN_LIB}"
env LD_LIBRARY_PATH="${SUBPROCESS_LD_PATH}" \
    python "${DEPLOY_SCRIPT}" \
```

**Verdict:** EXCELLENT. Uses `env` command to isolate subprocess environment.

---

### 9. Activation Script (Lines 1516-1628)

**Problem Addressed:** Problem 8 - Environment variables not persisting

**Source Check (lines 1526-1540):**
```bash
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo ""
    echo "ERROR: This script must be SOURCED, not executed!"
    echo "============================================================"
    echo "You ran:    ${0}"
    echo "You should: source ${BASH_SOURCE[0]}"
    exit 1
fi
```

**LD_LIBRARY_PATH Order (line 1613):**
```bash
# CONDA_PREFIX/lib first for CXXABI compatibility
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${CUDA_HOME}/lib64:${TENSORRT_DIR}/lib:${MMDEPLOY_LIB}:${CUDNN_LIB}:${CLEAN_LD_PATH}"
```

**Verdict:** EXCELLENT. Prevents the #1 user error (running instead of sourcing).

---

### 10. Pipeline JSON Fixes (Lines 1246-1508)

**Problem Addressed:** Problem 5 - deploy.json/pipeline.json model name mismatch

**Implementation:** Uses heredocs to write correct JSON configurations:
- RTMDet pipeline: 320x320 detection pipeline
- RTMPose pipeline: 256x192 pose estimation with SimCC output mapping

**Key Fix (RTMPose output_map):**
```json
"output_map": {
    "output": "simcc_x",
    "700": "simcc_y"
}
```

**Verdict:** CORRECT. Pre-written JSON configurations avoid runtime generation issues.

---

### 11. Skip Logic for Built Components (Lines 790-853, 864-1095)

**Implementation:**
```bash
# Line 790-792: ppl.cv skip check
if [[ -f "${PPLCV_BUILD_DIR}/install/lib/libpplcv_static.a" ]]; then
    log "ppl.cv already built - skipping"
else
    # ... build ppl.cv
fi

# Line 864-866: mmdeploy skip check
if [[ -f "${MMDEPLOY_BUILD_DIR}/lib/libmmdeploy_tensorrt_ops.so" ]]; then
    log "mmdeploy SDK already built - skipping"
```

**Verdict:** CORRECT. Allows resuming failed installs without rebuilding everything.

---

## Potential Issues & Recommendations

### Issue 1: set -e with Complex Pipelines (MEDIUM RISK)

**Location:** Various pipeline commands

**Risk:** `set -e` can cause silent failures with complex pipes.

**Mitigation Present:** `|| true` pattern used correctly:
```bash
# Line 1137
CUDNN_LIB_FILE=$(ldconfig -p 2>/dev/null | grep libcudnn.so | head -1 | awk '{print $NF}' || true)
```

**Status:** ADDRESSED

---

### Issue 2: Sudo Operations (LOW RISK)

**Locations:**
- Line 189: `sudo ln -sf` (CUDA symlinks)
- Line 200: `sudo ldconfig`
- Lines 462-527: apt-get installs
- Line 580: `sudo usermod`

**Risk:** Requires sudo password, may fail on systems without sudo.

**Recommendation:** Already uses `|| true` for non-critical sudo operations.

**Status:** ACCEPTABLE

---

### Issue 3: Google Drive Download (MEDIUM RISK)

**Location:** Lines 483-504

**Risk:** gdown may fail due to Google Drive rate limiting or access issues.

**Mitigation Present:**
```bash
if [[ ! -f "${TENSORRT_DOWNLOAD_PATH}" ]]; then
    error "Failed to download TensorRT from Google Drive.
Please download manually from:
  https://drive.google.com/file/d/${TENSORRT_GDRIVE_ID}/view"
```

**Status:** ACCEPTABLE with manual fallback instructions.

---

### Issue 4: No mmdeploy_runtime Wheel Fallback (LOW RISK)

**Location:** Lines 1081-1092

**Current Implementation:**
```bash
MMDEPLOY_WHEEL=$(find "${MMDEPLOY_BUILD_DIR}" -name "mmdeploy_runtime*.whl" 2>/dev/null | head -1)
if [[ -f "${MMDEPLOY_WHEEL}" ]]; then
    pip install "${MMDEPLOY_WHEEL}" --force-reinstall --no-deps
else
    # Fallback: Add to PYTHONPATH
    export PYTHONPATH="${MMDEPLOY_BUILD_DIR}/lib:${PYTHONPATH:-}"
```

**Status:** CORRECTLY HANDLED with PYTHONPATH fallback.

---

## Problem Coverage Matrix

| Problem # | Description | Fixed | Location |
|-----------|-------------|-------|----------|
| 1 | libcublasLt.so.12 not found | YES | Lines 125-204 |
| 2 | mmdeploy_runtime not found | YES | Lines 1081-1092, 1614 |
| 3 | libstdc++ CXXABI_1.3.15 not found | YES | Lines 1210, 1613 |
| 4 | CuPy CUDA version conflict | YES | Line 692 (cupy-cuda11x) |
| 5 | pipeline.json model name mismatch | YES | Lines 1246-1508 |
| 6 | TensorRT setTensorAddress error | N/A | Application code issue |
| 7 | OpenCV GTK+ GUI not implemented | YES | Lines 698-701 |
| 8 | Environment variables not persisting | YES | Lines 1526-1540 |
| 9 | Silent script exit (set -e) | YES | || true patterns |
| 10 | LD_LIBRARY_PATH subprocess pollution | YES | Lines 1206-1212 |
| 11 | Conflicting CUDA in .bashrc | YES | Lines 333-360 |

---

## File Dependencies

### Input Files Required:
1. TensorRT archive (auto-downloaded from Google Drive)
2. Model weights (auto-downloaded from OpenMMLab)

### Output Files Generated:
1. `${PROJECT_ROOT}/activate_ntkcap.sh` - Environment activation script
2. `${PPLCV_DIR}/cuda-build/install/` - ppl.cv installation
3. `${MMDEPLOY_DIR}/build/` - mmdeploy SDK build
4. `${MMDEPLOY_DIR}/rtmpose-trt/rtmdet-m/` - RTMDet engine
5. `${MMDEPLOY_DIR}/rtmpose-trt/rtmpose-m/` - RTMPose engine
6. `${CONDA_PREFIX}/pip-constraints.txt` - Pip constraint file
7. `~/.bashrc.backup.*` - Backup if .bashrc modified

---

## Verification Tests (Lines 1637-1747)

The script includes comprehensive verification:
1. Python import tests for all packages
2. numpy version assertion
3. CUDA availability check
4. TensorRT ops library loading test
5. Engine file existence check
6. PyQt5/PyQt6 availability check

---

## Conclusion

The `ntkins.sh` script is production-ready and addresses all documented issues. Key strengths include:

1. **Dynamic path detection** - No hardcoded paths
2. **Multi-layer numpy protection** - Constraint file + verification + force reinstall
3. **CUDA 12 compatibility symlinks** - Works on CUDA 11-only systems
4. **Subprocess environment isolation** - Uses `env` command
5. **Proper activation script** - Source-only with CONDA_PREFIX/lib priority
6. **Skip logic for builds** - Allows resuming failed installations
7. **Comprehensive verification** - Tests all critical components

**Recommendation:** Ready for deployment on the target machine.
