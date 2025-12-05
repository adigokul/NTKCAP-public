# Merge Notes: laptop_v Branch

## Summary

The `laptop_v` branch is the **original USB-only camera implementation** before AE400 depth camera support was added to master. It uses the DirectShow backend for USB webcam handling on Windows laptops.

**Commits:**
- `64db8e39` - "successfully test the whole process on laptop"

**Statistics:**
- 32 files changed
- 700 insertions, 3,690 deletions (net reduction of ~3,000 lines)

---

## Key Changes

### 1. Camera System Simplification

#### CameraProcess.py
The camera initialization has been **significantly simplified**:

**Before (master):**
- Supported both USB and AE400 (depth camera) modes
- Required `cam_type` and `cam_config` parameters
- Had separate `_init_usb_camera()`, `_init_ae400_camera()`, `_read_frame()`, `_release_camera()` methods
- ~170 lines

**After (laptop_v):**
- USB-only with **DirectShow backend** (`cv2.CAP_DSHOW`)
- No extra parameters needed
- Single initialization block
- ~70 lines

Key optimization:
```python
# Uses DirectShow backend (Windows-specific) for better USB camera handling
cap = cv2.VideoCapture(self.cam_id, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduces latency
```

### 2. NTKCAP_GUI.py Cleanup

**Removed features:**
- Camera mode toggle button (AE400/Webcam switch)
- Calibration center marker toggle (`btn_toggle_marker`)
- AE400 configuration reading and IP-based camera setup
- Related event handlers: `toggle_calibration_marker()`, `on_camera_mode_toggled()`

**Changed:**
- Button text colors changed from `white` to `black` for better visibility
- Camera process creation simplified (explicit p1, p2, p3, p4 instead of loop)

### 3. Config Simplification

#### config/config.json
```json
// Before (master) - Complex with AE400 support
{
    "cam": {
        "type": "ae400",
        "list": [0, 1, 2, 3],
        "ae400": {
            "ips": ["192.168.0.100", "192.168.3.100", ...],
            "openni2_base": "NTK_CAP/ThirdParty/OpenNI2"
        }
    }
}

// After (laptop_v) - Simple USB only
{
    "cam": {
        "number": 4,
        "list": [],
        "resolution": [1920, 1080],
        "name": "HD camera"
    }
}
```

### 4. Setup.ps1 Changes

**Removed:**
- `-SkipToPostMMDeploy` parameter
- Early configuration questions (installation method, CUDA handling)
- Some advanced TensorRT deployment logic

**Changed:**
- Simplified CUDA warning messages
- MMPose installation reverted to simpler `pip install -r requirements.txt`
- Removed chumpy workaround (uses standard install)
- Restored pycuda installation (not skipped)

### 5. Deleted Files

The following files/features were removed as they're not needed for laptop deployment:

| File | Purpose |
|------|---------|
| `CAMERA_INTEGRATION_SUMMARY.md` | AE400 integration documentation |
| `README_multicam.md` | Multi-camera setup guide |
| `REMOTE_CALCULATION_README.md` | Remote calculation documentation |
| `diagnose_ae400.py` | AE400 diagnostic tool |
| `multicam_test_rgb.py` | Multi-camera testing |
| `remote_calculate.py` | Remote calculation server |
| `trigger_remote_calculation.py` | Remote calculation trigger |
| `test_remote_trigger.bat` | Remote trigger test script |
| `cnimes_RT/check_cygnus_kernel.py` | Cygnus kernel checker |
| `cnimes_RT/realtime_emg_plot.py` | Real-time EMG plotting |
| `config/calculation_example.json` | Calculation config example |
| `config/remote_server_example.json` | Remote server config example |
| `rebuild_trt.ps1` | TensorRT rebuild script |
| `rebuild_trt_models.bat` | TensorRT models rebuild batch |
| `install/scripts/fix_encoding.ps1` | Encoding fix script |
| `install/scripts/setup_conda_path.ps1` | Conda path setup |

### 6. Restored Files

| File | Purpose |
|------|---------|
| `install/scripts/windows_cuda_vs_version_handle.ps1` | CUDA/VS version handling |
| `test_cameras_simple.py` | Simple camera test script |

---

## Impact Assessment

### Advantages of laptop_v
1. **DirectShow backend** - Better USB camera compatibility on Windows laptops
2. **Reduced latency** - Buffer size set to 1 for real-time performance
3. **Tested on laptop** - Verified working on laptop hardware

### What's Removed (features added to master after laptop_v branched)
1. AE400 depth camera support
2. Remote calculation capability
3. Camera mode switching at runtime
4. Calibration marker overlay feature
5. Some advanced setup options

---

## Merge Recommendation

**Proceed with merge** - The laptop_v branch provides optimized USB camera handling suitable for laptop deployments. The removed features (AE400, remote calculation) can be re-added later if needed.

**Post-merge actions needed:**
1. Verify USB cameras work correctly with DirectShow backend
2. Update documentation to reflect USB-only camera support
3. Consider keeping `rebuild_trt.ps1` if TensorRT model rebuilding is still needed
