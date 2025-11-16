# AE400 相機整合完成報告

## 概述

成功在 NTKCAP_GUI 中整合了 AE400 深度相機（RGB 模式）與 USB Webcam 的切換功能。

## 實現日期
2025-11-16

## 修改的文件

### 1. config/config.json
**修改內容：**
- 添加 `type` 字段（"usb" 或 "ae400"）
- 添加 `ae400` 配置區塊，包含：
  - `ips`: AE400 相機的 IP 地址列表
  - `openni2_base`: OpenNI2 庫的相對路徑

**關鍵代碼：**
```json
{
    "cam": {
        "type": "usb",
        "ae400": {
            "ips": ["192.168.0.100", "192.168.3.100", "192.168.5.100", "192.168.7.100"],
            "openni2_base": "NTK_CAP/ThirdParty/OpenNI2"
        }
    }
}
```

### 2. GUI_source/CameraProcess.py
**修改內容：**
- 添加 `cam_type` 和 `cam_config` 參數到 `__init__()`
- 實現 `_init_usb_camera()` - 初始化 USB 相機
- 實現 `_init_ae400_camera()` - 初始化 AE400 相機
- 實現 `_read_frame(cap)` - 統一的幀讀取接口
- 實現 `_release_camera(cap)` - 統一的資源釋放接口
- 修改 `run()` 方法使用新的初始化邏輯

**關鍵功能：**
- 支援雙模式初始化和讀取
- 自動處理 RGB ↔ BGR 轉換（AE400 輸出 RGB888）
- 自動調整解析度到 1920x1080
- 完整的錯誤處理

### 3. NTKCAP_GUI.py
**修改位置：**

#### a. 狀態列按鈕（第 496-537 行）
- 添加 `btn_camera_mode` 切換按鈕
- 樣式：藍色（Webcam）/ 綠色（AE400）
- 從 config.json 讀取初始狀態

#### b. 事件處理函數（第 539-585 行）
- 實現 `on_camera_mode_toggled()`
- 更新 config.json
- 提示用戶重啟相機以應用變更

#### c. opencamera() 函數（第 1232-1344 行）
- 讀取相機配置
- 根據 `type` 準備相機配置
- 使用迴圈創建 4 個 CameraProcess 實例
- 傳遞 `cam_type` 和 `cam_config` 參數

### 4. README_multicam.md
**更新內容：**
- 添加 GUI 切換功能使用說明
- 更新技術細節和代碼架構
- 添加故障排除指南
- 添加性能優化建議

### 5. 新增測試文件
- `test_paths.py` - 驗證 OpenNI2 路徑配置
- `multicam_test_rgb.py` - 獨立的 AE400 測試程式

## 功能特點

### ✅ 已實現
1. **一鍵切換**：狀態列按鈕即時切換相機類型
2. **自動配置**：切換後自動保存到 config.json
3. **雙模式支援**：完整支援 USB 和 AE400 兩種相機
4. **統一接口**：所有相機功能（錄製、校準等）保持一致
5. **錯誤處理**：完整的異常處理和用戶提示
6. **路徑管理**：自動處理 OpenNI2 路徑，無需硬編碼

### ⚠️ 限制
1. **僅 RGB 流**：目前僅使用 AE400 的彩色流，深度資料未整合
2. **全局切換**：4 個相機必須使用同一類型（不支援混合模式）
3. **需要重啟相機**：切換後需要關閉並重新打開相機

## 使用方法

### 快速開始
1. 啟動 NTKCAP_GUI.py
2. 點擊狀態列的 "Camera: Webcam" 按鈕
3. 切換到 "Camera: AE400" 模式
4. 點擊 "Open Camera"

### 配置要求
- OpenNI2 庫位於：`NTK_CAP/ThirdParty/OpenNI2/`
- 每個 AE400 相機需要獨立的資料夾（以 IP 命名）
- 已安裝 `openni` Python 包

## 測試建議

### 1. USB 模式測試
```bash
python NTKCAP_GUI.py
# 確保按鈕顯示 "Camera: Webcam" (藍色)
# 點擊 Open Camera
# 驗證 4 個 USB 相機視窗正常顯示
```

### 2. AE400 模式測試
```bash
# 先測試路徑
python test_paths.py

# 測試獨立腳本
python multicam_test_rgb.py

# 測試 GUI
python NTKCAP_GUI.py
# 點擊狀態列按鈕切換到 "Camera: AE400" (綠色)
# 重啟相機
# 驗證 4 個 AE400 相機視窗正常顯示
```

### 3. 切換測試
1. 在 Webcam 模式下打開相機
2. 關閉相機
3. 切換到 AE400 模式
4. 重新打開相機
5. 驗證正確顯示 AE400 畫面
6. 重複以上步驟反向切換

## 技術細節

### 初始化流程
```
NTKCAP_GUI.opencamera()
  ├─ 讀取 config.json
  ├─ 根據 type 準備 cam_configs[]
  ├─ 創建 4 個 CameraProcess
  │   └─ 傳遞 cam_type 和 cam_config
  │
  └─ CameraProcess.run()
      ├─ if cam_type == 'usb':
      │   └─ cap = _init_usb_camera()
      ├─ elif cam_type == 'ae400':
      │   └─ cap = _init_ae400_camera()
      │       └─ openni2.initialize(path)
      │
      └─ while True:
          └─ ret, frame = _read_frame(cap)
              ├─ USB: cap.read() → BGR
              └─ AE400: stream.read_frame() → RGB → BGR
```

### 幀格式轉換
```python
# AE400 輸出 RGB888，需轉換為 BGR
rgb = np.asarray(buf).reshape((h, w, 3))
bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

# 自動調整解析度
if (w != 1920) or (h != 1080):
    bgr = cv2.resize(bgr, (1920, 1080), interpolation=cv2.INTER_LINEAR)
```

## 已知問題與解決方案

### 問題 1：AE400 FPS 較低
**原因：** `wait_for_any_stream()` timeout 設置為 2000ms
**解決：** 可調整 timeout 或優化網路連接

### 問題 2：多進程 OpenNI2 初始化
**原因：** OpenNI2 不能在主進程初始化後傳遞給子進程
**解決：** 在每個 CameraProcess 的 `run()` 方法中獨立初始化

### 問題 3：RGB/BGR 格式不一致
**原因：** USB 輸出 BGR，AE400 輸出 RGB
**解決：** 在 `_read_frame()` 中統一轉換為 BGR

## 後續改進建議

### 短期（1-2 週）
- [ ] 添加相機連接狀態指示器
- [ ] 實現 AE400 自動重連機制
- [ ] 優化 AE400 FPS 性能

### 中期（1-2 月）
- [ ] 整合 AE400 深度資料流
- [ ] 支援混合模式（USB + AE400）
- [ ] 添加相機參數調整界面

### 長期（3+ 月）
- [ ] 支援更多相機類型（RealSense、Kinect 等）
- [ ] 實現熱插拔檢測
- [ ] 添加相機校準工具

## 文件結構
```
NTKCAP/
├── config/
│   └── config.json                    [修改] 添加 type 和 ae400 配置
├── GUI_source/
│   └── CameraProcess.py               [修改] 支援雙相機模式
├── NTK_CAP/ThirdParty/OpenNI2/
│   ├── 192.168.0.100/
│   ├── 192.168.3.100/
│   ├── 192.168.5.100/
│   └── 192.168.7.100/
├── NTKCAP_GUI.py                      [修改] 添加切換按鈕和邏輯
├── multicam_test_rgb.py               [新增] 獨立測試腳本
├── test_paths.py                      [新增] 路徑驗證工具
├── README_multicam.md                 [修改] 完整使用說明
└── CAMERA_INTEGRATION_SUMMARY.md      [新增] 本文件
```

## 版本資訊
- **項目**: NTKCAP Motion Capture System
- **功能**: AE400 Depth Camera Integration (RGB only)
- **版本**: 1.0.0
- **Python**: >= 3.10, < 3.13
- **依賴**: PyQt6, opencv-python, numpy, openni

## 聯絡資訊
如有問題或建議，請查閱：
- [README_multicam.md](README_multicam.md) - 詳細使用說明
- [NTKCAP_GUI.py](NTKCAP_GUI.py) - 主程式
- [GUI_source/CameraProcess.py](GUI_source/CameraProcess.py) - 相機處理模組
