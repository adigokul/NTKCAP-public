# 多相機系統使用說明

## 檔案
- [multicam_test_rgb.py](multicam_test_rgb.py) - 獨立的多相機 RGB 測試程式
- [test_paths.py](test_paths.py) - 路徑配置測試工具
- [NTKCAP_GUI.py](NTKCAP_GUI.py) - 主 GUI 程式（已整合相機切換功能）
- [GUI_source/CameraProcess.py](GUI_source/CameraProcess.py) - 相機處理模組（支援 USB 和 AE400）
- [config/config.json](config/config.json) - 相機配置文件

## 主要功能

### 1. GUI 相機模式切換（推薦使用）

在 NTKCAP_GUI 的狀態列添加了 **Camera Mode** 切換按鈕，可以在以下兩種模式間切換：

- **Webcam 模式**（藍色按鈕）：使用 USB 網絡攝影機
- **AE400 模式**（綠色按鈕）：使用 AE400 深度相機的 RGB 流

**使用方法：**
1. 啟動 NTKCAP_GUI.py
2. 在底部狀態列找到 "Camera: Webcam" 或 "Camera: AE400" 按鈕
3. 點擊按鈕切換相機類型
4. 關閉並重新打開相機以應用變更

**特點：**
- ✓ **一鍵切換**：無需手動編輯配置文件
- ✓ **即時保存**：切換後自動更新 config.json
- ✓ **狀態提示**：按鈕顏色和文字顯示當前模式
- ✓ **安全設計**：相機打開時會提示用戶需要重啟相機

### 2. 獨立測試腳本

**multicam_test_rgb.py** - 用於測試 AE400 相機

- ✓ **自動路徑檢測**：不需要手動修改用戶名或路徑
- ✓ **相對路徑**：使用項目內的 `NTK_CAP/ThirdParty/OpenNI2` 資料夾
- ✓ **可移植性**：在任何電腦上都可以運行
- ✓ **路徑驗證**：啟動時會顯示並驗證所有相機的 OpenNI2 路徑

### 3. 雙模式相機支援

系統現在支援兩種相機類型的無縫切換：

| 特性 | USB Webcam | AE400 Depth Camera |
|------|------------|-------------------|
| 識別方式 | 設備索引 (0, 1, 2, 3) | IP 地址 |
| 解析度 | 1920x1080 | 1920x1080 (可調整) |
| 幀率 | 30 FPS | 30 FPS |
| 錄製功能 | 完整支援 | 完整支援 |
| RGB 輸出 | ✓ | ✓ |
| Depth 輸出 | ✗ | ✓ (未整合) |

## 使用方法

### 方法 1：使用 GUI（推薦）

1. **啟動主程式**
   ```bash
   python NTKCAP_GUI.py
   ```

2. **切換相機模式**
   - 在底部狀態列找到 "Camera: Webcam" 按鈕
   - 點擊切換到 "Camera: AE400" 模式（或反之）
   - 按鈕會變色（藍色=Webcam，綠色=AE400）

3. **打開相機**
   - 點擊 "Open Camera" 按鈕
   - 系統會根據當前模式初始化相應的相機
   - 4 個相機視窗會顯示實時畫面

4. **使用相機功能**
   - 所有原有功能（錄製、校準、追蹤等）保持不變
   - AE400 模式下僅使用 RGB 流，深度資料未整合

### 方法 2：測試 AE400 相機（獨立腳本）

1. **測試路徑配置**
   ```bash
   python test_paths.py
   ```

   應該會看到：
   ```
   ============================================================
   路徑配置測試
   ============================================================
   ✓ 192.168.0.100 -> C:\Users\MyUser\Desktop\NTKCAP\...
   ✓ 192.168.3.100 -> ...
   ✓ 192.168.5.100 -> ...
   ✓ 192.168.7.100 -> ...
   ```

2. **運行多相機測試**
   ```bash
   python multicam_test_rgb.py
   ```
   - 會打開 4 個獨立視窗顯示 AE400 RGB 畫面
   - 按 `q` 或 `ESC` 關閉任一視窗
   - `Ctrl+C` 結束所有視窗

### 方法 3：手動編輯配置文件

編輯 [config/config.json](config/config.json)：

```json
{
    "cam": {
        "number": 4,
        "type": "usb",  // 改為 "ae400" 以使用深度相機
        "list": [0, 1, 2, 3],
        "resolution": [1920, 1080],
        "name": "HD camera",
        "ae400": {
            "ips": [
                "192.168.0.100",
                "192.168.3.100",
                "192.168.5.100",
                "192.168.7.100"
            ],
            "openni2_base": "NTK_CAP/ThirdParty/OpenNI2"
        }
    }
}
```

## 相機 IP 地址
腳本配置了 4 台相機：
- 192.168.0.100
- 192.168.3.100
- 192.168.5.100
- 192.168.7.100

如需修改，編輯 [multicam_test_rgb.py:18-23](multicam_test_rgb.py#L18-L23) 中的 `CAM_IPS` 列表。

## 控制
- **退出單個視窗**：在任何視窗按 `q` 或 `ESC`
- **退出全部**：按 `Ctrl+C` 或關閉所有視窗

## 技術細節

### 配置文件格式

[config/config.json](config/config.json) 結構：

```json
{
    "cam": {
        "number": 4,              // 相機數量
        "type": "usb",            // 相機類型: "usb" 或 "ae400"
        "list": [0, 1, 2, 3],     // USB 相機的設備索引
        "resolution": [1920, 1080],
        "name": "HD camera",
        "ae400": {
            "ips": [              // AE400 相機的 IP 地址列表
                "192.168.0.100",
                "192.168.3.100",
                "192.168.5.100",
                "192.168.7.100"
            ],
            "openni2_base": "NTK_CAP/ThirdParty/OpenNI2"  // OpenNI2 庫的相對路徑
        }
    }
}
```

### 代碼架構

**CameraProcess 雙模式支援：**

```python
# USB 模式
cap = cv2.VideoCapture(device_index)
ret, frame = cap.read()  # BGR 格式

# AE400 模式
openni2.initialize(openni2_path)
dev = openni2.Device.open_any()
stream = dev.create_color_stream()
stream.start()

frame_data = stream.read_frame()
rgb = np.asarray(buf).reshape((h, w, 3))
bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)  # 轉換為 BGR
```

**關鍵修改文件：**
1. [config/config.json](config/config.json) - 添加 `type` 和 `ae400` 配置
2. [GUI_source/CameraProcess.py](GUI_source/CameraProcess.py) - 實現 `_init_usb_camera()`, `_init_ae400_camera()`, `_read_frame()`
3. [NTKCAP_GUI.py](NTKCAP_GUI.py) - 添加切換按鈕和 `on_camera_mode_toggled()` 事件處理

## 故障排除

### 1. 缺少 openni 模組
```bash
poetry install
# 或
pip install openni numpy opencv-python
```

### 2. AE400 相機無法初始化

**檢查 OpenNI2 路徑：**
```
NTKCAP/
└── NTK_CAP/
    └── ThirdParty/
        └── OpenNI2/
            ├── 192.168.0.100/
            │   ├── OpenNI2.dll
            │   ├── OpenNI.ini
            │   └── OpenNI2/Drivers/
            ├── 192.168.3.100/
            ├── 192.168.5.100/
            └── 192.168.7.100/
```

運行測試：
```bash
python test_paths.py
```

### 3. GUI 按鈕無反應

- 確認 config.json 格式正確（有效的 JSON）
- 檢查 console 輸出的錯誤訊息
- 確認 `config_path` 指向正確的目錄

### 4. 相機切換後畫面異常

- 確保已關閉並重新打開相機
- 檢查 OpenNI2 驅動是否正確安裝
- 確認 AE400 相機的 IP 地址設定正確
- 查看 console 輸出的初始化訊息

### 5. USB 相機索引錯誤

如果 USB 相機無法開啟，可能是設備索引不正確：

1. 手動測試相機索引：
   ```python
   import cv2
   for i in range(10):
       cap = cv2.VideoCapture(i)
       if cap.isOpened():
           print(f"Camera found at index {i}")
           cap.release()
   ```

2. 更新 config.json 中的 `cam.list` 欄位

### 6. 性能問題

**AE400 FPS 較低：**
- AE400 使用 `wait_for_any_stream()` 可能比 USB 相機慢
- 考慮調整 timeout 參數（目前為 2000ms）
- 確保網路連接穩定（千兆乙太網路）

**記憶體使用過高：**
- 檢查共享記憶體緩衝區大小（預設 4 幀）
- 確認沒有記憶體洩漏（長時間運行）

## 未來改進

- [ ] 整合 AE400 的深度資料流
- [ ] 支援混合模式（部分 USB + 部分 AE400）
- [ ] 添加相機連接狀態監控
- [ ] 實現自動重連機制（參考 multicam_test_rgb.py）
- [ ] 支援更多相機類型（RealSense 等）

## 相關資源

- OpenNI2 文檔: https://github.com/occipital/openni2
- AE400 用戶手冊: 查看相機附帶文檔
- NTKCAP 主程式: [NTKCAP_GUI.py](NTKCAP_GUI.py)
