# 多相機 RGB 測試腳本使用說明

## 檔案
- [multicam_test_rgb.py](multicam_test_rgb.py) - 主程式
- [test_paths.py](test_paths.py) - 路徑配置測試工具

## 主要改進

### 移除硬編碼路徑
之前的版本使用硬編碼的路徑：
```python
BASE_DIR = r"C:/Users/000296/Desktop/AE400-4cam"
```

現在使用自動路徑檢測：
```python
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR
OPENNI2_BASE = PROJECT_ROOT / "NTK_CAP" / "ThirdParty" / "OpenNI2"
```

### 特點
- ✓ **自動路徑檢測**：不需要手動修改用戶名或路徑
- ✓ **相對路徑**：使用項目內的 `NTK_CAP/ThirdParty/OpenNI2` 資料夾
- ✓ **可移植性**：在任何電腦上都可以運行，只要在項目根目錄執行
- ✓ **路徑驗證**：啟動時會顯示並驗證所有相機的 OpenNI2 路徑

## 使用方法

### 1. 測試路徑配置
```bash
python test_paths.py
```

應該會看到類似輸出：
```
============================================================
路徑配置測試
============================================================
腳本目錄: C:\Users\MyUser\Desktop\NTKCAP
項目根目錄: C:\Users\MyUser\Desktop\NTKCAP
OpenNI2 基礎路徑: C:\Users\MyUser\Desktop\NTKCAP\NTK_CAP\ThirdParty\OpenNI2
OpenNI2 基礎路徑存在: True

相機 OpenNI2 路徑檢查:
------------------------------------------------------------
✓ 192.168.0.100        -> ...
✓ 192.168.3.100        -> ...
✓ 192.168.5.100        -> ...
✓ 192.168.7.100        -> ...
```

### 2. 運行多相機測試
```bash
python multicam_test_rgb.py
```

### 3. 安裝依賴（如果需要）
```bash
poetry install
```

或使用 pip：
```bash
pip install openni numpy opencv-python
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

## 故障排除

### 缺少 openni 模組
```bash
poetry install
# 或
pip install openni
```

### 找不到相機路徑
確保 OpenNI2 資料夾結構如下：
```
NTKCAP/
└── NTK_CAP/
    └── ThirdParty/
        └── OpenNI2/
            ├── 192.168.0.100/
            ├── 192.168.3.100/
            ├── 192.168.5.100/
            └── 192.168.7.100/
```

每個 IP 資料夾應包含：
- `OpenNI2.dll`
- `OpenNI.ini`
- `OpenNI2/Drivers/` 資料夾
