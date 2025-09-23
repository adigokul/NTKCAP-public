## NTKCAP EMG Integration Guide

### 🎯 功能總結
EMG事件記錄系統已成功整合到NTKCAP GUI的Record Task和Stop Record按鈕中。

### 📋 使用方式

#### 1. 基本操作流程
1. **開啟攝影機** - 點擊 "Open Cameras"
2. **選擇患者** - 在Patient List中選擇患者ID
3. **輸入任務名稱** - 在Task name欄位輸入任務名稱
4. **開始錄製** - 點擊 "Record Task" 按鈕
   - ✅ 同時啟動動作捕捉錄製
   - ✅ 同時啟動EMG錄製
   - ✅ 自動添加"Recording Start"事件標記
5. **停止錄製** - 點擊 "Stop Record" 按鈕
   - ✅ 同時停止動作捕捉錄製
   - ✅ 同時停止EMG錄製
   - ✅ 自動添加"Recording Stop"事件標記

#### 2. EMG數據儲存位置
```
Patient_data/
└── [Patient_ID]/
    └── [YYYY_MM_DD]/
        └── raw_data/
            └── [Task_Name]/
                ├── videos/          # 動作捕捉影片
                └── emg_data.csv     # EMG數據檔案
```

#### 3. EMG設備設定
- **預設WebSocket位址**: `ws://localhost:31278/ws`
- **預設通道數**: 8通道
- **數據格式**: Cygnus相容CSV格式

### 🔧 手動添加事件標記
在錄製過程中，可以調用以下方法添加自定義事件：
```python
# 在GUI中調用（例如綁定到按鍵或按鈕）
self.add_emg_event_marker(141, "特定動作開始")
self.add_emg_event_marker(142, "特定動作結束")
```

### 📊 EMG數據格式
生成的CSV檔案包含：
- **時間戳記** (timestamp)
- **EMG通道數據** (Ch1-Ch8)
- **事件標記** (Event欄位)
- **事件ID** (Event ID)
- **事件描述** (Event Description)

### ⚙️ 故障排除

#### EMG連接失敗
如果看到 `hostname is invalid` 錯誤：
1. 確認Cygnus EMG軟體正在運行
2. 確認WebSocket服務在 `ws://localhost:31278/ws` 啟動
3. 檢查防火牆設定
4. 確認EMG設備已正確連接
5. 確認WebSocket URI格式正確（需要包含 `ws://` 前綴和 `/ws` 後綴）

#### 修改EMG設定
在GUI的`__init__`方法中可以修改：
```python
self.emg_uri = "ws://localhost:31278/ws"  # EMG WebSocket位址
self.emg_channel_count = 8               # EMG通道數
```

### 📝 事件ID建議
- **100**: 錄製開始
- **200**: 錄製結束  
- **141**: 特定動作/測試開始
- **142**: 特定動作/測試結束
- **999**: 手動標記事件

### ✅ 驗證EMG整合
1. 日誌顯示 `🎯 Starting EMG recording` - EMG啟動成功
2. 日誌顯示 `📁 Output file: [路徑]` - 檔案路徑正確
3. 如果EMG設備連接，會看到 `✅ EMG connection established`
4. 錄製結束時會看到 `EMG recording stopped successfully`

### 🎉 整合完成狀態
- ✅ Record Task按鈕已整合EMG錄製
- ✅ Stop Record按鈕已整合EMG停止
- ✅ 自動事件標記(開始/結束)
- ✅ CSV格式與Cygnus相容
- ✅ 錯誤處理和fallback機制
- ✅ 手動事件標記功能

現在NTKCAP系統同時支援動作捕捉和EMG錄製，兩者完全同步！

(NTKCAP) PS D:\NTKCAP> python NTK_CAP\script_py\emg_localhost.py --help       
EMG WebSocket Data Reader with Event Markers
==================================================
usage: emg_localhost.py [-h] [--uri URI] [--timeout TIMEOUT]
                        [--output OUTPUT] [--continuous] [--test-events]      
                        [--scan-frequency] [--test-samples TEST_SAMPLES]      

EMG WebSocket Data Reader

options:
  -h, --help            show this help message and exit
  --uri URI, -u URI     Direct WebSocket URI specification, skip auto scan    
                        (e.g.: ws://localhost:31278/ws)
  --timeout TIMEOUT, -t TIMEOUT
                        Connection timeout in seconds (default: 5)
  --output OUTPUT, -o OUTPUT
                        Output CSV file path (default: auto-generated
                        timestamp filename)
  --continuous, -c      Continuous mode: continuously receive and save data   
                        until manual stop
  --test-events, -te    Test mode: test EMG recording with event markers      
  --scan-frequency, -sf
                        Enable frequency scanning mode for auto-discovery     
  --test-samples TEST_SAMPLES, -ts TEST_SAMPLES
                        Number of samples for test mode (default: 3000)       

Usage Examples:
  python emg_localhost.py                              # Auto scan, single read (quick mode)
  python emg_localhost.py --scan-frequency             # Auto scan with full frequency range
  python emg_localhost.py --uri ws://localhost:31278/ws  # Direct URI specification
  python emg_localhost.py -u ws://192.168.1.100:31278   # Use short parameter 
  python emg_localhost.py --uri localhost:31278         # Auto add ws:// prefix
  python emg_localhost.py -c -o emg_data.csv            # Continuous mode with output file
  python emg_localhost.py -u localhost:31278 -c         # Continuous mode auto filename
  python emg_localhost.py --test-events                 # Test EMG with event markers (3000 samples)
  python emg_localhost.py -te --test-samples 5000       # Test with 5000 samples
  python emg_localhost.py -te -sf --test-samples 1000   # Test with frequency scan
