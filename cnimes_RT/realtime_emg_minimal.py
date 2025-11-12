"""realtime_emg_minimal.py

最簡版本的即時 EMG 監控工具
完全不依賴 matplotlib、scipy 等可能有相容性問題的套件

僅使用：
- 標準庫 (json, time, threading, etc.)
- websocket-client
- numpy (僅用於基本陣列操作)
"""
import argparse
import json
import time
import threading
import os
import sys
from collections import deque
import math

# 嘗試載入 numpy，如果失敗就使用內建 list
try:
    import numpy as np
    HAS_NUMPY = True
    print(f"NumPy {np.__version__} 載入成功")
except ImportError:
    HAS_NUMPY = False
    print("NumPy 不可用，使用純 Python 實作")

# 載入 websocket-client
try:
    import websocket
    HAS_WEBSOCKET = True
    print("websocket-client 載入成功")
except ImportError:
    HAS_WEBSOCKET = False
    print("錯誤: 需要安裝 websocket-client")
    print("執行: pip install websocket-client")

CHANNEL_NAMES = [
    'Tibialis_anterior_right', 'Rectus_Femoris_right', 'Biceps_femoris_right', 'Gastrocnemius_right',
    'Tibialis_anterior_left', 'Rectus_Femoris_left', 'Biceps_femoris_left', 'Gastrocnemius_left'
]

class MinimalEMGMonitor:
    """最簡版本的 EMG 監控器"""
    
    def __init__(self, uri="ws://localhost:31278/ws", max_samples=50):
        self.uri = uri
        self.max_samples = max_samples
        
        # 使用 deque 或 list 儲存資料
        self.data_buffers = []
        for i in range(8):  # 8 個通道
            if HAS_NUMPY:
                self.data_buffers.append(deque(maxlen=max_samples))
            else:
                self.data_buffers.append([])
        
        # 統計資料
        self.packet_count = 0
        self.total_samples = 0
        self.connection_status = "未連接"
        self.last_data_time = 0
        
        # 執行緒鎖
        self.lock = threading.Lock()
        
        # WebSocket 相關
        self.ws = None
        self.ws_thread = None
        self.running = False
        
    def add_data_point(self, channel, value):
        """新增資料點"""
        try:
            processed_value = abs(float(value))  # 簡單處理：取絕對值
            
            if HAS_NUMPY:
                self.data_buffers[channel].append(processed_value)
            else:
                # 使用 list 並手動限制長度
                self.data_buffers[channel].append(processed_value)
                if len(self.data_buffers[channel]) > self.max_samples:
                    self.data_buffers[channel].pop(0)
                    
        except (ValueError, TypeError):
            # 如果無法轉換為數字，忽略
            pass
    
    def get_channel_stats(self, channel):
        """取得通道統計資料"""
        with self.lock:
            if len(self.data_buffers[channel]) == 0:
                return {
                    'current': 0.0,
                    'max': 0.0,
                    'min': 0.0,
                    'avg': 0.0,
                    'samples': 0
                }
            
            data = list(self.data_buffers[channel])
            
            return {
                'current': data[-1] if data else 0.0,
                'max': max(data) if data else 0.0,
                'min': min(data) if data else 0.0,
                'avg': sum(data) / len(data) if data else 0.0,
                'samples': len(data)
            }
    
    def on_message(self, ws, message):
        """處理 WebSocket 訊息"""
        try:
            data_dict = json.loads(message)
            
            if "contents" in data_dict:
                contents = data_dict["contents"]
                if isinstance(contents, list) and len(contents) > 0:
                    
                    with self.lock:
                        for item in contents:
                            if "eeg" in item and isinstance(item["eeg"], list):
                                eeg_data = item["eeg"]
                                
                                # 處理每個通道的資料
                                for channel_idx, value in enumerate(eeg_data[:8]):  # 最多8個通道
                                    self.add_data_point(channel_idx, value)
                                
                                self.total_samples += 1
                        
                        self.packet_count += 1
                        self.last_data_time = time.time()
                        
                        # 每100個封包顯示一次狀態
                        if self.packet_count % 100 == 0:
                            print(f"[INFO] 已處理 {self.packet_count} 個封包，{self.total_samples} 個樣本")
                            
        except json.JSONDecodeError:
            # JSON 解析錯誤，忽略
            pass
        except Exception as e:
            # 其他錯誤，記錄但繼續執行
            print(f"[WARNING] 資料處理錯誤: {e}")
    
    def on_error(self, ws, error):
        print(f"[ERROR] WebSocket 錯誤: {error}")
        self.connection_status = f"錯誤: {str(error)[:50]}"
    
    def on_close(self, ws, close_status_code, close_msg):
        print("[INFO] WebSocket 連接已關閉")
        self.connection_status = "連接已關閉"
        
    def on_open(self, ws):
        print(f"[SUCCESS] WebSocket 已連接到 {self.uri}")
        self.connection_status = "已連接"
    
    def start_websocket(self):
        """啟動 WebSocket 連接"""
        if not HAS_WEBSOCKET:
            print("[ERROR] websocket-client 未安裝")
            return False
        
        try:
            self.ws = websocket.WebSocketApp(
                self.uri,
                on_open=self.on_open,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close
            )
            
            self.ws_thread = threading.Thread(target=self._run_websocket, daemon=True)
            self.ws_thread.start()
            
            return True
            
        except Exception as e:
            print(f"[ERROR] WebSocket 啟動失敗: {e}")
            return False
    
    def _run_websocket(self):
        """執行 WebSocket 連接"""
        try:
            self.ws.run_forever()
        except Exception as e:
            print(f"[ERROR] WebSocket 執行錯誤: {e}")
    
    def create_text_bar(self, value, max_value=1.0, width=40):
        """建立文字進度條"""
        if max_value <= 0:
            max_value = 1.0
            
        normalized = min(value / max_value, 1.0)
        filled_width = int(normalized * width)
        
        bar = "█" * filled_width + "░" * (width - filled_width)
        return bar
    
    def display_realtime(self):
        """即時顯示 EMG 資料"""
        print("\n=== 即時 EMG 監控 ===")
        print("按 Ctrl+C 停止\n")
        
        try:
            while self.running:
                # 清除螢幕
                os.system('cls' if os.name == 'nt' else 'clear')
                
                # 顯示標題和狀態
                print("=" * 80)
                print(f"即時 EMG 監控 - 狀態: {self.connection_status}")
                print(f"WebSocket: {self.uri}")
                print(f"封包數: {self.packet_count} | 樣本數: {self.total_samples}")
                
                # 顯示資料新鮮度
                if self.last_data_time > 0:
                    data_age = time.time() - self.last_data_time
                    freshness = "🟢 即時" if data_age < 1 else f"🟡 {data_age:.1f}s前"
                    print(f"資料新鮮度: {freshness}")
                
                print("=" * 80)
                
                # 計算所有通道的最大值用於正規化
                all_max = 0.0
                channel_stats = []
                
                for i in range(8):
                    stats = self.get_channel_stats(i)
                    channel_stats.append(stats)
                    if stats['max'] > all_max:
                        all_max = stats['max']
                
                if all_max == 0:
                    all_max = 1.0
                
                # 顯示每個通道
                for i, stats in enumerate(channel_stats):
                    name = CHANNEL_NAMES[i] if i < len(CHANNEL_NAMES) else f'Channel_{i+1}'
                    
                    # 建立進度條
                    bar = self.create_text_bar(stats['current'], all_max, 30)
                    
                    # 顯示資訊
                    print(f"{name:25} | {stats['current']:7.3f} |{bar}| "
                          f"Max:{stats['max']:7.3f} Avg:{stats['avg']:7.3f} ({stats['samples']:3d} samples)")
                
                print("=" * 80)
                
                # 顯示總體統計
                if any(stats['samples'] > 0 for stats in channel_stats):
                    active_channels = sum(1 for stats in channel_stats if stats['samples'] > 0)
                    total_current = sum(stats['current'] for stats in channel_stats)
                    total_avg = sum(stats['avg'] for stats in channel_stats)
                    
                    print(f"總計 - 活躍通道: {active_channels}/8 | "
                          f"即時總和: {total_current:.3f} | 平均總和: {total_avg:.3f}")
                else:
                    print("等待 EMG 資料...")
                
                print("按 Ctrl+C 停止監控")
                print()
                
                time.sleep(0.5)  # 更新頻率: 2Hz
                
        except KeyboardInterrupt:
            print("\n[INFO] 使用者中斷監控")
        except Exception as e:
            print(f"\n[ERROR] 顯示錯誤: {e}")
    
    def run(self):
        """執行監控"""
        print(f"[INFO] 啟動 EMG 監控...")
        
        # 啟動 WebSocket
        if not self.start_websocket():
            print("[ERROR] 無法啟動 WebSocket")
            return False
        
        # 等待連接
        print("[INFO] 等待 WebSocket 連接...")
        time.sleep(2)
        
        # 開始監控
        self.running = True
        try:
            self.display_realtime()
        finally:
            self.running = False
            if self.ws:
                self.ws.close()
        
        return True

def test_connection(uri, timeout=5):
    """測試連接"""
    print(f"[INFO] 測試連接: {uri}")
    
    if not HAS_WEBSOCKET:
        print("[ERROR] websocket-client 未安裝")
        return False
    
    try:
        ws = websocket.WebSocket()
        ws.settimeout(timeout)
        ws.connect(uri)
        ws.close()
        print("[SUCCESS] 連接測試成功")
        return True
    except Exception as e:
        print(f"[ERROR] 連接測試失敗: {e}")
        return False

def main():
    print("=" * 50)
    print("最簡版本即時 EMG 監控工具")
    print("=" * 50)
    print(f"NumPy: {'✓ ' + np.__version__ if HAS_NUMPY else '✗ 不可用'}")
    print(f"WebSocket: {'✓ 可用' if HAS_WEBSOCKET else '✗ 不可用'}")
    print("=" * 50)
    
    parser = argparse.ArgumentParser(description='Minimal Realtime EMG Monitor')
    parser.add_argument('--uri', default='ws://localhost:31278/ws', 
                       help='WebSocket URI (預設: ws://localhost:31278/ws)')
    parser.add_argument('--test', action='store_true', 
                       help='僅測試連接，不開始監控')
    args = parser.parse_args()

    print(f"[INFO] WebSocket URI: {args.uri}")
    
    try:
        if args.test:
            # 僅測試連接
            success = test_connection(args.uri)
            if success:
                print("[SUCCESS] 連接測試通過")
            else:
                print("[ERROR] 連接測試失敗")
            return
        
        # 建立監控器
        monitor = MinimalEMGMonitor(uri=args.uri)
        
        # 執行監控
        success = monitor.run()
        
        if not success:
            print("\n[ERROR] 監控執行失敗")
            print("\n請檢查:")
            print("1. EMG 裝置已連接並運行")
            print("2. WebSocket 伺服器已啟動")
            print("3. URI 位址正確")
            print("4. 網路連接正常")
            print("\n安裝必要套件:")
            print("pip install websocket-client")
            
    except KeyboardInterrupt:
        print('\n[INFO] 使用者中斷程式')
    except Exception as e:
        print(f'\n[ERROR] 程式錯誤: {e}')
        import traceback
        traceback.print_exc()
    finally:
        print("\n[INFO] 程式結束")

if __name__ == '__main__':
    main()