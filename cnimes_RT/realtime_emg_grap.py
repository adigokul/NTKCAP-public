"""realtime_emg_plot_simple.py

簡化版的即時 EMG 繪圖工具 - 避免相容性問題

使用純 Python 和基本庫實現即時 EMG 資料顯示
"""
import argparse
import threading
import time
import json
import sys
import os
from collections import deque
import math

try:
    import tkinter as tk
    from tkinter import ttk, messagebox
    HAS_TKINTER = True
except ImportError:
    HAS_TKINTER = False
    print("警告: 無法載入 tkinter，將使用文字模式")

try:
    import websocket
    HAS_WEBSOCKET = True
except ImportError:
    HAS_WEBSOCKET = False
    print("錯誤: 需要安裝 websocket-client")
    print("請執行: pip install websocket-client")

CHANNEL_NAMES = [
    'Tibialis_anterior_right', 'Rectus_Femoris_right', 'Biceps_femoris_right', 'Gastrocnemius_right',
    'Tibialis_anterior_left', 'Rectus_Femoris_left', 'Biceps_femoris_left', 'Gastrocnemius_left'
]

class SimpleEMGPlotter:
    def __init__(self, uri="ws://localhost:31278/ws", max_samples=50):
        self.uri = uri
        self.max_samples = max_samples
        self.running = False
        
        # 資料緩衝區
        self.data_buffer = [deque(maxlen=max_samples) for _ in range(8)]
        self.lock = threading.Lock()
        
        # WebSocket 相關
        self.ws = None
        self.ws_thread = None
        
        # 統計資料
        self.packet_count = 0
        self.last_update = time.time()
        
    def on_message(self, ws, message):
        try:
            data_dict = json.loads(message)
            if "contents" in data_dict:
                contents = data_dict["contents"]
                if isinstance(contents, list) and len(contents) > 0:
                    # 提取 EMG 資料
                    for item in contents:
                        if "eeg" in item and isinstance(item["eeg"], list):
                            eeg = item["eeg"]
                            with self.lock:
                                for i, value in enumerate(eeg[:8]):  # 最多8個通道
                                    # 簡單的絕對值處理
                                    processed_value = abs(float(value))
                                    self.data_buffer[i].append(processed_value)
                                self.packet_count += 1
        except Exception as e:
            # 忽略解析錯誤
            pass

    def on_error(self, ws, error):
        print(f"WebSocket 錯誤: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        print("WebSocket 連接已關閉")

    def on_open(self, ws):
        print(f"WebSocket 已連接到 {self.uri}")

    def start_ws(self):
        if not HAS_WEBSOCKET:
            print("錯誤: websocket-client 未安裝")
            return False
            
        try:
            self.ws = websocket.WebSocketApp(self.uri,
                                           on_open=self.on_open,
                                           on_message=self.on_message,
                                           on_error=self.on_error,
                                           on_close=self.on_close)
            self.ws_thread = threading.Thread(target=self._run_ws, daemon=True)
            self.ws_thread.start()
            return True
        except Exception as e:
            print(f"WebSocket 啟動錯誤: {e}")
            return False

    def _run_ws(self):
        try:
            self.ws.run_forever()
        except Exception as e:
            print(f"WebSocket 執行錯誤: {e}")

    def get_current_data(self):
        """取得目前的資料"""
        with self.lock:
            data = []
            for i in range(8):
                if len(self.data_buffer[i]) > 0:
                    # 取得最新的值
                    data.append(list(self.data_buffer[i]))
                else:
                    data.append([0.0])
            return data, self.packet_count

class TkinterEMGPlotter(SimpleEMGPlotter):
    """使用 Tkinter 的圖形化版本"""
    
    def __init__(self, uri="ws://localhost:31278/ws", max_samples=50):
        super().__init__(uri, max_samples)
        self.root = None
        self.canvas = None
        self.labels = []
        self.progress_bars = []
        
    def create_gui(self):
        self.root = tk.Tk()
        self.root.title("即時 EMG 監控")
        self.root.geometry("800x600")
        self.root.configure(bg='black')
        
        # 標題
        title_label = tk.Label(self.root, text="即時 EMG 資料監控", 
                              font=("Arial", 16, "bold"), 
                              fg='white', bg='black')
        title_label.pack(pady=10)
        
        # 連接狀態
        self.status_label = tk.Label(self.root, text=f"連接中... ({self.uri})", 
                                   font=("Arial", 10), 
                                   fg='yellow', bg='black')
        self.status_label.pack()
        
        # 統計資料
        self.stats_label = tk.Label(self.root, text="封包數: 0", 
                                  font=("Arial", 10), 
                                  fg='cyan', bg='black')
        self.stats_label.pack()
        
        # 通道框架
        channels_frame = tk.Frame(self.root, bg='black')
        channels_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # 為每個通道創建顯示
        colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        for i in range(8):
            # 通道框架
            channel_frame = tk.Frame(channels_frame, bg='black')
            channel_frame.pack(fill='x', pady=2)
            
            # 通道名稱
            name = CHANNEL_NAMES[i] if i < len(CHANNEL_NAMES) else f'Channel {i+1}'
            label = tk.Label(channel_frame, text=f"{name}:", 
                           font=("Arial", 9), 
                           fg='white', bg='black', width=25, anchor='w')
            label.pack(side='left')
            self.labels.append(label)
            
            # 進度條
            progress = ttk.Progressbar(channel_frame, length=400, mode='determinate')
            progress.pack(side='left', fill='x', expand=True, padx=(10, 10))
            self.progress_bars.append(progress)
            
            # 數值顯示
            value_label = tk.Label(channel_frame, text="0.000", 
                                 font=("Arial", 9, "bold"), 
                                 fg=colors[i], bg='black', width=8)
            value_label.pack(side='right')
            self.labels.append(value_label)
        
        # 控制按鈕
        button_frame = tk.Frame(self.root, bg='black')
        button_frame.pack(pady=10)
        
        self.start_button = tk.Button(button_frame, text="開始", 
                                    command=self.toggle_connection,
                                    font=("Arial", 12), 
                                    bg='green', fg='white')
        self.start_button.pack(side='left', padx=5)
        
        quit_button = tk.Button(button_frame, text="結束", 
                              command=self.quit_app,
                              font=("Arial", 12), 
                              bg='red', fg='white')
        quit_button.pack(side='left', padx=5)
        
        # 開始更新循環
        self.running = True
        self.update_display()
        
    def toggle_connection(self):
        if not self.running:
            if self.start_ws():
                self.start_button.config(text="停止", bg='red')
                self.running = True
            else:
                messagebox.showerror("錯誤", "無法連接到 WebSocket 伺服器")
        else:
            self.running = False
            if self.ws:
                self.ws.close()
            self.start_button.config(text="開始", bg='green')
            
    def update_display(self):
        if self.running:
            try:
                data, packet_count = self.get_current_data()
                
                # 更新統計
                self.stats_label.config(text=f"封包數: {packet_count}")
                
                # 計算最大值用於正規化
                max_values = []
                for channel_data in data:
                    if len(channel_data) > 0:
                        max_val = max(channel_data)
                        max_values.append(max_val)
                    else:
                        max_values.append(0.0)
                
                overall_max = max(max_values) if max_values else 1.0
                if overall_max == 0:
                    overall_max = 1.0
                
                # 更新每個通道
                for i in range(8):
                    if len(data[i]) > 0:
                        current_value = data[i][-1]  # 最新值
                        normalized_value = (current_value / overall_max) * 100
                        
                        # 更新進度條
                        self.progress_bars[i]['value'] = normalized_value
                        
                        # 更新數值標籤 (每兩個label一組：名稱和數值)
                        value_label_index = i * 2 + 1
                        if value_label_index < len(self.labels):
                            self.labels[value_label_index].config(text=f"{current_value:.3f}")
                
                # 更新連接狀態
                if packet_count > 0:
                    self.status_label.config(text=f"已連接 ({self.uri})", fg='green')
                else:
                    self.status_label.config(text=f"等待資料... ({self.uri})", fg='yellow')
                    
            except Exception as e:
                print(f"顯示更新錯誤: {e}")
        
        # 排程下次更新
        if self.root:
            self.root.after(100, self.update_display)  # 100ms 更新一次
    
    def quit_app(self):
        self.running = False
        if self.ws:
            self.ws.close()
        if self.root:
            self.root.quit()
            self.root.destroy()
    
    def run(self):
        if not HAS_TKINTER:
            print("錯誤: Tkinter 不可用，無法建立圖形介面")
            return False
            
        try:
            self.create_gui()
            
            # 自動開始連接
            if self.start_ws():
                self.start_button.config(text="停止", bg='red')
                self.status_label.config(text=f"已連接 ({self.uri})", fg='green')
            
            self.root.mainloop()
            return True
            
        except Exception as e:
            print(f"GUI 錯誤: {e}")
            return False

class TextEMGPlotter(SimpleEMGPlotter):
    """文字模式版本"""
    
    def run(self):
        print("開始文字模式 EMG 監控...")
        print("按 Ctrl+C 停止")
        
        if not self.start_ws():
            print("錯誤: 無法連接到 WebSocket")
            return False
        
        try:
            while True:
                data, packet_count = self.get_current_data()
                
                # 清除螢幕 (Windows/Unix 相容)
                os.system('cls' if os.name == 'nt' else 'clear')
                
                print("=" * 60)
                print(f"即時 EMG 監控 - 封包數: {packet_count}")
                print(f"WebSocket: {self.uri}")
                print("=" * 60)
                
                for i in range(8):
                    name = CHANNEL_NAMES[i] if i < len(CHANNEL_NAMES) else f'Ch{i+1}'
                    if len(data[i]) > 0:
                        current_value = data[i][-1]
                        # 簡單的文字圖表
                        bar_length = int(current_value * 50) if current_value < 1 else 50
                        bar = "█" * bar_length
                        print(f"{name:25}: {current_value:8.3f} |{bar}")
                    else:
                        print(f"{name:25}: {'0.000':>8} |")
                
                print("=" * 60)
                print("按 Ctrl+C 停止")
                
                time.sleep(0.5)  # 0.5秒更新一次
                
        except KeyboardInterrupt:
            print("\n\n使用者中斷，程式結束")
            return True
        except Exception as e:
            print(f"執行錯誤: {e}")
            return False

def main():
    print("簡化版即時 EMG 繪圖工具")
    print("=" * 40)
    
    parser = argparse.ArgumentParser(description='Simple Realtime EMG plot from WebSocket')
    parser.add_argument('--uri', default='ws://localhost:31278/ws', help='WebSocket URI')
    parser.add_argument('--text', action='store_true', help='使用文字模式 (不使用圖形介面)')
    args = parser.parse_args()

    print(f"WebSocket URI: {args.uri}")
    
    try:
        if args.text or not HAS_TKINTER:
            print("使用文字模式...")
            plotter = TextEMGPlotter(uri=args.uri)
        else:
            print("使用圖形介面模式...")
            plotter = TkinterEMGPlotter(uri=args.uri)
        
        success = plotter.run()
        
        if not success:
            print("\n請確認:")
            print("1. EMG 裝置已連接並運行")
            print("2. WebSocket 伺服器已啟動") 
            print("3. URI 位址正確")
            print("4. 已安裝 websocket-client: pip install websocket-client")
            
    except KeyboardInterrupt:
        print('\n使用者中斷程式')
    except Exception as e:
        print(f'錯誤: {e}')
    finally:
        print("程式結束")

if __name__ == '__main__':
    main()