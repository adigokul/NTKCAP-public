"""realtime_emg_plot.py

簡單的即時 EMG 繪圖工具。

- 會連到 WebSocket (預設 ws://localhost:31278/ws)
- 使用專案中的 emg_localhost.process_data_from_websocket 取得每通道 50 樣本的 envelope
- 用 matplotlib 的 FuncAnimation 做即時更新

依賴：websocket-client, numpy, matplotlib, scipy
"""
import argparse
import threading
import time
import json
import sys
import os
from collections import deque

import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # 使用 TkAgg 後端避免相容性問題
import matplotlib.pyplot as plt
import websocket

# 嘗試使用專案中的 emg 處理程式
try:
    from data_streaming.EMG import emg_localhost
except Exception:
    try:
        # 添加路徑以找到 emg_localhost
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(os.path.dirname(current_dir))
        emg_path = os.path.join(parent_dir, 'NTK_CAP', 'script_py')
        if emg_path not in sys.path:
            sys.path.append(emg_path)
        import emg_localhost
    except Exception as e:
        print(f"警告：無法載入 emg_localhost 模組: {e}")
        print("將使用簡化版本的 EMG 處理")
        emg_localhost = None


CHANNEL_NAMES = [
    'Tibialis_anterior_right', 'Rectus_Femoris_right', 'Biceps_femoris_right', 'Gastrocnemius_right',
    'Tibialis_anterior_left', 'Rectus_Femoris_left', 'Biceps_femoris_left', 'Gastrocnemius_left'
]


class EMGWebSocketPlot:
    def __init__(self, uri="ws://localhost:31278/ws", max_samples=50):
        self.uri = uri
        self.max_samples = max_samples

        # shared buffer for the latest filtered envelope (channels x samples)
        self.filtered = np.zeros((8, self.max_samples))
        self.lock = threading.Lock()

        # filter states passed to emg_localhost functions
        self.bp_parameter = np.zeros((8, 8))
        self.nt_parameter = np.zeros((8, 2))
        self.lp_parameter = np.zeros((8, 4))

        self.ws = None
        self.ws_thread = None

    def on_message(self, ws, message):
        try:
            if emg_localhost is not None:
                # 使用原始的 emg_localhost 處理
                emg_array, self.bp_parameter, self.nt_parameter, self.lp_parameter = emg_localhost.process_data_from_websocket(
                    message, self.bp_parameter, self.nt_parameter, self.lp_parameter
                )
            else:
                # 簡化版本的 EMG 處理
                emg_array = self._simple_process_message(message)
            
            if isinstance(emg_array, np.ndarray) and emg_array.size:
                # emg_array expected shape (channels, samples)
                with self.lock:
                    # if shape matches, copy; otherwise try to adapt
                    if emg_array.shape == self.filtered.shape:
                        self.filtered = emg_array.copy()
                    else:
                        # pad or crop channels/samples as needed
                        ch = min(self.filtered.shape[0], emg_array.shape[0])
                        sm = min(self.filtered.shape[1], emg_array.shape[1])
                        self.filtered[:ch, :sm] = emg_array[:ch, :sm]
        except Exception as e:
            # 若 message 不是 JSON 或處理錯誤，忽略
            # print("EMG on_message error:", e)
            return

    def _simple_process_message(self, message):
        """簡化版本的 EMG 資料處理（當無法載入 emg_localhost 時使用）"""
        try:
            data_dict = json.loads(message)
            if "contents" in data_dict:
                contents = data_dict["contents"]
                if isinstance(contents, list) and len(contents) > 0:
                    # 提取 EMG 資料
                    emg_values = np.zeros((8, self.max_samples))
                    j = 0
                    for item in contents:
                        if "eeg" in item and isinstance(item["eeg"], list):
                            eeg = item["eeg"]
                            actual_channels = min(len(eeg), 8)
                            for i in range(actual_channels):
                                if j < self.max_samples:
                                    # 簡單的絕對值處理（模擬 envelope）
                                    emg_values[i, j] = abs(eeg[i])
                            j += 1
                            if j >= self.max_samples:
                                break
                    
                    if j > 0:
                        return emg_values[:, :j]
            
            return np.array([])
        except Exception:
            return np.array([])

    def on_error(self, ws, error):
        print("WebSocket error:", error)

    def on_close(self, ws, close_status_code, close_msg):
        print("WebSocket closed", close_status_code, close_msg)

    def on_open(self, ws):
        print("WebSocket connected to", self.uri)

    def start_ws(self):
        self.ws = websocket.WebSocketApp(self.uri,
                                         on_open=self.on_open,
                                         on_message=self.on_message,
                                         on_error=self.on_error,
                                         on_close=self.on_close)
        self.ws_thread = threading.Thread(target=self._run_ws, daemon=True)
        self.ws_thread.start()

    def _run_ws(self):
        # run_forever will reconnect by default on abnormal close
        try:
            self.ws.run_forever()
        except Exception as e:
            print("WebSocket run_forever exception:", e)

    def plot(self):
        # 使用預設樣式避免 seaborn-dark 相容性問題
        plt.style.use('default')
        
        # 設定深色主題
        plt.rcParams.update({
            'figure.facecolor': 'black',
            'axes.facecolor': 'black',
            'axes.edgecolor': 'white',
            'axes.labelcolor': 'white',
            'xtick.color': 'white',
            'ytick.color': 'white',
            'text.color': 'white'
        })
        
        fig, axes = plt.subplots(8, 1, figsize=(10, 8), sharex=True)
        fig.patch.set_facecolor('black')
        
        lines = []
        x = np.arange(self.max_samples)
        
        # 確保 axes 是陣列
        if not isinstance(axes, np.ndarray):
            axes = [axes]
        
        colors = ['cyan', 'yellow', 'magenta', 'green', 'red', 'blue', 'orange', 'pink']
        
        for i, ax in enumerate(axes):
            color = colors[i % len(colors)]
            line, = ax.plot(x, np.zeros_like(x), color=color, lw=1)
            ax.set_ylim(-0.01, 1.0)
            ax.set_ylabel(CHANNEL_NAMES[i] if i < len(CHANNEL_NAMES) else f'Ch{i}', 
                         color='white', fontsize=8)
            ax.tick_params(colors='white')
            ax.set_facecolor('black')
            lines.append(line)

        axes[-1].set_xlabel('sample (window)', color='white')

        def update(frame):
            try:
                with self.lock:
                    data = self.filtered.copy()
                
                for i, line in enumerate(lines):
                    if i < len(axes) and i < data.shape[0]:
                        line.set_ydata(data[i])
                        # auto scale (optional): keep lower bound 0
                        ymin = max(0, np.min(data[i]) - 0.01)
                        ymax = max(np.max(data[i]) + 0.01, 0.1)  # 確保最小範圍
                        if ymin < ymax:
                            axes[i].set_ylim(ymin, ymax)
                return lines
            except Exception as e:
                print(f"Update error: {e}")
                return lines

        try:
            from matplotlib.animation import FuncAnimation
            self.ani = FuncAnimation(fig, update, interval=100, blit=False)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Animation error: {e}")
            # 如果動畫失敗，至少顯示靜態圖
            plt.show()


def main():
    print("即時 EMG 繪圖工具")
    print("=" * 30)
    
    parser = argparse.ArgumentParser(description='Realtime EMG plot from WebSocket')
    parser.add_argument('--uri', default='ws://localhost:31278/ws', help='WebSocket URI')
    args = parser.parse_args()

    print(f"連接到 WebSocket: {args.uri}")
    
    try:
        plotter = EMGWebSocketPlot(uri=args.uri)
        plotter.start_ws()

        # wait a bit for connection
        print("等待連接...")
        time.sleep(1.0)
        
        print("開始繪圖... (按 Ctrl+C 或關閉視窗停止)")
        plotter.plot()
        
    except KeyboardInterrupt:
        print('\n使用者中斷程式')
    except Exception as e:
        print(f'錯誤: {e}')
        print("\n請確認:")
        print("1. EMG 裝置已連接並運行")
        print("2. WebSocket 伺服器已啟動")
        print("3. URI 位址正確")
    finally:
        print("程式結束")


if __name__ == '__main__':
    main()
