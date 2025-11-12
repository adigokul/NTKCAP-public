#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EMG Qt Viewer (PyQt5 + pyqtgraph)
---------------------------------
- Realtime 8-channel EMG stacked plots
- Style approximates the provided reference image:
  * white background
  * dotted light-gray grid
  * per-channel small title on the left (channel name)
  * shared x-axis in seconds
  * y-axis labeled "Voltage (uV)"
  * vertical orange reference line you can move or hide
- WebSocket input compatible with your previous script (same JSON schema),
  and optionally with "emg_localhost.process_data_from_websocket" if available.

Run:
  pip install PyQt5 pyqtgraph websocket-client numpy
  python emg_qt_viewer.py --uri ws://localhost:31278/ws

Keys:
  M : toggle marker line
  R : reset autoscale for all panels
  G : toggle grid
"""
import sys
import os
import json
import argparse
import threading
import time
from collections import deque

import numpy as np

# -------------------- Optional emg_localhost (lazy import) --------------------
# We avoid importing emg_localhost at module import time because it may import
# compiled dependencies (scipy, etc.) that are incompatible with the runtime
# NumPy version. Instead, lazily try to import it when we actually need it.
emg_localhost = None
_emg_localhost_attempted = False

def get_emg_localhost():
    """Try to import emg_localhost once and cache the result.

    Returns the module if available and import succeeded, otherwise None.
    Import errors are caught and printed but won't crash the program.
    """
    global emg_localhost, _emg_localhost_attempted
    if _emg_localhost_attempted:
        return emg_localhost
    _emg_localhost_attempted = True
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(os.path.dirname(current_dir))
        emg_path = os.path.join(parent_dir, 'NTK_CAP', 'script_py')
        if os.path.isdir(emg_path) and emg_path not in sys.path:
            sys.path.append(emg_path)
        import importlib
        emg_localhost = importlib.import_module('emg_localhost')  # type: ignore
        print("✅ emg_localhost module loaded successfully")
    except Exception as e:
        emg_localhost = None
        print(f"⚠️  emg_localhost unavailable: {e}")
    return emg_localhost

# -------------------- WebSocket client --------------------
try:
    import websocket  # websocket-client
    HAS_WS = True
except Exception:
    HAS_WS = False

from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg


CHANNEL_NAMES = [
    # Order to match the reference image (top to bottom)
    "LGA", "RGA", "LTA", "RTA", "LBF", "RBF", "LRF", "RRF"
]


class EMGBuffer:
    """Thread-safe ring buffer for 8xN EMG samples."""
    def __init__(self, n_channels=8, max_samples=2000):
        self.n_channels = n_channels
        self.max_samples = max_samples
        self.buf = np.zeros((n_channels, max_samples), dtype=np.float32)
        self.count = 0  # how many total samples have been written
        self.lock = threading.Lock()

    def append_frame(self, frame: np.ndarray):
        """
        frame: shape (n_channels, L) or (n_channels,) or (L,) broadcastable to channels
        """
        with self.lock:
            if frame.ndim == 1:
                frame = frame.reshape(self.n_channels, 1)
            ch = min(self.n_channels, frame.shape[0])
            L = frame.shape[1]
            if L >= self.max_samples:
                # keep the most recent max_samples
                self.buf[:, :] = frame[-ch:, -self.max_samples:]
                self.count += L
                return
            # roll left and insert at the end
            self.buf = np.roll(self.buf, -L, axis=1)
            self.buf[:, -L:] = frame[:ch, :L]
            self.count += L

    def get(self):
        """Copy current buffer (8 x max_samples)."""
        with self.lock:
            return self.buf.copy(), self.count


class WSReader(QtCore.QObject):
    """WebSocket client running in a background thread; emits new arrays."""
    new_frame = QtCore.pyqtSignal(np.ndarray)  # (n_channels, L)
    notify = QtCore.pyqtSignal(str)  # 新增: 用於通知訊息

    def __init__(self, uri, parent=None):
        super().__init__(parent)
        self.uri = uri
        self._stop = threading.Event()
        self._thread = None
        self.status = "未連接"

    def start(self):
        if not HAS_WS:
            self.status = "websocket-client 未安裝"
            return False
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return True

    def stop(self):
        self._stop.set()

    # Simplified parser if emg_localhost is absent
    def _simple_parse(self, message: str, max_block=64):
        try:
            data_dict = json.loads(message)
            if "contents" not in data_dict:
                return None
            contents = data_dict["contents"]
            # we accumulate sequential items (up to max_block columns)
            cols = []
            for item in contents:
                if "eeg" in item and isinstance(item["eeg"], list):
                    v = np.array(item["eeg"][:8], dtype=np.float32)
                    cols.append(v.reshape(8, 1))
                    if len(cols) >= max_block:
                        break
            if not cols:
                return None
            return np.concatenate(cols, axis=1)  # (8, L)
        except Exception:
            return None

    def _run(self):
        try:
            def on_open(ws):
                self.status = f"已連接: {self.uri}"
                print(f"✅ WebSocket connected: {self.uri}")
                self.notify.emit(f"WebSocket 已連接: {self.uri}")

            def on_close(ws, code, msg):
                self.status = "連接已關閉"
                print(f"⚠️  WebSocket closed: code={code}, msg={msg}")
                self.notify.emit(f"WebSocket 連接已關閉: code={code}, msg={msg}")

            def on_error(ws, err):
                self.status = f"錯誤: {err}"
                print(f"❌ WebSocket error: {err}")
                self.notify.emit(f"WebSocket 錯誤: {err}")

            def on_message(ws, message):
                try:
                    print(f"🔵 Raw WebSocket message: {message}")  # Log raw message
                    arr = None

                    # Try lazy-loading emg_localhost. If it's unavailable or fails,
                    # fall back to the built-in simple parser.
                    module = get_emg_localhost()
                    if module is not None:
                        try:
                            result = module.process_data_from_websocket(
                                message, None, None, None
                            )
                            # Check if result is valid before unpacking
                            if result and isinstance(result, list) and len(result) > 0:
                                emg_array = result[0]
                                if emg_array is not None:
                                    arr = np.asarray(emg_array, dtype=np.float32)
                        except Exception as e:
                            print(f"⚠️  emg_localhost parsing failed: {e}")
                            self.notify.emit(f"emg_localhost 解析失敗: {e}")
                            arr = None
                    
                    # If emg_localhost parsing failed or unavailable, try simple parser
                    if arr is None:
                        arr = self._simple_parse(message)
                    
                    # Only emit if we got valid data
                    if arr is not None and arr.size > 0:
                        self.new_frame.emit(arr)
                    else:
                        print("⚠️  Parsed data is invalid or empty.")
                        self.notify.emit("收到的資料無效或為空。")
                except Exception as e:
                    print(f"❌ Error parsing WebSocket message: {e}")
                    self.notify.emit(f"WebSocket 訊息解析錯誤: {e}")
                    import traceback
                    traceback.print_exc()

            ws = websocket.WebSocketApp(
                self.uri, on_open=on_open, on_message=on_message,
                on_error=on_error, on_close=on_close
            )
            self.status = "連線中…"
            ws.run_forever()
        except Exception as e:
            self.status = f"錯誤: {e}"
            self.notify.emit(f"WebSocket 執行錯誤: {e}")


class EMGQtViewer(QtWidgets.QMainWindow):
    def __init__(self, uri: str, sampling_rate: int = 1000, max_samples: int = 3000,
                 show_marker=True, marker_sec=5.0, parent=None):
        super().__init__(parent)
        self.setWindowTitle("EMG Realtime Viewer (PyQt5 + pyqtgraph)")
        self.resize(1280, 720)

        # Data/WS
        self.sampling_rate = float(sampling_rate)
        self.buf = EMGBuffer(8, max_samples)
        self.ws = WSReader(uri=uri)
        self.ws.new_frame.connect(self._on_new_frame)
        self.ws.notify.connect(self.show_notification)  # 連接通知 signal

        # UI
        self._build_ui(show_marker=show_marker, marker_sec=marker_sec)
        self._build_notification_label()

        # Timer to refresh plots
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._refresh)
        self.timer.start(33)  # ~30 FPS

        # start WS
        self.ws.start()

    def show_notification(self, msg: str, duration_ms: int = 3000):
        # 以右上角浮動文字顯示通知
        self.notification_label.setText(msg)
        self.notification_label.adjustSize()
        self._move_notification_label()
        self.notification_label.show()
        # 設定自動隱藏
        QtCore.QTimer.singleShot(duration_ms, self.notification_label.hide)

    def _move_notification_label(self):
        margin = 12
        w = self.notification_label.width()
        h = self.notification_label.height()
        self.notification_label.move(self.width() - w - margin, margin)

    def _build_notification_label(self):
        self.notification_label = QtWidgets.QLabel(self)
        self.notification_label.setStyleSheet(
            "background: rgba(255,255,200,0.9); color: #333; border: 1px solid #ccc; "
            "padding: 6px 12px; border-radius: 8px; font-size: 12pt;"
        )
        self.notification_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.notification_label.hide()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._move_notification_label()

    # -------------------- UI --------------------
    def _build_ui(self, show_marker=True, marker_sec=5.0):
        cw = QtWidgets.QWidget(self)
        self.setCentralWidget(cw)
        layout = QtWidgets.QVBoxLayout(cw)

        # status line
        self.status_label = QtWidgets.QLabel("狀態：初始化中", self)
        self.status_label.setStyleSheet("color: #333;")
        layout.addWidget(self.status_label)

        # graphics area
        self.glw = pg.GraphicsLayoutWidget()
        self.glw.setBackground('w')  # 或 '#ffffff'
        layout.addWidget(self.glw, 1)

        # global style
        pg.setConfigOptions(antialias=True)
        self.setStyleSheet("""QMainWindow { background: #ffffff; }""")

        # Eight stacked plots, linked by x-axis
        self.plots = []
        self.curves = []
        self.marker_lines = []
        self.show_grid = True

        x = np.arange(self.buf.max_samples, dtype=np.float32) / self.sampling_rate
        self.x_seconds = x - x[-1]  # right-aligned time axis like [-T..0]

        for i, name in enumerate(CHANNEL_NAMES):
            p = self.glw.addPlot(row=i, col=0)
            p.setMenuEnabled(False)
            p.setClipToView(True)
            p.setDownsampling(auto=True)
            p.setMouseEnabled(x=True, y=False)
            p.setLabel('left', name, **{'color': '#fc1414', 'size': '14pt', 'bold': True})
            if i == len(CHANNEL_NAMES) - 1:
                p.setLabel('bottom', "Time (sec)", **{'color': '#666', 'size': '9pt'})
            else:
                p.getAxis('bottom').setStyle(showValues=False)

            # y-axis label on the right for the top plot only (shared meaning)
            if i == 0:
                right_axis = pg.AxisItem('right')
                right_axis.setLabel('Voltage (uV)', color='#666', **{'size': '9pt'})
                p.layout.addItem(right_axis, 2, 2)
                # The right axis won't auto-scale; it's a label-only visual cue.

            # grid (light dotted)
            p.showGrid(x=self.show_grid, y=self.show_grid, alpha=0.25)
            # ensure left axis and text are visible on a light background
            p.getAxis('left').setPen(pg.mkPen("#333333"))
            try:
                p.getAxis('left').setTextPen(pg.mkPen("#333333"))
            except Exception:
                # older pyqtgraph versions may not have setTextPen; ignore
                pass
            # increase left axis width so labels don't get clipped
            try:
                p.getAxis('left').setWidth(80)
            except Exception:
                pass
            p.getAxis('bottom').setPen(pg.mkPen("#000000"))

            # curve
            pen = pg.mkPen(color=(220,220, 220), width=3)
            c = p.plot(self.x_seconds, np.zeros_like(self.x_seconds), pen=pen)
            self.curves.append(c)
            self.plots.append(p)

            # vertical marker
            ml = pg.InfiniteLine(pos=-marker_sec, angle=90, pen=pg.mkPen("#ffae00", width=2))
            ml.setVisible(show_marker)
            p.addItem(ml)
            self.marker_lines.append(ml)

            # tighten margins to mimic compact stack
            p.setContentsMargins(5, 2, 5, 2)
            p.setYRange(-40000, 40000, padding=0.0)  # 固定震幅範圍
            # 不自動縮放

        # Link x ranges
        for p in self.plots[1:]:
            p.setXLink(self.plots[0])
        self.plots[0].setXRange(self.x_seconds[0], self.x_seconds[-1], padding=0.0)

        # shortcuts
        QtWidgets.QShortcut(QtGui.QKeySequence("M"), self, activated=self._toggle_marker)
        QtWidgets.QShortcut(QtGui.QKeySequence("R"), self, activated=self._reset_view)
        QtWidgets.QShortcut(QtGui.QKeySequence("G"), self, activated=self._toggle_grid)

    # -------------------- Slots --------------------
    def _on_new_frame(self, arr: np.ndarray):
        # Assume arr in volts or microvolts? If volts, convert to microvolts for display
        # You can change the scale here as needed.
        self.buf.append_frame(arr)

    def _refresh(self):
        data, count = self.buf.get()
        # Build right-aligned time axis
        T = data.shape[1] / self.sampling_rate
        x = np.linspace(-T, 0.0, data.shape[1], dtype=np.float32)

        for i, c in enumerate(self.curves):
            y = data[i]
            c.setData(x, y)
            # 不自動縮放 y 軸

        self.status_label.setText(f"狀態：{self.ws.status} | 收到樣本: {count}")

    # -------------------- Helpers --------------------
    def _toggle_marker(self):
        vis = not self.marker_lines[0].isVisible()
        for ml in self.marker_lines:
            ml.setVisible(vis)

    def _reset_view(self):
        # reset X to full window, Y autoscale next refresh
        data, _ = self.buf.get()
        T = data.shape[1] / self.sampling_rate
        for p in self.plots:
            p.setXRange(-T, 0.0, padding=0.0)

    def _toggle_grid(self):
        self.show_grid = not self.show_grid
        for p in self.plots:
            p.showGrid(x=self.show_grid, y=self.show_grid, alpha=0.25)


def main():
    parser = argparse.ArgumentParser(description="PyQt5 EMG viewer")
    parser.add_argument("--uri", default="ws://localhost:31278/ws", help="WebSocket URI")
    parser.add_argument("--fs", type=int, default=60, help="Sampling rate (Hz)")
    parser.add_argument("--max", dest="max_samples", type=int, default=3000,
                        help="Max samples kept per channel")
    parser.add_argument("--marker", action="store_true", help="Show vertical marker line")
    parser.add_argument("--marker_sec", type=float, default=5.0, help="Marker position in seconds (from right, negative)")
    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("EMG Viewer")

    win = EMGQtViewer(uri=args.uri,
                      sampling_rate=args.fs,
                      max_samples=args.max_samples,
                      show_marker=args.marker,
                      marker_sec=args.marker_sec)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
