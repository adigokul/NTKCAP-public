import cv2
import time
import numpy as np
from PyQt6.QtGui import *
from PyQt6.QtCore import *
from multiprocessing import Event, shared_memory, Manager, Queue, Array, Lock, Process

class UpdateThread(QThread):
    update_signal = pyqtSignal(QImage)
    def __init__(self, cam_id, start_evt, stop_evt, queue_kp, shm_kp):
        super().__init__()
        ### define
        self.shared_name_kp = shm_kp
        self.start_evt = start_evt
        self.stop_evt = stop_evt
        self.queue_kp = queue_kp
        self.ThreadActive = False
        self.RecordActive = False
        self.cam_id = cam_id
        self.frame_id = 0
        self.out = None
        self.buffer_length = 4
        self.scale_size = None
        self.show_calibration_marker = False  # 校正紅點標記（只影響顯示）

    def run(self):
        shape = (1080, 1920, 3)
        dtype = np.uint8
        existing_shm = shared_memory.SharedMemory(name=self.shared_name_kp)
        shared_array = np.ndarray((self.buffer_length,) + shape, dtype=dtype, buffer=existing_shm.buf)
        self.start_evt.wait()
        self.ThreadActive = True
        while self.ThreadActive:
            try:
                # Clear queue backlog to get the most recent frame
                idx = None
                while True:
                    try:
                        idx = self.queue_kp.get_nowait()
                    except:
                        break
                
                # If no frames available, wait for new one
                if idx is None:
                    idx = self.queue_kp.get(timeout=1)
                    
            except:
                time.sleep(0.01)
                continue
            frame = shared_array[idx, : ]
            self.update_signal.emit(self.convert_to_qimage(frame, self.scale_size))

    def stop(self):
        if self.stop_evt.is_set():
            self.ThreadActive = False

    def convert_to_qimage(self, frame, scale_size):
        # 重要：複製幀以避免影響原始數據
        display_frame = frame.copy()

        # 如果需要顯示校正標記，在複製的幀上畫紅點
        if self.show_calibration_marker:
            height, width = display_frame.shape[:2]
            center_x, center_y = width // 2, height // 2
            # 畫紅色圓點（半徑10像素）
            cv2.circle(display_frame, (center_x, center_y), 10, (0, 0, 255), -1)
            # 畫紅色十字線
            cv2.line(display_frame, (center_x - 30, center_y), (center_x + 30, center_y), (0, 0, 255), 2)
            cv2.line(display_frame, (center_x, center_y - 30), (center_x, center_y + 30), (0, 0, 255), 2)

        Image = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        height, width, channel = Image.shape
        bytesPerline = channel * width
        ConvertToQtFormat = QImage(Image, width, height, bytesPerline, QImage.Format.Format_RGB888)
        Pic = ConvertToQtFormat.scaled(QSize(scale_size[0], scale_size[1]), Qt.AspectRatioMode.KeepAspectRatio)
        return Pic