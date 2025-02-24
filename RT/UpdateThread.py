from NTK_CAP.script_py.NTK_Cap import *
from check_extrinsic import *
from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *
from multiprocessing import Event, shared_memory, Manager, Queue, Array, Lock, Process
import cv2
import time
import numpy as np

class UpdateThread(QThread):
    update_signal = pyqtSignal(QImage)
    def __init__(self, cam_id, sync_frame_show_shm_name, draw_frame_queue, start_evt, stop_evt):
        super().__init__()
        ### define
        self.cam_id = cam_id
        self.sync_frame_show_shm_name = sync_frame_show_shm_name
        self.draw_frame_queue = draw_frame_queue
        self.start_evt = start_evt
        self.stop_evt = stop_evt
        self.scale_size = None
        self.frame_shm_shape = (1080, 1920, 3)
        self.buffer_length = 20

        self.ThreadActive = False
        self.RecordActive = False

    def run(self):
        
        existing_shm = shared_memory.SharedMemory(name=self.sync_frame_show_shm_name)
        shared_array = np.ndarray((self.buffer_length,) + self.frame_shm_shape, dtype=np.uint8, buffer=existing_shm.buf)
        self.start_evt.wait()
        self.ThreadActive = True
        while self.ThreadActive:
            t1 = time.time()
            try:
                idx_get = self.draw_frame_queue.get(timeout=0.01)
            except:
                continue
            frame = shared_array[idx_get, : ]
            self.update_signal.emit(self.convert_to_qimage(frame))
    
    def stop(self):
        if self.stop_evt.is_set():
            self.ThreadActive = False

    def convert_to_qimage(self, frame):
        Image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = Image.shape
        bytesPerline = channel * width
        ConvertToQtFormat = QImage(Image, width, height, bytesPerline, QImage.Format.Format_RGB888)
        Pic = ConvertToQtFormat.scaled(QSize(self.scale_size[0], self.scale_size[1]), Qt.AspectRatioMode.KeepAspectRatio)
        return Pic