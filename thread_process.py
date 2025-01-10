import cv2, time
from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *
import numpy as np
from multiprocessing import shared_memory
import os

class UpdateThread(QThread):
    update_signal = pyqtSignal(QImage)

    def __init__(self, shared_name, cam_id, start_evt, rec_evt, stop_evt, queue):
        super().__init__()
        ### define
        self.shared_name = shared_name
        self.start_evt = start_evt
        self.rec_evt = rec_evt
        self.stop_evt = stop_evt
        self.queue = queue
        self.ThreadActive = False
        self.RecordActive = False
        self.cam_id = cam_id
        self.frame_id = 0
        self.out = None
        self.data = {}
        self.buffer_length = 4
        self.scale_size = None

    def run(self):
        shape = (1080, 1920, 3)
        dtype = np.uint8
        existing_shm = shared_memory.SharedMemory(name=self.shared_name)
        shared_array = np.ndarray((self.buffer_length,) + shape, dtype=dtype, buffer=existing_shm.buf)
        self.start_evt.wait()
        self.ThreadActive = True

        while self.ThreadActive:
            try:
                idx = self.queue.get(timeout=1)
            except:
                time.sleep(0.01)
                continue
            frame = shared_array[idx, : ]
            if self.rec_evt.is_set():
                cv2.putText(frame, str(self.frame_id), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                self.out.write(frame)
                self.frame_id += 1

            self.update_signal.emit(self.convert_to_qimage(frame, self.scale_size))
            
        if self.out: self.out.release()

    def record(self, output_file):
        width = 1920
        height = 1080
        fps = 30
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        if os.path.exists(output_file):
            os.remove(output_file)
        self.out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
        self.frame_id = 0

    def stop(self):
        if self.stop_evt.is_set():
            self.ThreadActive = False

    def stop_record(self):
        self.out.release()

    def convert_to_qimage(self, frame, scale_size):
        Image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = Image.shape
        bytesPerline = channel * width
        ConvertToQtFormat = QImage(Image, width, height, bytesPerline, QImage.Format.Format_RGB888)
        Pic = ConvertToQtFormat.scaled(QSize(scale_size[0], scale_size[1]), Qt.AspectRatioMode.KeepAspectRatio)
        return Pic

class TimeThread(QThread):
    update_time_signal = pyqtSignal(str)
    def __init__(self, start_time):
        super().__init__()
        self.start_time = start_time
        self.running = True

    def run(self):
        while self.running:
            elapsed_time = (time.perf_counter() - self.start_time)
            formatted_time = f"{elapsed_time:.2f}"
            self.update_time_signal.emit(formatted_time)
            time.sleep(0.1)  

    def stop(self):
        self.running = False