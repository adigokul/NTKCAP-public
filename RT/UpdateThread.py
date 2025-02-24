from NTK_CAP.script_py.NTK_Cap import *
from check_extrinsic import *
from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *
from multiprocessing import Event, shared_memory, Manager, Queue, Array, Lock, Process
import cv2
import time
import numpy as np
VISUALIZATION_CFG = dict(
    halpe26=dict(
        skeleton=[(15, 13), (13, 11), (11,19),(16, 14), (14, 12), (12,19),
                  (17,18), (18,19), (18,5), (5,7), (7,9), (18,6), (6,8),
                  (8,10), (1,2), (0,1), (0,2), (1,3), (2,4), (3,5), (4,6),
                  (15,20), (15,22), (15,24),(16,21),(16,23), (16,25)],
        palette=[(51, 153, 255), (0, 255, 0), (255, 128, 0)],
        link_color=[
            1, 1, 1, 2, 2, 2, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2
        ],
        point_color=[
            0, 0, 0, 0, 0, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2
        ],
        sigmas=[
            0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072,
            0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089, 0.026,
            0.026, 0.066, 0.079, 0.079, 0.079, 0.079, 0.079, 0.079
        ]
    )
)
sigmas = VISUALIZATION_CFG['halpe26']['sigmas']
palette = VISUALIZATION_CFG['halpe26']['palette']
skeleton = VISUALIZATION_CFG['halpe26']['skeleton']
link_color = VISUALIZATION_CFG['halpe26']['link_color']
point_color = VISUALIZATION_CFG['halpe26']['point_color']
class UpdateThread(QThread):
    update_signal = pyqtSignal(QImage)
    def __init__(self, cam_id, sync_frame_show_shm_name, draw_frame_queue, sync_tracker_show_shm_name, start_evt, stop_evt):
        super().__init__()
        ### define
        self.cam_id = cam_id
        self.sync_frame_show_shm_name = sync_frame_show_shm_name
        self.draw_frame_queue = draw_frame_queue
        self.start_evt = start_evt
        self.stop_evt = stop_evt
        self.scale_size = None
        self.frame_shm_shape = (1080, 1920, 3)
        self.keypoints_sync_shm_shape = (1, 26, 3)
        self.buffer_length = 20
        self.sync_tracker_show_shm_name = sync_tracker_show_shm_name
        self.ThreadActive = False
        self.RecordActive = False

    def run(self):
        
        existing_shm = shared_memory.SharedMemory(name=self.sync_frame_show_shm_name)
        shared_array = np.ndarray((self.buffer_length,) + self.frame_shm_shape, dtype=np.uint8, buffer=existing_shm.buf)
        existing_shm_keypoints = shared_memory.SharedMemory(name=self.sync_tracker_show_shm_name)
        shared_array_keypoints = np.ndarray((self.buffer_length,) + self.keypoints_sync_shm_shape, dtype=np.float32, buffer=existing_shm_keypoints.buf)
        self.start_evt.wait()
        self.ThreadActive = True
        while self.ThreadActive:
            t1 = time.time()
            try:
                idx_get = self.draw_frame_queue.get(timeout=0.01)
            except:
                continue
            frame = shared_array[idx_get, : ]
            keypoints = shared_array_keypoints[idx_get, : ]
            scores = keypoints[..., 2]
            keypoints = np.round(keypoints[..., :2], 3)
            self.draw_frame(frame, keypoints, scores, palette, skeleton, link_color, point_color)
            self.update_signal.emit(self.convert_to_qimage(frame))
            print(time.time()-t1)
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
    def draw_frame(self, frame, keypoints, scores, palette, skeleton, link_color, point_color):
        keypoints = keypoints.astype(int)
        # cv2.putText(frame, (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (30, 144, 255), 4, cv2.LINE_AA)
        for kpts, score in zip(keypoints, scores):
            show = [1] * len(kpts)
            for (u, v), color in zip(skeleton, link_color):
                if score[u] > 0.5 and score[v] > 0.5:
                    cv2.line(frame, kpts[u], tuple(kpts[v]), palette[color], 2, cv2.LINE_AA)
                else:
                    show[u] = show[v] = 0
            for kpt, show, color in zip(kpts, show, point_color):
                if show:
                    cv2.circle(frame, kpt, 1, palette[color], 2, cv2.LINE_AA)