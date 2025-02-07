from PyQt6.QtWebEngineWidgets import QWebEngineView
from NTK_CAP.script_py.NTK_Cap import *
from check_extrinsic import *
from PyQt6.QtWidgets import *
from PyQt6 import uic
from PyQt6.QtGui import *
from PyQt6.QtCore import *
from datetime import datetime
from multiprocessing import Event, shared_memory, Manager, Queue, Array, Lock, Process
from mmdeploy_runtime import PoseTracker
import copy
import sys
import os
import cv2
import time
import numpy as np
from collections import deque
import asyncio
'''
找問題 : # !!!!
version : cap.read -> sync -> tracker -> update label + triangulation
'''
VISUALIZATION_CFG = dict(
    halpe26=dict(
        skeleton=[(15,13), (13,11), (11,19),(16,14), (14,12), (12,19),
                  (17,18), (18,19), (18,5), (5,7), (7,9), (18,6), (6,8),
                  (8,10), (1,2), (0,1), (0,2), (1,3), (2,4), (3,5), (4,6),
                  (15,20), (15,22), (15,24),(16,21),(16,23), (16,25)],
        # palette=[(51,153,255), (0,255,0), (255,128,0)],
        palette=[(128,128,128), (51,153,255), (192,192,192)],
        link_color=[
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        ],
        point_color=[
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        ],
        sigmas=[
            0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072,
            0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089, 0.026,
            0.026, 0.066, 0.079, 0.079, 0.079, 0.079, 0.079, 0.079
        ]))

det_model_path = os.path.join(os.getcwd(),"NTK_CAP", "ThirdParty", "mmdeploy", "rtmpose-trt", "rtmdet-nano")
pose_model_path = os.path.join(os.getcwd(),"NTK_CAP", "ThirdParty", "mmdeploy", "rtmpose-trt", "rtmpose-m")
device ="cuda"

sigmas = VISUALIZATION_CFG['halpe26']['sigmas']
palette = VISUALIZATION_CFG['halpe26']['palette']
skeleton = VISUALIZATION_CFG['halpe26']['skeleton']
link_color = VISUALIZATION_CFG['halpe26']['link_color']
point_color = VISUALIZATION_CFG['halpe26']['point_color']

class Triangulation(Process):
    def __init__(self, keypoints_sync_shm_name, sync_tracker_queue, start_evt, stop_evt, *args, **kwargs):
        super().__init__()
        self.keypoints_sync_shm_name = keypoints_sync_shm_name
        self.sync_tracker_queue = sync_tracker_queue
        self.buffer_length = 20
        self.keypoints_sync_shm_shape = (4, 1, 26, 3)
        self.start_evt = start_evt
        self.stop_evt = stop_evt
    def run(self):
        existing_shm_keypoints = shared_memory.SharedMemory(name=self.keypoints_sync_shm_name)
        shared_array_keypoints = np.ndarray((self.buffer_length,) + self.keypoints_sync_shm_shape, dtype=np.float32, buffer=existing_shm_keypoints.buf)
        self.start_evt.wait()
        idx = 0
        while self.start_evt.is_set():
            try:
                idx_get = self.sync_tracker_queue.get(timeout=0.03)
            except:
                time.sleep(0.01)
                continue
            keypoints = shared_array_keypoints[idx_get, : ] # shape = (4, 1, 26, 3)
            # triangulation

class PreTri_Process(Process):
    def __init__(self, sync_tracker_shm_name0, sync_tracker_shm_name1, sync_tracker_shm_name2, sync_tracker_shm_name3, sync_tracker_queue_lst, keypoints_sync_shm_name, sync_tracker_queue, start_evt, stop_evt, *args, **kwargs):
        super().__init__()
        self.sync_tracker_shm_name0 = sync_tracker_shm_name0
        self.sync_tracker_shm_name1 = sync_tracker_shm_name1
        self.sync_tracker_shm_name2 = sync_tracker_shm_name2
        self.sync_tracker_shm_name3 = sync_tracker_shm_name3
        self.sync_tracker_shm_lst = []
        self.sync_tracker_queue_lst = sync_tracker_queue_lst
        self.keypoints_sync_shm_name = keypoints_sync_shm_name
        self.keypoints_sync_shm_shape = (4, 1, 26, 3)
        self.keypoints_shm_shape = (1, 26, 3)
        self.sync_tracker_queue = sync_tracker_queue
        self.buffer_length = 20
        self.start_evt = start_evt
        self.stop_evt = stop_evt
    def run(self):
        self.sync_tracker_shm_lst.clear()
        existing_shm_keypoints = shared_memory.SharedMemory(name=self.keypoints_sync_shm_name)
        shared_array_keypoints = np.ndarray((self.buffer_length,) + self.keypoints_sync_shm_shape, dtype=np.float32, buffer=existing_shm_keypoints.buf)
        existing_shm_tracker0 = shared_memory.SharedMemory(name=self.sync_tracker_shm_name0)
        shared_array_tracker0 = np.ndarray((self.buffer_length,) + self.keypoints_shm_shape, dtype=np.float32, buffer=existing_shm_tracker0.buf)
        existing_shm_tracker1 = shared_memory.SharedMemory(name=self.sync_tracker_shm_name1)
        shared_array_tracker1 = np.ndarray((self.buffer_length,) + self.keypoints_shm_shape, dtype=np.float32, buffer=existing_shm_tracker1.buf)
        existing_shm_tracker2 = shared_memory.SharedMemory(name=self.sync_tracker_shm_name2)
        shared_array_tracker2 = np.ndarray((self.buffer_length,) + self.keypoints_shm_shape, dtype=np.float32, buffer=existing_shm_tracker2.buf)
        existing_shm_tracker3 = shared_memory.SharedMemory(name=self.sync_tracker_shm_name3)
        shared_array_tracker3 = np.ndarray((self.buffer_length,) + self.keypoints_shm_shape, dtype=np.float32, buffer=existing_shm_tracker3.buf)
        self.sync_tracker_shm_lst.append(shared_array_tracker0)
        self.sync_tracker_shm_lst.append(shared_array_tracker1)
        self.sync_tracker_shm_lst.append(shared_array_tracker2)
        self.sync_tracker_shm_lst.append(shared_array_tracker3)
        
        idx = 0
        count = 0
        time.sleep(1)
        while self.start_evt.is_set():
            not_get = [0, 1, 2, 3]
            keypoints_frame = [None] * 4
            # cap_s = time.time()

            while not_get:
                i = not_get[0]
                try:
                    idx_get = self.sync_tracker_queue_lst[i].get_nowait()
                    not_get.pop(0)
                    keypoints = self.sync_tracker_shm_lst[i][idx_get]
                    keypoints_frame[i] = keypoints
                except Exception as e:
                    not_get.append(not_get.pop(0))
            np.copyto(shared_array_keypoints[idx, :], keypoints_frame)
            self.sync_tracker_queue.put(idx)
            idx = (idx+1) % self.buffer_length
            print("send to triangulation", count)
            count += 1

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
        self.buffer_length = 10

        self.ThreadActive = False
        self.RecordActive = False

    def run(self):
        
        existing_shm = shared_memory.SharedMemory(name=self.sync_frame_show_shm_name)
        shared_array = np.ndarray((self.buffer_length,) + self.frame_shm_shape, dtype=np.uint8, buffer=existing_shm.buf)
        self.start_evt.wait()
        self.ThreadActive = True
        while self.ThreadActive:
            try:
                idx_get = self.draw_frame_queue.get(timeout=0.05)
            except:
                time.sleep(0.01)
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

class Tracker_Process(Process):
    def __init__(self, cam_id, sync_frame_shm_name, sync_frame_queue, sync_tracker_shm_name, sync_tracker_queue, sync_frame_show_shm_name, draw_frame_queue, start_evt, stop_evt, *args, **kwargs):
        super().__init__()
        self.cam_id = cam_id
        self.sync_frame_shm_name = sync_frame_shm_name
        self.sync_frame_queue = sync_frame_queue
        self.sync_tracker_shm_name = sync_tracker_shm_name
        self.sync_tracker_queue = sync_tracker_queue
        self.sync_frame_show_shm_name = sync_frame_show_shm_name
        self.draw_frame_queue = draw_frame_queue
        self.start_evt = start_evt
        self.stop_evt = stop_evt

        self.frame_shm_shape = (1080, 1920, 3)
        self.buffer_length = 20
        self.tracker_shm_shape = (1, 26, 3)
    def run(self):
        # RTM model
        tracker = PoseTracker(det_model=det_model_path,pose_model=pose_model_path,device_name=device)
        state = tracker.create_state(det_interval=1, det_min_bbox_size=100, keypoint_sigmas=sigmas, pose_max_num_bboxes=1)

        existing_shm_frame = shared_memory.SharedMemory(name=self.sync_frame_shm_name)
        shared_array_frame = np.ndarray((self.buffer_length,) + self.frame_shm_shape, dtype=np.uint8, buffer=existing_shm_frame.buf)
        
        existing_shm_tracker = shared_memory.SharedMemory(name=self.sync_tracker_shm_name)
        shared_array_tracker = np.ndarray((self.buffer_length,) + self.tracker_shm_shape, dtype=np.float32, buffer=existing_shm_tracker.buf)

        existing_shm_frame_show = shared_memory.SharedMemory(name=self.sync_frame_show_shm_name)
        shared_array_frame_show = np.ndarray((10,) + self.frame_shm_shape, dtype=np.uint8, buffer=existing_shm_frame_show.buf)

        keypoints_template = np.full((1, 26, 3), np.nan, dtype=np.float32)
        
        idx, idx_show = 0, 0
        self.start_evt.wait()
        count = 0
        while self.start_evt.is_set():
            try:
                idx_get = self.sync_frame_queue.get(timeout=0.03)
            except:
                time.sleep(0.01)
                continue
            frame = shared_array_frame[idx_get, : ]
            keypoints, _ = tracker(state, frame, detect=-1)[:2]
            if keypoints.shape == (0, 0, 3): 
                np.copyto(shared_array_tracker[idx, :], keypoints_template)
            else:
                np.copyto(shared_array_tracker[idx, :], keypoints)
            print("Tracker", self.cam_id, count)
            self.sync_tracker_queue.put(idx)
            idx = (idx+1) % self.buffer_length
            count += 1
            scores = keypoints[..., 2]
            keypoints = np.round(keypoints[..., :2], 3)
            self.draw_frame(frame, keypoints, scores, palette, skeleton, link_color, point_color)
            np.copyto(shared_array_frame_show[idx_show,:], frame)
            self.draw_frame_queue.put(idx_show)
            idx_show = (idx_show+1) % 10
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

class SyncProcess(Process):
    def __init__(self, raw_frame_shm_name0, raw_frame_shm_name1, raw_frame_shm_name2, raw_frame_shm_name3, sync_frame_shm_name0, sync_frame_shm_name1, sync_frame_shm_name2, sync_frame_shm_name3, time_stamp_array_lst, camera_queue_lst, sync_frame_queue_lst, start_evt, stop_evt, *args, **kwargs):
        super().__init__()
        self.raw_frame_shm_name0 = raw_frame_shm_name0
        self.raw_frame_shm_name1 = raw_frame_shm_name1
        self.raw_frame_shm_name2 = raw_frame_shm_name2
        self.raw_frame_shm_name3 = raw_frame_shm_name3
        self.sync_frame_shm_name0 = sync_frame_shm_name0
        self.sync_frame_shm_name1 = sync_frame_shm_name1
        self.sync_frame_shm_name2 = sync_frame_shm_name2
        self.sync_frame_shm_name3 = sync_frame_shm_name3
        self.time_stamp_array_lst = time_stamp_array_lst
        self.camera_queue_lst = camera_queue_lst
        self.sync_frame_queue_lst = sync_frame_queue_lst
        self.start_evt = start_evt
        self.stop_evt = stop_evt
        self.buffer_length = 20
        self.frame_shm_shape = (1080, 1920, 3)
        self.shm_raw_frame_lst, self.shm_sync_frame_lst = [], []
        self.TR = 50 # ms
        self.frame_sync_buffer = deque(maxlen=30)
        self.time_stamp_sync_buffer = deque(maxlen=30)
    
    def run(self):
        self.shm_raw_frame_lst.clear()
        self.shm_sync_frame_lst.clear()
        
        existing_shm_raw_frame0 = shared_memory.SharedMemory(name=self.raw_frame_shm_name0)
        shared_array_raw_frame0 = np.ndarray((self.buffer_length,) + self.frame_shm_shape, dtype=np.uint8, buffer=existing_shm_raw_frame0.buf)

        existing_shm_raw_frame1 = shared_memory.SharedMemory(name=self.raw_frame_shm_name1)
        shared_array_raw_frame1 = np.ndarray((self.buffer_length,) + self.frame_shm_shape, dtype=np.uint8, buffer=existing_shm_raw_frame1.buf)

        existing_shm_raw_frame2 = shared_memory.SharedMemory(name=self.raw_frame_shm_name2)
        shared_array_raw_frame2 = np.ndarray((self.buffer_length,) + self.frame_shm_shape, dtype=np.uint8, buffer=existing_shm_raw_frame2.buf)

        existing_shm_raw_frame3 = shared_memory.SharedMemory(name=self.raw_frame_shm_name3)
        shared_array_raw_frame3 = np.ndarray((self.buffer_length,) + self.frame_shm_shape, dtype=np.uint8, buffer=existing_shm_raw_frame3.buf)
        self.shm_raw_frame_lst.append(shared_array_raw_frame0)
        self.shm_raw_frame_lst.append(shared_array_raw_frame1)
        self.shm_raw_frame_lst.append(shared_array_raw_frame2)
        self.shm_raw_frame_lst.append(shared_array_raw_frame3)

        existing_shm_sync_frame0 = shared_memory.SharedMemory(name=self.sync_frame_shm_name0)
        shared_array_sync_frame0 = np.ndarray((self.buffer_length,) + self.frame_shm_shape, dtype=np.uint8, buffer=existing_shm_sync_frame0.buf)

        existing_shm_sync_frame1 = shared_memory.SharedMemory(name=self.sync_frame_shm_name1)
        shared_array_sync_frame1 = np.ndarray((self.buffer_length,) + self.frame_shm_shape, dtype=np.uint8, buffer=existing_shm_sync_frame1.buf)

        existing_shm_sync_frame2 = shared_memory.SharedMemory(name=self.sync_frame_shm_name2)
        shared_array_sync_frame2 = np.ndarray((self.buffer_length,) + self.frame_shm_shape, dtype=np.uint8, buffer=existing_shm_sync_frame2.buf)

        existing_shm_sync_frame3 = shared_memory.SharedMemory(name=self.sync_frame_shm_name3)
        shared_array_sync_frame3 = np.ndarray((self.buffer_length,) + self.frame_shm_shape, dtype=np.uint8, buffer=existing_shm_sync_frame3.buf)
        self.shm_sync_frame_lst.append(shared_array_sync_frame0)
        self.shm_sync_frame_lst.append(shared_array_sync_frame1)
        self.shm_sync_frame_lst.append(shared_array_sync_frame2)
        self.shm_sync_frame_lst.append(shared_array_sync_frame3)

        self.start_evt.wait()
        time_stamp_frame = [None] * 4
        raw_frame_all = [None] * 4
        time_stamp_frame_sync = []
        raw_frame_sync = []
        next_queue = [0, 0, 0, 0]
        check = 0
        idx = 0
        count = 0
        time.sleep(1)
        while self.start_evt.is_set():
            not_get = [0, 1, 2, 3]
            time_stamp_frame = [None] * 4
            raw_frame_all = [None] * 4
            time_stamp_frame_sync.clear()
            raw_frame_sync.clear()

            while not_get:
                i = not_get[0]
                try:
                    idx_get = self.camera_queue_lst[i].get_nowait()
                    not_get.pop(0)
                    time_stamp = self.time_stamp_array_lst[i][idx_get]
                    raw_frame = self.shm_raw_frame_lst[i][idx_get, :]
                    time_stamp_frame[i] = time_stamp
                    raw_frame_all[i] = raw_frame
                except Exception as e:
                    not_get.append(not_get.pop(0))

            print(time_stamp_frame)
            
            if self.check_follow_up(time_stamp_frame):
                self.frame_sync_buffer.clear()
                self.time_stamp_sync_buffer.clear()
                next_queue = [0, 0, 0, 0]
                for i in range(4):
                    np.copyto(self.shm_sync_frame_lst[i][idx, :], raw_frame_all[i])
                    self.sync_frame_queue_lst[i].put(idx)
                    
                print(count, time_stamp_frame)
            else:
                self.frame_sync_buffer.append(raw_frame_all.copy())
                self.time_stamp_sync_buffer.append(time_stamp_frame.copy())
                for pos_idx, pos in enumerate(next_queue):
                    time_stamp_frame_sync.append(self.time_stamp_sync_buffer[pos][pos_idx])
                    raw_frame_sync.append(self.frame_sync_buffer[pos][pos_idx])
                for i in range(4):
                    
                    np.copyto(self.shm_sync_frame_lst[i][idx, :], raw_frame_sync[i])
                    self.sync_frame_queue_lst[i].put(idx)

                print(count, time_stamp_frame_sync)
                # send to shared memory for triangulation
                for i in range(4):
                    if max(time_stamp_frame_sync)-time_stamp_frame_sync[i] > self.TR:
                        next_queue[i] += 1
                        check += 1
                if check == 0:
                    self.frame_sync_buffer.popleft()
                    self.time_stamp_sync_buffer.popleft()
                else:
                    check = 0
                if max(time_stamp_frame) - min(time_stamp_frame) > self.TR:
                    try:
                        _ = self.camera_queue_lst[time_stamp_frame.index(min(time_stamp_frame))].get(timeout=0.05)
                    except:
                        continue
            count += 1
            idx = (idx+1) % self.buffer_length

            print(next_queue)
    def check_follow_up(self, time_stamp_frame):
        return all(max(time_stamp_frame) - ts <= self.TR for ts in time_stamp_frame)  
        
class CameraProcess(Process):
    def __init__(self, cam_id, camera_shm, time_stamp_array, camera_queue, unify_start_time, start_cam_evt, stop_cam_evt, *args, **kwargs):
        super().__init__()

        self.cam_id = cam_id
        self.cam_shm_name = camera_shm
        self.time_stamp_array = time_stamp_array

        self.cam_q = camera_queue
        self.start_evt = start_cam_evt
        self.stop_evt = stop_cam_evt
        self.buffer_length = 20
        self.start_time = unify_start_time
        # Camera
        self.frame_width = 1920
        self.frame_height = 1080
        self.fps = 30
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.cam_shm_shape = (1080, 1920, 3)
        self.frame_id = 0

    def run(self):
        idx = 0
        existing_shm_frame = shared_memory.SharedMemory(name=self.cam_shm_name)
        shared_array_frame = np.ndarray((self.buffer_length,) + self.cam_shm_shape, dtype=np.uint8, buffer=existing_shm_frame.buf)
        
        cap = cv2.VideoCapture(self.cam_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        self.start_evt.wait()
        count = 0
        _, _ = cap.read()
        _, _ = cap.read()
        _, _ = cap.read()
        _, _ = cap.read()
        _, _ = cap.read()
        while self.start_evt.is_set():
            cap_s = time.time() - self.start_time
            ret, frame = cap.read()
            cap_e = time.time() - self.start_time
            cal_time = (cap_e+cap_s)*500
        
            np.copyto(shared_array_frame[idx,:], frame)
            self.time_stamp_array[idx] = cal_time
            print("camera", self.cam_id, count, cal_time)
            count += 1
            self.cam_q.put(idx)
            idx = (idx+1) % self.buffer_length
            

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        uic.loadUi("NTKCAP_rt_test.ui", self)
        self.btnOpenCamera: QPushButton = self.findChild(QPushButton, "btnOpenCamera")
        self.btnOpenCamera.clicked.connect(self.OpenCamera)
        self.btnCloseCamera: QPushButton = self.findChild(QPushButton, "btnCloseCamera")
        self.btnCloseCamera.clicked.connect(self.CloseCamera)
        # Params for OpenCamera
        self.camera_proc_lst = [] # 4 processes for reading and saving frames
        self.camera_shm_lst = [] # 4 shms for frames from cap.read()
        self.camera_shm_shape = (1080, 1920, 3) # shape for camera_shm
        self.buffer_length = 20 # buffer length
        self.tracker_sync_proc_lst = []
        self.tracker_sync_shm_lst = []
        self.tracker_sync_shm_shape = (1, 26, 3) # !!!!
        self.worker_proc_lst = []
        self.camera_opened = False
        self.keypoints_sync_shm_shape = (4, 1, 26, 3) # !!!
        self.sync_frame_shm_lst = []
    def image_update_slot(self, image, label):
        label.setPixmap(QPixmap.fromImage(image))
    def OpenCamera(self):
        # Initialization
        self.camera_proc_lst = []
        self.camera_shm_lst = []
        self.tracker_sync_proc_lst = []
        self.tracker_sync_shm_lst = []
        self.camera_queue = [Queue() for _ in range(4)]
        self.sync_frame_queue = [Queue() for _ in range(4)]
        self.tracker_queue = [Queue() for _ in range(4)]
        self.draw_frame_queue = [Queue() for _ in range(4)]
        self.sync_tracker_queue = Queue()
        self.start_camera_evt = Event()
        self.stop_camera_evt = Event()
        self.time_stamp_array_lst = []
        self.camera_opened = True
        self.time_sync_process = None
        self.pre_tri_process = None
        self.trangulation = None
        start_time = time.time()
        self.sync_frame_shm_lst = []
        self.sync_frame_show_shm_lst = []
        self.threads = []
        self.keypoints_sync_shm = shared_memory.SharedMemory(create=True, size=int(np.prod(self.keypoints_sync_shm_shape) * np.dtype(np.float32).itemsize * self.buffer_length)) # for triangulation
        for i in range(4):
            camera_shm = shared_memory.SharedMemory(create=True, size=int(np.prod(self.camera_shm_shape) * np.dtype(np.uint8).itemsize * self.buffer_length))
            self.camera_shm_lst.append(camera_shm)
            time_stamp_array = Array('d', [0.0] * 20)
            self.time_stamp_array_lst.append(time_stamp_array)
            sync_frame_shm = shared_memory.SharedMemory(create=True, size=int(np.prod(self.camera_shm_shape) * np.dtype(np.uint8).itemsize * self.buffer_length))
            self.sync_frame_shm_lst.append(sync_frame_shm)
            tracker_sync_shm = shared_memory.SharedMemory(create=True, size=int(np.prod(self.tracker_sync_shm_shape) * np.dtype(np.float32).itemsize * self.buffer_length))
            self.tracker_sync_shm_lst.append(tracker_sync_shm)
            frame_show_shm = shared_memory.SharedMemory(create=True, size=int(np.prod(self.camera_shm_shape) * np.dtype(np.uint8).itemsize * 10))
            self.sync_frame_show_shm_lst.append(frame_show_shm)
            p_camera = CameraProcess(
                i,
                self.camera_shm_lst[i].name,
                self.time_stamp_array_lst[i],
                self.camera_queue[i],
                start_time,
                self.start_camera_evt,
                self.stop_camera_evt
            )
            self.camera_proc_lst.append(p_camera)

            worker = Tracker_Process(
                i,
                self.sync_frame_shm_lst[i].name,
                self.sync_frame_queue[i],
                self.tracker_sync_shm_lst[i].name,
                self.tracker_queue[i],
                self.sync_frame_show_shm_lst[i].name,
                self.draw_frame_queue[i],
                self.start_camera_evt,
                self.stop_camera_evt
            )
            self.worker_proc_lst.append(worker)
            thread = UpdateThread(
                i,
                self.sync_frame_show_shm_lst[i].name,
                self.draw_frame_queue[i],
                self.start_camera_evt,
                self.stop_camera_evt
            )
            self.threads.append(thread)
        self.label_cam = {0:self.Camera1, 1:self.Camera2, 2:self.Camera3, 3:self.Camera4}
        for i in range(4):
            label = self.label_cam[i]
            self.threads[i].scale_size = [label.size().width(), label.size().height()]
            self.threads[i].update_signal.connect(lambda image, label=label: self.image_update_slot(image, label))
        self.pre_tri_process = PreTri_Process(
            self.tracker_sync_shm_lst[0].name,
            self.tracker_sync_shm_lst[1].name,
            self.tracker_sync_shm_lst[2].name,
            self.tracker_sync_shm_lst[3].name,
            self.tracker_queue,
            self.keypoints_sync_shm.name,
            self.sync_tracker_queue,
            self.start_camera_evt,
            self.stop_camera_evt
        )
        self.trangulation = Triangulation(
            self.keypoints_sync_shm.name,
            self.sync_tracker_queue,
            self.start_camera_evt,
            self.stop_camera_evt
        )
        self.time_sync_process = SyncProcess(
            self.camera_shm_lst[0].name, # shm for receiving frames from four camera processes
            self.camera_shm_lst[1].name,
            self.camera_shm_lst[2].name,
            self.camera_shm_lst[3].name,
            self.sync_frame_shm_lst[0].name,
            self.sync_frame_shm_lst[1].name,
            self.sync_frame_shm_lst[2].name,
            self.sync_frame_shm_lst[3].name,
            self.time_stamp_array_lst,
            self.camera_queue,
            self.sync_frame_queue,
            self.start_camera_evt,
            self.stop_camera_evt
        )
        # start processes
        for process in self.camera_proc_lst:
            process.start()
        self.time_sync_process.start()
        for worker in self.worker_proc_lst:
            worker.start()
        self.pre_tri_process.start()
        for thread in self.threads:
            thread.start()
        self.start_camera_evt.set()
    def CloseCamera(self):
        if not self.camera_opened: return
        # release share memory: 4 camera cap.read() to sync shm + 4 synced to tracker shm + 4 tracker to preparation for triangulation shm + 1 send to triangulation + 4 update pyqt label
        for shm in self.camera_shm_lst: # 4 camera cap.read() to sync shm
            shm.close()
            shm.unlink()
        for shm in self.sync_frame_shm_lst: # 4 synced to tracker shm
            shm.close()
            shm.unlink()
        for shm in self.tracker_sync_shm_lst:  # 4 tracker to preparation for triangulation shm
            shm.close()
            shm.unlink()
        for shm in self.sync_frame_show_shm_lst: # 4 update pyqt label
            shm.close()
            shm.unlink()
        self.keypoints_sync_shm.close() # 1 send to triangulation
        self.keypoints_sync_shm.unlink()
        # release share array: 4 for time stamps
        for array in self.time_stamp_array_lst: # 4 for time stamps
            del array
        # release queue: 4 camera cap.read() to sync queue + 4 synced to tracker queue + 4 tracker to preparation for triangulation queue + 1 send to triangulation queue + 4 send to Qthread for updating GUI label queue
        for queue in self.camera_queue: # 4 camera cap.read() to sync queue
            queue.close()
        for queue in self.sync_frame_queue: # 4 synced to tracker queue
            queue.close()
        for queue in self.tracker_queue: # 4 tracker to preparation for triangulation queue
            queue.close()
        for queue in self.draw_frame_queue: # 4 send to Qthread for update label queue
            queue.close()
        self.sync_tracker_queue.close() # 1 send to triangulation queue
        # release Qthread for updating GUI labels
        for thread in self.threads:
            thread.stop()
        for thread in self.threads:
            thread.wait()
        
        # release list for share array, share memory, and QThread
        self.camera_shm_lst.clear()
        self.time_stamp_array_lst.clear()
        self.tracker_sync_shm_lst.clear()
        self.sync_frame_shm_lst.clear()
        self.sync_frame_show_shm_lst.clear()
        self.threads.clear()
        # terminate processes: 4 camera processes + 1 time stamp sync process + 4 tracker processes + 1 preparation for triangulation process
        for process in self.camera_proc_lst: # 4 camera processes
            if process.is_alive():
                process.terminate()
                process.join()
        for worker in self.worker_proc_lst: # 4 tracker processes
            if process.is_alive():
                process.terminate()
                process.join()
        self.time_sync_process.terminate() # 1 time stamp sync process
        self.pre_tri_process.terminate() # 1 preparation for triangulation process
        # clear list for 4 camera processes & 4 tracker processes
        self.camera_proc_lst.clear() # 4 camera processes
        self.worker_proc_lst.clear() # 4 tracker processes
    def closeEvent(self, event):
        self.CloseCamera()
if __name__ == "__main__":
    App = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(App.exec())