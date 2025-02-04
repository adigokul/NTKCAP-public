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

'''
找問題 : # !!!!
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
class SyncProcess(Process):
    def __init__(self, tracker_sync_shm_lst, time_stamp_sync_shm_lst, tracker_queue_lst, start_evt, stop_evt, *args, **kwargs):
        super().__init__()
        self.tracker_sync_shm_lst = tracker_sync_shm_lst
        self.time_stamp_sync_shm_lst = time_stamp_sync_shm_lst
        self.tracker_q_lst = tracker_queue_lst
        self.start_evt = start_evt
        self.stop_evt = stop_evt
        self.buffer_length = 4
        self.tracker_sync_shm_shape = (1, 26, 3)
        self.time_stamp_shm_shape = 2
        self.shm_tracker_lst = []
        self.shm_time_stamp_lst = []
    def run(self):
        self.shm_tracker_lst = []
        self.shm_time_stamp_lst = []
        
        for i in range(len(self.tracker_sync_shm_lst)):
            
            existing_shm_tracker, shared_array_tracker, existing_shm_time_stamp_sync, shared_array_time_stamp_sync = None, None, None, None
            existing_shm_tracker = shared_memory.SharedMemory(name=self.tracker_sync_shm_lst[i].name)
            shared_array_tracker = np.ndarray((self.buffer_length,) + self.tracker_sync_shm_shape, dtype=np.float32, buffer=existing_shm_tracker.buf)
            self.shm_tracker_lst.append(shared_array_tracker)
            existing_shm_time_stamp_sync = shared_memory.SharedMemory(name=self.time_stamp_sync_shm_lst[i].name)
            shared_array_time_stamp_sync = np.ndarray((self.buffer_length, self.time_stamp_shm_shape), dtype=np.float64, buffer=existing_shm_time_stamp_sync.buf)
            self.shm_time_stamp_lst.append(shared_array_time_stamp_sync)
        idx = 0
        while self.start_evt.is_set():
            not_get = [0, 1, 2, 3]

            while not_get:
                    i = not_get[0]
                    try:
                        idx_get = self.tracker_q_lst[i].get(timeout=0.005)
                        # print(f"Queue {i} index: {idx_get}")
                        not_get.pop(0)
                        time_stamp = self.shm_time_stamp_lst[i][idx_get, :]
                        tracker = self.shm_tracker_lst[i][idx_get, :]
                        
                        # print(f"{i} Timestamp: {(time_stamp[0]+time_stamp[1])*500}")
                    except Exception as e:
                        # print(f"Queue {i} failed with error: {e}")
                        not_get.append(not_get.pop(0))
                    

class Tracker_Process(Process):
    def __init__(self, raw_frame_shm, time_stamp_shm, time_stamp_sync_shm, tracker_sync_shm, cam_q, tracker_q, start_evt, stop_evt, *args, **kwargs):
        super().__init__()
        self.raw_frame_shm_name = raw_frame_shm
        self.time_stamp_shm_name = time_stamp_shm
        self.time_stamp_sync_shm_name = time_stamp_sync_shm
        self.tracker_sync_shm_name = tracker_sync_shm
        self.camera_q = cam_q
        self.tracker_q = tracker_q
        self.start_evt = start_evt
        self.stop_evt = stop_evt
        self.cam_shm_shape = (1080, 1920, 3)
        self.time_stamp_shm_shape = 2
        self.buffer_length = 4
        self.tracker_sync_shm_shape = (1, 26, 3)
    def run(self):
        # RTM model
        tracker = PoseTracker(det_model=det_model_path,pose_model=pose_model_path,device_name=device)
        state = tracker.create_state(det_interval=1, det_min_bbox_size=100, keypoint_sigmas=sigmas, pose_max_num_bboxes=1)

        existing_shm_frame = shared_memory.SharedMemory(name=self.raw_frame_shm_name)
        shared_array_frame = np.ndarray((self.buffer_length,) + self.cam_shm_shape, dtype=np.uint8, buffer=existing_shm_frame.buf)
        existing_shm_time_stamp = shared_memory.SharedMemory(name=self.time_stamp_shm_name)
        shared_array_time_stamp = np.ndarray((self.buffer_length, self.time_stamp_shm_shape), dtype=np.float64, buffer=existing_shm_time_stamp.buf)
        existing_shm_tracker = shared_memory.SharedMemory(name=self.tracker_sync_shm_name)
        shared_array_tracker = np.ndarray((self.buffer_length,) + self.tracker_sync_shm_shape, dtype=np.float32, buffer=existing_shm_tracker.buf)
        existing_shm_time_stamp_sync = shared_memory.SharedMemory(name=self.time_stamp_sync_shm_name)
        shared_array_time_stamp_sync = np.ndarray((self.buffer_length, self.time_stamp_shm_shape), dtype=np.float64, buffer=existing_shm_time_stamp_sync.buf)
        idx = 0
        while self.start_evt.is_set():
            try:
                idx_get = self.camera_q.get(timeout=1)
            except:
                time.sleep(0.01)
                continue
            frame = shared_array_frame[idx_get, : ]
            time_stamp = shared_array_time_stamp[idx_get, :]
            keypoints, _ = tracker(state, frame, detect=-1)[:2]
            if keypoints.shape == (0, 0, 3): 
                np.copyto(shared_array_tracker[idx, :], np.full((1, 26, 3), np.nan, dtype=np.float32))
            else: 
                np.copyto(shared_array_tracker[idx, :], keypoints)
            # print(np.mean(time_stamp)*1000)
            np.copyto(shared_array_time_stamp_sync[idx, :], time_stamp)
            self.tracker_q.put(idx)
            idx = (idx+1) % self.buffer_length
            scores = keypoints[..., 2]
            keypoints = np.round(keypoints[..., :2], 3)

            self.draw_frame(frame, keypoints, scores, palette, skeleton, link_color, point_color)
            # np.copyto(shared_array_kp[idx,:], frame)
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
        
        
class CameraProcess(Process):
    def __init__(self, cam_id, camera_shm, time_stamp_shm, camera_queue, start_cam_evt, stop_cam_evt, *args, **kwargs):
        super().__init__()

        self.cam_id = cam_id
        self.cam_shm_name = camera_shm
        self.time_stamp_shm_name = time_stamp_shm

        self.cam_q = camera_queue
        self.start_evt = start_cam_evt
        self.stop_evt = stop_cam_evt
        self.buffer_length = 4
        # Camera
        self.frame_width = 1920
        self.frame_height = 1080
        self.fps = 30
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.cam_shm_shape = (1080, 1920, 3)
        self.time_stamp_shm_shape = 2
        self.frame_id = 0

    def run(self):
        start_time = time.time()
        idx = 0
        existing_shm_frame = shared_memory.SharedMemory(name=self.cam_shm_name)
        shared_array_frame = np.ndarray((self.buffer_length,) + self.cam_shm_shape, dtype=np.uint8, buffer=existing_shm_frame.buf)
        existing_shm_time_stamp = shared_memory.SharedMemory(name=self.time_stamp_shm_name)
        shared_array_time_stamp = np.ndarray((self.buffer_length, self.time_stamp_shm_shape), dtype=np.float64, buffer=existing_shm_time_stamp.buf)
        cap = cv2.VideoCapture(self.cam_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        self.start_evt.wait()
        while True:
            cap_s = time.time() - start_time
            ret, frame = cap.read()
            cap_e = time.time() - start_time
            cal_time = (cap_e+cap_s)*500
            np.copyto(shared_array_frame[idx,:], frame)
            print(float(f"{cal_time:.3f}"))
            
            np.copyto(shared_array_time_stamp[idx, :], np.array([float(f"{cap_s:.3f}"), float(f"{cap_e:.3f}")], dtype=np.float64))
            self.cam_q.put(idx)
            idx = (idx+1) % self.buffer_length
            

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        uic.loadUi("NTKCAP_rt_test.ui", self)
        self.btnOpenCamera: QPushButton = self.findChild(QPushButton, "btnOpenCamera")
        self.btnOpenCamera.clicked.connect(self.OpenCamera)
        # Params for OpenCamera
        self.camera_proc_lst = [] # 4 processes for reading and saving frames
        self.camera_shm_lst = [] # 4 shms for frames from cap.read()
        self.camera_shm_shape = (1080, 1920, 3) # shape for camera_shm
        self.time_stamp_shm_shape = 2
        self.buffer_length = 4 # buffer length for camera_shm
        self.tracker_sync_proc_lst = []
        self.tracker_sync_shm_lst = []
        self.tracker_sync_shm_shape = (1, 26, 3) # !!!!
        self.worker_proc_lst = []
        self.camera_opened = False
        self.time_stamp_sync_shm_lst = []
    def OpenCamera(self):
        # Initialization
        self.camera_proc_lst = []
        self.camera_shm_lst = []
        self.tracker_sync_proc_lst = []
        self.tracker_sync_shm_lst = []
        self.camera_queue = [Queue() for _ in range(4)]
        self.tracker_queue = [Queue() for _ in range(4)]
        self.start_camera_evt = Event()
        self.stop_camera_evt = Event()
        self.time_stamp_shm_lst = []
        self.time_stamp_sync_shm_lst = []
        self.camera_opened = True
        self.time_sync_process = None
        for i in range(4):
            camera_shm = shared_memory.SharedMemory(create=True, size=int(np.prod(self.camera_shm_shape) * np.dtype(np.uint8).itemsize * self.buffer_length))
            self.camera_shm_lst.append(camera_shm)
            time_stamp_array = Array('d', [0.0] * 4)
            time_stamp_shm = shared_memory.SharedMemory(create=True, size=int(self.time_stamp_shm_shape * np.dtype(np.float64).itemsize * self.buffer_length))
            self.time_stamp_shm_lst.append(time_stamp_shm)
            tracker_sync_shm = shared_memory.SharedMemory(create=True, size=int(np.prod(self.tracker_sync_shm_shape) * np.dtype(np.float32).itemsize * self.buffer_length))
            self.tracker_sync_shm_lst.append(tracker_sync_shm)
            time_stamp_sync_shm = shared_memory.SharedMemory(create=True, size=int(self.time_stamp_shm_shape * np.dtype(np.float64).itemsize * self.buffer_length))
            self.time_stamp_sync_shm_lst.append(time_stamp_sync_shm)
        
            p_camera = CameraProcess(
                i,
                self.camera_shm_lst[i].name,
                self.time_stamp_shm_lst[i].name,
                self.camera_queue[i],
                self.start_camera_evt,
                self.stop_camera_evt
            )
            self.camera_proc_lst.append(p_camera)
            worker = Tracker_Process(
                self.camera_shm_lst[i].name,
                self.time_stamp_shm_lst[i].name,
                self.time_stamp_sync_shm_lst[i].name,
                self.tracker_sync_shm_lst[i].name,
                self.camera_queue[i],
                self.tracker_queue[i],
                self.start_camera_evt,
                self.stop_camera_evt
            )
            self.worker_proc_lst.append(worker)
        for process in self.camera_proc_lst:
            process.start()
        for worker in self.worker_proc_lst:
            worker.start()
        self.time_sync_process = SyncProcess(
            self.tracker_sync_shm_lst, 
            self.time_stamp_sync_shm_lst,
            self.tracker_queue,
            self.start_camera_evt, 
            self.stop_camera_evt
        )
        self.time_sync_process.start()
        self.start_camera_evt.set()
    def closeEvent(self, event):
        for shm in self.camera_shm_lst:
            shm.close()
            shm.unlink()
        for shm in self.time_stamp_shm_lst:
            shm.close()
            shm.unlink()
        for shm in self.time_stamp_sync_shm_lst:
            shm.close()
            shm.unlink()
        for shm in self.tracker_sync_shm_lst:
            shm.close()
            shm.unlink()
        for queue in self.camera_queue:
            queue.close()
        for queue in self.tracker_queue:
            queue.close()
        self.camera_shm_lst.clear()
        self.time_stamp_shm_lst.clear()
        self.tracker_sync_shm_lst.clear()
        for process in self.camera_proc_lst:
            if process.is_alive():
                process.terminate()
                process.join()
        for worker in self.worker_proc_lst:
            if process.is_alive():
                process.terminate()
                process.join()
        self.time_sync_process.terminate()
        self.camera_proc_lst.clear()
        self.worker_proc_lst.clear()
if __name__ == "__main__":
    App = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(App.exec())