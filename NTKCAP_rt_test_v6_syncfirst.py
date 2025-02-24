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
import cupy as cp
import cv2
import time
import numpy as np
from collections import deque
import asyncio
from anytree import Node, RenderTree
import glob

from Pose2Sim.common import computeP, weighted_triangulation, reprojection, \
euclidean_distance, natural_sort, euclidean_dist_with_multiplication, camera2point_dist,computemap,undistort_points1,find_camera_coordinate
from Pose2Sim.skeletons import *

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
            # print("send to triangulation", count)
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
            # print("Tracker", self.cam_id, count)
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

            # print(time_stamp_frame)
            
            if self.check_follow_up(time_stamp_frame):
                self.frame_sync_buffer.clear()
                self.time_stamp_sync_buffer.clear()
                next_queue = [0, 0, 0, 0]
                for i in range(4):
                    np.copyto(self.shm_sync_frame_lst[i][idx, :], raw_frame_all[i])
                    self.sync_frame_queue_lst[i].put(idx)
                    
                # print(count, time_stamp_frame)
            else:
                self.frame_sync_buffer.append(raw_frame_all.copy())
                self.time_stamp_sync_buffer.append(time_stamp_frame.copy())
                for pos_idx, pos in enumerate(next_queue):
                    time_stamp_frame_sync.append(self.time_stamp_sync_buffer[pos][pos_idx])
                    raw_frame_sync.append(self.frame_sync_buffer[pos][pos_idx])
                for i in range(4):
                    
                    np.copyto(self.shm_sync_frame_lst[i][idx, :], raw_frame_sync[i])
                    self.sync_frame_queue_lst[i].put(idx)

                # print(count, time_stamp_frame_sync)
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

            # print(next_queue)
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
            # print("camera", self.cam_id, count, cal_time)
            count += 1
            self.cam_q.put(idx)
            idx = (idx+1) % self.buffer_length
            
class Triangulation(Process):
    def __init__(self, keypoints_sync_shm_name, sync_tracker_queue, start_evt, stop_evt, config, *args, **kwargs):
        super().__init__()
        self.keypoints_sync_shm_name = keypoints_sync_shm_name
        self.sync_tracker_queue = sync_tracker_queue
        self.buffer_length = 20
        self.keypoints_sync_shm_shape = (4, 1, 26, 3)
        self.start_evt = start_evt
        self.stop_evt = stop_evt
        self.config_dict = config
        calib_file = r"C:\Users\MyUser\Desktop\NTKCAP\Patient_data\HealthCare020\2024_12_06\2025_02_18_00_02_calculated\1005_1\calib-2d\Calib_easymocap.toml"
        config=r"C:\Users\MyUser\Desktop\NTKCAP\NTK_CAP\template\Empty_project\User\Config.toml"
    
        config_dict = toml.load(config)
        self.likelihood_threshold = config_dict.get('triangulation').get('likelihood_threshold')
        calib = toml.load(calib_file)
        self.cam_coord=cp.array([find_camera_coordinate(calib[list(calib.keys())[i]]) for i in range(4)])

        self.P = computeP(calib_file)
        mappingx, mappingy = computemap(calib_file)
        self.mappingx = cp.array(mappingx)
        self.mappingy = cp.array(mappingy)
    def run(self):
        combinations_4 = [[0, 1, 2, 3]]
        combinations_3 = [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]
        combinations_2 = [[2, 3], [1, 3], [1, 2], [0, 3], [0, 2], [0, 1]]
        existing_shm_keypoints = shared_memory.SharedMemory(name=self.keypoints_sync_shm_name)
        shared_array_keypoints = np.ndarray((self.buffer_length,) + self.keypoints_sync_shm_shape, dtype=np.float32, buffer=existing_shm_keypoints.buf)
        self.start_evt.wait()
        idx = 0
        coord1 = []
        coord2 = []
        coord3 = []
        coord4 = []
        cal_path = r"C:\Users\MyUser\Desktop\NTKCAP\Patient_data\HealthCare020\2024_12_06\2025_02_18_00_02_calculated\1005_1\pose-2d-tracked"
        template = ["pose_cam1_json", "pose_cam2_json", "pose_cam3_json", "pose_cam4_json"]
        path1 = os.path.join(cal_path, template[0])
        path2 = os.path.join(cal_path, template[1])
        path3 = os.path.join(cal_path, template[2])
        path4 = os.path.join(cal_path, template[3])
        for i in os.listdir(path1):
            path = os.path.join(path1, i)
            with open(path, 'r') as f:
                data1 = json.load(f)
                coord1.append(data1)
        for i in os.listdir(path2):
            path = os.path.join(path2, i)
            with open(path, 'r') as f:
                data2 = json.load(f)
                coord2.append(data2)
        for i in os.listdir(path3):
            path = os.path.join(path3, i)
            with open(path, 'r') as f:
                data3 = json.load(f)
                coord3.append(data3)
        for i in os.listdir(path4):
            path = os.path.join(path4, i)
            with open(path, 'r') as f:
                data4 = json.load(f)
                coord4.append(data4)
        c1 = np.array(coord1[0]['people'][0]['pose_keypoints_2d'])
        c2 = np.array(coord2[0]['people'][0]['pose_keypoints_2d'])
        c3 = np.array(coord3[0]['people'][0]['pose_keypoints_2d'])
        c4 = np.array(coord4[0]['people'][0]['pose_keypoints_2d'])
        arr1 = c1.reshape(26, 3)
        arr2 = c2.reshape(26, 3)
        arr3 = c3.reshape(26, 3)
        arr4 = c4.reshape(26, 3)
        stacked = np.stack([arr1, arr2, arr3, arr4], axis=0)
        keypoints_ids = [19, 12, 14, 16, 21, 23, 25, 11, 13, 15, 20, 22, 24, 18, 17, 0, 6, 8, 10, 5, 7, 9]
        prep = stacked[:, keypoints_ids, :]
        P = cp.array(self.P)
        P_cam_0_comb4 = cp.concatenate([P[0][0].reshape(1, -1), P[1][0].reshape(1, -1), P[2][0].reshape(1, -1), P[3][0].reshape(1, -1)], axis=0)
        P_cam_1_comb4 = cp.concatenate([P[0][1].reshape(1, -1), P[1][1].reshape(1, -1), P[2][1].reshape(1, -1), P[3][1].reshape(1, -1)], axis=0)
        P_cam_2_comb4 = cp.concatenate([P[0][2].reshape(1, -1), P[1][2].reshape(1, -1), P[2][2].reshape(1, -1), P[3][2].reshape(1, -1)], axis=0)
        P_cam_comb4 = cp.stack([P_cam_0_comb4, P_cam_1_comb4, P_cam_2_comb4], axis=0)
        comb_3 = cp.array(combinations_3)
        P_cam_comb3 = cp.stack([
            cp.stack([cp.concatenate([P[d][j].reshape(1, -1) for d in combinations_3[i]], axis=0) for j in range(3)], axis=0)
            for i in range(4)
        ], axis=0)

        combinations_2 = [[2, 3], [1, 3], [1, 2], [0, 3], [0, 2], [0, 1]] 
        comb_2 = cp.array(combinations_2)
        P_cam_comb2 = cp.stack([
            cp.stack([cp.concatenate([P[d][j].reshape(1, -1) for d in combinations_2[i]], axis=0) for j in range(3)], axis=0)
            for i in range(6)
        ], axis=0)
        
        while self.start_evt.is_set():
            try:
                idx_get = self.sync_tracker_queue.get(timeout=0.03)
            except:
                time.sleep(0.01)
                continue
            keypoints = shared_array_keypoints[idx_get, : ] # shape = (4, 1, 26, 3)
            t1 = time.time()
            result = cp.array(np.transpose(prep, (1, 0, 2)))
            prep_4 = cp.expand_dims(result, axis=(0, 2))
            prep_3 = cp.expand_dims(result[:, combinations_3, :], axis=0)
            prep_2 = cp.expand_dims(result[:, combinations_2, :], axis=0)
            prep_4, prep_3, prep_2 = self.undistort_points_cupy(prep_4, prep_3, prep_2)
            
            prep_4like=cp.min(prep_4[:,:,:,:,2],axis=3)
            prep_3like=cp.min(prep_3[:,:,:,:,2],axis=3)
            prep_2like=cp.min(prep_2[:,:,:,:,2],axis=3)
            prep_like = cp.concatenate((prep_4like,prep_3like,prep_2like),axis =2)
            
            A4,A3,A2 = self.create_A(prep_4, prep_3, prep_2, P)
            
            Q4,Q3,Q2 = self.find_Q(A4,A3,A2)
            
            Q = cp.concatenate((Q4,Q3,Q2),axis = 2)
            Q3_bug,Q2_bug = self.create_QBug(self.likelihood_threshold,prep_3like,Q3,prep_2like,Q2)
            

            real_dist4, real_dist3, real_dist2 = self.find_real_dist_error(self.P, self.cam_coord, prep_4, Q4, prep_3, Q3, prep_2, Q2, Q3_bug, Q2_bug, P_cam_comb4, P_cam_comb3, comb_3, P_cam_comb2, comb_2)
            
            ## delete the liklelihoood vlue which is too low
            real_dist = cp.concatenate((real_dist4,real_dist3,real_dist2),axis = 2)
            loc = cp.where(prep_like < self.likelihood_threshold)
            #import pdb;pdb.set_trace()
            real_dist[loc] = cp.inf
            # Find the index of the first non-inf element along axis 2
            
            non_inf_mask = ~cp.isinf(real_dist)
            min_locations_nan = cp.argmax(non_inf_mask, axis=2)
            real_dist_dynamic = cp.copy(real_dist)
            list_dynamic_mincam=  {'Hip':4,'RHip':4,'RKnee':3,'RAnkle':3,'RBigToe':3,'RSmallToe':3,'RHeel':3,'LHip':4,'LKnee':3,'LAnkle':3,'LBigToe':3,'LSmallToe':3,'LHeel':3,'Neck':3,'Head':2,'Nose':2,'RShoulder':3,'RElbow':3,'RWrist':3,'LShoulder':3,'LElbow':3,'LWrist':3}
            list_dynamic_mincam_prep = [self.map_to_listdynamic(value) for value in list_dynamic_mincam.values()]
            ## setting the list dynamic
            
            for i in range(22):    
                real_dist_dynamic[:,i,list_dynamic_mincam_prep[i]] = cp.inf
            
            ## find the minimum combination
            temp_shape = cp.shape(Q)
            checkinf = cp.min(real_dist_dynamic,axis =2)
            min_locations = cp.argmin(real_dist_dynamic, axis=2)
            loc =cp.where(checkinf==cp.inf)
            min_locations[loc] = min_locations_nan[loc]
            batch_indices, time_indices = cp.meshgrid(cp.arange(temp_shape[0]), cp.arange(temp_shape[1]), indexing='ij')
            Q_selected = Q[batch_indices, time_indices, min_locations]
            Q_selected = Q_selected[:,:,0:3]
            Q_selected = cp.asnumpy(Q_selected)
            Q_tot_gpu = [Q_selected[i].ravel() for i in range(Q_selected.shape[0])]
            print(time.time()-t1)
    def create_QBug(self, likelihood_threshold,prep_3like,Q3,prep_2like,Q2):
        loc = cp.where(prep_3like < likelihood_threshold)
        prep_3like[loc] = cp.inf
        # Find the index of the first non-inf element along axis 2
        non_inf_mask3 = ~cp.isinf(prep_3like)
        min_locations_nan3 = cp.argmax(non_inf_mask3, axis=2)
        batch_indices = cp.arange(Q3.shape[0])[:, None]  # Shape: (184, 1)
        time_indices = cp.arange(Q3.shape[1])[None, :]   # Shape: (1, 22)

    # Use advanced indexing to extract the desired values
        selected_slices = Q3[batch_indices, time_indices, min_locations_nan3, :]
        
    # Add an additional dimension to match the shape (184, 22, 1, 4)
        Q3_bug = selected_slices[:, :, cp.newaxis, :]

        loc = cp.where(prep_2like < likelihood_threshold)
        prep_2like[loc] = cp.inf
        # Find the index of the first non-inf element along axis 2
        non_inf_mask2 = ~cp.isinf(prep_2like)
        min_locations_nan2 = cp.argmax(non_inf_mask2, axis=2)
        batch_indices = cp.arange(Q2.shape[0])[:, None]  # Shape: (184, 1)
        time_indices = cp.arange(Q2.shape[1])[None, :]   # Shape: (1, 22)

    # Use advanced indexing to extract the desired values
        selected_slices = Q2[batch_indices, time_indices, min_locations_nan2, :]
        
    # Add an additional dimension to match the shape (184, 22, 1, 4)
        Q2_bug = selected_slices[:, :, cp.newaxis, :]
        return Q3_bug,Q2_bug
    def find_Q(self, A4,A3,A2):
        f = cp.shape(A4)
        A_flat = A4.reshape(-1, 8, 4)  # Shape: (250 * 22 * 1, 8, 4)

        # Step 2: Perform SVD in batch using CuPy
        U, S, Vt = cp.linalg.svd(A_flat, full_matrices=False)  # Batched SVD

        # Step 3: Compute Q
        # Transpose Vt to get V
        V = Vt.transpose(0, 2, 1)  # Shape: (batch_size, 4, 4)

        # Extract and compute Q
        Q = cp.array([
            V[:, 0, 3] / V[:, 3, 3],
            V[:, 1, 3] / V[:, 3, 3],
            V[:, 2, 3] / V[:, 3, 3],
            cp.ones(V.shape[0])  # Add 1 as the last element of Q
        ]).T  # Shape: (batch_size, 4)

        # Step 4: Reshape Q back to (250, 22, 1, 4)
        Q4 = Q.reshape(f[0], f[1], f[2], 4)
        f = cp.shape(A3)
        A_flat = A3.reshape(-1, 6, 4)  # Shape: (250 * 22 * 1, 8, 4)

        # Step 2: Perform SVD in batch using CuPy
        U, S, Vt = cp.linalg.svd(A_flat, full_matrices=False)  # Batched SVD

        # Step 3: Compute Q
        # Transpose Vt to get V
        V = Vt.transpose(0, 2, 1)  # Shape: (batch_size, 4, 4)

        # Extract and compute Q
        Q = cp.array([
            V[:, 0, 3] / V[:, 3, 3],
            V[:, 1, 3] / V[:, 3, 3],
            V[:, 2, 3] / V[:, 3, 3],
            cp.ones(V.shape[0])  # Add 1 as the last element of Q
        ]).T  # Shape: (batch_size, 4)

        # Step 4: Reshape Q back to (250, 22, 1, 4)
        Q3 = Q.reshape(f[0], f[1], f[2], 4)
        
        f = cp.shape(A2)
        A_flat = A2.reshape(-1, 4, 4)  # Shape: (250 * 22 * 1, 8, 4)

        # Step 2: Perform SVD in batch using CuPy
        U, S, Vt = cp.linalg.svd(A_flat, full_matrices=False)  # Batched SVD
        # Step 3: Compute Q
        # Transpose Vt to get V
        V = Vt.transpose(0, 2, 1)  # Shape: (batch_size, 4, 4)

        # Extract and compute Q
        Q = cp.array([
            V[:, 0, 3] / V[:, 3, 3],
            V[:, 1, 3] / V[:, 3, 3],
            V[:, 2, 3] / V[:, 3, 3],
            cp.ones(V.shape[0])  # Add 1 as the last element of Q
        ]).T  # Shape: (batch_size, 4)

        # Step 4: Reshape Q back to (250, 22, 1, 4)
        Q2 = Q.reshape(f[0], f[1], f[2], 4)


        return Q4,Q3,Q2
    def create_A(self, prep_4,prep_3,prep_2,P):
        # Elements
        t1 = time.time()
        sh = np.shape(prep_4)
        elements = [0, 1, 2, 3]

        # Total elements
        n = len(elements)
        combinations_4 = [elements]  # No deletions
        combinations_3 = [elements[:i] + elements[i+1:] for i in range(len(elements))]  # Delete one element
        combinations_2 = [elements[:i] + elements[i+1:j] + elements[j+1:] for i in range(len(elements)) for j in range(i+1, len(elements))]  # Delete two elements
        results = []

        # Iterate over the range 4 for c
        for c in range(4):
            # Compute the first part
            part1 = (P[c][0] - prep_4[:, :, :, c, 0:1] * P[c][2]) * prep_4[:, :, :, c, 2:3]
            # Compute the second part
            part2 = (P[c][1] - prep_4[:, :, :, c, 1:2] * P[c][2]) * prep_4[:, :, :, c, 2:3]
            
            # Append the results along the last axis
            results.append(part1)
            temp = cp.array(part1)
            results.append(part2)
        # Concatenate all results along the new axis
        A4 = cp.stack(results, axis=3)  # Concatenate along the 4th dimension

        results1 = []
        final_results =[]
        # Iterate over the range 3 for c
        for i in range(4):
            results1 = []
            for c in range(3):
                camera_index = combinations_3[i]
                d = camera_index[c]
                
                # Compute the first part
                part1 = (P[d][0] - prep_3[:, :, i, c, 0:1] * P[d][2]) * prep_3[:, :, i, c, 2:3]
                # Compute the second part
                part2 = (P[d][1] - prep_3[:, :, i, c, 1:2] * P[d][2]) * prep_3[:, :, i, c, 2:3]
                #import pdb;pdb.set_trace()
                # Append the results along the last axis
                results1.append(part1)           
                results1.append(part2)
            inner_results_stacked = cp.stack(results1, axis=2) 
            final_results.append(inner_results_stacked)
            
        A3 = cp.stack(final_results, axis=2) 
        
        results1 = []
        final_results =[]
        # Iterate over the range 3 for c
        for i in range(6):
            results1 = []
            for c in range(2):
                camera_index = combinations_2[i]
                d = camera_index[c]
                
                # Compute the first part
                part1 = (P[d][0] - prep_2[:, :, i, c, 0:1] * P[d][2]) * prep_2[:, :, i, c, 2:3]
                # Compute the second part
                part2 = (P[d][1] - prep_2[:, :, i, c, 1:2] * P[d][2]) * prep_2[:, :, i, c, 2:3]
                # Append the results along the last axis
                results1.append(part1)           
                results1.append(part2)
            inner_results_stacked = cp.stack(results1, axis=2) 
            final_results.append(inner_results_stacked)
            
        A2 = cp.stack(final_results, axis=2)

        return A4,A3,A2
    def find_real_dist_error(self, P, cam_coord, prep_4, Q4, prep_3, Q3, prep_2, Q2, Q3_bug, Q2_bug, P_cam_comb4, P_cam_comb3, comb_3, P_cam_comb2, comb_2):
    
        result_c4 = cp.einsum('cik,btk->cit', P_cam_comb4, Q4[:,:,0,:])
        
        x_c4 = result_c4[0] / result_c4[2]
        y_c4 = result_c4[1] / result_c4[2]
        dist_c4 = cp.sqrt(cp.sum((Q4[:, :, 0, 0:3] - cp.stack(cam_coord, axis=0)[:, None, :]) ** 2, axis=-1))
        
        X_c4 = cp.expand_dims(x_c4.T, axis=(0, 2))
        Y_c4 = cp.expand_dims(y_c4.T, axis=(0, 2))
        final_dist_c4 = cp.expand_dims(dist_c4.T, axis=(0, 2))

        rpj_coor_4 = cp.stack((X_c4,Y_c4), axis = -1)
        
        x1, y1 = prep_4[:, :, :, :, 0], prep_4[:, :, :, :, 1]
        x2, y2 = rpj_coor_4[:, :, :, :, 0], rpj_coor_4[:, :, :, :, 1]
        # Compute the Euclidean distance
        rpj = cp.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        real_dist  = final_dist_c4*rpj
        real_dist4 = cp.max(real_dist, axis=-1)

        # real_dist3
        

        result_c3 = cp.einsum('ncik,nbtk->ncit', P_cam_comb3, Q3.transpose(2, 0, 1, 3))
        x_c3 = result_c3[:, 0] / result_c3[:, 2]
        y_c3 = result_c3[:, 1] / result_c3[:, 2]

        dist_c3 = cp.sqrt(cp.sum((Q3_bug.transpose(2, 0, 1, 3)[:, :, :, 0:3] - cam_coord[comb_3][:, :, None, :]) ** 2, axis=-1))
        X_c3 = cp.expand_dims(x_c3.transpose(2, 0, 1), axis=0)
        Y_c3 = cp.expand_dims(y_c3.transpose(2, 0, 1), axis=0)
        final_dist_c3 = cp.expand_dims(dist_c3.transpose(2, 0, 1), axis=0)

        rpj_coor_3 = cp.stack((X_c3,Y_c3),axis = -1)
        
        x1, y1 = prep_3[:, :, :, :, 0], prep_3[:, :, :, :, 1]
        x2, y2 = rpj_coor_3[:, :, :, :, 0], rpj_coor_3[:, :, :, :, 1]
        # Compute the Euclidean distance
        rpj = cp.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        real_dist  = final_dist_c3*rpj
        real_dist3 = cp.max(real_dist, axis=-1)
        
        result_c2 = cp.einsum('ncik,nbtk->ncit', P_cam_comb2, Q2.transpose(2, 0, 1, 3))

        x_c2 = result_c2[:, 0] / result_c2[:, 2]
        y_c2 = result_c2[:, 1] / result_c2[:, 2]

        dist_c2 = cp.sqrt(cp.sum((Q2_bug.transpose(2, 0, 1, 3)[:, :, :, 0:3] - cam_coord[comb_2][:, :, None, :]) ** 2, axis=-1))
        X_c2 = cp.expand_dims(x_c2.transpose(2, 0, 1), axis=0)
        Y_c2 = cp.expand_dims(y_c2.transpose(2, 0, 1), axis=0)
        final_dist_c2 = cp.expand_dims(dist_c2.transpose(2, 0, 1), axis=0)  
        
        
        #[754.2223150392452, 1165.0827450392762]
        rpj_coor_2 = cp.stack((X_c2,Y_c2),axis = -1)
        
        x1, y1 = prep_2[:, :, :, :, 0], prep_2[:, :, :, :, 1]
        x2, y2 = rpj_coor_2[:, :, :, :, 0], rpj_coor_2[:, :, :, :, 1]
        # Compute the Euclidean distance
        rpj = cp.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        real_dist  = final_dist_c2*rpj
        real_dist2 = cp.max(real_dist, axis=-1)

        return real_dist4, real_dist3, real_dist2    
    def map_to_listdynamic(self, value):
        if value == 4:
            return [1,2,3,4,5,6,7,8,9,10]  # Map 4 to [0]
        elif value == 3:
            return [5,6,7,8,9,10]  # Map 3 to [1, 2, 3, 4]
        elif value == 2:
            return []  # Map 2 to [5, 6, 7, 8, 9, 10]      
            # triangulation
    def undistort_points_cupy(self, prep_4,prep_3,prep_2):
        
        x = prep_4[:, :, :, :, 0]  # Shape: (155, 22, 4, 3)
        y = prep_4[:, :, :, :, 1]  # Shape: (155, 22, 4, 3)
        likelihood = prep_4[:, :, :, :, 2]  # Shape: (155, 22, 4, 3)

        # Perform bilinear interpolation on x and y
        x_undistorted = self.bilinear_interpolate_cupy(self.mappingx, x, y)  # Shape: (155, 22, 4, 3)
        y_undistorted = self.bilinear_interpolate_cupy(self.mappingy, x, y)  # Shape: (155, 22, 4, 3)

        # Combine results into a single array
        prep_4 = cp.stack([x_undistorted, y_undistorted, likelihood], axis=-1)  # Shape: (155, 22, 4, 3, 3)


        # Extract x, y, and likelihood values
        x = prep_3[:, :, :, :, 0]  # Shape: (155, 22, 4, 3)
        y = prep_3[:, :, :, :, 1]  # Shape: (155, 22, 4, 3)
        likelihood = prep_3[:, :, :, :, 2]  # Shape: (155, 22, 4, 3)

        # Perform bilinear interpolation on x and y
        x_undistorted = self.bilinear_interpolate_cupy(self.mappingx, x, y)  # Shape: (155, 22, 4, 3)
        y_undistorted = self.bilinear_interpolate_cupy(self.mappingy, x, y)  # Shape: (155, 22, 4, 3)

        # Combine results into a single array
        prep_3 = cp.stack([x_undistorted, y_undistorted, likelihood], axis=-1)  # Shape: (155, 22, 4, 3, 3)


        x = prep_2[:, :, :, :, 0]  # Shape: (155, 22, 4, 3)
        y = prep_2[:, :, :, :, 1]  # Shape: (155, 22, 4, 3)
        likelihood = prep_2[:, :, :, :, 2]  # Shape: (155, 22, 4, 3)

        # Perform bilinear interpolation on x and y
        x_undistorted = self.bilinear_interpolate_cupy(self.mappingx, x, y)  # Shape: (155, 22, 4, 3)
        y_undistorted = self.bilinear_interpolate_cupy(self.mappingy, x, y)  # Shape: (155, 22, 4, 3)

        # Combine results into a single array
        prep_2 = cp.stack([x_undistorted, y_undistorted, likelihood], axis=-1)  # Shape: (155, 22, 4, 3, 3)
        return prep_4, prep_3, prep_2

    def bilinear_interpolate_cupy(self, map, x, y):
        
        # Get integer coordinates surrounding the point
        x0 = cp.floor(x).astype(cp.int32)
        y0 = cp.floor(y).astype(cp.int32)
        # Ensure coordinates are within bounds
        x0 = cp.clip(x0,0, map.shape[1] - 2)
        y0 = cp.clip(y0,0, map.shape[0] - 2)

        x1 = x0 + 1
        y1 = y0 + 1

        # Use cp.take_along_axis for advanced indexing
        Ia = map[y0, x0]
        Ib = map[y0, x1]
        Ic = map[y1, x0]
        Id = map[y1, x1]

        # Interpolation weights
        wa = (x1 - x) * (y1 - y)
        wb = (x - x0) * (y1 - y)
        wc = (x1 - x) * (y - y0)
        wd = (x - x0) * (y - y0)

        # Calculate the interpolated value
        return wa * Ia + wb * Ib + wc * Ic + wd * Id
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
        
        # Params for triangulation
        config = os.path.join(os.getcwd(), "NTK_CAP", "template", "Empty_project", "User", "Config.toml")
        if type(config)==dict:
            self.config_dict = config
        else:
            self.config_dict = self.read_config_file(config)
    def read_config_file(self, config):
        config_dict = toml.load(config)
        return config_dict
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
            self.stop_camera_evt,
            self.config_dict
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
        self.trangulation.start()
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
        self.trangulation.terminate()
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