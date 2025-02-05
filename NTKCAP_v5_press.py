from PyQt6.QtWebEngineWidgets import QWebEngineView
import logging
import time
from datetime import datetime
import os
import cv2
from NTK_CAP.script_py.NTK_Cap import *
from check_extrinsic import *
from NTK_CAP.script_py.kivy_file_chooser import select_directories_and_return_list
import traceback
from functools import partial
import sys
from PyQt6.QtWidgets import *
from PyQt6 import uic
from PyQt6.QtGui import *
from PyQt6.QtCore import *
import json
from datetime import datetime
from multiprocessing import Event, shared_memory, Manager, Queue, Array, Lock, Process
from mmdeploy_runtime import PoseTracker
import copy
import socket
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pyqtgraph as pg
# VISUALIZATION_CFG = dict(
#     halpe26=dict(
#         skeleton=[(15,13), (13,11), (11,19),(16,14), (14,12), (12,19),
#                   (17,18), (18,19), (18,5), (5,7), (7,9), (18,6), (6,8),
#                   (8,10), (1,2), (0,1), (0,2), (1,3), (2,4), (3,5), (4,6),
#                   (15,20), (15,22), (15,24),(16,21),(16,23), (16,25)],
#         # palette=[(51,153,255), (0,255,0), (255,128,0)],
#         palette=[(128,128,128), (51,153,255), (192,192,192)],
#         link_color=[
#             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
#         ],
#         point_color=[
#             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
#         ],
#         sigmas=[
#             0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072,
#             0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089, 0.026,
#             0.026, 0.066, 0.079, 0.079, 0.079, 0.079, 0.079, 0.079
#         ]))
VISUALIZATION_CFG = dict(
    coco=dict(
        skeleton=[(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11),
                  (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2),
                  (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)],
        palette=[(255, 128, 0), (255, 153, 51), (255, 178, 102), (230, 230, 0),
                 (255, 153, 255), (153, 204, 255), (255, 102, 255),
                 (255, 51, 255), (102, 178, 255), (51, 153, 255),
                 (255, 153, 153), (255, 102, 102), (255, 51, 51),
                 (153, 255, 153), (102, 255, 102), (51, 255, 51), (0, 255, 0),
                 (0, 0, 255), (255, 0, 0), (255, 255, 255)],
        link_color=[
            0, 0, 0, 0, 7, 7, 7, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 16, 16
        ],
        point_color=[16, 16, 16, 16, 16, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0],
        sigmas=[
            0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072,
            0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089
        ]),
    coco_wholebody=dict(
        skeleton=[(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11),
                  (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2),
                  (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (15, 17),
                  (15, 18), (15, 19), (16, 20), (16, 21), (16, 22), (91, 92),
                  (92, 93), (93, 94), (94, 95), (91, 96), (96, 97), (97, 98),
                  (98, 99), (91, 100), (100, 101), (101, 102), (102, 103),
                  (91, 104), (104, 105), (105, 106), (106, 107), (91, 108),
                  (108, 109), (109, 110), (110, 111), (112, 113), (113, 114),
                  (114, 115), (115, 116), (112, 117), (117, 118), (118, 119),
                  (119, 120), (112, 121), (121, 122), (122, 123), (123, 124),
                  (112, 125), (125, 126), (126, 127), (127, 128), (112, 129),
                  (129, 130), (130, 131), (131, 132)],
        palette=[(51, 153, 255), (0, 255, 0), (255, 128, 0), (255, 255, 255),
                 (255, 153, 255), (102, 178, 255), (255, 51, 51)],
        link_color=[
            1, 1, 2, 2, 0, 0, 0, 0, 1, 2, 1, 2, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
            2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 1, 1, 1,
            1, 2, 2, 2, 2, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 1, 1, 1, 1
        ],
        point_color=[
            0, 0, 0, 0, 0, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2,
            2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
            3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
            3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
            3, 3, 3, 3, 2, 2, 2, 2, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 1, 1,
            1, 1, 3, 2, 2, 2, 2, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 1, 1, 1, 1
        ],
        sigmas=[
            0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072,
            0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089, 0.068,
            0.066, 0.066, 0.092, 0.094, 0.094, 0.042, 0.043, 0.044, 0.043,
            0.040, 0.035, 0.031, 0.025, 0.020, 0.023, 0.029, 0.032, 0.037,
            0.038, 0.043, 0.041, 0.045, 0.013, 0.012, 0.011, 0.011, 0.012,
            0.012, 0.011, 0.011, 0.013, 0.015, 0.009, 0.007, 0.007, 0.007,
            0.012, 0.009, 0.008, 0.016, 0.010, 0.017, 0.011, 0.009, 0.011,
            0.009, 0.007, 0.013, 0.008, 0.011, 0.012, 0.010, 0.034, 0.008,
            0.008, 0.009, 0.008, 0.008, 0.007, 0.010, 0.008, 0.009, 0.009,
            0.009, 0.007, 0.007, 0.008, 0.011, 0.008, 0.008, 0.008, 0.01,
            0.008, 0.029, 0.022, 0.035, 0.037, 0.047, 0.026, 0.025, 0.024,
            0.035, 0.018, 0.024, 0.022, 0.026, 0.017, 0.021, 0.021, 0.032,
            0.02, 0.019, 0.022, 0.031, 0.029, 0.022, 0.035, 0.037, 0.047,
            0.026, 0.025, 0.024, 0.035, 0.018, 0.024, 0.022, 0.026, 0.017,
            0.021, 0.021, 0.032, 0.02, 0.019, 0.022, 0.031
        ]),
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
# !!!!
det_model_path = os.path.join(os.getcwd(),"NTK_CAP", "ThirdParty", "mmdeploy", "rtmpose-trt", "rtmdet-nano")
pose_model_path = os.path.join(os.getcwd(),"NTK_CAP", "ThirdParty", "mmdeploy", "rtmpose-trt", "rtmpose-m")
device ="cuda"

sigmas = VISUALIZATION_CFG['halpe26']['sigmas']
palette = VISUALIZATION_CFG['halpe26']['palette']
skeleton = VISUALIZATION_CFG['halpe26']['skeleton']
link_color = VISUALIZATION_CFG['halpe26']['link_color']
point_color = VISUALIZATION_CFG['halpe26']['point_color']
class TrackerProcess(Process):
    def __init__(self, start_evt, cam_id, stop_evt, queue_cam, queue_kp, shm, shm_kp, *args, **kwargs):
        super().__init__()
        self.start_evt = start_evt
        self.cam_id = cam_id
        self.stop_evt = stop_evt
        self.queue_cam = queue_cam
        self.queue_kp = queue_kp
        self.shm = shm
        self.buffer_length = 4
        self.shm_kp = shm_kp
        self.recording = False
        self.frame_width = 1920
        self.frame_height = 1080
        
    def run(self):
        shape = (1080, 1920, 3)
        existing_shm = shared_memory.SharedMemory(name=self.shm)
        shared_array = np.ndarray((self.buffer_length,) + shape, dtype=np.uint8, buffer=existing_shm.buf)
        np.set_printoptions(precision=4, suppress=True)
        tracker = PoseTracker(det_model=det_model_path,pose_model=pose_model_path,device_name=device)
        state = tracker.create_state(det_interval=1, det_min_bbox_size=100, keypoint_sigmas=sigmas, pose_max_num_bboxes=1)
        existing_shm_kp = shared_memory.SharedMemory(name=self.shm_kp)
        shared_array_kp = np.ndarray((self.buffer_length,) + shape, dtype=np.uint8, buffer=existing_shm_kp.buf)
        idx = 0
        while self.start_evt.is_set():
            try:
                idx_get = self.queue_cam.get(timeout=1)
            except:
                time.sleep(0.01)
                continue
            frame = shared_array[idx_get, : ]
            keypoints, _ = tracker(state, frame, detect=-1)[:2]
            scores = keypoints[..., 2]
            keypoints = np.round(keypoints[..., :2], 3)
            
            self.draw_frame(frame, keypoints, scores, palette, skeleton, link_color, point_color)
            np.copyto(shared_array_kp[idx,:], frame)
            self.queue_kp.put(idx) # !!!!idx?
            idx = (idx+1) % self.buffer_length
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
    def __init__(self, shared_name, cam_id, start_evt, task_rec_evt, apose_rec_evt, calib_rec_evt, stop_evt, task_stop_rec_evt, calib_video_save_path, queue, address_base, record_task_name, *args, **kwargs):
        super().__init__()
        ### define
        self.shared_name = shared_name
        self.cam_id = cam_id
        self.start_evt = start_evt
        self.task_rec_evt = task_rec_evt
        self.apose_rec_evt = apose_rec_evt
        self.calib_rec_evt = calib_rec_evt
        self.stop_evt = stop_evt
        self.task_stop_rec_evt = task_stop_rec_evt
        self.queue = queue
        self.frame_count = 0
        self.buffer_length = 4
        self.delay_time = 0.2
        self.shared_dict_record = record_task_name
        self.record_date = datetime.now().strftime("%Y_%m_%d")
        self.calib_video_save_path = calib_video_save_path
        ### initialization
        self.recording = False
        self.start_time = None
        self.frame_width = 1920
        self.frame_height = 1080
        self.fps = 30
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.patientID_path = address_base
        self.frame_id = 0
        self.time_stamp = None
        self.record_path = None
    def run(self):
        shape = (1080, 1920, 3)
        # shared memory setup
        idx = 0 # the current index of frames in shared memory
        existing_shm = shared_memory.SharedMemory(name=self.shared_name)
        shared_array = np.ndarray((self.buffer_length,) + shape, dtype=np.uint8, buffer=existing_shm.buf)

        cap = cv2.VideoCapture(self.cam_id)
        (width, height) = (1920, 1080)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.start_evt.wait()

        while True:
            cap_s = time.time()
            ret, frame = cap.read()
            cap_e = time.time()
            
            if not ret or self.stop_evt.is_set():
                break
            
            if self.task_stop_rec_evt.is_set() and self.recording:
                self.recording = False
                self.start_time = None
                self.frame_count = 0
                self.out.release()
                np.save(os.path.join(self.record_path, f"{self.cam_id+1}_dates.npy"), self.time_stamp)
                
            if self.apose_rec_evt.is_set():
                if not self.recording:
                    self.record_path = None
                    self.recording = True
                    self.record_path = os.path.join(self.patientID_path, self.shared_dict_record['name'], self.record_date, 'raw_data', "Apose", "videos")
                    self.out = cv2.VideoWriter(os.path.join(self.record_path, f"{self.cam_id+1}.mp4"), self.fourcc, self.fps, (self.frame_width, self.frame_height))
                    self.frame_id = 0
                self.out.write(frame)
                if self.frame_id < 9:
                    self.frame_id += 1
                else:
                    self.recording = False
                    self.frame_count = 0
                    self.out.release()
                    self.apose_rec_evt.clear()
            if self.calib_rec_evt.is_set():
                self.record_path = None
                self.out = cv2.VideoWriter(os.path.join(self.calib_video_save_path, f"{self.cam_id+1}.mp4"), self.fourcc, self.fps, (self.frame_width, self.frame_height))
                self.out.write(frame)
                self.out.release()
                self.calib_rec_evt.clear()
            if self.task_rec_evt.is_set():
                if not self.recording:
                    self.record_path = None
                    self.time_stamp = None
                    self.recording = True
                    self.record_path = os.path.join(self.patientID_path, self.shared_dict_record['name'], self.record_date, 'raw_data', self.shared_dict_record['task_name'], 'videos')
                    self.out = cv2.VideoWriter(os.path.join(self.record_path, f"{self.cam_id+1}.mp4"), self.fourcc, self.fps, (self.frame_width, self.frame_height))
                    self.frame_id = 0
                    self.start_time = self.shared_dict_record['start_time']
                    self.time_stamp = np.empty((1000000, 2))
                self.out.write(frame)
                dt_str1 = cap_s - self.start_time
                dt_str2 = cap_e - self.start_time
                self.time_stamp[self.frame_id] =[np.array(float( f"{dt_str1 :.3f}")),np.array(float( f"{dt_str2 :.3f}"))]
                self.frame_id += 1
            np.copyto(shared_array[idx,:], frame)     
            self.queue.put(idx)
            idx = (idx+1) % self.buffer_length

        cap.release()
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

    def run(self):
        shape = (1080, 1920, 3)
        dtype = np.uint8
        existing_shm = shared_memory.SharedMemory(name=self.shared_name_kp)
        shared_array = np.ndarray((self.buffer_length,) + shape, dtype=dtype, buffer=existing_shm.buf)
        self.start_evt.wait()
        self.ThreadActive = True
        while self.ThreadActive:
            try:
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
        Image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = Image.shape
        bytesPerline = channel * width
        ConvertToQtFormat = QImage(Image, width, height, bytesPerline, QImage.Format.Format_RGB888)
        Pic = ConvertToQtFormat.scaled(QSize(scale_size[0], scale_size[1]), Qt.AspectRatioMode.KeepAspectRatio)
        return Pic

class VideoPlayer(QThread):
    frame_changed = pyqtSignal(int)
    data_ready = pyqtSignal(np.ndarray, np.ndarray, int, int)
    def __init__(self, labels: list[QLabel], web_viewer: QWebEngineView, plot, parent=None):
        super().__init__(parent)
        self.video_path = None
        self.label1, self.label2, self.label3, self.label4 = labels[0], labels[1], labels[2], labels[3]
        self.frame_count = None
        self.timer = QTimer()
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer.timeout.connect(self.run)
        self.result_load_videos_allframes = []
        self.web_view_widget = web_viewer
        self.progress = 0
        self.start_server()

        self.plot = plot
    def load_video(self, cal_task_path: str):
        self.result_load_videos_allframes = []
        self.progress = 0
        for i in range(4):
            video_path = os.path.join(cal_task_path, 'videos_pose_estimation_repj_combine', f"{i+1}.mp4")
            cap = cv2.VideoCapture(video_path)
            self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.result_load_videos_allframes.append([])
            while True:
                ret, frame = cap.read()
                if not ret: break                
                self.result_load_videos_allframes[i].append(frame)
            cap.release()

    def np2qimage(self, img):
        resized_frame = cv2.cvtColor(cv2.resize(img, (self.label2.width(), self.label2.height()), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        height, width, channel = resized_frame.shape
        bytes_per_line = channel * width
        q_image = QImage(resized_frame, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        return pixmap
    def result_load_gait_figures(self, cal_task_path: str):
        result_post_analysis_imgs_data_path = os.path.join(cal_task_path, 'post_analysis', "raw_data.npz")
        data = np.load(result_post_analysis_imgs_data_path, allow_pickle=True)
        self.knee_data = data['Knee'].item()
        self.hip_data = data['Hip'].item()
        self.ankle_data = data['Ankle'].item()
        self.Stride  = data['Stride'].item()
        self.Speed = data['Speed'].item()
        R_knee = self.knee_data['R_knee']
        L_knee = self.knee_data['L_knee']
        
        self.plot.showGrid(x=True, y=True, alpha=0.2)  # Enable grid
        
    def set_slider_value(self, progress):
        self.progress = progress
        if self.progress !=0:
            if int(progress * 100) >= 100:
                self.progress = 0
                self.frame_changed.emit(0)
                self.web_view_widget.page().runJavaScript("resetAnimation();")
                
            else:
                self.frame_changed.emit(round(self.progress * 100))

    def run(self):        
        cap_b = time.perf_counter()      
        self.web_view_widget.page().runJavaScript(
            """
            (function() {
                if (typeof window.getAnimationProgress === 'function') {
                    return getAnimationProgress();
                } else {
                    return 0;
                }
            })();
            """,
            self.set_slider_value
        )
        
        self.label1.setPixmap(self.np2qimage(self.result_load_videos_allframes[0][int(self.progress * self.frame_count)]))
        self.label2.setPixmap(self.np2qimage(self.result_load_videos_allframes[1][int(self.progress * self.frame_count)]))
        self.label3.setPixmap(self.np2qimage(self.result_load_videos_allframes[2][int(self.progress * self.frame_count)]))
        self.label4.setPixmap(self.np2qimage(self.result_load_videos_allframes[3][int(self.progress * self.frame_count)]))
        
    def slider_changed(self):
        self.web_view_widget.page().runJavaScript(f"window.updateAnimationProgress({self.progress});")     
        self.label1.setPixmap(self.np2qimage(self.result_load_videos_allframes[0][int(self.progress * self.frame_count)]))
        self.label2.setPixmap(self.np2qimage(self.result_load_videos_allframes[1][int(self.progress * self.frame_count)]))
        self.label3.setPixmap(self.np2qimage(self.result_load_videos_allframes[2][int(self.progress * self.frame_count)]))
        self.label4.setPixmap(self.np2qimage(self.result_load_videos_allframes[3][int(self.progress * self.frame_count)]))
    def hip_flexion_plot(self):
        self.scatter_ball_R, self.scatter_ball_L, self.xline = None, None, None
        R_Hip = self.hip_data['R_Hip']
        L_Hip = self.hip_data['L_Hip']
        loc_max_finalR = self.hip_data['loc_max_finalR']
        loc_max_finalL = self.hip_data['loc_max_finalL']
        
        # Clear any existing plots
        self.plot.clear()
        self.plot.setYRange(min(min(np.concatenate([R_Hip,L_Hip])), min(np.concatenate([R_Hip,L_Hip]))) - 10, max(max(np.concatenate([R_Hip,L_Hip])), max(np.concatenate([R_Hip,L_Hip]))) + 10)
        self.plot.setXRange(0, len(R_Hip))  # Full data range for the plot
        #slider.valueChanged.disconnect()
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.addLegend()  # Re-add legend

        # Plot right and left Hip data
        self.plot.plot(R_Hip, pen=pg.mkPen('r', width=3), name="Right Hip")
        self.plot.plot(L_Hip, pen=pg.mkPen('b', width=3), name="Left Hip")

        # Plot max points
        self.plot.plot(loc_max_finalR, R_Hip[loc_max_finalR], pen=None, symbol='t', symbolSize=15, symbolBrush=(0, 255, 0))
        self.plot.plot(loc_max_finalL, L_Hip[loc_max_finalL], pen=None, symbol='t', symbolSize=15, symbolBrush=(0, 255, 255))

        # Plot midpoint stars
        midpoint_R = loc_max_finalR[len(loc_max_finalR) // 2]
        midpoint_L = loc_max_finalL[len(loc_max_finalL) // 2]
        self.plot.plot([midpoint_R], [R_Hip[midpoint_R]], pen=None, symbol='star', symbolSize=30, symbolBrush=(255, 0, 255))
        self.plot.plot([midpoint_L], [L_Hip[midpoint_L]], pen=None, symbol='star', symbolSize=30, symbolBrush=(255, 0, 255))

        # Create new scatter balls
        self.scatter_ball_R = self.plot.plot([0], [R_Hip[0]], pen=None, symbol='o', symbolSize=15, symbolBrush='r')
        self.scatter_ball_L = self.plot.plot([0], [L_Hip[0]], pen=None, symbol='o', symbolSize=15, symbolBrush='b')

        # Create a new vertical timeline (xline)
        self.xline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('k', width=2))
        self.plot.addItem(self.xline)

        # Update the initial plot
    def update_hip_flexion(self):
        frame_index = int(self.progress * self.frame_count)
        # Move the vertical timeline and scatter balls
        self.xline.setPos(frame_index)
        self.scatter_ball_R.setData([frame_index], [self.hip_data['R_Hip'][frame_index]])
        self.scatter_ball_L.setData([frame_index], [self.hip_data['L_Hip'][frame_index]])
    def knee_flexion_plot(self):
        self.scatter_ball_R, self.scatter_ball_L, self.xline = None, None, None
        R_knee = self.knee_data['R_knee']
        L_knee = self.knee_data['L_knee']
        loc_max_finalR = self.knee_data['loc_max_finalR']
        loc_max_finalL = self.knee_data['loc_max_finalL']
            
        
        # Clear any existing plots
        self.plot.clear()
        
        #slider.valueChanged.disconnect()
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.addLegend()  # Re-add legend
        self.plot.setYRange(min(min(np.concatenate([R_knee, L_knee])), min(np.concatenate([R_knee, L_knee]))) - 10, max(max(np.concatenate([R_knee, L_knee])), max(np.concatenate([R_knee, L_knee]))) + 10)
        self.plot.setXRange(0, len(R_knee))  # Full data range for the plot


        # Plot right and left knee data
        self.plot.plot(R_knee, pen=pg.mkPen('r', width=3), name="Right Knee")
        self.plot.plot(L_knee, pen=pg.mkPen('b', width=3), name="Left Knee")

        # Plot max points
        self.plot.plot(loc_max_finalR, R_knee[loc_max_finalR], pen=None, symbol='t', symbolSize=15, symbolBrush=(0, 255, 0))
        self.plot.plot(loc_max_finalL, L_knee[loc_max_finalL], pen=None, symbol='t', symbolSize=15, symbolBrush=(0, 255, 255))

        # Plot midpoint stars
        midpoint_R = loc_max_finalR[len(loc_max_finalR) // 2]
        midpoint_L = loc_max_finalL[len(loc_max_finalL) // 2]
        self.plot.plot([midpoint_R], [R_knee[midpoint_R]], pen=None, symbol='star', symbolSize=30, symbolBrush=(255, 0, 255))
        self.plot.plot([midpoint_L], [L_knee[midpoint_L]], pen=None, symbol='star', symbolSize=30, symbolBrush=(255, 0, 255))

        # Create new scatter balls
        self.scatter_ball_R = self.plot.plot([0], [R_knee[0]], pen=None, symbol='o', symbolSize=15, symbolBrush='r')
        self.scatter_ball_L = self.plot.plot([0], [L_knee[0]], pen=None, symbol='o', symbolSize=15, symbolBrush='b')

        # Create a new vertical timeline (xline)
        self.xline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('k', width=2))
        self.plot.addItem(self.xline)

    # Update the animation based on slider value
    def update_knee_flexion(self):
        frame_index = int(self.progress * self.frame_count)
        # Move the vertical timeline and scatter balls
        self.xline.setPos(frame_index)
        self.scatter_ball_R.setData([frame_index], [self.knee_data['R_knee'][frame_index]])
        self.scatter_ball_L.setData([frame_index], [self.knee_data['L_knee'][frame_index]])
    def ankle_flexion_plot(self):
        self.scatter_ball_R, self.scatter_ball_L, self.xline = None, None, None
        R_ankle = self.ankle_data['R_ankle']
        L_ankle = self.ankle_data['L_ankle']
        loc_max_finalR = self.ankle_data['loc_max_finalR']
        loc_max_finalL = self.ankle_data['loc_max_finalL']

        # Clear any existing plots
        self.plot.clear()
        #slider.valueChanged.disconnect()
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.addLegend()  # Re-add legend
        self.plot.setYRange(min(min(np.concatenate([R_ankle,L_ankle])), min(np.concatenate([R_ankle,L_ankle]))) - 10, max(max(np.concatenate([R_ankle,L_ankle])), max(np.concatenate([R_ankle,L_ankle]))) + 10)
        self.plot.setXRange(0, len(R_ankle))  # Full data range for the plot
        # Plot right and left ankle data
        self.plot.plot(R_ankle, pen=pg.mkPen('r', width=3), name="Right ankle")
        self.plot.plot(L_ankle, pen=pg.mkPen('b', width=3), name="Left ankle")

        # Plot max points
        self.plot.plot(loc_max_finalR, R_ankle[loc_max_finalR], pen=None, symbol='t', symbolSize=15, symbolBrush=(0, 255, 0))
        self.plot.plot(loc_max_finalL, L_ankle[loc_max_finalL], pen=None, symbol='t', symbolSize=15, symbolBrush=(0, 255, 255))

        # Plot midpoint stars
        midpoint_R = loc_max_finalR[len(loc_max_finalR) // 2]
        midpoint_L = loc_max_finalL[len(loc_max_finalL) // 2]
        self.plot.plot([midpoint_R], [R_ankle[midpoint_R]], pen=None, symbol='star', symbolSize=30, symbolBrush=(255, 0, 255))
        self.plot.plot([midpoint_L], [L_ankle[midpoint_L]], pen=None, symbol='star', symbolSize=30, symbolBrush=(255, 0, 255))

        # Create new scatter balls
        self.scatter_ball_R = self.plot.plot([0], [R_ankle[0]], pen=None, symbol='o', symbolSize=15, symbolBrush='r')
        self.scatter_ball_L = self.plot.plot([0], [L_ankle[0]], pen=None, symbol='o', symbolSize=15, symbolBrush='b')

        # Create a new vertical timeline (xline)
        self.xline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('k', width=2))
        self.plot.addItem(self.xline)

    def update_ankle_flexion(self):
        frame_index = int(self.progress * self.frame_count)
        # Move the vertical timeline and scatter balls
        self.xline.setPos(frame_index)
        self.scatter_ball_R.setData([frame_index], [self.ankle_data['R_ankle'][int(frame_index)]])
        self.scatter_ball_L.setData([frame_index], [self.ankle_data['L_ankle'][int(frame_index)]])
    def speed_plot(self):
        self.scatter_ball_B, self.scatter_ball_mean, self.xline, self.scatter_ball_flunc = None, None, None, None
        self.plot.clear()
        self.plot.addLegend(offset=(-10, 10)) 
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        
        B = self.Speed['B']
        mean_velocity = self.Speed['mean_velocity']
        flunc_velocity = self.Speed['flunc_velocity']
        interp_speed = self.Speed['interp_speed']
        bound = self.Speed['bound']
        bound_start_end = self.Speed['bound_start_end']
        rms_final_steady = self.Speed['rms_final_steady']
        rms_start_end = self.Speed['rms_start_end']
        rms_All = self.Speed['rms_All']
        self.plot.setYRange(min(min(np.concatenate([B,flunc_velocity])), min(np.concatenate([B,flunc_velocity]))) -0.1, max(max(np.concatenate([B,flunc_velocity])), max(np.concatenate([B,flunc_velocity]))) + 0.1)
        self.plot.setXRange(0, len(B))  # Full data range for the plot
        # Plot lines
        self.plot.plot(mean_velocity, pen=pg.mkPen('b', width=3), name="Mean Velocity")
        self.plot.plot(flunc_velocity, pen=pg.mkPen('orange', width=3), name="Fluctuation Velocity")
        self.plot.plot(B, pen=pg.mkPen('#CCCC00', width=3), name="Raw Data")

        self.xline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('k', width=2))
        self.plot.addItem(self.xline)

        self.scatter_ball_mean = self.plot.plot([0], [mean_velocity[0]], pen=None, symbol='o', symbolSize=15, symbolBrush='b')
        self.scatter_ball_B = self.plot.plot([0], [B[0]], pen=None, symbol='o', symbolSize=15, symbolBrush='#CCCC00')
        self.scatter_ball_flunc = self.plot.plot([0], [flunc_velocity[0]], pen=None, symbol='o', symbolSize=15, symbolBrush='orange')
        # Draw vertical lines (equivalent to axvline)
        for x in [min(bound), max(bound)]:
            self.plot.addItem(pg.InfiniteLine(pos=x, angle=90, pen=pg.mkPen('#00FF00', style=Qt.PenStyle.DashLine, width=2)))


        for x in [min(bound_start_end), max(bound_start_end)]:
            self.plot.addItem(pg.InfiniteLine(pos=x, angle=90, pen=pg.mkPen('dimgrey', style=Qt.PenStyle.DashLine, width=2)))

        # Text annotations
        text_1 = pg.TextItem(f'rms steady = {rms_final_steady:.5f}', color='k', anchor=(0, 0), border=None); text_1.setFont(pg.QtGui.QFont("Arial", 12))
        text_2 = pg.TextItem(f'rms start end = {rms_start_end:.5f}', color='k', anchor=(0, 0), border=None); text_2.setFont(pg.QtGui.QFont("Arial", 12))
        text_3 = pg.TextItem(f'rms All = {rms_All:.5f}', color='k', anchor=(0, 0), border=None); text_3.setFont(pg.QtGui.QFont("Arial", 12))
        text_4 = pg.TextItem(f'max speed = {np.max(mean_velocity):.5f}', color='k', anchor=(0, 0), border=None); text_4.setFont(pg.QtGui.QFont("Arial", 12))
        # Position text items on the plot
        self.plot.addItem(text_1, ignoreBounds=True)
        self.plot.addItem(text_2, ignoreBounds=True)
        self.plot.addItem(text_3, ignoreBounds=True)
        self.plot.addItem(text_4, ignoreBounds=True)

        text_1.setPos(5, np.max(mean_velocity) )
        text_2.setPos(5, np.max(mean_velocity) -0.1)
        text_3.setPos(5, np.max(mean_velocity) - 0.2)
        text_4.setPos(5, np.max(mean_velocity) - 0.3)

    def update_speed(self):
        frame_index = int(self.progress * self.frame_count)
        self.xline.setPos(frame_index)
        self.scatter_ball_B.setData([frame_index], [self.Speed['B'][frame_index]])
        self.scatter_ball_mean.setData([frame_index], [self.Speed['mean_velocity'][frame_index]])
        self.scatter_ball_flunc.setData([frame_index], [self.Speed['flunc_velocity'][frame_index]])
    def stride_plot(self):
        self.scatter_ball_heel_L, self.scatter_ball_heel_R = None, None
        self.plot.clear()
        self.plot.addLegend()  # Re-add legend
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        
        R_heel = self.Stride['R_heel']
        L_heel = self.Stride['L_heel']
        frame_R_heel_sground = self.Stride['frame_R_heel_sground']
        frame_L_heel_sground = self.Stride['frame_L_heel_sground']

        xmr = self.Stride['xmr']
        xml = self.Stride['xml']
        ymr = self.Stride['ymr']
        yml = self.Stride['yml']
        pace_r = self.Stride['pace_r']
        pace_l = self.Stride['pace_l']
        mid_r_index = self.Stride['mid_r_index']
        mid_l_index = self.Stride['mid_l_index']
        self.plot.setXRange(min(min(np.concatenate([R_heel[:, 0],L_heel[:, 0]])), min(np.concatenate([R_heel[:, 0],L_heel[:, 0]]))) -0.1, max(max(np.concatenate([R_heel[:, 0],L_heel[:, 0]])), max(np.concatenate([R_heel[:, 0],L_heel[:, 0]]))) + 0.1)
        self.plot.setYRange(min(min(np.concatenate([R_heel[:, 2],L_heel[:, 2]])), min(np.concatenate([R_heel[:, 2],L_heel[:, 2]]))) -0.1, max(max(np.concatenate([R_heel[:, 2],L_heel[:, 2]])), max(np.concatenate([R_heel[:, 2],L_heel[:, 2]]))) + 0.1)

        # Scatter plots for heel traces
        self.plot.plot(R_heel[:, 0], R_heel[:, 2], pen=None, symbol='o', symbolSize=10, symbolBrush='b', name='Right Heel Trace')
        self.plot.plot(L_heel[:, 0], L_heel[:, 2], pen=None, symbol='o', symbolSize=10, symbolBrush='orange', name='Left Heel Trace')

        # Scatter plots for heel strikes
        self.plot.plot(R_heel[frame_R_heel_sground, 0], R_heel[frame_R_heel_sground, 2], pen=None, symbol='o', symbolSize=15, symbolBrush='r', name='Right Heel Strikes')
        self.plot.plot(L_heel[frame_L_heel_sground, 0], L_heel[frame_L_heel_sground, 2], pen=None, symbol='o', symbolSize=15, symbolBrush='r', name='Left Heel Strikes')
        self.scatter_ball_heel_R = self.plot.plot([R_heel[0, 0]], [R_heel[0, 2]], pen=None, symbol='star', symbolSize=25, symbolBrush='k')
        self.scatter_ball_heel_L = self.plot.plot([L_heel[0, 0]], [L_heel[0, 2]], pen=None, symbol='star', symbolSize=25, symbolBrush='k')
        # Adding text annotations for pace_r and pace_l
        for i in range(len(xmr)):
            text_item = pg.TextItem(f'{pace_r[i]:.4f}', color='k')
            self.plot.addItem(text_item)
            text_item.setPos(xmr[i], ymr[i])

        for i in range(len(xml)):
            text_item = pg.TextItem(f'{pace_l[i]:.4f}', color='k')
            self.plot.addItem(text_item)
            text_item.setPos(xml[i], yml[i])

        # Highlighting specific points
        self.plot.plot([xmr[mid_r_index] - 0.01], [ymr[mid_r_index]], pen=None, symbol='t', symbolSize=20, symbolBrush='#00FF00')
        self.plot.plot([xml[mid_l_index] - 0.01], [yml[mid_l_index]], pen=None, symbol='t', symbolSize=20, symbolBrush='#00FF00')

    def update_stride(self):
        frame_index = int(self.progress * self.frame_count)
        self.scatter_ball_heel_R.setData([self.Stride['R_heel'][frame_index, 0]], [self.Stride['R_heel'][frame_index, 2]])
        self.scatter_ball_heel_L.setData([self.Stride['L_heel'][frame_index, 0]], [self.Stride['L_heel'][frame_index, 2]])
    def start_server(self):
        port = 8000
        if self.is_port_in_use(port):
            self.kill_process_using_port(port)
        print("Starting local server...")
        self.server_process = subprocess.Popen(["python", "-m", "http.server", str(port)],
                                                cwd=".", stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    def is_port_in_use(self, port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0
    def kill_process_using_port(self, port):
        try:
            result = subprocess.check_output(f"netstat -ano | findstr :{port}", shell=True, text=True)
            for line in result.splitlines():
                if "LISTENING" in line:
                    pid = line.split()[-1]
                    subprocess.run(f"taskkill /PID {pid} /F", shell=True)
                    
        except subprocess.CalledProcessError:
            pass
    
    def load_gltf_file_in_viewer(self, file_path):
        self.web_view_widget.setUrl(QUrl(f"http://localhost:8000/viewer.html?model={file_path}"))
        
        
class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        uic.loadUi('NTKCAP_GUI_dec.ui', self)
        self.showMaximized()
        # Initialization
        self.current_directory = os.getcwd()
        self.language = 'Chinese'
        self.config_path = os.path.join(self.current_directory, "config")                
        self.cam_num, self.resolution = self.getcamerainfo(self.config_path)        
        self.time_file_path = os.path.join(self.current_directory, "cali_time.txt")        
        self.mode_select = 'Recording'
        self.record_path = self.current_directory
        self.calibra_path = os.path.join(self.current_directory, "calibration")
        self.patient_path = os.path.join(self.current_directory, "Patient_data")
        self.calib_toml_path = os.path.join(self.calibra_path, "Calib.toml")
        self.extrinsic_path = os.path.join(self.calibra_path,"ExtrinsicCalibration")
        
        self.font_path = os.path.join(self.current_directory, "NTK_CAP", "ThirdParty", "Noto_Sans_HK", "NotoSansHK-Bold.otf")
        self.show_result_path = None
        self.err_calib_extri.setText(read_err_calib_extri(self.current_directory)) # Show ext calib err
        self.label_cam = None
        self.tabWidgetRight = self.findChild(QTabWidget, "tabWidget_2")
        self.tabWidgetRight.hide()
        self.tabWidgetLeft = self.findChild(QTabWidget, "tabWidget")
        self.tabWidgetLeft.currentChanged.connect(self.left_tab_changed)
        self.btn_shortcut_load_videos = self.findChild(QPushButton, "btn_shortcut_load_videos")
        self.btn_shortcut_load_videos.clicked.connect(self.button_shortcut_load_videos)
        self.btn_shortcut_new_recording = self.findChild(QPushButton, "btn_shortcut_new_recording")
        self.btn_shortcut_new_recording.clicked.connect(self.button_shortcut_new_recording)
        self.btn_shortcut_calculation = self.findChild(QPushButton, "btn_shortcut_calculation")
        self.btn_shortcut_calculation.clicked.connect(self.button_shortcut_calculation)
        # self.btn_calibration_folder.clicked.connect(self.create_calibration_ask) # create new calib confirmation
        
        self.btn_extrinsic_record.clicked.connect(self.button_extrinsic_record)
        self.btn_extrinsic_record.setEnabled(False)
        self.btn_extrinsic_manual_calculate.clicked.connect(self.button_extrinsic_manual_calculate)
        self.btn_config.clicked.connect(self.button_config)
        self.btn_pg1_reset.clicked.connect(self.reset_setup_tab)
        self.btn_new_patient.clicked.connect(self.button_create_new_patient)
        # calibration
        self.calib_save_path = os.path.join(self.extrinsic_path, "videos")
        
        # Open cameras
        self.camera_proc_lst = []
        self.tracker_proc_lst = []
        self.shm_lst = []
        self.shm_kp_lst = []
        self.btnOpenCamera.clicked.connect(self.opencamera)
        self.btnCloseCamera.clicked.connect(self.closeCamera)
        self.btnCloseCamera.setEnabled(False)
        self.shape = (1080, 1920, 3)
        self.buffer_length = 4
        self.camera_opened = False       
        
        # Record task
        self.record_list_widget_patient_id: QListWidget = self.findChild(QListWidget, "list_widget_patient_id_record")
        self.record_list_widget_patient_id.itemDoubleClicked.connect(self.lw_record_select_patientID)
        self.record_select_patientID = None
        self.record_task_name = None
        self.record_list_widget_patient_id_list_show()
        self.btnStartRecording.clicked.connect(self.startrecord_task)
        self.btnStartRecording.setEnabled(False)
        self.btnStopRecording.clicked.connect(self.stoprecord_task)
        self.btnStopRecording.setEnabled(False)
        self.record_opened = False
        self.record_enter_task_name: QLineEdit = self.findChild(QLineEdit, "record_enter_task_name")
        self.record_enter_task_name.setEnabled(False)
        self.record_enter_task_name_kb_listen = False
        self.record_enter_task_name.focusInEvent = self.record_enter_task_name_infocus
        self.record_enter_task_name.focusOutEvent = self.record_enter_task_name_outfocus
        self.list_widget_patient_task_record = self.findChild(QListWidget, "list_widget_patient_task_record")
        
        # record Apose
        self.btn_Apose_record: QPushButton = self.findChild(QPushButton, "btn_Apose_record")
        self.btn_Apose_record.clicked.connect(self.Apose_record_ask)
        self.btn_Apose_record.setEnabled(False)
        self.timer_apose = QTimer()
        self.timer_apose.timeout.connect(self.check_apose_finish)
        # Show result
        self.widget_gait_figure = self.findChild(QWidget, "widget_gait_figure")
        pg.setConfigOption('background', 'w') # White background
        pg.setConfigOption('foreground', 'k') # Set text and grid to black
        self.graphWidget = pg.GraphicsLayoutWidget()
        self.gait_figure_layout = QVBoxLayout(self.widget_gait_figure)
        self.gait_figure_layout.addWidget(self.graphWidget)
        plot = self.graphWidget.addPlot(title="Gait Figures")
        self.combobox_result_select_gait_figures = self.findChild(QComboBox, "combobox_result_select_gait_figures")
        self.combobox_result_select_gait_figures.currentIndexChanged.connect(self.result_select_gait_figures)
        self.result_current_gait_figures_index = 0
        self.select_result_cal_rpjvideo_labels = [
            self.findChild(QLabel, "rpjvideo1"),
            self.findChild(QLabel, "rpjvideo2"),
            self.findChild(QLabel, "rpjvideo3"),
            self.findChild(QLabel, "rpjvideo4")
        ]
        self.frame_label: QLabel = self.findChild(QLabel, "frame_show")
        self.playButton: QPushButton = self.findChild(QPushButton, "btn_result_video_playstop")
        self.playButton.clicked.connect(self.play_stop)
        self.playButton.setEnabled(False)
        self.result_video_slider: QSlider = self.findChild(QSlider, "VideoProgressSlider")
        self.result_video_slider.sliderMoved.connect(self.slider_changed)
        self.result_video_slider.setMaximum(100)
        self.result_video_slider.setMinimum(0)
        self.result_video_slider.setEnabled(False)
        self.current_frame = None
        self.video_players = []
        self.frame_rate = 30
        self.is_playing = False
        self.video_player = VideoPlayer(self.select_result_cal_rpjvideo_labels, self.result_web_view_widget, plot)
        self.video_player.frame_changed.connect(self.update_slider)
        self.result_select_patient_id, self.result_select_date, self.result_select_cal_time, self.result_select_task = None, None, None, None
        self.result_select_depth = 0
        
        self.result_cal_task: QListWidget = self.findChild(QListWidget, "result_cal_task")
        self.result_cal_task.itemClicked.connect(self.folder_selected)
        self.result_cal_task.itemDoubleClicked.connect(self.folder_clicked)
        self.btn_result_back_path = self.findChild(QPushButton, "btn_result_back_path")
        self.btn_result_back_path.clicked.connect(self.select_back_path)
        self.text_label_path_depth = self.findChild(QLabel, "text_label_path_depth")
        self.text_label_path_depth.setText(" ")
        self.label_result_patient_id = self.findChild(QLabel, "label_result_patient_id")
        self.label_result_record_date = self.findChild(QLabel, "label_result_record_date")
        self.label_result_calculation_time = self.findChild(QLabel, "label_result_calculation_time")
        self.label_result_task = self.findChild(QLabel, "label_result_task")
        self.result_load_folders()
        
        self.result_web_view_widget: QWebEngineView = self.findChild(QWebEngineView, "result_web_view_widget")
        
        # Calculation
        self.cal_select_patient_id, self.cal_select_date = None, None
        self.cal_select_depth = 0
        self.listwidget_select_cal_date = self.findChild(QListWidget, "listwidget_select_cal_date")
        self.listwidget_select_cal_date.itemDoubleClicked.connect(self.cal_select_folder_clicked)
        self.listwidget_selected_cal_date = self.findChild(QListWidget, "listwidget_selected_cal_date")
        self.listwidget_selected_cal_date.itemClicked.connect(self.cal_selected_folder_selcted)
        self.btn_cal_back_path = self.findChild(QPushButton, "btn_cal_back_path")
        self.btn_cal_back_path.clicked.connect(self.cal_select_back_path)
        self.btn_cal_back_path.setEnabled(False)
        self.cal_select_list = []
        self.listwidget_selected_cal_date_item_selected = None
        self.btn_cal_delete = self.findChild(QPushButton, "btn_cal_delete")
        self.btn_cal_delete.clicked.connect(self.cal_selected_folder_selcted_delete)
        self.btn_cal_start_cal = self.findChild(QPushButton, "btn_cal_start_cal")
        self.btn_cal_start_cal.clicked.connect(self.btn_pre_marker_calculate)
        self.btn_cal_start_cal.setEnabled(True)
        self.cal_load_folders()
        self.marker_calculate_process = None
        self.timer_marker_calculate = QTimer()
        self.timer_marker_calculate.timeout.connect(self.check_cal_finish)
    def button_shortcut_calculation(self):
        self.left_tab_changed(1)
        self.tabWidgetRight.setCurrentIndex(1)
    def button_shortcut_load_videos(self):
        self.left_tab_changed(2)
        self.tabWidgetLeft.setCurrentIndex(2)
    def button_shortcut_new_recording(self):
        self.left_tab_changed(1)
        self.tabWidgetLeft.setCurrentIndex(1)
        self.tabWidgetRight.setCurrentIndex(0)
    def left_tab_changed(self, index):
        if index == 1:
            self.tabWidgetRight.show()
        else:
            self.tabWidgetRight.hide()
    def reset_setup_tab(self):
        self.record_select_patientID = None
        self.record_task_name = None
        self.list_widget_patient_id_record.clearSelection()
        self.record_enter_task_name.setText("")
        self.record_enter_task_name.setEnabled(False)
        self.btn_Apose_record.setEnabled(False)
        self.btnStartRecording.setEnabled(False)
        self.btnStopRecording.setEnabled(False)
    # start record
    def keyPressEvent(self, event):
        if (self.record_enter_task_name_kb_listen) and (event.key() == 16777220):
            if os.path.exists(os.path.join(self.patient_path, self.record_select_patientID, datetime.now().strftime("%Y_%m_%d"), "raw_data", self.record_enter_task_name.text())):
                reply = QMessageBox.question(
                    self, 
                    "task name already exists",
                    "Are you sure to cover it?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.Yes:
                    self.record_task_name = self.record_enter_task_name.text()
                    self.btnStartRecording.setEnabled(True)
                    self.btnStopRecording.setEnabled(True)
                    self.btn_Apose_record.setEnabled(False)
                    self.label_log.setText(f"Current task name : {self.record_task_name}")
            else:
                self.record_task_name = self.record_enter_task_name.text()
                self.btnStartRecording.setEnabled(True)
                self.btnStopRecording.setEnabled(True)
                self.btn_Apose_record.setEnabled(False)
                self.label_log.setText(f"Current task name : {self.record_task_name}")

    def record_enter_task_name_infocus(self, event):
        self.record_enter_task_name_kb_listen = True

    def record_enter_task_name_outfocus(self, event):
        self.record_enter_task_name_kb_listen = False
    def lw_patient_task_record(self):
        self.list_widget_patient_task_record.clear()
        patient_today_task_recorded = os.path.join(self.patient_path, self.record_select_patientID, datetime.now().strftime("%Y_%m_%d"))
        if os.path.exists(patient_today_task_recorded):
            patient_tasks_recorded_list = [item for item in os.listdir(os.path.join(patient_today_task_recorded, "raw_data"))]
            for item in patient_tasks_recorded_list:
                self.list_widget_patient_task_record.addItem(item)
        else:
            self.label_log.setText("No tasks recorded today")
    def lw_record_select_patientID(self, item):
        self.record_select_patientID = item.text()
        self.record_enter_task_name.setEnabled(True)
        self.btn_Apose_record.setEnabled(True)
        self.record_task_name = None
        self.lw_patient_task_record()
        self.label_selected_patient.setText(f"Selected patient : {self.record_select_patientID}")
    def record_list_widget_patient_id_list_show(self):
        self.record_list_widget_patient_id.clear()
        patientID_list = [item for item in os.listdir(self.patient_path)]
        for item in patientID_list:
            self.record_list_widget_patient_id.addItem(item)
    def button_create_new_patient(self):
        text, ok = QInputDialog.getText(self, 'New subject', 'Enter patient ID:')
        
        if ok:
            if os.path.exists(os.path.join(self.patient_path, text)):
                reply = QMessageBox.question(
                    self, 
                    "Patient ID already exists",
                    "Are you sure to cover the ID?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.Yes:
                    shutil.rmtree(os.path.join(self.patient_path, text))
                    os.mkdir(os.path.join(self.patient_path, text))
                    self.label_log.setText(f"Patient ID {text} is selected")
                    self.record_list_widget_patient_id_list_show()
                    self.btn_Apose_record.setEnabled(True)
                    self.record_select_patientID = text
                    self.label_selected_patient.setText(f"Selected patient : {self.record_select_patientID}")
                elif reply == QMessageBox.StandardButton.No:
                    self.button_create_new_patient()
            else:
                self.record_select_patientID = text
                os.mkdir(os.path.join(self.patient_path, text))
                self.label_log.setText(f"Patient ID {text} is selected")
                self.record_list_widget_patient_id_list_show()
                self.btn_Apose_record.setEnabled(True)
                self.label_selected_patient.setText(f"Selected patient : {self.record_select_patientID}")
    def check_apose_finish(self):
        if not self.apose_rec_evt1.is_set() and not self.apose_rec_evt2.is_set() and not self.apose_rec_evt3.is_set() and not self.apose_rec_evt4.is_set():
            self.shared_dict_record_name.clear()
            self.record_opened = False
            self.list_widget_patient_id_record.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
            self.timer_apose.stop()
            self.btn_pg1_reset.setEnabled(True)
            self.label_log.setText("Apose finished!")
    def update_Apose_note(self):
        olddir_meetnote = os.path.join(self.config_path, 'meetnote_layout.json')
        newdir_meetnote = os.path.join(self.patient_path, self.record_select_patientID, datetime.now().strftime("%Y_%m_%d"), 'raw_data', 'Meet_note.json')
        shutil.copy2(olddir_meetnote, newdir_meetnote)
    
    def Apose_record(self):
        config_name = os.path.join(self.config_path, "config.json")
        cali_time_file_path = os.path.join(self.current_directory, "cali_time.txt") # self.time_file_path
        toml_file_path = self.calib_toml_path
        cali_file_path = self.calibra_path
        save_path_date = os.path.join(self.patient_path, self.record_select_patientID, datetime.now().strftime("%Y_%m_%d"))
        save_path_Apose = os.path.join(save_path_date, "raw_data", "Apose")
        time_file_path = os.path.join(save_path_Apose, "recordtime.txt")
        save_path_videos = os.path.join(save_path_Apose, "videos")
        os.makedirs(save_path_videos)
        if os.path.exists(cali_time_file_path):
            shutil.copy(cali_time_file_path, save_path_date)
            self.label_log.setText("已成功複製 cali_time.txt")
        else:
            self.label_log.setText("cali_time.txt 不存在")

        if os.path.exists(cali_file_path):
            if os.path.exists(os.path.join(save_path_date, "raw_data", "calibration")):
                shutil.rmtree(os.path.join(save_path_date, "raw_data", "calibration"))
            shutil.copytree(cali_file_path, os.path.join(save_path_date, "raw_data", "calibration"))
            self.label_log.setText("已成功複製 calibration資料夾")
        else:
            self.label_log.setText("calibration 資料夾 不存在")
        if os.path.exists(time_file_path):
            with open(time_file_path, "r") as file:
                formatted_datetime = file.read().strip()
        else:
            now = datetime.now()
            formatted_datetime = now.strftime("%Y_%m_%d_%H%M")
            with open(time_file_path, "w") as file:
                file.write(formatted_datetime)
        with open(config_name, 'r') as f:
            data = json.load(f)
        num_cameras = data['cam']['list']
        self.shared_dict_record_name['name'] = self.record_select_patientID
        self.btn_Apose_record.setEnabled(False)
        self.record_opened = True
        self.list_widget_patient_id_record.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        self.record_enter_task_name.setEnabled(False)
        self.apose_rec_evt1.set()
        self.apose_rec_evt2.set()
        self.apose_rec_evt3.set()
        self.apose_rec_evt4.set()
        self.timer_apose.start(1500)
    def Apose_record_ask(self):
        if (not self.camera_opened):
            QMessageBox.information(self, "Cameras are not opened", "Please open cameras first！")
            return
        elif self.record_opened:
            QMessageBox.information(self, "Record action already exists", "There is another record task！")
            return
        elif not self.record_select_patientID:
            QMessageBox.information(self, "Patient ID not selected", "Please select patient ID first！")
            return
            
        if os.path.exists(os.path.join(self.patient_path, self.record_select_patientID, datetime.now().strftime("%Y_%m_%d"), "raw_data", "Apose")):
            reply = QMessageBox.question(
                self, 
                "Apose already exists",
                "Are you sure to create new Apose?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.btn_pg1_reset.setEnabled(False)
                shutil.rmtree(os.path.join(self.patient_path, self.record_select_patientID, datetime.now().strftime("%Y_%m_%d"), "raw_data", "Apose"))
                self.Apose_record() 
                self.update_Apose_note()
                self.label_log.setText(f"Create new Apose for {self.record_select_patientID}")
        else:
            self.Apose_record()
            
    def startrecord_task(self):
        if (not self.camera_opened) or (self.record_opened) or (not self.record_select_patientID) or (not self.record_task_name): 
            return
        if not os.path.exists(os.path.join(self.patient_path, self.record_select_patientID, datetime.now().strftime("%Y_%m_%d"), "raw_data", self.record_task_name)):
            os.makedirs(os.path.join(self.patient_path, self.record_select_patientID, datetime.now().strftime("%Y_%m_%d"), "raw_data", self.record_task_name, 'videos'))
        self.btn_pg1_reset.setEnabled(False)
        self.shared_dict_record_name['name'] = self.record_select_patientID
        self.shared_dict_record_name['task_name'] = self.record_task_name
        self.shared_dict_record_name['start_time'] = time.time()
        self.task_stop_rec_evt.clear()
        self.btnStartRecording.setEnabled(False)
        self.task_rec_evt.set()
        self.record_opened = True
        self.record_enter_task_name.setEnabled(False)
        self.list_widget_patient_id_record.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        self.label_log.setText("Recording start")
    def stoprecord_task(self): 
        if not self.record_opened:
            return
        self.record_opened = False
        self.task_stop_rec_evt.set()
        self.task_rec_evt.clear()
        self.shared_dict_record_name.clear()
        self.record_enter_task_name.clear()
        self.record_enter_task_name.setEnabled(True)
        self.list_widget_patient_id_record.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.btn_pg1_reset.setEnabled(True)
        self.btnStopRecording.setEnabled(False)
        self.btnStartRecording.setEnabled(True)
        self.label_log.setText("Record end")
        self.lw_patient_task_record()
    def image_update_slot(self, image, label):
        label.setPixmap(QPixmap.fromImage(image))
    # open cameras
    def opencamera(self):
        self.camera_opened = True
        self.manager = Manager()
        self.task_rec_evt = Event()
        self.apose_rec_evt1 = Event()
        self.calib_rec_evt1 = Event()
        self.apose_rec_evt2 = Event()
        self.calib_rec_evt2 = Event()
        self.apose_rec_evt3 = Event()
        self.calib_rec_evt3 = Event()
        self.apose_rec_evt4 = Event()
        self.calib_rec_evt4 = Event()
        self.task_stop_rec_evt = Event()
        self.start_evt = Event()
        self.stop_evt = Event()
        self.camera_proc_lst = []
        self.tracker_proc_lst = []
        self.threads = []
        self.shared_dict_record_name = self.manager.dict()
        self.queue = [Queue() for _ in range(4)]
        self.queue_kp = [Queue() for _ in range(4)]
        self.shm_lst = []
        self.shm_kp_lst = []
        self.btnCloseCamera.setEnabled(True)
        self.btnOpenCamera.setEnabled(False)
        for i in range(4):
            shm = shared_memory.SharedMemory(create=True, size=int(np.prod(self.shape) * np.dtype(np.uint8).itemsize * self.buffer_length))
            self.shm_lst.append(shm)
            
            shm1 = shared_memory.SharedMemory(create=True, size=int(np.prod(self.shape) * np.dtype(np.uint8).itemsize * self.buffer_length))
            self.shm_kp_lst.append(shm1)
            p1 = TrackerProcess(
                self.start_evt,
                i,
                self.stop_evt,
                self.queue[i],
                self.queue_kp[i],
                self.shm_lst[i].name,
                self.shm_kp_lst[i].name
            )
            self.tracker_proc_lst.append(p1)
            thread = UpdateThread(
                i,
                self.start_evt,
                self.stop_evt,
                self.queue_kp[i],
                self.shm_kp_lst[i].name
            )
            self.threads.append(thread)
        p1 = CameraProcess(
            self.shm_lst[0].name, 
            0,
            self.start_evt,
            self.task_rec_evt,
            self.apose_rec_evt1,
            self.calib_rec_evt1,
            self.stop_evt,
            self.task_stop_rec_evt,
            self.calib_save_path,
            self.queue[0],
            self.patient_path,
            self.shared_dict_record_name
        )
        self.camera_proc_lst.append(p1)
        p2 = CameraProcess(
            self.shm_lst[1].name, 
            1,
            self.start_evt,
            self.task_rec_evt,
            self.apose_rec_evt2,
            self.calib_rec_evt2,
            self.stop_evt,
            self.task_stop_rec_evt,
            self.calib_save_path,
            self.queue[1],
            self.patient_path,
            self.shared_dict_record_name
        )
        self.camera_proc_lst.append(p2)
        p3 = CameraProcess(
            self.shm_lst[2].name, 
            2,
            self.start_evt,
            self.task_rec_evt,
            self.apose_rec_evt3,
            self.calib_rec_evt3,
            self.stop_evt,
            self.task_stop_rec_evt,
            self.calib_save_path,
            self.queue[2],
            self.patient_path,
            self.shared_dict_record_name
        )
        self.camera_proc_lst.append(p3)
        p4 = CameraProcess(
            self.shm_lst[3].name, 
            3,
            self.start_evt,
            self.task_rec_evt,
            self.apose_rec_evt4,
            self.calib_rec_evt4,
            self.stop_evt,
            self.task_stop_rec_evt,
            self.calib_save_path,
            self.queue[3],
            self.patient_path,
            self.shared_dict_record_name
        )
        self.camera_proc_lst.append(p4)
        self.update()
        self.label_cam = {0:self.Camera1, 1:self.Camera2, 2:self.Camera3, 3:self.Camera4}
        for i in range(4):
            label = self.label_cam[i]
            self.threads[i].scale_size = [label.size().width(), label.size().height()]
            self.threads[i].update_signal.connect(lambda image, label=label: self.image_update_slot(image, label))

        for process in self.camera_proc_lst:
            process.start()
        for process in self.tracker_proc_lst:
            process.start()
        for thread in self.threads:
            thread.start()
        self.btn_extrinsic_record.setEnabled(True)
        self.start_evt.set()
    def closeCamera(self, is_main=True):
        if not self.camera_opened: return
        if self.record_opened: return
        self.btnOpenCamera.setEnabled(True)
        self.btn_extrinsic_record.setEnabled(False)
        self.stop_evt.set()
        for shm in self.shm_lst:
            shm.close()
            shm.unlink()
        for shm1 in self.shm_kp_lst:
            shm1.close()
            shm1.unlink()
        for queue in self.queue:
            queue.close()
        for queue in self.queue_kp:
            queue.close()
        self.shm_lst.clear()
        self.shm_kp_lst.clear()
        self.shared_dict_record_name.clear()

        for thread in self.threads:
            thread.stop()
        for thread in self.threads:
            thread.wait()
        
        for process in self.camera_proc_lst:
            if process.is_alive():
                process.terminate()
                process.join()
        for process in self.tracker_proc_lst:
            if process.is_alive():
                process.terminate()
                process.join()
        self.camera_proc_lst.clear()
        self.tracker_proc_lst.clear()
        self.camera_opened = False
        
        self.start_evt.clear()
        self.stop_evt.clear()
        self.task_rec_evt.clear()
        self.task_stop_rec_evt.clear()

        if hasattr(self, 'manager'):
            self.manager.shutdown()
        for i in range(self.cam_num):
            label = self.label_cam[i]
            black_pixmap = QPixmap(label.width(), label.height())
            black_pixmap.fill(QColor(0, 0, 0))
            label.setPixmap(black_pixmap)
    def button_config(self, instance):
        # self.label_log.text = "檢測Webcam ID並更新config"
        self.label_log.setText("detect Webcam ID and update config")
        camera_config_create(self.config_path)
        detect_camera_list = camera_config_update(self.config_path, 10, new_gui=True)
        self.label_log.setText(f"detect Webcam ID : {str(detect_camera_list)} and update config")
    def getcamerainfo(self, config_path):
        camera_config_path = os.path.join(config_path, 'config.json')
        with open(camera_config_path, 'r') as camconf:
            camera_config = json.load(camconf)
        return camera_config['cam']['number'], camera_config['cam']['resolution']

    def button_extrinsic_record(self):
        reply = QMessageBox.question(
            self, 
            "Confirm Action",
            "Are you sure to create new calibration?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            create_calibration_folder(self.current_directory)
            self.btn_pg1_reset.setEnabled(False)
            self.label_log.setText("create new extrinsic params")
            config_name = os.path.join(self.config_path, "config.json")
            time_file_path = os.path.join(self.record_path, "calibration", "calib_time.txt")
            with open(config_name, 'r') as f:
                data = json.load(f)
            num_cameras = data['cam']['list']
            self.calib_rec_evt1.set()
            self.calib_rec_evt2.set()
            self.calib_rec_evt3.set()
            self.calib_rec_evt4.set()
            if os.path.exists(time_file_path)==1:
                os.remove(time_file_path)
            now = datetime.now()
            formatted_datetime = now.strftime("%Y_%m_%d_%H%M")
            with open(time_file_path, "w") as file:
                file.write(formatted_datetime)
            while True:
                if not self.calib_rec_evt1.set() and not self.calib_rec_evt2.set() and not self.calib_rec_evt3.set() and not self.calib_rec_evt4.set():
                    self.label_log.setText("Finish extrinsic record")
                    self.button_extrinsic_calculate()
                    self.label_log.setText("Finish extrinsic calculation")
                    self.btn_pg1_reset.setEnabled(True)
                    self.err_calib_extri.setText(read_err_calib_extri(self.current_directory))
                    break
        
    def button_extrinsic_calculate(self):
        self.label_log.setText("calculating extrinsic")
        try:
            err_list = calib_extri(self.current_directory,0)
            self.label_log.setText('calculate finished')            
            self.err_calib_extri.text = err_list            
        except:            
            self.label_log.setText('check intrinsic and extrinsic exist')
            self.err_calib_extri.text = 'no calibration file found'          
    def button_extrinsic_manual_calculate(self, instance):
        try:
            def remove_folder_with_contents(path):
                # Check if the directory exists
                if os.path.isdir(path):
                    # Recursively delete the directory and all its contents
                    shutil.rmtree(path)
            remove_folder_with_contents(os.path.join(self.extrinsic_path,'images'))
            remove_folder_with_contents(os.path.join(self.extrinsic_path,'yolo_backup'))
            remove_folder_with_contents(os.path.join(self.extrinsic_path,'chessboard'))
            
            err_list =calib_extri(self.current_directory,1)
            self.label_log.text = 'calculate finished'
            self.err_calib_extri.text = err_list     
            self.err_calib_extri.xetText(read_err_calib_extri(self.current_directory)) 
        except:
            # self.label_log.text = '檢查是否有拍攝以及計算內參，以及是否有拍攝外參'
            self.label_log.text = 'check intrinsic and extrinsic exist'
            self.err_calib_extri.text = 'no calibration file found'
    def check_cal_finish(self):
        if self.marker_calculate_process:
            if not self.marker_calculate_process.is_alive():
                self.timer_marker_calculate.stop()
                self.label_calculation_status.setText("caculation is finished")
                self.btn_cal_start_cal.setEnabled(True)
    def btn_pre_marker_calculate(self):
        if self.cal_select_list == []:
            return
        else:
            cur_dir = copy.deepcopy(self.current_directory)
            cal_list = copy.deepcopy(self.cal_select_list)
            # self.closeCamera()
            self.marker_calculate_process = Process(target=mp_marker_calculate, args=(cur_dir, cal_list))
            self.marker_calculate_process.start()
            self.cal_select_list = []
            self.label_calculation_status.setText("start calculating")
            self.btn_cal_start_cal.setEnabled(False)
            self.cal_show_selected_folder()
            self.timer_marker_calculate.start(1000)
    # Calculation tab 
    def cal_select_back_path(self):
        if self.cal_select_depth == 1:
            self.cal_select_depth -= 1
            self.cal_select_patient_id = None
            self.btn_cal_back_path.setEnabled(False)
        
        self.cal_load_folders()
    def cal_load_folders(self):
        if self.cal_select_depth == 0:
            self.listwidget_select_cal_date.clear()
            items = [item for item in os.listdir(self.patient_path)]
            for item in items:
                self.listwidget_select_cal_date.addItem(item)
        elif self.cal_select_depth == 1:            
            self.listwidget_select_cal_date.clear()
            items = [item for item in os.listdir(os.path.join(self.patient_path, self.cal_select_patient_id))]
            for item in items:
                self.listwidget_select_cal_date.addItem(item)
    def cal_show_selected_folder(self):
        self.listwidget_selected_cal_date.clear()        
        for item in self.cal_select_list:
            self.listwidget_selected_cal_date.addItem(item)
    def cal_select_folder_clicked(self, item):
        if self.cal_select_depth == 0:
            self.cal_select_patient_id = item.text()
            self.btn_cal_back_path.setEnabled(True)
            self.cal_select_depth += 1
            self.cal_load_folders()
        elif self.cal_select_depth == 1:
            self.cal_select_date = item.text()
            if os.path.join(self.patient_path, self.cal_select_patient_id, self.cal_select_date) not in self.cal_select_list:
                self.cal_select_list.append(os.path.join(self.patient_path, self.cal_select_patient_id, self.cal_select_date))
            self.cal_show_selected_folder()
            self.cal_select_date = None
            self.cal_select_depth = 0
            self.btn_cal_back_path.setEnabled(False)
            self.cal_load_folders()
            return
            
    def cal_selected_folder_selcted(self, item):
        self.listwidget_selected_cal_date_item_selected = item.text()
    def cal_selected_folder_selcted_delete(self):
        self.cal_select_list.remove(self.listwidget_selected_cal_date_item_selected)
        self.cal_show_selected_folder()
    # Show result tab 
    def select_back_path(self):
        if self.result_select_depth == 1:
            self.result_select_depth -= 1
            self.result_select_patient_id = None
            self.btn_result_back_path.setEnabled(False)
        elif self.result_select_depth == 2:
            self.result_select_depth -= 1
            self.result_select_date = None
        elif self.result_select_depth == 3:
            self.result_select_depth -= 1
            self.result_select_cal_time = None
        elif self.result_select_depth == 4:
            self.result_select_depth -= 1
            self.result_select_task = None
        self.result_load_folders()
    def folder_selected(self, item):
        pass
    def result_load_folders(self):        
        if self.result_select_depth == 0:   
            self.result_cal_task.clear()   
            items = [item for item in os.listdir(self.patient_path)]
            for item in items:
                self.result_cal_task.addItem(item)
            self.label_result_patient_id.setText(" ")
            self.label_result_record_date.setText(" ")
            self.label_result_calculation_time.setText(" ")
            self.label_result_task.setText(" ")
            
        elif self.result_select_depth == 1:
            self.result_cal_task.clear()
            items = [item for item in os.listdir(os.path.join(self.patient_path, self.result_select_patient_id))]
            for item in items:
                self.result_cal_task.addItem(item)
            self.label_result_record_date.setText(" ")
            self.label_result_patient_id.setText(self.result_select_patient_id)
        elif self.result_select_depth == 2:
            self.result_cal_task.clear()
            items = [item for item in os.listdir(os.path.join(self.patient_path, self.result_select_patient_id, self.result_select_date))]
            for item in items:
                self.result_cal_task.addItem(item)
            self.label_result_calculation_time.setText(" ")
            self.label_result_record_date.setText(self.result_select_date)
        elif self.result_select_depth == 3:
            self.result_cal_task.clear()
            items = [item for item in os.listdir(os.path.join(self.patient_path, self.result_select_patient_id, self.result_select_date, self.result_select_cal_time))]
            for item in items:
                self.result_cal_task.addItem(item)
            self.label_result_task.setText(" ")
            self.label_result_calculation_time.setText(self.result_select_cal_time)
        elif self.result_select_depth == 4:
            self.label_result_task.setText(self.result_select_task)
            self.select_result_cal_task(self.patient_path, self.result_select_patient_id, self.result_select_date, self.result_select_cal_time, self.result_select_task)

    def folder_clicked(self, item):
        if self.result_select_depth == 0:
            self.result_select_patient_id = item.text()
            self.btn_result_back_path.setEnabled(True)
        elif self.result_select_depth == 1:
            self.result_select_date = item.text()
        elif self.result_select_depth == 2:
            self.result_select_cal_time = item.text()
        elif self.result_select_depth == 3:
            self.result_select_task = item.text()
        elif self.result_select_depth == 4:
            return
        self.result_select_depth += 1
        self.result_load_folders()

    def update_slider(self):
        self.result_video_slider.setValue(int(self.video_player.progress * 100))
        self.frame_label.setText(str(int(self.video_player.progress * 100)))
    def select_result_cal_task(self, patient_path, result_select_patient_id, result_select_date, result_select_cal_time, result_select_task):
        self.show_result_path = os.path.join(patient_path, result_select_patient_id, result_select_date, result_select_cal_time, result_select_task)
        self.video_player.load_video(self.show_result_path)
        self.video_player.result_load_gait_figures(self.show_result_path)
        self.video_player.load_gltf_file_in_viewer(f"./Patient_data/{result_select_patient_id}/{result_select_date}/{result_select_cal_time}/{result_select_task}/model.gltf")
        self.playButton.setEnabled(True)
        self.result_video_slider.setEnabled(True)
    def result_disconnet_gait_figures(self):
        if self.result_current_gait_figures_index == 1:#knee flexion
            self.result_video_slider.valueChanged.disconnect(self.video_player.update_hip_flexion)
        elif self.result_current_gait_figures_index ==2:
            self.result_video_slider.valueChanged.disconnect(self.video_player.update_knee_flexion)          
        elif self.result_current_gait_figures_index ==3:
            self.result_video_slider.valueChanged.disconnect(self.video_player.update_ankle_flexion)
        elif self.result_current_gait_figures_index ==4:
            self.result_video_slider.valueChanged.disconnect(self.video_player.update_speed)
        elif self.result_current_gait_figures_index ==5:
            self.result_video_slider.valueChanged.disconnect(self.video_player.update_stride)
    def result_select_gait_figures(self, index):

        # self.result_disconnet_gait_figures()
        if index == 1:#knee flexion
            self.video_player.hip_flexion_plot()# Clear and replot everything
            self.result_video_slider.valueChanged.connect(self.video_player.update_hip_flexion)
            self.result_current_gait_figures_index = 1
        elif index ==2:
            self.video_player.knee_flexion_plot()       
            self.result_video_slider.valueChanged.connect(self.video_player.update_knee_flexion)
            self.result_current_gait_figures_index = 2
        elif index ==3:
            self.video_player.ankle_flexion_plot()
            self.result_video_slider.valueChanged.connect(self.video_player.update_ankle_flexion)
            self.result_current_gait_figures_index = 3
        elif index ==4:
            self.video_player.speed_plot()
            self.result_video_slider.valueChanged.connect(self.video_player.update_speed)
            self.result_current_gait_figures_index = 4
        elif index ==5:
            self.video_player.stride_plot()
            self.result_video_slider.valueChanged.connect(self.video_player.update_stride)
            self.result_current_gait_figures_index = 5
    def play_stop(self):
        if self.is_playing:
            self.result_web_view_widget.page().runJavaScript("stopAnimation();")
            self.video_player.timer.stop()
            self.is_playing = False
        else:
            self.result_web_view_widget.page().runJavaScript("startAnimation();")
            self.video_player.timer.start(15)
            self.is_playing = True
    
    def slider_changed(self, value):
        if self.is_playing:
            self.play_stop()
        self.video_player.progress = value/100
        self.video_player.slider_changed()
        self.result_video_slider.setValue(value)
        self.frame_label.setText(str(value))
        
    def closeEvent(self, event):
        self.closeCamera()
        if self.marker_calculate_process and self.marker_calculate_process.is_alive():
            self.marker_calculate_process.terminate()
            self.marker_calculate_process.join()
        
if __name__ == "__main__":
    App = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(App.exec())