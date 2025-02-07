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
# !!!! result load switch & Apose retake & HTML screen
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