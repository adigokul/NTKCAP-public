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
        state = tracker.create_state(det_interval=1, det_min_bbox_size=100, keypoint_sigmas=sigmas, pose_max_num_bboxes=3)
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
            
            if self.task_stop_rec_evt.is_set():
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
        self.data = {}
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
    def __init__(self, labels: list[QLabel], web_viewer: QWebEngineView, parent=None):
        super().__init__(parent)
        self.video_path = None
        self.label1, self.label2, self.label3, self.label4 = labels[0], labels[1], labels[2], labels[3]
        self.frame_count = None
        self.timer = QTimer()
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer.timeout.connect(self.run)
        self.result_load_videos_allframes = []
        self.web_view_widget = web_viewer
        self.progress = None
        self.start_server()
    def load_video(self, cal_task_path: str):
        self.result_load_videos_allframes = []
        self.progress = None
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
    def set_slider_value(self, progress):
        self.progress = progress
        if self.progress is not None:
            if int(progress * 100) >= 100:
                self.progress = 0
                self.frame_changed.emit(0)
                self.web_view_widget.page().runJavaScript("resetAnimation();")
                
            else:
                self.frame_changed.emit(int(self.progress * 100))
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
        uic.loadUi('NTKCAP_GUI.ui', self)
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

        # record Apose
        self.btn_Apose_record: QPushButton = self.findChild(QPushButton, "btn_Apose_record")
        self.btn_Apose_record.clicked.connect(self.Apose_record_ask)
        self.btn_Apose_record.setEnabled(False)
        self.timer_apose = QTimer()
        self.timer_apose.timeout.connect(self.check_apose_finish)
        # Show result
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
        self.video_player = VideoPlayer(self.select_result_cal_rpjvideo_labels, self.result_web_view_widget)
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
        self.result_load_folders()
        self.result_gait_figures = []
        self.result_gait_figures_names = []
        self.result_gait_figures_idx = []
        self.btn_right_gait_figure: QPushButton = self.findChild(QPushButton, "btn_right_gait_figure")
        self.btn_right_gait_figure.clicked.connect(self.result_next_gait_figure)
        self.btn_right_gait_figure.setEnabled(False)
        self.btn_left_gait_figure: QPushButton = self.findChild(QPushButton, "btn_left_gait_figure")
        self.btn_left_gait_figure.clicked.connect(self.result_former_gait_figure)
        self.btn_left_gait_figure.setEnabled(False)
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
            else:
                self.record_task_name = self.record_enter_task_name.text()
                self.btnStartRecording.setEnabled(True)
                self.btnStopRecording.setEnabled(True)
                self.btn_Apose_record.setEnabled(False)

    def record_enter_task_name_infocus(self, event):
        self.record_enter_task_name_kb_listen = True

    def record_enter_task_name_outfocus(self, event):
        self.record_enter_task_name_kb_listen = False

    def lw_record_select_patientID(self, item):
        self.record_select_patientID = item.text()
        self.record_enter_task_name.setEnabled(True)
        self.btn_Apose_record.setEnabled(True)
        self.record_task_name = None
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
                elif reply == QMessageBox.StandardButton.No:
                    self.button_create_new_patient()
            else:
                self.record_select_patientID = text
                os.mkdir(os.path.join(self.patient_path, text))
                self.label_log.setText(f"Patient ID {text} is selected")
                self.record_list_widget_patient_id_list_show()
                self.btn_Apose_record.setEnabled(True)
                
    def check_apose_finish(self):
        if not self.apose_rec_evt.is_set():
            self.shared_dict_record_name.clear()
            self.record_opened = False
            self.list_widget_patient_id_record.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
            self.timer_apose.stop()
            self.btn_pg1_reset.setEnabled(True)
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
            
            shutil.copytree(cali_file_path, os.path.join(save_path_date,"calibration"))
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
        self.apose_rec_evt.set()
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
                self.Apose_record(self.config_path, self.current_directory, self.record_select_patientID, datetime.now().strftime("%Y_%m_%d")) 
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
        self.record_select_patientID = None
        self.record_enter_task_name.clear()
        self.record_enter_task_name.setEnabled(True)
        self.list_widget_patient_id_record.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.btn_pg1_reset.setEnabled(True)
        self.btnStopRecording.setEnabled(False)
        self.btnStartRecording.setEnabled(True)
        self.label_log.setText("Recording end")
    def image_update_slot(self, image, label):
        label.setPixmap(QPixmap.fromImage(image))
    # open cameras
    def opencamera(self):
        self.camera_opened = True
        self.manager = Manager()
        self.task_rec_evt = Event()
        self.apose_rec_evt = Event()
        self.calib_rec_evt = Event()
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
            p = CameraProcess(
                self.shm_lst[i].name, 
                i,
                self.start_evt,
                self.task_rec_evt,
                self.apose_rec_evt,
                self.calib_rec_evt,
                self.stop_evt,
                self.task_stop_rec_evt,
                self.calib_save_path,
                self.queue[i],
                self.patient_path,
                self.shared_dict_record_name
            )
            self.camera_proc_lst.append(p)
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
            self.calib_rec_evt.set()
            if os.path.exists(time_file_path)==1:
                os.remove(time_file_path)
            now = datetime.now()
            formatted_datetime = now.strftime("%Y_%m_%d_%H%M")
            with open(time_file_path, "w") as file:
                file.write(formatted_datetime)
            while True:
                if not self.calib_rec_evt.set():
                    self.label_log.setText("Finish extrinsic record")
                    self.button_extrinsic_calculate()
                    self.label_log.setText("Finish extrinsic calculation")
                    self.btn_pg1_reset.setEnabled(True)
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
        print(1)
    def result_load_folders(self):        
        if self.result_select_depth == 0:   
            self.result_cal_task.clear()   
            items = [item for item in os.listdir(self.patient_path)]
            for item in items:
                self.result_cal_task.addItem(item)
            self.text_label_path_depth.setText(" ")   
        elif self.result_select_depth == 1:
            self.result_cal_task.clear()
            items = [item for item in os.listdir(os.path.join(self.patient_path, self.result_select_patient_id))]
            for item in items:
                self.result_cal_task.addItem(item)
            self.text_label_path_depth.setText(f"{self.result_select_patient_id}")
        elif self.result_select_depth == 2:
            self.result_cal_task.clear()
            items = [item for item in os.listdir(os.path.join(self.patient_path, self.result_select_patient_id, self.result_select_date))]
            for item in items:
                self.result_cal_task.addItem(item)
            self.text_label_path_depth.setText(f"{self.result_select_patient_id}/{self.result_select_date}")
        elif self.result_select_depth == 3:
            self.result_cal_task.clear()
            items = [item for item in os.listdir(os.path.join(self.patient_path, self.result_select_patient_id, self.result_select_date, self.result_select_cal_time))]
            for item in items:
                self.result_cal_task.addItem(item)
            self.text_label_path_depth.setText(f"{self.result_select_patient_id}/{self.result_select_date}/{self.result_select_cal_time}")
        elif self.result_select_depth == 4:
            self.text_label_path_depth.setText(f"{self.result_select_patient_id}/{self.result_select_date}/{self.result_select_cal_time}/{self.result_select_task}")
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
        self.result_load_gait_figures()
        
        self.video_player.load_gltf_file_in_viewer(f"./Patient_data/{result_select_patient_id}/{result_select_date}/{result_select_cal_time}/{result_select_task}/model.gltf")
        self.playButton.setEnabled(True)
        self.result_video_slider.setEnabled(True)
        
    def result_load_gait_figures(self):
        self.label_gait_figure.clear()
        self.result_gait_figures_idx = 0
        self.result_gait_figures = []
        self.result_gait_figures_names = []
        self.btn_left_gait_figure.setEnabled(True)
        self.btn_right_gait_figure.setEnabled(True)
        result_post_analysis_imgs_folder_path = os.path.join(self.show_result_path, 'post_analysis')
        for png_file in os.listdir(result_post_analysis_imgs_folder_path):
            if png_file.endswith('.png'):
                self.result_gait_figures_names.append(png_file.split(self.result_select_patient_id)[0].split('for')[0])
                img = cv2.imread(os.path.join(result_post_analysis_imgs_folder_path, png_file))
                resized_img = cv2.resize(img, (self.label_gait_figure.size().width(), self.label_gait_figure.size().height()), interpolation=cv2.INTER_LINEAR)
                img_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
                height, width, channel = img_rgb.shape
                bytesPerline = channel * width
                ConvertToQtFormat = QImage(img_rgb, width, height, bytesPerline, QImage.Format.Format_RGB888)
                pixmap = QPixmap.fromImage(ConvertToQtFormat)
                self.result_gait_figures.append(pixmap)
        self.label_gait_figure.setPixmap(self.result_gait_figures[0])
        self.label_gait_figure_name.setText(self.result_gait_figures_names[0])
    def result_update_gait_figure(self):
        self.label_gait_figure.clear()
        self.label_gait_figure.setPixmap(self.result_gait_figures[self.result_gait_figures_idx])
        self.label_gait_figure_name.clear()
        self.label_gait_figure_name.setText(self.result_gait_figures_names[self.result_gait_figures_idx])

    def result_former_gait_figure(self):
        if self.result_gait_figures_idx > 0:
            self.result_gait_figures_idx -= 1
        else:
            self.result_gait_figures_idx = 9
        self.result_update_gait_figure()
    def result_next_gait_figure(self):
        if self.result_gait_figures_idx < 9:
            self.result_gait_figures_idx += 1
        else:
            self.result_gait_figures_idx = 0
        self.result_update_gait_figure()
    
    def play_stop(self):
        if self.is_playing:
            self.result_web_view_widget.page().runJavaScript("stopAnimation();")
            self.video_player.timer.stop()
            self.is_playing = False
        else:
            self.result_web_view_widget.page().runJavaScript("startAnimation();")
            self.video_player.timer.start(33)
            self.is_playing = True
    
    def slider_changed(self, value):
        if self.is_playing:
            self.play_stop()
        print(value)
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