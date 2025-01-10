from PyQt6 import uic
from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *
from multiprocessing import Event, shared_memory, Manager, Queue, Array, Lock
import numpy as np
from scipy.signal import butter, filtfilt
import sys, cv2, time
import os
import json
import warnings
from thread_process import CameraThread, UpdateThread, TimeThread
from camera_process import CameraProcess
from typing import Any
from datetime import datetime
# from mmdeploy_runtime import PoseTracker
# from numba import jit
# from numba.np.extensions import cross2d
# from numpy.lib.stride_tricks import sliding_window_view

# set the different parts of model
VISUALIZATION_CFG = dict(
    halpe26=dict(
        skeleton=[(15,13), (13,11), (11,19),(16,14), (14,12), (12,19),
                  (17,18), (18,19), (18,5), (5,7), (7,9), (18,6), (6,8),
                  (8,10), (1,2), (0,1), (0,2), (1,3), (2,4), (3,5), (4,6),
                  (15,20), (15,22), (15,24),(16,21),(16,23), (16,25)],
        palette=[(51,153,255), (0,255,0), (255,128,0)],
        link_color=[
            1, 1, 1, 2, 2, 2, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2
        ],
        point_color=[
            0, 0, 0, 0, 0, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 2, 2, 2, 1, 2, 1, 2, 1, 2
        ],
        sigmas=[
            0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072,
            0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089, 0.026,
            0.026, 0.066, 0.079, 0.079, 0.079, 0.079, 0.079, 0.079
        ]))

# set the angle combination for different keypoints of human body
BODY_CFG = dict(
    trunk=(18,19),
    pelvis=(11,19,12),
    hip=(19,11,13,12,14),
    knee=(11,12,13,14,15,16),
    left_hip=(19,11,13),
    right_hip=(19,12,14),
    left_knee=(11,13,15),
    right_knee=(12,14,16),
    left_thigh=(11,13),
    right_thigh=(12,14),
    heel=(24,25),
    left_foot=(13,15,20,22,24),
    right_foot=(14,16,21,23,25),
    empty=()
)

# the model path for person recognition and pose inference, using device cuda
# det_model_path = os.path.join(os.getcwd(),"NTK_FB", "ThirdParty", "mmdeploy", "rtmpose-trt", "rtmdet-m")
det_model_path = os.path.join(os.getcwd(),"NTK_FB", "ThirdParty", "mmdeploy", "rtmpose-trt_fp16", "rtmdet-m")
pose_model_path = os.path.join(os.getcwd(),"NTK_FB", "ThirdParty", "mmdeploy", "rtmpose-trt", "rtmpose-m")
device ="cuda"

sigmas = VISUALIZATION_CFG['halpe26']['sigmas']
palette = VISUALIZATION_CFG['halpe26']['palette']
skeleton = VISUALIZATION_CFG['halpe26']['skeleton']
link_color = VISUALIZATION_CFG['halpe26']['link_color']
point_color = VISUALIZATION_CFG['halpe26']['point_color']

# class NumpyEncoder(json.JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, np.ndarray):
#             return obj.tolist()
#         if isinstance(obj, np.integer):
#             return int(obj)
#         if isinstance(obj, np.floating):
#             return float(obj)
#         return super(NumpyEncoder, self).default(obj)

class NumpyEncoder(json.JSONEncoder):

    def __init__(self, *args, **kwargs):
        self._indent = kwargs.get('indent', 2)
        super().__init__(*args, **kwargs)
    
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)
    
    def encode(self, obj):
        json_str = super().encode(obj)
        
        if not self._indent:
            return json_str
        
        parsed = json.loads(json_str)
        return self._format_object(parsed)
    
    def _is_keypoints_data(self, obj):
        """檢查是否為keypoints格式的數據"""
        if not isinstance(obj, list) or len(obj) != 2:
            return False
        return all(isinstance(n, (int, float)) for n in obj)
    
    def _is_keypoints_list(self, obj):
        """檢查是否為多組keypoints的列表"""
        if not isinstance(obj, list):
            return False
        return all(self._is_keypoints_data(item) for item in obj)
    
    def _format_object(self, obj, level=0):
        indent = ' ' * (self._indent * level)
        next_indent = ' ' * (self._indent * (level + 1))
        
        if isinstance(obj, dict):
            if not obj:
                return "{}"
            
            items = []
            for k, v in obj.items():
                formatted_value = self._format_object(v, level + 1)
                items.append(f'{next_indent}"{k}": {formatted_value}')
            return "{\n" + ",\n".join(items) + f"\n{indent}}}"
        
        elif isinstance(obj, list):
            if not obj:
                return "[]"
            
            # 檢查是否為單個座標對
            if self._is_keypoints_data(obj):
                return f"[{obj[0]}, {obj[1]}]"
            
            # 檢查內部是否包含多組座標列表
            if all(isinstance(x, list) and self._is_keypoints_list(x) for x in obj):
                # 處理巢狀的 keypoints 列表
                nested_lists = []
                for sublist in obj:
                    points = [f"[{x[0]}, {x[1]}]" for x in sublist]
                    nested_lists.append(next_indent + "[ " + ", ".join(points) + " ]")
                return "[\n" + ",\n".join(nested_lists) + f"\n{indent}]"
            
            # 檢查是否為直接的 keypoints 列表
            if self._is_keypoints_list(obj):
                points = [f"[{x[0]}, {x[1]}]" for x in obj]
                return "[ " + ", ".join(points) + " ]"
            
            # 處理一般列表
            is_simple = all(isinstance(x, (int, float, str, bool)) for x in obj)
            if is_simple and len(obj) > 0:
                return "[" + ", ".join(json.dumps(x) for x in obj) + "]"
            
            items = [next_indent + self._format_object(x, level + 1) for x in obj]
            return "[\n" + ",\n".join(items) + f"\n{indent}]"
        
        return json.dumps(obj)

class MainWindow(QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        uic.loadUi('ui_test.ui', self)
        self.setWindowTitle('NTK Feedback System')
        self.win = self.graphLayoutWidget
        self.win2 = self.graphLayoutWidget_2
        self.timer = QTimer(timerType=Qt.TimerType.PreciseTimer)
        np.set_printoptions(precision=4, suppress=True)

        self.QTableWidgets = [
            self.tableWidget,
            self.tableWidget_2,
            self.tableWidget_3,
            self.tableWidget_average,
        ]

        for tablewidget in self.QTableWidgets:
            tablewidget.setSpan(0, 0, 1, 3)
            tablewidget.setSpan(9, 0, 1, 3)
            tablewidget.setSpan(18, 0, 1, 3)
            tablewidget.setColumnWidth(0, 250)
            for i in range(19, 24):
                tablewidget.setSpan(i, 1, 1, 2)

        
        self.tableWidget_4.setSpan(0, 0, 1, 4)
        self.tableWidget_4.setSpan(9, 0, 1, 4)
        self.tableWidget_4.setSpan(18, 0, 1, 4)
        self.tableWidget_4.setColumnWidth(0, 25)
        self.tableWidget_4.setColumnWidth(1, 230)
        for i in range(19, 24):
            self.tableWidget_4.setSpan(i, 2, 1, 2)


        self.tableWidget_4.itemChanged.connect(lambda item: self.check_table(item))
        self.checkbox = [
            self.checkBox_16,
            self.checkBox_17,
            self.checkBox_18
        ]

        self.combobox_camera = [
            self.comboBox_frontal,
            self.comboBox_left,
            self.comboBox_right
        ]

        width = self.win.size().width()
        height = self.win.size().height()
        self.avg_height = int(height/4)

        ################################################################
        ### Push button
        # camera calibration
        self.push_check_cam.clicked.connect(self.check_camera_task)                             
        self.push_open_cam.clicked.connect(self.open_camera_task)
        self.push_close_cam.clicked.connect(self.close_camera_task)
        self.push_refresh.clicked.connect(self.refresh_cam_task)
        # control panel
        self.push_patient.clicked.connect(self.open_patient_window)
        self.push_start.clicked.connect(self.main_start_task)
        self.push_stop.clicked.connect(self.main_stop_task)
        self.push_record.clicked.connect(self.main_record_task)
        self.push_stop_record.clicked.connect(self.main_stop_record_task)
        self.push_select_color.clicked.connect(self.select_skl_color)
        # patient information
        self.push_create.clicked.connect(self.create_data_task)
        self.push_select.clicked.connect(self.select_data_task)
        self.push_confirm.clicked.connect(self.confirm_task)
        self.push_switch_patient.clicked.connect(self.switch_patient)

        # calibration
        self.push_calibration_create.clicked.connect(self.create_calibration_task)
        self.push_calibration_select.clicked.connect(self.select_calibration_date_task)
        self.push_calibration_select_page.clicked.connect(lambda: self.stackedWidget_calib.setCurrentIndex(0))
        self.push_calib_start.clicked.connect(self.sub_start_task)
        self.push_calib_rec.clicked.connect(self.sub_record_task)
        self.push_calib_stop.clicked.connect(self.sub_stop_task)
        self.push_calib_stop_rec.clicked.connect(self.sub_stop_record_task)
        self.push_calibration_calculate.clicked.connect(lambda: self.calc_calibration_task(self.calibration_name))
        self.push_calculate_average.clicked.connect(self.calc_average)

        ### check box
        for i, checkBox in enumerate(self.frontal):
            checkBox.stateChanged.connect(lambda state, idx=i: self.checkFrontalPlot(state, idx))
        for i, checkBox in enumerate(self.sagittal):
            checkBox.stateChanged.connect(lambda state, idx=i: self.checkSagittalPlot(state, idx))

        self.groupBox_18.clicked.connect(lambda : self.check_box(self.patient.groupBox))
        self.groupBox_19.clicked.connect(lambda : self.check_box(self.patient.groupBox_2))
        self.groupBox_20.clicked.connect(lambda : self.check_box(self.patient.groupBox_3))

        self.checkBox_frontal.clicked.connect(self.update_camera_layout)
        self.checkBox_left.clicked.connect(self.update_camera_layout)
        self.checkBox_right.clicked.connect(self.update_camera_layout)

        ### signal & slot
        ### timer
        self.timer.setInterval(33) # milliseconds, 1 sec = 1000 millisecond
        self.proc_lst = []
        self.sub_proc_lst = []
        self.main_thread_1 = []
        self.main_thread_2 = []
        self.main_thread_3 = []
        self.main_thread_4 = []
        self.sub_thread_1 = []
        self.sub_thread_2 = []
        self.sub_thread_3 = []
        self.sub_thread_4 = []
        self.shm_lst = []
        self.sub_shm_lst = []
        self.initialization_shm_lst = []
        self.shape = (1080, 1920, 3)
        self.dtype = np.uint8
        self.buffer_length = 4
        patient_width = self.patient.size().width()      
        self.patient_size_dic = {1:patient_width, 2:patient_width/2 , 3:patient_width/3}

        self.save_path = os.path.join(os.getcwd(),"Patient_data")
        self.current_patient = "unknown"
        self.calibration_date = "unknown"
        self.trial_name = "unknown"
        # self.check_cam = False
        self.camera_opened = False
        self.record_opened = False
        self.cam_number = [0,1,2]
        self.create_patient_folders()
        self.initialization_task()
        self.lock = Lock()
        ######################################################
    def switch_patient(self):
        self.stackedWidget_patient.setCurrentIndex(0)
        self.stackedWidget_calib.setCurrentIndex(0)
        self.current_patient = "unknown"
        self.calibration_date = "unknown"
        self.trial_name = "unknown"

    def check_box(self, groupbox):
        plane_group = self.sender()
        if plane_group.isChecked():
            groupbox.show()
        else:
            groupbox.hide()
        self.adjust_groupbox_size()

    # display how long the calibration task spends
    def update_time_task(self, time, label):
        label.setText(f"{time}s")

    # change the page as different push button is clicked
    def calibration_change_task(self, index):
        self.stackedWidget.setCurrentIndex(index)
        self.calibration_name = self.comboBox_calibration.currentText()

    def select_calibration_date_task(self):
        calibration_path = os.path.join(self.save_path, self.current_patient, "calibration")
        dialog = QDialog()
        dialog.setWindowTitle("Select the date of calibration file")
        list_view = QListView()
        model = QFileSystemModel()
        model.setRootPath(calibration_path)
        list_view.setModel(model)
        list_view.setRootIndex(model.index(calibration_path))
        confirm_button = QPushButton("Confirm")
        def on_confirm():
            index = list_view.currentIndex()
            if index.isValid():
                selected_file = model.filePath(index)
                self.calibration_date = os.path.basename(selected_file)
                self.label_calibration_date_val.setText(self.calibration_date)
                QMessageBox.information(self, "Selected File", f"The selected file is:\n{self.calibration_date}")
                self.stackedWidget_calib.setCurrentIndex(1)
            else:
                QMessageBox.warning(self, "No Selection", "No file selected.")
            dialog.accept()
        
        confirm_button.clicked.connect(on_confirm)
        layout = QVBoxLayout()
        layout.addWidget(list_view)
        layout.addWidget(confirm_button)  
        dialog.setLayout(layout)
        dialog.exec()
        
    def select_data_task(self):
        dialog = QDialog()
        dialog.setWindowTitle("Select the patient")
        list_view = QListView()
        model = QFileSystemModel()
        model.setRootPath(self.save_path)
        list_view.setModel(model)
        list_view.setRootIndex(model.index(self.save_path))
        confirm_button = QPushButton("Confirm")
        def on_confirm():
            index = list_view.currentIndex()
            if index.isValid():
                selected_file = model.filePath(index)
                self.current_patient = os.path.basename(selected_file)
                self.patient_name.setText(self.current_patient)
                QMessageBox.information(self, "Selected File", f"The selected file is:\n{selected_file}")
                self.stackedWidget_patient.setCurrentIndex(1)
                self.stackedWidget_calib.setCurrentIndex(0)
            else:
                QMessageBox.warning(self, "No Selection", "No file selected.")
            dialog.accept()
        
        confirm_button.clicked.connect(on_confirm)
        layout = QVBoxLayout()
        layout.addWidget(list_view)
        layout.addWidget(confirm_button)
        dialog.setLayout(layout)
        dialog.exec()
    
    def create_calibration_task(self):
        now = datetime.now()
        formatted_datetime = now.strftime("%Y_%m_%d_%H_%M")
        calibration_path = os.path.join(self.save_path, self.current_patient, "calibration", f"{formatted_datetime}")

        if os.path.exists(calibration_path):
            QMessageBox.warning(self, "Warning", "Calibration data already exists", QMessageBox.StandardButton.Ok)
            return

        os.makedirs(calibration_path, exist_ok=True)
        self.calibration_date = f"{formatted_datetime}"
        for i in range(1,4):
            os.makedirs(os.path.join(calibration_path, f"Calibration{i}"), exist_ok=True)
        self.label_calibration_date_val.setText(self.calibration_date)
        self.stackedWidget_calib.setCurrentIndex(1)
        
        QMessageBox.information(self, "Information",f"File {calibration_path} is created successfully",QMessageBox.StandardButton.Ok)

    def create_data_task(self):
        if not self.lineEdit_name.text() or not self.lineEdit_phone.text():
            QMessageBox.warning(self, "Warning", "Please enter whole patient data", QMessageBox.StandardButton.Ok)
            return
        
        self.current_patient = self.lineEdit_name.text()
        patient_path = os.path.join(self.save_path, self.current_patient)

        if os.path.exists(patient_path):
            QMessageBox.warning(self, "Warning", "Patient data already exists", QMessageBox.StandardButton.Ok)
            return

        os.makedirs(patient_path, exist_ok=True)
        with open(os.path.join(patient_path, "info.txt"), 'w') as file:
            file.write(f"{self.comboBox_gender.currentText()}\n{self.lineEdit_phone.text()}")
        self.patient_name.setText(self.current_patient)
        self.create_patient_folders()
        QMessageBox.information(self, "Information","Patient data is created successfully",QMessageBox.StandardButton.Ok)
        self.stackedWidget_patient.setCurrentIndex(1)

    def create_patient_folders(self):
        patient_path = os.path.join(self.save_path, self.current_patient)
        calibration_path = os.path.join(patient_path, "calibration")
        os.makedirs(os.path.join(patient_path, "trial"), exist_ok=True)
    
    # def confirm_task(self):
    #     if not self.lineEdit_trial.text():
    #         QMessageBox.warning(self, "Warning", "Please enter trial name", QMessageBox.StandardButton.Ok)
    #         return

    #     self.trial_name = self.lineEdit_trial.text()
    #     trial_path = os.path.join(self.save_path, self.current_patient, "trial", self.trial_name)
        
    #     if not os.path.exists(trial_path):
    #         os.makedirs(trial_path, exist_ok=True)
    #         QMessageBox.information(self, "File information", "The trial folder is created successfully", QMessageBox.StandardButton.Ok)
    #         return

    #     reply = QMessageBox.question(self, "Question", "Trial name already exists. Would you like to continue?", QMessageBox.StandardButton.No | QMessageBox.StandardButton.Yes)
    
    #     if reply == QMessageBox.StandardButton.Yes:
    #         QMessageBox.information(self, "File information", f"Using existing trial folder: {self.trial_name}", QMessageBox.StandardButton.Ok)    
    def start_task(self, is_main=True):
        if self.camera_opened:
            return

        self.camera_opened = True

        self.manager = Manager()
        self.start_evt = Event()
        self.rec_evt = Event()
        self.stop_evt = Event()
        self.stop_rec_evt = Event()
        
        self.proc_lst = []
        self.threads = []
        self.shared_dicts = [self.manager.dict() for _ in range(4)]
        self.queue = [Queue() for _ in range(4)]
        self.shm_lst = []
        # shared memory
        for i in range(4):
            shm = shared_memory.SharedMemory(create=True, size=int(np.prod(self.shape) * np.dtype(self.dtype).itemsize * self.buffer_length))
            self.shm_lst.append(shm)

            p = CameraProcess(
                self.shm_lst[i].name, 
                self.shared_dicts[i], 
                i,
                self.start_evt, 
                self.rec_evt, 
                self.stop_evt, 
                self.stop_rec_evt, 
                self.queue[i]
            )
            self.proc_lst.append(p)

            thread = UpdateThread(
                self.shm_lst[i].name,
                i,
                self.start_evt,
                self.rec_evt,
                self.stop_evt,
                self.queue[i],
            )
            self.threads.append(thread)            
        self.update()
        # !!!!
        label_cam = {0:self.patient.label_frontal_1, 1:self.patient.label_left_1, 2:self.patient.label_right_1}
        for i in range(4):
            label = label_cam[i]
            self.threads[i].update_signal.connect(lambda image, label=label: self.image_update_slot(image, label))
            self.threads[i].size(label.size().width(), label.size().height())    

        for process in self.proc_lst:
            process.start()

        for thread in self.threads:
            thread.start()

        self.start_evt.set()

        if is_main:
            self.adjust_groupbox_size()
        else:
            self.push_check_cam.setEnabled(False)
            self.push_open_cam.setEnabled(False)
            self.push_calib_start.setEnabled(False)
            self.push_calib_stop.setEnabled(True)
            self.push_calib_rec.setEnabled(True)

    def stop_task(self, is_main=True):
        if not self.camera_opened:
            return
        
        if self.record_opened:
            return

        self.stop_evt.set()
        
        for shm in self.shm_lst:
            shm.close()
            shm.unlink()
        
        for queue in self.queue:
            queue.close()
        self.shm_lst.clear()
        self.shared_dicts.clear()

        self.timer.stop()

        for thread in self.threads:
            thread.stop()
        for thread in self.threads:
            thread.wait()
        
        for process in self.proc_lst:
            if process.is_alive():
                process.terminate()
                process.join()

        self.proc_lst.clear()
        self.camera_opened = False
        
        self.start_evt.clear()
        self.stop_evt.clear()
        self.rec_evt.clear()
        self.stop_rec_evt.clear()

        if hasattr(self, 'manager'):
            self.manager.shutdown()

        self.push_check_cam.setEnabled(True)
        self.push_calib_start.setEnabled(True)
        self.push_calib_stop.setEnabled(False)
        self.push_calib_rec.setEnabled(False)

    def record_task(self, is_main=True):
        if not self.camera_opened:
            return
        
        if self.record_opened:
            return
        
        path = os.path.join(self.save_path, self.current_patient, "trial", self.trial_name) if is_main else os.path.join(self.save_path, self.current_patient, "calibration", self.calibration_date, self.calibration_name)
        if any(file.endswith(".mp4") for file in os.listdir(path)):
            reply = QMessageBox.question(self, "Question", "The calibration file already exists. Would you like to continue?", QMessageBox.StandardButton.No | QMessageBox.StandardButton.Yes)
            if reply == QMessageBox.StandardButton.No:
                return

        self.stop_rec_evt.clear()
        self.push_calib_rec.setEnabled(False)
        self.push_calib_stop_rec.setEnabled(True)

        for i in self.cam_number:
            self.threads[i].record(os.path.join(path, plane_names[i]))

        self.rec_evt.set()

        self.start_time = time.perf_counter()
        self.time_thread = TimeThread(self.start_time)
        self.time_thread.update_time_signal.connect(lambda time:self.update_time_task(time, self.label_record_time))
        
        self.time_thread.start()
        self.label_recording.setText("Recording")
        self.record_opened = True

    def stop_record_task(self, is_main=True):
        if not self.record_opened:
            return
        self.record_opened = False
        self.label_recording.setText("Finished")
        self.rec_evt.clear()
        self.stop_rec_evt.set()
        self.push_calib_rec.setEnabled(True)
        self.push_calib_stop_rec.setEnabled(False)
        camera_names = {"Frontal Plane":"camera_frontal.json", "Left Sagittal Plane":"camera_left.json", "Right Sagittal Plane":"camera_right.json"}
        plane_order = ["Frontal Plane", "Left Sagittal Plane", "Right Sagittal Plane"]
        for plane in plane_order:
            cam_id = self.plane_to_cam_id[plane]
            self.save_data_task(self.shared_dicts[cam_id], os.path.join(self.save_path, self.current_patient, "calibration", self.calibration_date, self.calibration_name, camera_names[plane]))
            self.shared_dicts[cam_id].clear()
        
        self.time_thread.stop()
        for thread in self.threads:
            thread.stop_record()

    def sub_start_task(self):
        self.start_task(is_main=False)
        
    def sub_stop_task(self):
        self.stop_task()

    def sub_record_task(self):
        self.record_task()

    def sub_stop_record_task(self):
        self.stop_record_task()

    def main_record_task(self):
        if not self.camera_opened:
            return
        
        if self.record_opened:
            return
        
        if self.trial_name=="unknown":
            QMessageBox.warning(self, "Warning", "The trial name is unknown", QMessageBox.StandardButton.Ok)
            return

        path = os.path.join(self.save_path, self.current_patient, "trial", self.trial_name)
        if any(file.endswith(".mp4") for file in os.listdir(path)):
            QMessageBox.warning(self, "Warning", "The file already exists", QMessageBox.StandardButton.Ok)
            return
        
        self.stop_rec_evt.clear()

        plane_names = ["Frontal.mp4", "Left.mp4", "Right.mp4"]
        for i in self.cam_number:
            self.main_threads[i].record(os.path.join(path, plane_names[i]))
        
        self.rec_evt.set()

        self.start_time = time.perf_counter()
        self.time_thread = TimeThread(self.start_time)
        self.time_thread.update_time_signal.connect(lambda time:self.update_time_task(time, self.label_record_trial_time))
        self.time_thread.start()
        self.label_recording_trial.setText("Recording")
        self.record_opened = True

    def main_stop_record_task(self):
        self.record_task()

        if not self.record_opened:
            return
        self.record_opened = False
        self.label_recording_trial.setText("Finished")
        self.rec_evt.clear()
        self.stop_rec_evt.set()
        camera_names = {"Frontal Plane":"camera_frontal.json", "Left Sagittal Plane":"camera_left.json", "Right Sagittal Plane":"camera_right.json"}
        plane_order = ["Frontal Plane", "Left Sagittal Plane", "Right Sagittal Plane"]
        for plane in plane_order:
            cam_id = self.plane_to_cam_id[plane]
            self.save_data_task(self.main_shared_dicts[cam_id], os.path.join(self.save_path, self.current_patient,"trial", self.trial_name, camera_names[plane]))
            self.main_shared_dicts[cam_id].clear()
        
        self.time_thread.stop()
        for thread in self.main_threads:
            thread.stop_record()

    def main_stop_task(self):
        if not self.camera_opened:
            return
        
        self.stop_evt.set()

        for shm in self.shm_lst:
            shm.close()
            shm.unlink()

        for queue in self.main_queue:
            queue.close()
        self.shm_lst.clear()
        self.main_shared_dicts.clear()

        self.timer.stop()
        for thread in self.main_threads:
            thread.stop()
        for thread in self.main_threads:
            thread.wait()

        for process in self.proc_lst:
            if process.is_alive():
                process.terminate()
                process.join()

        self.proc_lst.clear()
        self.camera_opened = False
        
        self.start_evt.clear()
        self.stop_evt.clear()
        self.rec_evt.clear()
        self.stop_rec_evt.clear()

        if hasattr(self, 'main_manager'):
            self.main_manager.shutdown()

    def save_data_task(self, shared_dict, file_path):
        data = dict(shared_dict)  # Convert manager.dict to regular dict

        s = time.perf_counter()
        with open(file_path, 'w') as f:
            json.dump(data, f, cls=NumpyEncoder, indent=4)
        file_name = os.path.basename(file_path)
        integer_key_count = sum(1 for key in data if isinstance(key, int))
        print(f"Saved data for {file_name} with {integer_key_count} frames")
        e = time.perf_counter()
        print(f"Saved data took {e-s} seconds")

    def get_label_size(self, label):
        return label.size().width(), label.size().height()
    
    def adjust_groupbox_size(self):

        visible_groupboxes = [box for box in [self.patient.groupBox, self.patient.groupBox_2, self.patient.groupBox_3] if box.isVisible()]
        min_width = int(self.patient_size_dic.get(len(visible_groupboxes), 100))

        for box in visible_groupboxes:
            box.setMinimumWidth(min_width)
            box.setMaximumWidth(min_width)
        try:
            label_cam = {0:self.patient.label_frontal_1, 1:self.patient.label_left_1, 2:self.patient.label_right_1}
            for i in range(3):
                label = label_cam[i]
                self.main_threads[i].size(label.size().width(), label.size().height())
            self.update()
        except:
            pass
    
    def check_camera_task(self):
        cam_list = []
        for i in range(3):  
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    cam_list.append(str(i))
            cap.release()

        check_cam = len(cam_list) == 3

        if check_cam:
            message = "All the cameras are detected"
            self.push_calib_start.setEnabled(True)
            self.push_open_cam.setEnabled(True)
        else:
            message = f"Please check the camera connection, found: {', '.join(cam_list)}"

        QMessageBox.information(self, "Information", message, QMessageBox.StandardButton.Ok)

    def open_camera_task(self):        
        # if not self.check_cam:
        #     QMessageBox.warning(self, "Warning", "Please check cameras first", QMessageBox.StandardButton.Ok)
        #     return
        
        if self.camera_opened:
            return
        
        
        self.push_refresh.setEnabled(True)
        self.push_close_cam.setEnabled(True)
        self.push_check_cam.setEnabled(False)
        self.push_open_cam.setEnabled(False)

        self.thread1 = CameraThread(0)
        self.thread2 = CameraThread(1)
        self.thread3 = CameraThread(2)
        self.thread4 = CameraThread(3)
        self.thread1.start()
        self.thread2.start()
        self.thread3.start()
        self.thread4.start()
        self.update_camera_layout()

        self.camera_opened = True

    def close_camera_task(self):
        if not self.camera_opened:
            return

        self.push_refresh.setEnabled(False)
        self.push_close_cam.setEnabled(False)
        self.push_check_cam.setEnabled(True)
        self.push_open_cam.setEnabled(True)
        
        self.thread1.stop()
        self.thread2.stop()
        self.thread3.stop()
        self.thread4.stop()
        self.thread1.wait()
        self.thread2.wait()
        self.thread3.wait()
        self.thread4.wait()
        self.thread1 = None
        self.thread2 = None
        self.thread3 = None
        self.thread4 = None
        self.camera_opened = False

    def image_update_slot(self, image, label):
        label.setPixmap(QPixmap.fromImage(image))

    def closeEvent(self, event):
        self.timer.stop()
        self.stop_evt.set()
        for shm, shared_dict in zip(self.shm_lst, self.main_shared_dicts):
            shm.close()
            shm.unlink()
            shared_dict.clear()
        self.sub_stop_evt.set()
        for shm, shared_dict in zip(self.sub_shm_lst, self.sub_shared_dicts):
            shm.close()
            shm.unlink()
            shared_dict.clear()
        for process in self.proc_lst:
            if process.is_alive():
                process.terminate()
                process.join()

        for sub_process in self.sub_proc_lst:
            if sub_process.is_alive():
                sub_process.terminate()
                sub_process.join()
        #!!!!
        for thread in [self.main_thread_1, self.main_thread_2, self.main_thread_3]:
            if isinstance(thread, QThread) and thread.isRunning():
                thread.stop()
                thread.wait()

        for thread in [self.sub_thread_1, self.sub_thread_2, self.sub_thread_3]:
            if isinstance(thread, QThread) and thread.isRunning():
                thread.stop()
                thread.wait()

        event.accept()

def main():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    app = QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec())
 
if __name__ == '__main__':
    main()