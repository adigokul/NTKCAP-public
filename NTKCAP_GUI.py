import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
import json
import copy
import sys
import threading
from datetime import datetime
import pyqtgraph as pg
from PyQt6 import uic
from PyQt6.QtGui import *
from PyQt6.QtCore import *
from PyQt6.QtWidgets import *
from PyQt6.QtWebEngineWidgets import QWebEngineView
from multiprocessing import Event, shared_memory, Manager, Queue, Process
from check_extrinsic import *
from NTK_CAP.script_py.NTK_Cap import *
from GUI_source.TrackerProcess import TrackerProcess
from GUI_source.CameraProcess import CameraProcess
from GUI_source.UpdateThread import UpdateThread
from GUI_source.VideoPlayer import VideoPlayer
from NTK_CAP.script_py.emg_localhost import EMGEventRecorder, detect_channel_count

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        uic.loadUi('GUI_source/NTKCAP_GUI_dec.ui', self)
        self.showMaximized()
        self.setWindowTitle("NTKCAP")
        # Initialization
        self.current_directory = os.getcwd()
        self.config_path = os.path.join(self.current_directory, "config")                
        self.cam_num, self.resolution = self.getcamerainfo(self.config_path)        
        self.time_file_path = os.path.join(self.current_directory, "cali_time.txt")        
        self.record_path = self.current_directory
        self.calibra_path = os.path.join(self.current_directory, "calibration")
        self.patient_path = os.path.join(self.current_directory, "Patient_data")
        self.multi_person_path = os.path.join(self.patient_path, "multi_person")
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
        self.label_cam = {0:self.Camera1, 1:self.Camera2, 2:self.Camera3, 3:self.Camera4}  
        
        # Record task
        # multi person
        self.btn_multi_person = self.findChild(QPushButton, "btn_multi_person")
        self.btn_multi_person.toggled.connect(self.on_multi_person_toggled)
        self.btn_multi_person.setCheckable(True)
        self.multi_person = False
        self.multi_person_record_list = []
        self.list_widget_multi_record_list = self.findChild(QListWidget, "list_widget_multi_record_list")
        self.list_widget_multi_record_list.itemDoubleClicked.connect(self.lw_multi_subjects_selected_del_select)
        self.list_widget_multi_record_list.setVisible(False)
        self.Camera1.clicked.connect(lambda: self.CameraLabel_click(self.Camera1))
        self.Camera2.clicked.connect(lambda: self.CameraLabel_click(self.Camera2))
        self.Camera3.clicked.connect(lambda: self.CameraLabel_click(self.Camera3))
        self.Camera4.clicked.connect(lambda: self.CameraLabel_click(self.Camera4))
        self.BoundaryOpened = []
        self.btn_record_boundary_cam = self.findChild(QPushButton, "btn_record_boundary_cam")
        self.btn_record_boundary_cam.clicked.connect(self.record_boundary_cam)
        self.btn_record_boundary_cam.setCheckable(True)
        self.btn_record_boundary_cam.setVisible(False)
        self.label_boundaryCam = self.findChild(QLabel, "label_boundaryCam")
        self.label_boundaryCam.setVisible(False)
        self.lw_select_del = None
        self.btn_multi_subjects_selected_del = self.findChild(QPushButton, "btn_multi_subjects_selected_del")
        self.btn_multi_subjects_selected_del.clicked.connect(self.lw_multi_subjects_selected_del)
        self.btn_multi_subjects_selected_del.setEnabled(False)
        self.btn_multi_subjects_selected_del.setVisible(False)
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
        self.btn_multi_match_gui = self.findChild(QPushButton, "btn_multi_match_gui")
        self.btn_multi_match_gui.clicked.connect(self.multi_match_gui)
        self.btn_multi_match_gui.setEnabled(True)
        self.btn_boundary_gui = self.findChild(QPushButton, "btn_boundary_gui")
        self.btn_boundary_gui.clicked.connect(self.boundary_gui)
        self.btn_boundary_gui.setEnabled(True)
        self.logview = self.findChild(QPlainTextEdit, "logview")
        self.logview.setReadOnly(True)
        self.logview.setVisible(False)
        # record Apose
        self.btn_Apose_record: QPushButton = self.findChild(QPushButton, "btn_Apose_record")
        self.btn_Apose_record.clicked.connect(self.Apose_record_ask)
        self.btn_Apose_record.setEnabled(False)
        self.timer_apose = QTimer()
        self.timer_apose.timeout.connect(self.check_apose_finish)
        # Show result
        self.btn_result_refresh = self.findChild(QPushButton, "btn_result_refresh")
        self.btn_result_refresh.clicked.connect(self.result_load_folders)
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
        self.label_cal_selected_tasks = self.findChild(QLabel, "label_cal_selected_tasks")
        self.checkBox_fast_cal = self.findChild(QCheckBox, "checkBox_fast_cal")
        self.checkBox_fast_cal.setChecked(False)
        self.checkBox_fast_cal.toggled.connect(self.on_fast_calculation)
        self.fast_cal = False
        self.checkBox_gait = self.findChild(QCheckBox, "checkBox_gait")
        self.checkBox_gait.setChecked(True)
        self.checkBox_gait.toggled.connect(self.on_gait_calculation)
        self.gait = True
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
        
        # EMG Event Recording System
        self.emg_recorder = None
        self.emg_recording_active = False
        self.emg_thread_active = False
        self.emg_thread = None
        self.emg_uri = "ws://localhost:31278/ws"  # Complete WebSocket URI
        self.emg_channel_count = 8  # Default channel count
        
        # EEG Event Recording System
        self.eeg_recorder = None
        self.eeg_recording_active = False
        self.eeg_thread_active = False
        self.eeg_thread = None
        self.eeg_uri = "ws://127.0.0.1:31279/ws"  # EEG WebSocket URI
        self.eeg_channel_count = 32  # EEG typically has more channels
        
        # Bio-signal Recording Control Checkboxes
        self.checkBox_emg_recording = self.findChild(QCheckBox, "checkBox_emg_recording")
        self.checkBox_eeg_recording = self.findChild(QCheckBox, "checkBox_eeg_recording")
        self.emg_recording_enabled = False
        self.eeg_recording_enabled = False
        
        # Connect checkbox signals
        if self.checkBox_emg_recording:
            self.checkBox_emg_recording.toggled.connect(self.on_emg_recording_toggled)
        if self.checkBox_eeg_recording:
            self.checkBox_eeg_recording.toggled.connect(self.on_eeg_recording_toggled)
    # Bio-signal Recording Control
    def on_emg_recording_toggled(self, checked):
        """Handle EMG recording checkbox toggle"""
        self.emg_recording_enabled = checked
        if checked:
            print("✅ EMG recording enabled")
        else:
            print("❌ EMG recording disabled")
    
    def on_eeg_recording_toggled(self, checked):
        """Handle EEG recording checkbox toggle"""
        self.eeg_recording_enabled = checked
        if checked:
            print("✅ EEG recording enabled")
        else:
            print("❌ EEG recording disabled")
    
    # Main page shortcut
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
        self.multi_person_record_list.clear()
        self.lw_multi_subjects_selected_show()
        self.record_select_patientID = None
        self.record_task_name = None
        self.list_widget_patient_id_record.clearSelection()
        self.record_enter_task_name.setText("")
        self.record_enter_task_name.setEnabled(False)
        self.btn_Apose_record.setEnabled(False)
        self.btnStartRecording.setEnabled(False)
        self.btnStopRecording.setEnabled(False)
        self.label_selected_patient.setText("Selected patient :")
        self.label_log.setText("Please select a patient ID")
        self.result_load_folders()
        self.list_widget_patient_task_record.clear()
    def record_boundary_cam(self):
        if self.btn_record_boundary_cam.isChecked():
            self.Camera1.setClickable(True)
            self.Camera2.setClickable(True)
            self.Camera3.setClickable(True)
            self.Camera4.setClickable(True)
        else:
            self.BoundaryOpened = []
            self.label_boundaryCam.clear()
            self.Camera1.setClickable(False)
            self.Camera2.setClickable(False)
            self.Camera3.setClickable(False)
            self.Camera4.setClickable(False)
            
    def multi_match_gui(self):
        subprocess.run(["python", os.path.join(self.current_directory, 'GUI_source', 'match_multi_GUI.py'), self.multi_person_path])
    def boundary_gui(self):
        subprocess.run(["python", os.path.join(self.current_directory, 'GUI_source', 'boundary_GUI.py')])
    def on_multi_person_toggled(self, checked): # mp func is checked or not
        if (not self.camera_opened):
            QMessageBox.information(self, "Cameras are not opened", "Please open cameras first！")
            return
        if self.multi_person:
            reply = QMessageBox.question(
                self, 
                "Confirm Action",
                "Exit multi person mode?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.list_widget_multi_record_list.setVisible(False)
                self.label_boundaryCam.setVisible(False)
                self.btn_record_boundary_cam.setVisible(False)
                self.btn_multi_subjects_selected_del.setVisible(False)
                self.record_select_patientID = None
                self.label_selected_patient.setText(f"Selected patient :")
                self.record_enter_task_name.setEnabled(False)
                self.list_widget_patient_task_record.clear()
                self.multi_person = False
                self.multi_person_record_list = []
                self.lw_multi_subjects_selected_show()
                self.btn_multi_person.blockSignals(True)
                self.btn_multi_person.setChecked(False)
                self.btn_multi_person.blockSignals(False)
            else:
                self.list_widget_multi_record_list.setVisible(True)
                self.label_boundaryCam.setVisible(True)
                self.btn_record_boundary_cam.setVisible(True)
                self.btn_multi_subjects_selected_del.setVisible(True)
                self.btn_multi_person.blockSignals(True)
                self.btn_multi_person.setChecked(True)
                self.btn_multi_person.blockSignals(False)             
        else:
            reply = QMessageBox.question(
                self, 
                "Multi person mode",
                "Have you completed the Apose of all subjects?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.list_widget_multi_record_list.setVisible(True)
                self.label_boundaryCam.setVisible(True)
                self.btn_record_boundary_cam.setVisible(True)
                self.btn_multi_subjects_selected_del.setVisible(True)
                self.record_select_patientID = "multi_person"
                self.label_selected_patient.setText(f"Selected patient : {self.record_select_patientID}")
                self.record_enter_task_name.setEnabled(True)
                self.lw_patient_task_record()
                self.multi_person = True
                self.btn_multi_person.blockSignals(True)
                self.btn_multi_person.setChecked(True)
                self.btn_multi_person.blockSignals(False)
            else:
                self.list_widget_multi_record_list.setVisible(False)
                self.label_boundaryCam.setVisible(False)
                self.btn_record_boundary_cam.setVisible(False)
                self.btn_multi_subjects_selected_del.setVisible(False)
                self.btn_multi_person.blockSignals(True)
                self.btn_multi_person.setChecked(False)
                self.btn_multi_person.blockSignals(False)   
        
    def lw_multi_subjects_selected_del_select(self, item):
        self.lw_select_del = item.text()
        self.btn_multi_subjects_selected_del.setEnabled(True)
    
    def lw_multi_subjects_selected_del(self):
        self.multi_person_record_list.remove(self.lw_select_del)
        self.lw_select_del = None
        self.btn_multi_subjects_selected_del.setEnabled(False)
        self.lw_multi_subjects_selected_show()
    def lw_multi_subjects_selected_show(self):
        self.list_widget_multi_record_list.clear()
        if self.multi_person_record_list:
            for item in self.multi_person_record_list:
                self.list_widget_multi_record_list.addItem(item)
            
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
        if self.multi_person:
            if (item.text() != 'multi_person') and (item.text() not in self.multi_person_record_list):
                self.multi_person_record_list.append(item.text())
                self.record_enter_task_name.setEnabled(True)
                self.lw_multi_subjects_selected_show()
        else:
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
            if item != 'multi_person':
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
            self.record_enter_task_name.setEnabled(True)

    def update_Apose_note(self):
        olddir_meetnote = os.path.join(self.config_path, 'meetnote_layout.json')
        newdir_meetnote = os.path.join(self.patient_path, self.record_select_patientID, datetime.now().strftime("%Y_%m_%d"), 'raw_data', 'Meet_note.json')
        shutil.copy2(olddir_meetnote, newdir_meetnote)
    
    def Apose_record(self):
        cali_file_path = self.calibra_path
        save_path_date = os.path.join(self.patient_path, self.record_select_patientID, datetime.now().strftime("%Y_%m_%d"))
        save_path_Apose = os.path.join(save_path_date, "raw_data", "Apose")
        time_file_path = os.path.join(save_path_Apose, "recordtime.txt")
        save_path_videos = os.path.join(save_path_Apose, "videos")
        os.makedirs(save_path_videos)
        if os.path.exists(self.time_file_path):
            shutil.copy(self.time_file_path, save_path_date)
            self.label_log.setText("Successfully copied cali_time.txt")
        else:
            self.label_log.setText("cali_time.txt doesn't exist")
        if os.path.exists(cali_file_path):
            if os.path.exists(os.path.join(save_path_date, "raw_data", "calibration")):
                shutil.rmtree(os.path.join(save_path_date, "raw_data", "calibration"))
            shutil.copytree(cali_file_path, os.path.join(save_path_date, "raw_data", "calibration"))
            self.label_log.setText("Successfully copied calibration folder")
        else:
            self.label_log.setText("Calibration folder doesn't exist")
        if os.path.exists(time_file_path):
            with open(time_file_path, "r") as file:
                formatted_datetime = file.read().strip()
        else:
            now = datetime.now()
            formatted_datetime = now.strftime("%Y_%m_%d_%H%M")
            with open(time_file_path, "w") as file:
                file.write(formatted_datetime)

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
        self.task_stop_rec_evt.clear()
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
        if self.multi_person:   
            self.shared_dict_record_name['name'] = self.record_select_patientID
            self.shared_dict_record_name['task_name'] = self.record_task_name
            self.shared_dict_record_name['start_time'] = time.time()            
            if self.multi_person_record_list is not None:
                multi_name_folder_path = os.path.join(self.patient_path, self.record_select_patientID, datetime.now().strftime("%Y_%m_%d"), "raw_data", self.record_task_name, 'name')
                os.makedirs(multi_name_folder_path)
                with open(os.path.join(multi_name_folder_path, "name.txt"), "w") as f:
                    for name in self.multi_person_record_list:                
                        f.write(name + "\n")
            with open(os.path.join(multi_name_folder_path, "Boundary.txt"), "w", encoding="utf-8") as f:
                for cam_id in self.BoundaryOpened:
                    f.write(f"{int(cam_id)}\n")
            # continue
        else:
            self.shared_dict_record_name['name'] = self.record_select_patientID
            self.shared_dict_record_name['task_name'] = self.record_task_name
            self.shared_dict_record_name['start_time'] = time.time()
        self.task_stop_rec_evt.clear()
        self.btnStartRecording.setEnabled(False)
        self.task_rec_evt.set()
        self.record_opened = True
        self.record_enter_task_name.setEnabled(False)
        self.list_widget_patient_id_record.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        
        # Start EMG and EEG Recording (only if checkboxes are checked)
        emg_started = False
        eeg_started = False
        
        try:
            # Add the script_py directory to Python path
            script_py_path = os.path.join(os.path.dirname(__file__), 'NTK_CAP', 'script_py')
            if script_py_path not in sys.path:
                sys.path.append(script_py_path)
            
            from NTK_CAP.script_py.emg_localhost import EMGEventRecorder
            
            # Start EMG Recording only if checkbox is checked
            if self.emg_recording_enabled and self.checkBox_emg_recording.isChecked():
                emg_output_path = os.path.join(self.patient_path, self.record_select_patientID, 
                                             datetime.now().strftime("%Y_%m_%d"), "raw_data", 
                                             self.record_task_name, "emg_data.csv")
                # Use cumulative timestamps by default (True)
                self.emg_recorder = EMGEventRecorder(self.emg_uri, emg_output_path, self.emg_channel_count, use_cumulative_timestamp=True)
                emg_started = self.emg_recorder.start_recording()
            
            # Start EEG Recording only if checkbox is checked
            if self.eeg_recording_enabled and self.checkBox_eeg_recording.isChecked():
                eeg_output_path = os.path.join(self.patient_path, self.record_select_patientID, 
                                             datetime.now().strftime("%Y_%m_%d"), "raw_data", 
                                             self.record_task_name, "eeg_data.csv")
                # Use cumulative timestamps by default (True)  
                self.eeg_recorder = EMGEventRecorder(self.eeg_uri, eeg_output_path, self.eeg_channel_count, use_cumulative_timestamp=True)
                eeg_started = self.eeg_recorder.start_recording()
            
            if emg_started:
                self.emg_recording_active = True
                # Start EMG data processing thread
                self.emg_thread_active = True
                self.emg_thread = threading.Thread(target=self._emg_processing_loop, daemon=True)
                self.emg_thread.start()
                print("✅ EMG recording started successfully")
            
            if eeg_started:
                self.eeg_recording_active = True
                # Start EEG data processing thread
                self.eeg_thread_active = True
                self.eeg_thread = threading.Thread(target=self._eeg_processing_loop, daemon=True)
                self.eeg_thread.start()
                print("✅ EEG recording started successfully")
            
            # Update status message based on what was started
            if emg_started and eeg_started:
                self.label_log.setText("Recording start (EMG + EEG)")
            elif emg_started:
                self.label_log.setText("Recording start (EMG only)")
            elif eeg_started:
                self.label_log.setText("Recording start (EEG only)")
            elif not self.emg_recording_enabled and not self.eeg_recording_enabled:
                self.label_log.setText("Recording start (No bio-signal recording selected)")
            else:
                self.label_log.setText("Recording start (Bio-signal connection failed)")
                
        except Exception as e:
            print(f"❌ EMG/EEG recording failed to start: {e}")
            self.emg_recording_active = False
            self.eeg_recording_active = False
            self.label_log.setText("Recording start (Bio-signal recording failed)")
    
    def stoprecord_task(self): 
        if not self.record_opened:
            return
        
        # Stop EMG and EEG Recording
        emg_stopped = False
        eeg_stopped = False
        
        # Stop EMG Recording
        if self.emg_recording_active and self.emg_recorder:
            try:
                # Add recording stop event marker
                self.emg_recorder.add_event_marker(141, "Recording End")
                
                # Stop the EMG processing thread
                self.emg_thread_active = False
                if hasattr(self, 'emg_thread') and self.emg_thread is not None:
                    # Wait a bit for the thread to finish
                    self.emg_thread.join(timeout=2.0)
                
                # Stop EMG recording
                self.emg_recorder.stop_recording()
                self.emg_recording_active = False
                emg_stopped = True
                print("✅ EMG recording stopped successfully")
            except Exception as e:
                print(f"❌ Error stopping EMG recording: {e}")
                # Force stop
                self.emg_thread_active = False
                self.emg_recording_active = False
        
        # Stop EEG Recording
        if self.eeg_recording_active and self.eeg_recorder:
            try:
                # Add recording stop event marker
                self.eeg_recorder.add_event_marker(141, "Recording End")
                
                # Stop the EEG processing thread
                self.eeg_thread_active = False
                if hasattr(self, 'eeg_thread') and self.eeg_thread is not None:
                    # Wait a bit for the thread to finish
                    self.eeg_thread.join(timeout=2.0)
                
                # Stop EEG recording
                self.eeg_recorder.stop_recording()
                self.eeg_recording_active = False
                eeg_stopped = True
                print("✅ EEG recording stopped successfully")
            except Exception as e:
                print(f"❌ Error stopping EEG recording: {e}")
                # Force stop
                self.eeg_thread_active = False
                self.eeg_recording_active = False
        
        self.record_opened = False
        self.task_stop_rec_evt.set()
        self.task_rec_evt.clear()
        self.shared_dict_record_name.clear()
        self.record_enter_task_name.clear()
        self.record_enter_task_name.setEnabled(True)
        self.list_widget_patient_id_record.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.btn_pg1_reset.setEnabled(True)
        self.btnStopRecording.setEnabled(False)
        
        if emg_stopped and eeg_stopped:
            self.label_log.setText("Record end (EMG + EEG)")
        elif emg_stopped:
            self.label_log.setText("Record end (EMG only)")
        elif eeg_stopped:
            self.label_log.setText("Record end (EEG only)")
        else:
            self.label_log.setText("Record end")
            
        self.lw_patient_task_record()
        self.cal_load_folders()
        
    def add_bio_event_marker(self, event_id=None, event_description="Manual Event"):
        """Add an event marker to both EMG and EEG recordings"""
        emg_success = False
        eeg_success = False
        
        if event_id is None:
            event_id = 999  # Default event ID for manual events
        
        # Add event to EMG recording
        if self.emg_recording_active and self.emg_recorder:
            try:
                self.emg_recorder.add_event_marker(event_id, event_description)
                emg_success = True
                print(f"EMG Event added: ID={event_id}, Description='{event_description}'")
            except Exception as e:
                print(f"Error adding EMG event marker: {e}")
        
        # Add event to EEG recording
        if self.eeg_recording_active and self.eeg_recorder:
            try:
                self.eeg_recorder.add_event_marker(event_id, event_description)
                eeg_success = True
                print(f"EEG Event added: ID={event_id}, Description='{event_description}'")
            except Exception as e:
                print(f"Error adding EEG event marker: {e}")
        
        if emg_success or eeg_success:
            return True
        else:
            print("No bio-signal recording active, cannot add event marker")
            return False
    
    # Keep the old method name for backward compatibility
    def add_emg_event_marker(self, event_id=None, event_description="Manual Event"):
        """Legacy method - redirects to add_bio_event_marker"""
        return self.add_bio_event_marker(event_id, event_description)
    
    def _emg_processing_loop(self):
        """Background thread for EMG data processing"""
        print("🎯 EMG processing thread started")
        
        first_data_received = False
        
        while self.emg_thread_active and self.emg_recording_active and self.emg_recorder:
            try:
                # Process one data packet with timeout
                if self.emg_recorder.process_and_save_data(timeout=0.1):
                    # Successfully processed data
                    if not first_data_received:
                        # Add recording start event marker on first successful data reception
                        self.emg_recorder.add_event_marker(130, "Recording Start")
                        first_data_received = True
                        print("✅ First EMG data received, added start event marker")
                    continue
                else:
                    # No data received within timeout, continue loop
                    time.sleep(0.01)  # Small delay to prevent busy waiting
                    
            except Exception as e:
                print(f"❌ EMG processing error: {e}")
                time.sleep(0.1)  # Wait before retrying
                
        print("🛑 EMG processing thread ended")
        
    def _eeg_processing_loop(self):
        """Background thread for EEG data processing"""
        print("🎯 EEG processing thread started")
        
        first_data_received = False
        
        while self.eeg_thread_active and self.eeg_recording_active and self.eeg_recorder:
            try:
                # Process one data packet with timeout
                if self.eeg_recorder.process_and_save_data(timeout=0.1):
                    # Successfully processed data
                    if not first_data_received:
                        # Add recording start event marker on first successful data reception
                        self.eeg_recorder.add_event_marker(130, "Recording Start")
                        first_data_received = True
                        print("✅ First EEG data received, added start event marker")
                    continue
                else:
                    # No data received within timeout, continue loop
                    time.sleep(0.01)  # Small delay to prevent busy waiting
                    
            except Exception as e:
                print(f"❌ EEG processing error: {e}")
                time.sleep(0.1)  # Wait before retrying
                
        print("🛑 EEG processing thread ended")
        
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
    def closeCamera(self):
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
        
    def CameraLabel_click(self, label):
        if label.objectName()[-1] in self.BoundaryOpened:
            return
        if len(self.BoundaryOpened) < 2:
            self.BoundaryOpened.append(label.objectName()[-1])
        else:
            self.BoundaryOpened.pop(0)
            self.BoundaryOpened.append(label.objectName()[-1])
        if self.BoundaryOpened != []:
            self.label_boundaryCam.setText('\n'.join(self.BoundaryOpened))
    def button_config(self, instance):
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
            self.task_stop_rec_evt.clear()
            self.btn_pg1_reset.setEnabled(False)
            self.label_log.setText("create new extrinsic params")
            time_file_path = os.path.join(self.record_path, "calibration", "calib_time.txt")
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
            err_list = calib_extri(self.current_directory, 0)
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
            err_list = calib_extri(self.current_directory,1)
            self.label_log.text = 'calculate finished'
            self.err_calib_extri.text = err_list
            self.err_calib_extri.setText(read_err_calib_extri(self.current_directory))
        except:
            self.label_log.text = 'check intrinsic and extrinsic exist'
            self.err_calib_extri.text = 'no calibration file found'

    def check_cal_finish(self):
        if self.marker_calculate_process:
            if not self.marker_calculate_process.is_alive():
                self.cal_show_selected_folder()
                self.timer_marker_calculate.stop()
                self.label_calculation_status.setText("Calculation is finished")
                self.btn_cal_start_cal.setEnabled(True)
                self.result_load_folders()

    def btn_pre_marker_calculate(self):
        if self.cal_select_list == []:
            return
        else:
            cur_dir = copy.deepcopy(self.current_directory)
            cal_list = copy.deepcopy(self.cal_select_list)
            self.closeCamera()
            # mp_marker_calculate(cur_dir, cal_list, self.fast_cal, self.gait)
            self.marker_calculate_process = Process(target=mp_marker_calculate, args=(cur_dir, cal_list, self.fast_cal, self.gait))
            self.marker_calculate_process.start()
            self.cal_select_list = []
            self.label_calculation_status.setText("Calculating")
            self.btn_cal_start_cal.setEnabled(False)
            self.timer_marker_calculate.start(1000)

    # Calculation tab
    def on_fast_calculation(self, checked):
        if checked:
            QMessageBox.information(self, "Notification", "Fast calculation only supports single person mode")
            self.fast_cal = True
        else:
            self.fast_cal = False

    def on_gait_calculation(self, checked):
        if checked:
            self.gait = True
        else:
            self.gait = False

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
            self.label_cal_selected_tasks.setText(f"Selected Tasks: {len(self.cal_select_list)}")    
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
        self.label_cal_selected_tasks.setText(f"Selected Tasks: {len(self.cal_select_list)}")    
    
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
        self.combobox_result_select_gait_figures.setCurrentIndex(0)
        self.video_player.load_gltf_file_in_viewer(f"./Patient_data/{result_select_patient_id}/{result_select_date}/{result_select_cal_time}/{result_select_task}/model.gltf")
        self.playButton.setEnabled(True)
        self.result_video_slider.setEnabled(True)

    def result_disconnet_gait_figures(self):
        if self.result_current_gait_figures_index == 1: # Hip flexion
            self.result_video_slider.valueChanged.disconnect(self.video_player.update_hip_flexion)
        elif self.result_current_gait_figures_index ==2: # Knee flexion
            self.result_video_slider.valueChanged.disconnect(self.video_player.update_knee_flexion)          
        elif self.result_current_gait_figures_index ==3: # Ankle flexion
            self.result_video_slider.valueChanged.disconnect(self.video_player.update_ankle_flexion)
        elif self.result_current_gait_figures_index ==4: # Speed
            self.result_video_slider.valueChanged.disconnect(self.video_player.update_speed)
        elif self.result_current_gait_figures_index ==5: # Stride
            self.result_video_slider.valueChanged.disconnect(self.video_player.update_stride)

    def result_select_gait_figures(self, index):
        if index == 1: # Hip flexion
            self.video_player.hip_flexion_plot()#
            self.result_video_slider.valueChanged.connect(self.video_player.update_hip_flexion)
            self.result_current_gait_figures_index = 1
        elif index ==2: # Knee flexion
            self.video_player.knee_flexion_plot()       
            self.result_video_slider.valueChanged.connect(self.video_player.update_knee_flexion)
            self.result_current_gait_figures_index = 2
        elif index ==3: # Ankle flexion
            self.video_player.ankle_flexion_plot()
            self.result_video_slider.valueChanged.connect(self.video_player.update_ankle_flexion)
            self.result_current_gait_figures_index = 3
        elif index ==4: # Speed
            self.video_player.speed_plot()
            self.result_video_slider.valueChanged.connect(self.video_player.update_speed)
            self.result_current_gait_figures_index = 4
        elif index ==5: # Stride
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
        # Stop EMG recording if active
        if self.emg_recording_active:
            try:
                self.emg_thread_active = False
                if hasattr(self, 'emg_thread') and self.emg_thread is not None:
                    self.emg_thread.join(timeout=1.0)
                if self.emg_recorder:
                    self.emg_recorder.stop_recording()
                print("EMG recording stopped on application close")
            except Exception as e:
                print(f"Error stopping EMG on close: {e}")
        
        # Stop EEG recording if active
        if self.eeg_recording_active:
            try:
                self.eeg_thread_active = False
                if hasattr(self, 'eeg_thread') and self.eeg_thread is not None:
                    self.eeg_thread.join(timeout=1.0)
                if self.eeg_recorder:
                    self.eeg_recorder.stop_recording()
                print("EEG recording stopped on application close")
            except Exception as e:
                print(f"Error stopping EEG on close: {e}")
        
        self.closeCamera()
        if self.marker_calculate_process and self.marker_calculate_process.is_alive():
            self.marker_calculate_process.terminate()
            self.marker_calculate_process.join()

if __name__ == "__main__":
    App = QApplication(sys.argv)
    App.setWindowIcon(QIcon("GUI_source/TeamLogo.jpg"))
    window = MainWindow()
    window.show()
    sys.exit(App.exec())