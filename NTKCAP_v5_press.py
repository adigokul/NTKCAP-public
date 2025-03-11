import os
import time
import json
import copy
import sys
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
        self.btn_multi_person.clicked.connect(self.multi_person_mode)
        self.multi_person = False
        self.multi_person_record_list = []
        self.list_widget_multi_record_list = self.findChild(QListWidget, "list_widget_multi_record_list")
        self.list_widget_multi_record_list.itemDoubleClicked.connect(self.lw_multi_subjects_selected_del_select)
        self.lw_select_del = None
        self.btn_multi_subjects_selected_del = self.findChild(QPushButton, "btn_multi_subjects_selected_del")
        self.btn_multi_subjects_selected_del.clicked.connect(self.lw_multi_subjects_selected_del)
        self.btn_multi_subjects_selected_del.setEnabled(False)
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
        self.checkBox_fast_cal = self.findChild(QCheckBox, "checkBox_fast_cal")
        self.checkBox_fast_cal.setChecked(False)
        self.checkBox_fast_cal.toggled.connect(self.on_fast_calculation)
        self.fast_cal = False
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
    def multi_person_mode(self):
        reply = QMessageBox.question(
            self, 
            "Multi person mode",
            "Have you completed the Apose of all subjects?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.btn_multi_person.setCheckable(True)
            self.btn_multi_person.setChecked(True)
        else:
            self.btn_multi_person.setCheckable(False)
            self.btn_multi_person.setChecked(False)
    def on_multi_person_toggled(self, checked):
        if checked:
            self.record_select_patientID = "multi_person"
            self.label_selected_patient.setText(f"Selected patient : {self.record_select_patientID}")
            self.record_enter_task_name.setEnabled(True)
            self.lw_patient_task_record()
            self.multi_person = True
        else:
            self.record_select_patientID = None
            self.label_selected_patient.setText(f"Selected patient :")
            self.record_enter_task_name.setEnabled(False)
            self.list_widget_patient_task_record.clear()
            self.multi_person = False
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
            if item.text() != 'multi_person':
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
        self.label_log.setText("Record end")
        self.lw_patient_task_record()
        self.cal_load_folders()
        
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
        # self.tri = Process(target=tri, args=())
        # self.tri.start()
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
            config_name = os.path.join(self.config_path, "config.json")
            time_file_path = os.path.join(self.record_path, "calibration", "calib_time.txt")
            with open(config_name, 'r') as f:
                data = json.load(f)
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
            self.marker_calculate_process = Process(target=mp_marker_calculate, args=(cur_dir, cal_list, self.fast_cal))
            self.marker_calculate_process.start()
            self.cal_select_list = []
            self.label_calculation_status.setText("Calculating")
            self.btn_cal_start_cal.setEnabled(False)
            self.timer_marker_calculate.start(1000)

    # Calculation tab
    def on_fast_calculation(self, checked):
        if checked:
            self.fast_cal = True
        else:
            self.fast_cal = False

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
        self.combobox_result_select_gait_figures.setCurrentIndex(0)
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
    App.setWindowIcon(QIcon("GUI_source/Team_logo.jpg"))
    window = MainWindow()
    window.show()
    sys.exit(App.exec())
