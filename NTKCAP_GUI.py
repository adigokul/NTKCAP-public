import multiprocessing
try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
import json
import copy
import sys
import threading
import queue
import multiprocessing
from datetime import datetime
import pyqtgraph as pg
from PyQt6 import uic
from PyQt6.QtGui import *
from PyQt6.QtCore import *
from PyQt6.QtWidgets import *
from PyQt6.QtWebEngineWidgets import QWebEngineView
from multiprocessing import Event, shared_memory, Manager, Queue, Process

# Add NTK_CAP script_py directory to Python path
script_py_path = os.path.join(os.path.dirname(__file__), 'NTK_CAP', 'script_py')
if script_py_path not in sys.path:
    sys.path.insert(0, script_py_path)

from check_extrinsic import *
from NTK_CAP.script_py.NTK_Cap import *
from GUI_source.TrackerProcess import TrackerProcess
from GUI_source.CameraProcess import CameraProcess
from GUI_source.UpdateThread import UpdateThread
from GUI_source.VideoPlayer import VideoPlayer

from PyQt6.QtWidgets import QDialog, QFormLayout, QLineEdit, QComboBox, QDialogButtonBox

import sqlite3
from PyQt6.QtWidgets import QDialog, QFormLayout, QLineEdit, QComboBox, QDialogButtonBox, QListWidget, QVBoxLayout, QLabel
# Connect to SQLite database
conn = sqlite3.connect('icd10/icd10.db')



# Define fuzzy search function
def fuzzy_search(query):
    cursor = conn.execute("""
        SELECT Code, CM2023_英文名稱, CM2023_中文名稱 
        FROM icd10_fts 
        WHERE CM2023_英文名稱 LIKE ? 
        OR CM2023_中文名稱 LIKE ? 
        OR Code LIKE ?
    """, (f'%{query}%', f'%{query}%', f'%{query}%'))
    results = cursor.fetchall()
    return results


class PatientDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("New Patient")
        layout = QFormLayout(self)

        # Fields for sex, height, weight, age
        self.sex_edit = QComboBox(self)
        self.sex_edit.addItems(["M", "F"])
        self.height_edit = QLineEdit(self)
        self.weight_edit = QLineEdit(self)
        self.age_edit = QLineEdit(self)

        # Symptoms input field and search results
        self.symptom_edit = QLineEdit(self)  # Symptoms input field
        self.results_list = QListWidget(self)  # Display search results

        # Layout setup
        layout.addRow("Sex:", self.sex_edit)
        layout.addRow("Height (cm):", self.height_edit)
        layout.addRow("Weight (kg):", self.weight_edit)
        layout.addRow("Age:", self.age_edit)
        layout.addRow("Symptoms:", self.symptom_edit)  # Added symptoms field
        layout.addWidget(self.results_list)  # Added the results list for ICD-10 search

        # Buttons for accepting or canceling the dialog
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        # Connect symptom input field to search function
        self.symptom_edit.textChanged.connect(self.search_symptoms)

        # Connect the itemClicked signal to select_result method
        self.results_list.itemClicked.connect(self.select_result)

    def search_symptoms(self):
        """Perform a fuzzy search based on the symptom entered."""
        query = self.symptom_edit.text()
        if query:
            # Perform the fuzzy search using ICD-10 database
            results = fuzzy_search(query)
            self.results_list.clear()  # Clear previous results
            if results:
                # Display ICD-10 code and description
                for result in results:
                    self.results_list.addItem(f"Code: {result[0]}, {result[1]} / {result[2]}")
            else:
                self.results_list.addItem("No matching results found.")

    def get_data(self):
        """Return the data entered in the dialog."""
        return {
            "sex": self.sex_edit.currentText(),
            "height": self.height_edit.text(),
            "weight": self.weight_edit.text(),
            "age": self.age_edit.text(),
            "symptoms": self.symptom_edit.text(),  # Get symptoms from input
            "symptom_code": None  # Placeholder for ICD-10 code, will be set when a result is selected
        }

    def select_result(self, item):
        """Handle selecting a symptom result from the list."""
        selected_text = item.text()
        # Extract ICD code and descriptions
        selected_code = selected_text.split(",")[0].split(":")[1].strip()
        selected_en_description = selected_text.split(",")[1].strip()
        selected_cn_description = selected_text.split("/")[1].strip() if "/" in selected_text else ""

        # Update the symptom input field with selected ICD-10 code and description
        self.symptom_edit.setText(f"Code: {selected_code}, {selected_en_description} / {selected_cn_description}")
        
        # Optionally, store the code for further use
        self.symptom_code = selected_code  # Store the selected ICD code
        self.results_list.clear()  # Clear results after selection

        print(f"Selected: {selected_code} - {selected_en_description} / {selected_cn_description}")

    def closeEvent(self, event):
        """Close the SQLite connection when the dialog is closed."""
        conn.close()
        event.accept()

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
        
        # Create necessary directories if they don't exist
        os.makedirs(self.patient_path, exist_ok=True)
        os.makedirs(self.multi_person_path, exist_ok=True)
        os.makedirs(self.calibra_path, exist_ok=True)
        os.makedirs(self.extrinsic_path, exist_ok=True)
        
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

        # 添加紅點標記切換按鈕（用於校正）
        self.btn_toggle_marker = QPushButton("Show Center Marker")
        self.btn_toggle_marker.setCheckable(True)
        self.btn_toggle_marker.setChecked(True)  # 默認顯示
        self.btn_toggle_marker.clicked.connect(self.toggle_calibration_marker)
        self.btn_toggle_marker.setEnabled(False)  # 相機未開啟時禁用

        # 將按鈕添加到外參校正區域的佈局中
        extrinsic_layout = self.findChild(QVBoxLayout, "verticalLayout_4")
        if extrinsic_layout:
            extrinsic_layout.insertWidget(0, self.btn_toggle_marker)
            print("✅ Calibration marker toggle button added to extrinsic calibration section")
        else:
            print("⚠️ Warning: Could not find verticalLayout_4 for calibration marker button")
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
        # Connect returnPressed signal for reliable Enter key handling
        self.record_enter_task_name.returnPressed.connect(self.on_task_name_entered)
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
        
        # Initialize task selection mode variables
        self.task_checkboxes = {}  # Store checkboxes for patient view
        self.current_patient_tasks = []  # Store current patient's tasks
        
        # Add new UI elements for task-based selection
        self.setup_task_selection_ui()
        
        self.cal_load_folders()
        self.marker_calculate_process = None
        self.marker_progress_queue = None  # Queue for receiving progress updates
        self.timer_marker_calculate = QTimer()
        self.timer_marker_calculate.timeout.connect(self.check_cal_finish)
        
        # Add calculation progress bar (next to RT Pose Detection button in status bar)
        self.calculation_progress_bar = QProgressBar()
        self.calculation_progress_bar.setMinimum(0)
        self.calculation_progress_bar.setMaximum(100)
        self.calculation_progress_bar.setValue(0)
        self.calculation_progress_bar.setTextVisible(True)
        self.calculation_progress_bar.setFormat("Calculation: %p%")
        self.calculation_progress_bar.setVisible(False)  # Hidden by default
        self.calculation_progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid grey;
                border-radius: 5px;
                text-align: center;
                background-color: #f0f0f0;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                width: 10px;
                margin: 0.5px;
            }
        """)
        self.statusBar().addPermanentWidget(self.calculation_progress_bar)
        print("✅ Calculation progress bar added to status bar")
        
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
        
        # Real-time Pose Detection Control - Create a button and add it to the UI
        self.btn_rt_pose_detection = QPushButton("RT Pose Detection: OFF")
        self.btn_rt_pose_detection.setCheckable(True)
        self.btn_rt_pose_detection.setChecked(False)  # Default to disabled
        self.btn_rt_pose_detection.setStyleSheet("""
            QPushButton {
                background-color: #ff6b6b;
                color: black;
                border: none;
                padding: 5px;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:checked {
                background-color: #51cf66;
            }
        """)
        self.rt_pose_detection_enabled = False  # Default to disabled
        
        # Add the button to the Configuration section (next to other buttons)
        # Find a suitable location in the UI to place the button
        try:
            # Try to find the Configuration section or a suitable parent
            config_widget = self.findChild(QWidget, "Configuration") or self.findChild(QWidget, "tab")
            if config_widget is None:
                # If no specific widget found, try to find the main widget
                config_widget = self.centralWidget()
            
            if config_widget:
                # Create or get the layout
                layout = config_widget.layout()
                if layout is None:
                    layout = QVBoxLayout()
                    config_widget.setLayout(layout)
                
                # Add the button to the layout
                layout.addWidget(self.btn_rt_pose_detection)
                print("✅ RT Pose Detection button added to UI")
            else:
                print("❌ Could not find suitable parent widget for RT Pose Detection button")
        except Exception as e:
            print(f"❌ Error adding RT Pose Detection button to UI: {e}")
            # If all else fails, just add it to the main window
            self.statusBar().addPermanentWidget(self.btn_rt_pose_detection)
        
        # Connect checkbox signals
        if self.checkBox_emg_recording:
            self.checkBox_emg_recording.toggled.connect(self.on_emg_recording_toggled)
        if self.checkBox_eeg_recording:
            self.checkBox_eeg_recording.toggled.connect(self.on_eeg_recording_toggled)
        
        # Connect RT pose detection button signal
        self.btn_rt_pose_detection.clicked.connect(self.on_rt_pose_detection_toggled)
        
        # Add RT Pose Detection button to the status bar (always visible)
        self.statusBar().addPermanentWidget(self.btn_rt_pose_detection)
        print("✅ RT Pose Detection button added to status bar")
        
        # Add Arrange Files button (for inverse analysis data)
        self.btn_arrange_files = QPushButton("Arrange Files to Inverse Analysis Data")
        self.btn_arrange_files.setStyleSheet("""
            QPushButton {
                background-color: #4285f4;
                color: black;
                border: none;
                padding: 5px;
                border-radius: 3px;
                font-weight: bold;
                min-width: 200px;
            }
            QPushButton:hover {
                background-color: #3367d6;
            }
            QPushButton:pressed {
                background-color: #2952b3;
            }
        """)
        self.btn_arrange_files.clicked.connect(self.arrange_files_to_inverse_analysis)
        self.statusBar().addPermanentWidget(self.btn_arrange_files)
        print("✅ Arrange Files button added to status bar")
        
        # Add Calculation Progress Bar next to RT Pose Detection button
        self.progress_bar_calculation = QProgressBar()
        self.progress_bar_calculation.setMaximum(100)
        self.progress_bar_calculation.setMinimum(0)
        self.progress_bar_calculation.setValue(0)
        self.progress_bar_calculation.setTextVisible(True)
        self.progress_bar_calculation.setFormat("Calculation: %p%")
        self.progress_bar_calculation.setStyleSheet("""
            QProgressBar {
                border: 2px solid grey;
                border-radius: 5px;
                text-align: center;
                min-width: 200px;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                width: 10px;
            }
        """)
        self.progress_bar_calculation.setVisible(False)  # Hidden by default
        self.statusBar().addPermanentWidget(self.progress_bar_calculation)
        print("✅ Calculation progress bar added to status bar")

        # Add Camera Mode toggle button to status bar
        self.btn_camera_mode = QPushButton("Camera: Webcam")
        self.btn_camera_mode.setCheckable(True)
        self.btn_camera_mode.setChecked(False)  # Default is webcam (unchecked)
        self.btn_camera_mode.setStyleSheet("""
            QPushButton {
                background-color: #4a90e2;
                color: white;
                border: none;
                padding: 5px 10px;
                border-radius: 3px;
                font-weight: bold;
                min-width: 180px;
            }
            QPushButton:checked {
                background-color: #51cf66;
            }
            QPushButton:hover {
                background-color: #357abd;
            }
            QPushButton:checked:hover {
                background-color: #40c057;
            }
        """)
        self.btn_camera_mode.clicked.connect(self.on_camera_mode_toggled)
        self.statusBar().addPermanentWidget(self.btn_camera_mode)

        # Initialize button state from config
        try:
            camera_config_path = os.path.join(self.config_path, 'config.json')
            with open(camera_config_path, 'r') as f:
                config = json.load(f)
            current_type = config['cam'].get('type', 'usb')
            poe_model = config['cam'].get('poe', {}).get('model', 'ae450').upper()
            if current_type in ('ae400', 'ae450', 'poe'):
                self.btn_camera_mode.setChecked(True)
                self.btn_camera_mode.setText(f"Camera: PoE ({poe_model})")
                print(f"✅ Camera Mode toggle button added to status bar (PoE {poe_model})")
            else:
                print("✅ Camera Mode toggle button added to status bar (Webcam)")
        except Exception as e:
            print(f"⚠️ Could not read camera type from config: {e}")
            print("✅ Camera Mode toggle button added to status bar (default: Webcam)")

    # Calibration Marker Toggle
    def toggle_calibration_marker(self):
        """切換校正紅點標記的顯示"""
        show_marker = self.btn_toggle_marker.isChecked()

        # 更新所有 UpdateThread 的標記狀態
        if hasattr(self, 'threads'):
            for thread in self.threads:
                thread.show_calibration_marker = show_marker

        # 更新按鈕文字
        if show_marker:
            self.btn_toggle_marker.setText("Hide Center Marker")
            print("✓ 校正標記已顯示")
        else:
            self.btn_toggle_marker.setText("Show Center Marker")
            print("✗ 校正標記已隱藏")

    # Camera Mode Toggle
    def on_camera_mode_toggled(self):
        """Handle camera mode toggle between Webcam and PoE (AE400/AE450)"""
        is_poe = self.btn_camera_mode.isChecked()

        # Update config file and button text
        try:
            camera_config_path = os.path.join(self.config_path, 'config.json')
            with open(camera_config_path, 'r') as f:
                config = json.load(f)

            poe_model = config['cam'].get('poe', {}).get('model', 'ae450').upper()

            if is_poe:
                self.btn_camera_mode.setText(f"Camera: PoE ({poe_model})")
                camera_type = "poe"
                print(f"✅ Switched to PoE camera mode ({poe_model}, RGB only)")
            else:
                self.btn_camera_mode.setText("Camera: Webcam")
                camera_type = "usb"
                print("✅ Switched to Webcam mode")

            config['cam']['type'] = camera_type

            with open(camera_config_path, 'w') as f:
                json.dump(config, f, indent=4)

            print(f"📝 Camera type updated in config: {camera_type}")

            # Warn user if cameras are currently open
            if self.camera_opened:
                QMessageBox.information(
                    self,
                    "Camera Mode Changed",
                    f"Camera mode changed to {'PoE Depth Camera (' + poe_model + ')' if is_poe else 'Webcam'}.\n\n"
                    "Please close and reopen cameras for changes to take effect."
                )
            else:
                if hasattr(self, 'label_log'):
                    self.label_log.setText(
                        f"Camera mode: {'PoE (' + poe_model + ')' if is_poe else 'Webcam'} - Open cameras to apply"
                    )

        except Exception as e:
            print(f"❌ Error updating camera config: {e}")
            QMessageBox.critical(self, "Error", f"Failed to update camera configuration:\n{e}")

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
    
    def on_rt_pose_detection_toggled(self):
        """Handle real-time pose detection button toggle"""
        self.rt_pose_detection_enabled = self.btn_rt_pose_detection.isChecked()
        
        if self.rt_pose_detection_enabled:
            print("✅ Real-time pose detection enabled")
            self.btn_rt_pose_detection.setText("RT Pose Detection: ON")
            if hasattr(self, 'label_log'):
                self.label_log.setText("RT pose detection enabled - May require ONNX Runtime")
        else:
            print("❌ Real-time pose detection disabled")
            self.btn_rt_pose_detection.setText("RT Pose Detection: OFF")
            if hasattr(self, 'label_log'):
                self.label_log.setText("RT pose detection disabled - Using TensorRT only")
        
        # If cameras are already open, restart tracker processes with new settings
        if self.camera_opened and hasattr(self, 'tracker_proc_lst'):
            try:
                print("🔄 Updating tracker processes with new pose detection settings...")
                
                # Stop existing tracker processes
                for process in self.tracker_proc_lst:
                    if process.is_alive():
                        process.terminate()
                        process.join(timeout=2)
                        if process.is_alive():
                            process.kill()
                
                # Clear the list
                self.tracker_proc_lst.clear()
                
                # Recreate tracker processes with new settings
                for i in range(4):
                    p1 = TrackerProcess(
                        self.start_evt,
                        i,
                        self.stop_evt,
                        self.queue[i],
                        self.queue_kp[i],
                        self.shm_lst[i].name,
                        self.shm_kp_lst[i].name,
                        enable_pose_detection=self.rt_pose_detection_enabled
                    )
                    self.tracker_proc_lst.append(p1)
                
                # Start the new tracker processes
                for process in self.tracker_proc_lst:
                    process.start()
                
                status_msg = "enabled" if self.rt_pose_detection_enabled else "disabled"
                print(f"✅ Tracker processes restarted with pose detection {status_msg}")
                if hasattr(self, 'label_log'):
                    self.label_log.setText(f"Tracker processes updated - RT pose detection {status_msg}")
                    
            except Exception as e:
                print(f"❌ Error updating tracker processes: {e}")
                if hasattr(self, 'label_log'):
                    self.label_log.setText(f"Error updating tracker processes: {e}")
        else:
            # If cameras are not open, just update the setting for next time
            print("📝 RT pose detection setting updated for next camera session")
    
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
        # Keep for backward compatibility, but primary handler is on_task_name_entered()
        if (self.record_enter_task_name_kb_listen) and (event.key() == 16777220):
            self.on_task_name_entered()

    def on_task_name_entered(self):
        """Handle task name entry when Enter is pressed - called by returnPressed signal"""
        if not self.record_select_patientID:
            self.label_log.setText("Please select a patient ID first")
            return
        
        task_name = self.record_enter_task_name.text().strip()
        if not task_name:
            self.label_log.setText("Please enter a task name")
            return
        
        task_path = os.path.join(self.patient_path, self.record_select_patientID, datetime.now().strftime("%Y_%m_%d"), "raw_data", task_name)
        
        if os.path.exists(task_path):
            reply = QMessageBox.question(
                self, 
                "task name already exists",
                "Are you sure to cover it?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                return
        
        # Set task name and enable recording buttons
        self.record_task_name = task_name
        self.btnStartRecording.setEnabled(True)
        self.btnStopRecording.setEnabled(True)
        self.btn_Apose_record.setEnabled(False)
        self.label_log.setText(f"Current task name : {self.record_task_name}")
        print(f"✓ Task name set: {self.record_task_name}, Record button enabled")

    def record_enter_task_name_infocus(self, event):
        """Called when task name field gains focus"""
        self.record_enter_task_name_kb_listen = True

    def record_enter_task_name_outfocus(self, event):
        """Called when task name field loses focus"""
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
            patient_id = text  # Store the original patient ID
            
            # The original logic stays the same
            if os.path.exists(os.path.join(self.patient_path, patient_id)):
                reply = QMessageBox.question(
                    self, 
                    "Patient ID already exists",
                    "Are you sure to overwrite the ID?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.Yes:
                    shutil.rmtree(os.path.join(self.patient_path, patient_id))
                    os.mkdir(os.path.join(self.patient_path, patient_id))
                    self.label_log.setText(f"Patient ID {patient_id} is selected")
                    self.record_list_widget_patient_id_list_show()
                    self.btn_Apose_record.setEnabled(True)
                    self.record_select_patientID = patient_id
                    self.label_selected_patient.setText(f"Selected patient : {self.record_select_patientID}")
                elif reply == QMessageBox.StandardButton.No:
                    self.button_create_new_patient()
            else:
                # Create patient directory as before
                self.record_select_patientID = patient_id
                os.mkdir(os.path.join(self.patient_path, patient_id))
                self.label_log.setText(f"Patient ID {patient_id} is selected")
                self.record_list_widget_patient_id_list_show()
                self.btn_Apose_record.setEnabled(True)
                self.label_selected_patient.setText(f"Selected patient : {self.record_select_patientID}")

            # Now collect additional data (sex, height, weight, age, symptoms)
            dialog = PatientDialog(self)
            print('Opening patient dialog...')
            
            if dialog.exec() == QDialog.DialogCode.Accepted:
                print('Dialog accepted.')
                data = dialog.get_data()  # Get the collected data
                
                # Save the additional data into the patient directory
                with open(os.path.join(self.patient_path, patient_id, "info.txt"), "a") as f:
                    print('Writing patient info...')
                    f.write(f"Sex: {data['sex']}\n")
                    f.write(f"Height: {data['height']}\n")
                    f.write(f"Weight: {data['weight']}\n")
                    f.write(f"Age: {data['age']}\n")
                    f.write(f"Symptoms: {data['symptoms']}\n")  # Add symptoms to the file

                # Update UI after adding the additional patient info
                self.label_log.setText(f"Patient {patient_id} info updated")
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

        # Read camera configuration
        camera_config_path = os.path.join(self.config_path, 'config.json')
        with open(camera_config_path, 'r') as f:
            config = json.load(f)

        cam_config = config['cam']
        camera_type = cam_config.get('type', 'usb')  # Default to usb
        print(f"📷 Opening cameras in {camera_type.upper()} mode")

        # Prepare camera-specific configs
        project_root = os.path.dirname(os.path.abspath(__file__))
        cam_configs = []

        if camera_type in ('ae400', 'ae450', 'poe'):
            # PoE camera mode (AE400/AE450/LIPSedge)
            # Try new 'poe' config first, fallback to 'ae400' for backwards compatibility
            poe_config = cam_config.get('poe', cam_config.get('ae400', {}))
            ips = poe_config.get('ips', [])
            openni2_base = poe_config.get('openni2_base', 'NTK_CAP/ThirdParty/OpenNI2')
            poe_model = poe_config.get('model', 'ae450')

            print(f"  PoE Camera Model: {poe_model.upper()}")

            for i in range(4):
                if i < len(ips):
                    openni2_path = os.path.join(project_root, openni2_base, ips[i])
                    cam_configs.append({
                        'type': 'poe',
                        'ip': ips[i],
                        'openni2_path': openni2_path,
                        'model': poe_model
                    })
                    print(f"  Cam {i+1}: {poe_model.upper()} @ {ips[i]}")
                else:
                    print(f"  Warning: No PoE config for camera {i+1}, using USB fallback")
                    cam_configs.append({'type': 'usb', 'device_index': i})
        else:
            # USB mode
            usb_indices = cam_config.get('list', [0, 1, 2, 3])
            for i in range(4):
                cam_configs.append({
                    'type': 'usb',
                    'device_index': usb_indices[i] if i < len(usb_indices) else i
                })
                print(f"  Cam {i+1}: USB device {usb_indices[i] if i < len(usb_indices) else i}")

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
                self.shm_kp_lst[i].name,
                enable_pose_detection=self.rt_pose_detection_enabled
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
        apose_events = [self.apose_rec_evt1, self.apose_rec_evt2, self.apose_rec_evt3, self.apose_rec_evt4]
        calib_events = [self.calib_rec_evt1, self.calib_rec_evt2, self.calib_rec_evt3, self.calib_rec_evt4]

        for i in range(4):
            p = CameraProcess(
                self.shm_lst[i].name,
                i,
                self.start_evt,
                self.task_rec_evt,
                apose_events[i],
                calib_events[i],
                self.stop_evt,
                self.task_stop_rec_evt,
                self.calib_save_path,
                self.queue[i],
                self.patient_path,
                self.shared_dict_record_name,
                cam_type=camera_type,
                cam_config=cam_configs[i]
            )
            self.camera_proc_lst.append(p)
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

        # 啟用校正紅點標記
        self.btn_toggle_marker.setEnabled(True)
        for thread in self.threads:
            thread.show_calibration_marker = True  # 默認顯示紅點
        self.btn_toggle_marker.setChecked(True)
        self.btn_toggle_marker.setText("Hide Center Marker")
        print("✓ 校正中心標記已啟用（方便對準）")

        self.start_evt.set()
    def closeCamera(self):
        if not self.camera_opened: return
        if self.record_opened: return

        # 禁用校正標記按鈕
        self.btn_toggle_marker.setEnabled(False)

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
        # Update progress from queue (real progress from calculation process)
        if self.marker_progress_queue:
            try:
                while not self.marker_progress_queue.empty():
                    progress_data = self.marker_progress_queue.get_nowait()
                    if isinstance(progress_data, dict):
                        progress = progress_data.get('progress', 0)
                        status = progress_data.get('status', '')
                        self.progress_bar_calculation.setValue(int(progress))
                        if status:
                            self.label_calculation_status.setText(status)
            except:
                pass
        
        if self.marker_calculate_process:
            if not self.marker_calculate_process.is_alive():
                self.cal_show_selected_folder()
                self.timer_marker_calculate.stop()
                self.label_calculation_status.setText("Calculation is finished")
                self.btn_cal_start_cal.setEnabled(True)
                self.result_load_folders()
                
                # Hide progress bar and set to 100%
                self.progress_bar_calculation.setValue(100)
                QTimer.singleShot(1000, lambda: self.progress_bar_calculation.setVisible(False))
                QTimer.singleShot(1000, lambda: self.progress_bar_calculation.setValue(0))
                
                # Clean up queue
                if self.marker_progress_queue:
                    self.marker_progress_queue.close()
                    self.marker_progress_queue = None

    def btn_pre_marker_calculate(self):
        if self.cal_select_list == []:
            return
        else:
            cur_dir = copy.deepcopy(self.current_directory)
            
            # Process selection list to handle both date-level and task-level selections
            cal_list_raw = copy.deepcopy(self.cal_select_list)
            cal_list = []
            task_filter_dict = {}  # Dictionary to store task filters for each date path
            
            for item in cal_list_raw:
                if "##" in item:
                    # Extract date path and task name
                    date_path, task_name = item.split("##", 1)
                    if date_path not in cal_list:
                        cal_list.append(date_path)
                        task_filter_dict[date_path] = []
                    task_filter_dict[date_path].append(task_name)
                else:
                    # Regular date path (all tasks)
                    if item not in cal_list:
                        cal_list.append(item)
                        task_filter_dict[item] = None  # None means all tasks
            
            self.closeCamera()
            
            # Show progress bar and start from 0
            self.progress_bar_calculation.setValue(0)
            self.progress_bar_calculation.setVisible(True)
            
            # Create Queue for real-time progress tracking
            self.marker_progress_queue = multiprocessing.Queue()
            
            # Debug: Print the paths being sent to calculation
            print(f"Debug: Sending {len(cal_list)} paths to calculation:")
            for path in cal_list:
                print(f"  - {path}")
                if path in task_filter_dict:
                    print(f"    Tasks: {task_filter_dict[path]}")
            
            # Use multiprocessing to avoid UI freeze with task filtering
            self.marker_calculate_process = Process(target=mp_marker_calculate, args=(cur_dir, cal_list, self.fast_cal, self.gait, self.marker_progress_queue, task_filter_dict))
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

    def setup_task_selection_ui(self):
        """Setup UI elements for task-based selection"""
        # Find the calculation tab widget
        cal_widget = None
        try:
            # Try to find the calculation tab or a suitable parent widget
            cal_widget = self.findChild(QWidget, "tab_calculation") or self.findChild(QWidget, "calculation")
            if cal_widget is None:
                # Fallback to finding by layout or other means
                cal_widget = self.centralWidget()
        except:
            cal_widget = self.centralWidget()
        
        if cal_widget:
            # Add "Select All Tasks" button
            self.btn_select_all_tasks = QPushButton("Select All Tasks")
            self.btn_select_all_tasks.setStyleSheet("""
                QPushButton {
                    background-color: #28a745;
                    color: black;
                    border: none;
                    padding: 8px;
                    border-radius: 4px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #218838;
                }
            """)
            self.btn_select_all_tasks.clicked.connect(self.select_all_tasks)
            
            # Add "Select All" checkbox for patient view
            self.checkbox_select_all_patient = QCheckBox("Select All in Patient")
            self.checkbox_select_all_patient.setVisible(False)  # Hidden by default
            self.checkbox_select_all_patient.toggled.connect(self.toggle_all_patient_tasks)
            
            # Add select all patient checkbox to status bar (only for patient view)
            try:
                self.statusBar().addPermanentWidget(self.checkbox_select_all_patient)
            except:
                # Fallback: add to main widget if possible
                pass

    def select_all_tasks(self):
        """Select all available tasks from current patient"""
        if not hasattr(self, 'patient_path') or not os.path.exists(self.patient_path):
            return
            
        if not hasattr(self, 'cal_select_patient_id') or not self.cal_select_patient_id:
            return
            
        original_count = len(self.cal_select_list)
        
        # Get current patient's path
        current_patient_path = os.path.join(self.patient_path, self.cal_select_patient_id)
        if not os.path.exists(current_patient_path):
            return
        
        # Get all date folders for current patient (exclude raw_data if it exists at patient level)
        date_folders = [item for item in os.listdir(current_patient_path) 
                       if os.path.isdir(os.path.join(current_patient_path, item)) and item != 'raw_data']
        
        # For each task in current patient, find its corresponding date folder and add to selection
        for task_checkbox_path in self.task_checkboxes.keys():
            if task_checkbox_path.startswith(current_patient_path):
                if task_checkbox_path not in self.cal_select_list:
                    self.cal_select_list.append(task_checkbox_path)
                # Update checkbox state
                if task_checkbox_path in self.task_checkboxes:
                    self.task_checkboxes[task_checkbox_path].setChecked(True)
        
        # Update UI
        self.cal_show_selected_folder()
        self.label_cal_selected_tasks.setText(f"Selected Tasks: {len(self.cal_select_list)}")
        
        # Update patient-level select all checkbox
        self.update_patient_select_all_state()
        
        # Show brief feedback in status
        added_count = len(self.cal_select_list) - original_count
        if hasattr(self, 'statusBar'):
            if added_count > 0:
                self.statusBar().showMessage(f"✅ Added {added_count} tasks (Total: {len(self.cal_select_list)})", 3000)
            else:
                self.statusBar().showMessage("ℹ️ All tasks were already selected", 2000)

    def on_select_all_patients_toggled(self, checked):
        """Handle Select All Patients checkbox toggle"""
        if checked:
            self.select_all_patients()
        else:
            self.deselect_all_patients()

    def on_select_all_dates_toggled(self, checked):
        """Handle Select All Dates checkbox toggle"""
        if checked:
            self.select_all_dates()
        else:
            self.deselect_all_dates()

    def on_select_all_tasks_toggled(self, checked):
        """Handle Select All Tasks checkbox toggle"""
        if checked:
            self.select_all_current_date_tasks()
        else:
            self.deselect_all_current_date_tasks()

    def select_all_patients(self):
        """Select all patients (all their dates and tasks)"""
        if not hasattr(self, 'patient_path') or not os.path.exists(self.patient_path):
            return
        
        for patient_name in os.listdir(self.patient_path):
            patient_path = os.path.join(self.patient_path, patient_name)
            if os.path.isdir(patient_path):
                for date_name in os.listdir(patient_path):
                    date_path = os.path.join(patient_path, date_name)
                    if os.path.isdir(date_path) and date_name != 'raw_data':
                        if date_path not in self.cal_select_list:
                            self.cal_select_list.append(date_path)
        
        self.cal_show_selected_folder()
        self.label_cal_selected_tasks.setText(f"Selected Tasks: {len(self.cal_select_list)}")

    def deselect_all_patients(self):
        """Deselect all patients"""
        self.cal_select_list.clear()
        self.cal_show_selected_folder()
        self.label_cal_selected_tasks.setText(f"Selected Tasks: {len(self.cal_select_list)}")

    def select_all_dates(self):
        """Select all dates for current patient"""
        if not hasattr(self, 'cal_select_patient_id') or not self.cal_select_patient_id:
            return
            
        patient_path = os.path.join(self.patient_path, self.cal_select_patient_id)
        if os.path.exists(patient_path):
            for date_name in os.listdir(patient_path):
                date_path = os.path.join(patient_path, date_name)
                if os.path.isdir(date_path) and date_name != 'raw_data':
                    if date_path not in self.cal_select_list:
                        self.cal_select_list.append(date_path)
        
        self.cal_show_selected_folder()
        self.label_cal_selected_tasks.setText(f"Selected Tasks: {len(self.cal_select_list)}")

    def deselect_all_dates(self):
        """Deselect all dates for current patient"""
        if not hasattr(self, 'cal_select_patient_id') or not self.cal_select_patient_id:
            return
            
        patient_path = os.path.join(self.patient_path, self.cal_select_patient_id)
        dates_to_remove = [path for path in self.cal_select_list if path.startswith(patient_path)]
        
        for date_path in dates_to_remove:
            self.cal_select_list.remove(date_path)
        
        self.cal_show_selected_folder()
        self.label_cal_selected_tasks.setText(f"Selected Tasks: {len(self.cal_select_list)}")

    def select_all_current_date_tasks(self):
        """Select current date (all tasks in this date)"""
        if not hasattr(self, 'cal_select_date_id') or not self.cal_select_date_id:
            return
            
        patient_path = os.path.join(self.patient_path, self.cal_select_patient_id)
        date_path = os.path.join(patient_path, self.cal_select_date_id)
        
        # Select all individual tasks for this date
        for task_checkbox_path, checkbox in self.task_checkboxes.items():
            if task_checkbox_path.startswith(date_path + "##"):
                if task_checkbox_path not in self.cal_select_list:
                    self.cal_select_list.append(task_checkbox_path)
                checkbox.setChecked(True)
        
        self.cal_show_selected_folder()
        self.label_cal_selected_tasks.setText(f"Selected Tasks: {len(self.cal_select_list)}")

    def deselect_all_current_date_tasks(self):
        """Deselect current date (all tasks in this date)"""
        if not hasattr(self, 'cal_select_date_id') or not self.cal_select_date_id:
            return
            
        patient_path = os.path.join(self.patient_path, self.cal_select_patient_id)
        date_path = os.path.join(patient_path, self.cal_select_date_id)
        
        # Deselect all individual tasks for this date
        tasks_to_remove = []
        for task_checkbox_path, checkbox in self.task_checkboxes.items():
            if task_checkbox_path.startswith(date_path + "##"):
                tasks_to_remove.append(task_checkbox_path)
                checkbox.setChecked(False)
        
        # Remove from selection list
        for task_path in tasks_to_remove:
            if task_path in self.cal_select_list:
                self.cal_select_list.remove(task_path)
        
        self.cal_show_selected_folder()
        self.label_cal_selected_tasks.setText(f"Selected Tasks: {len(self.cal_select_list)}")

    def select_all_current_patient_tasks(self):
        """Select all tasks for current patient"""
        if not hasattr(self, 'cal_select_patient_id') or not self.cal_select_patient_id:
            return
            
        current_patient_path = os.path.join(self.patient_path, self.cal_select_patient_id)
        
        # Select all tasks for current patient
        for task_checkbox_path in self.task_checkboxes.keys():
            if task_checkbox_path.startswith(current_patient_path):
                if task_checkbox_path not in self.cal_select_list:
                    self.cal_select_list.append(task_checkbox_path)
                # Update checkbox state
                if task_checkbox_path in self.task_checkboxes:
                    self.task_checkboxes[task_checkbox_path].setChecked(True)
        
        # Update UI
        self.cal_show_selected_folder()
        self.label_cal_selected_tasks.setText(f"Selected Tasks: {len(self.cal_select_list)}")
        self.update_patient_select_all_state()

    def deselect_all_current_patient_tasks(self):
        """Deselect all tasks for current patient"""
        if not hasattr(self, 'cal_select_patient_id') or not self.cal_select_patient_id:
            return
            
        current_patient_path = os.path.join(self.patient_path, self.cal_select_patient_id)
        
        # Deselect all tasks for current patient
        tasks_to_remove = []
        for task_path in self.cal_select_list:
            if task_path.startswith(current_patient_path):
                tasks_to_remove.append(task_path)
                # Update checkbox state
                if task_path in self.task_checkboxes:
                    self.task_checkboxes[task_path].setChecked(False)
        
        # Remove from selection list
        for task_path in tasks_to_remove:
            self.cal_select_list.remove(task_path)
        
        # Update UI
        self.cal_show_selected_folder()
        self.label_cal_selected_tasks.setText(f"Selected Tasks: {len(self.cal_select_list)}")
        self.update_patient_select_all_state()

    def toggle_all_patient_tasks(self, checked):
        """Toggle all tasks for current patient"""
        if self.cal_select_depth != 1 or not self.cal_select_patient_id:
            return
            
        patient_path = os.path.join(self.patient_path, self.cal_select_patient_id)
        if not os.path.exists(patient_path):
            return
            
        if checked:
            # Add all tasks for current patient
            for date_name in os.listdir(patient_path):
                date_path = os.path.join(patient_path, date_name)
                if os.path.isdir(date_path) and date_name != 'raw_data':
                    task_path = os.path.join(patient_path, date_name)
                    if task_path not in self.cal_select_list:
                        self.cal_select_list.append(task_path)
        else:
            # Remove all tasks for current patient
            patient_tasks = [path for path in self.cal_select_list 
                           if path.startswith(patient_path)]
            for task_path in patient_tasks:
                self.cal_select_list.remove(task_path)
        
        # Update checkboxes if they exist
        for checkbox in self.task_checkboxes.values():
            checkbox.setChecked(checked)
        
        # Update UI
        self.cal_show_selected_folder()
        self.label_cal_selected_tasks.setText(f"Selected Tasks: {len(self.cal_select_list)}")

    def cal_select_back_path(self):
        if self.cal_select_depth == 1:
            # Going back from date view to patient view
            self.cal_select_depth -= 1
            self.cal_select_patient_id = None
            self.btn_cal_back_path.setEnabled(False)
        elif self.cal_select_depth == 2:
            # Going back from task view to date view
            self.cal_select_depth -= 1
            self.cal_select_date_id = None
            # Keep back button enabled since we're still in navigation
        self.cal_load_folders()

    def cal_load_folders(self):
        # Update path display based on current depth
        if self.cal_select_depth == 0:
            self.text_label_path_depth.setText("D:\\NTKCAP\\Patient_data")
        elif self.cal_select_depth == 1:
            self.text_label_path_depth.setText(f"D:\\NTKCAP\\Patient_data\\{self.cal_select_patient_id}")
        elif self.cal_select_depth == 2:
            if hasattr(self, 'cal_select_date_id') and self.cal_select_date_id:
                # Get available tasks from raw_data to show task selection context
                patient_path = os.path.join(self.patient_path, self.cal_select_patient_id)
                date_path = os.path.join(patient_path, self.cal_select_date_id)
                raw_data_path = os.path.join(date_path, 'raw_data')
                
                if os.path.exists(raw_data_path):
                    tasks = [item for item in os.listdir(raw_data_path) 
                            if os.path.isdir(os.path.join(raw_data_path, item)) 
                            and item not in ['Apose', 'calibration']]
                    if tasks:
                        task_list = ", ".join(sorted(tasks))
                        self.text_label_path_depth.setText(f"D:\\NTKCAP\\Patient_data\\{self.cal_select_patient_id}\\{self.cal_select_date_id} (Tasks: {task_list})")
                    else:
                        self.text_label_path_depth.setText(f"D:\\NTKCAP\\Patient_data\\{self.cal_select_patient_id}\\{self.cal_select_date_id} (No tasks found)")
                else:
                    self.text_label_path_depth.setText(f"D:\\NTKCAP\\Patient_data\\{self.cal_select_patient_id}\\{self.cal_select_date_id}")
            else:
                self.text_label_path_depth.setText(f"D:\\NTKCAP\\Patient_data\\{self.cal_select_patient_id}")
            
        if self.cal_select_depth == 0:
            # Load patient list with Select All Patients
            self.listwidget_select_cal_date.clear()
            
            # Add "Select All Patients" checkbox as first item
            select_all_list_item = QListWidgetItem()
            self.listwidget_select_cal_date.addItem(select_all_list_item)
            
            select_all_widget = QWidget()
            select_all_layout = QHBoxLayout(select_all_widget)
            
            self.checkbox_select_all_patients = QCheckBox("Select All Patients")
            self.checkbox_select_all_patients.setStyleSheet("""
                QCheckBox {
                    font-weight: bold;
                    color: black;
                    background-color: #e1f5e1;
                    padding: 5px;
                    border-radius: 3px;
                }
            """)
            self.checkbox_select_all_patients.toggled.connect(self.on_select_all_patients_toggled)
            
            select_all_layout.addWidget(self.checkbox_select_all_patients)
            select_all_layout.setContentsMargins(5, 8, 5, 8)
            
            # Set proper size for the widget to avoid being cut off
            select_all_widget.setMinimumHeight(40)
            select_all_list_item.setSizeHint(QSize(0, 40))
            
            self.listwidget_select_cal_date.setItemWidget(select_all_list_item, select_all_widget)
            
            # Add separator
            separator_item = QListWidgetItem("─────────────────────")
            separator_item.setFlags(Qt.ItemFlag.NoItemFlags)
            separator_item.setBackground(QColor(240, 240, 240))
            self.listwidget_select_cal_date.addItem(separator_item)
            
            # Add patient list
            items = [item for item in os.listdir(self.patient_path)]
            for item in items:
                patient_item = QListWidgetItem(f"👤 {item}")
                self.listwidget_select_cal_date.addItem(patient_item)
                
            # Hide patient-specific checkbox at patient level
            if hasattr(self, 'checkbox_select_all_patient'):
                self.checkbox_select_all_patient.setVisible(False)
                
        elif self.cal_select_depth == 1:
            # Load date list for selected patient with Select All Dates
            self.listwidget_select_cal_date.clear()
            
            # Add "Select All Dates" checkbox
            select_all_list_item = QListWidgetItem()
            self.listwidget_select_cal_date.addItem(select_all_list_item)
            
            select_all_widget = QWidget()
            select_all_layout = QHBoxLayout(select_all_widget)
            
            self.checkbox_select_all_dates = QCheckBox("Select All Dates")
            self.checkbox_select_all_dates.setStyleSheet("""
                QCheckBox {
                    font-weight: bold;
                    color: #2d5016;
                    background-color: #e8f5e8;
                    padding: 5px;
                    border-radius: 3px;
                }
            """)
            self.checkbox_select_all_dates.toggled.connect(self.on_select_all_dates_toggled)
            
            select_all_layout.addWidget(self.checkbox_select_all_dates)
            select_all_layout.setContentsMargins(5, 8, 5, 8)
            
            # Set proper size for the widget to avoid being cut off
            select_all_widget.setMinimumHeight(40)
            select_all_list_item.setSizeHint(QSize(0, 40))
            
            self.listwidget_select_cal_date.setItemWidget(select_all_list_item, select_all_widget)
            
            # Add separator
            separator_item = QListWidgetItem("─────────────────────")
            separator_item.setFlags(Qt.ItemFlag.NoItemFlags)
            separator_item.setBackground(QColor(240, 240, 240))
            self.listwidget_select_cal_date.addItem(separator_item)
            
            # Add date folders
            patient_path = os.path.join(self.patient_path, self.cal_select_patient_id)
            if os.path.exists(patient_path):
                date_folders = [item for item in os.listdir(patient_path) 
                               if os.path.isdir(os.path.join(patient_path, item)) and item != 'raw_data']
                date_folders.sort()
                
                for date_folder in date_folders:
                    date_item = QListWidgetItem(f"📅 {date_folder}")
                    self.listwidget_select_cal_date.addItem(date_item)
                
        elif self.cal_select_depth == 2:
            # Load task list for selected date with checkboxes
            self.listwidget_select_cal_date.clear()
            self.task_checkboxes.clear()
            
            # Add "Select All Tasks" checkbox as first item
            select_all_list_item = QListWidgetItem()
            self.listwidget_select_cal_date.addItem(select_all_list_item)
            
            # Create select all widget with checkbox
            select_all_widget = QWidget()
            select_all_layout = QHBoxLayout(select_all_widget)
            
            self.checkbox_select_all_tasks = QCheckBox("Select All Tasks")
            self.checkbox_select_all_tasks.setStyleSheet("""
                QCheckBox {
                    font-weight: bold;
                    color: #2d5016;
                    background-color: #e8f5e8;
                    padding: 5px;
                    border-radius: 3px;
                }
                QCheckBox::indicator {
                    width: 18px;
                    height: 18px;
                }
            """)
            self.checkbox_select_all_tasks.toggled.connect(self.on_select_all_tasks_toggled)
            
            select_all_layout.addWidget(self.checkbox_select_all_tasks)
            select_all_layout.setContentsMargins(5, 8, 5, 8)
            
            # Set proper size for the widget to avoid being cut off
            select_all_widget.setMinimumHeight(40)
            select_all_list_item.setSizeHint(QSize(0, 40))
            
            self.listwidget_select_cal_date.setItemWidget(select_all_list_item, select_all_widget)
            
            # Add separator
            separator_item = QListWidgetItem("─────────────────────")
            separator_item.setFlags(Qt.ItemFlag.NoItemFlags)  # Make it non-selectable
            separator_item.setBackground(QColor(240, 240, 240))
            self.listwidget_select_cal_date.addItem(separator_item)
            
            # Get tasks from current selected date
            if hasattr(self, 'cal_select_date_id') and self.cal_select_date_id:
                patient_path = os.path.join(self.patient_path, self.cal_select_patient_id)
                date_path = os.path.join(patient_path, self.cal_select_date_id)
                raw_data_path = os.path.join(date_path, 'raw_data')
                
                if os.path.exists(raw_data_path):
                    # Get task folders (exclude 'Apose' and 'calibration')
                    tasks = [item for item in os.listdir(raw_data_path) 
                            if os.path.isdir(os.path.join(raw_data_path, item)) 
                            and item not in ['Apose', 'calibration']]
                    tasks.sort()
                    
                    # Add items with checkboxes
                    for task in tasks:
                        # Create a widget item
                        list_item = QListWidgetItem()
                        self.listwidget_select_cal_date.addItem(list_item)
                        
                        # Create a widget with checkbox and label
                        item_widget = QWidget()
                        item_layout = QHBoxLayout(item_widget)
                        
                        checkbox = QCheckBox(task)
                        
                        # Create unique task path that includes the task name
                        task_path = f"{date_path}##{task}"  # Use ## as separator to make it unique
                        
                        checkbox.setChecked(task_path in self.cal_select_list)
                        checkbox.toggled.connect(lambda checked, path=task_path: self.on_task_checkbox_toggled(checked, path))
                        
                        item_layout.addWidget(checkbox)
                        item_layout.setContentsMargins(5, 5, 5, 5)
                        
                        # Store checkbox reference
                        self.task_checkboxes[task_path] = checkbox
                        
                        # Set proper size for the widget
                        item_widget.setMinimumHeight(30)
                        list_item.setSizeHint(QSize(0, 30))
                        self.listwidget_select_cal_date.setItemWidget(list_item, item_widget)

    def on_task_checkbox_toggled(self, checked, task_path):
        """Handle individual task checkbox toggle"""
        if checked:
            if task_path not in self.cal_select_list:
                self.cal_select_list.append(task_path)
        else:
            if task_path in self.cal_select_list:
                self.cal_select_list.remove(task_path)
        
        # Update UI
        self.cal_show_selected_folder()
        self.label_cal_selected_tasks.setText(f"Selected Tasks: {len(self.cal_select_list)}")
        
        # Update select all checkbox states
        self.update_select_all_states()

    def update_select_all_states(self):
        """Update select all checkbox states for current depth"""
        if self.cal_select_depth == 2 and hasattr(self, 'cal_select_date_id'):
            # In task selection level
            patient_path = os.path.join(self.patient_path, self.cal_select_patient_id)
            date_path = os.path.join(patient_path, self.cal_select_date_id)
            
            # Count total tasks and selected tasks
            total_tasks = 0
            selected_tasks = 0
            for task_checkbox_path in self.task_checkboxes.keys():
                if task_checkbox_path.startswith(date_path + "##"):
                    total_tasks += 1
                    if task_checkbox_path in self.cal_select_list:
                        selected_tasks += 1
            
            # All tasks are selected if selected count equals total count
            all_selected = (selected_tasks == total_tasks) and (total_tasks > 0)
            
            # Update Select All Tasks checkbox
            if hasattr(self, 'checkbox_select_all_tasks'):
                # Temporarily disconnect to avoid recursive calls
                self.checkbox_select_all_tasks.toggled.disconnect()
                self.checkbox_select_all_tasks.setChecked(all_selected)
                self.checkbox_select_all_tasks.toggled.connect(self.on_select_all_tasks_toggled)

    def cal_show_selected_folder(self):
        self.listwidget_selected_cal_date.clear()        
        for item in self.cal_select_list:
            # Format display: if it contains ##, show the task name part
            if "##" in item:
                # Extract task name from path##task format
                path_part, task_name = item.split("##", 1)
                display_text = f"{path_part} (Task: {task_name})"
            else:
                display_text = item
            self.listwidget_selected_cal_date.addItem(display_text)

    def cal_select_folder_clicked(self, item):
        if self.cal_select_depth == 0:
            # Remove emoji prefix from patient name
            patient_name = item.text().replace("👤 ", "")
            
            # Entering date selection view
            self.cal_select_patient_id = patient_name
            self.btn_cal_back_path.setEnabled(True)
            self.cal_select_depth += 1
            self.cal_load_folders()
        elif self.cal_select_depth == 1:
            # Remove emoji prefix from date name
            date_name = item.text().replace("📅 ", "")
            
            # Entering task selection view
            self.cal_select_date_id = date_name
            self.cal_select_depth += 1
            self.cal_load_folders()
        elif self.cal_select_depth == 2:
            # In task view - tasks are handled by checkboxes, not double-click
            pass
            
    def cal_selected_folder_selcted(self, item):
        self.listwidget_selected_cal_date_item_selected = item.text()

    def cal_selected_folder_selcted_delete(self):
        self.cal_select_list.remove(self.listwidget_selected_cal_date_item_selected)
        self.cal_show_selected_folder()
        self.label_cal_selected_tasks.setText(f"Selected Tasks: {len(self.cal_select_list)}")    
    
    def arrange_files_to_inverse_analysis(self):
        """
        Arrange calculation results to Inverse_analysis_data folder with intelligent task completion
        Structure: Inverse_analysis_data/patient_name/task_name (selected tasks + auto-completion)
        """
        try:
            import shutil
            from pathlib import Path
            from datetime import datetime
            
            # Check if there are selected tasks
            if not self.cal_select_list:
                QMessageBox.warning(self, "Warning", "No tasks selected for arrangement!")
                return
            
            # Process selected tasks and build task selection map
            selected_tasks_map = {}  # {date_path: [task_names]}
            for selected_item in self.cal_select_list:
                if "##" in selected_item:
                    # Individual task selection (path##task)
                    date_path, task_name = selected_item.split("##", 1)
                    if date_path not in selected_tasks_map:
                        selected_tasks_map[date_path] = []
                    selected_tasks_map[date_path].append(task_name)
                else:
                    # Whole date selection (all tasks in date)
                    selected_tasks_map[selected_item] = None  # None means all tasks
            
            # Get current directory (project root)
            project_root = Path(self.current_directory)
            patient_data_path = project_root / "Patient_data"
            inverse_analysis_path = project_root / "Inverse_analysis_data"
            
            if not patient_data_path.exists():
                QMessageBox.warning(self, "Warning", "Patient_data folder not found!")
                return
            
            # Create Inverse_analysis_data folder if it doesn't exist
            inverse_analysis_path.mkdir(exist_ok=True)
            
            copied_count = 0
            completed_count = 0
            
            # Process each selected date path
            for date_path_str, required_tasks in selected_tasks_map.items():
                date_path = Path(date_path_str)
                
                # Extract patient name and date from path
                try:
                    patient_name = date_path.parts[-2]  # Patient_data/PATIENT/DATE
                    date_name = date_path.parts[-1]
                except IndexError:
                    continue
                
                # Create patient folder in inverse analysis
                target_patient_path = inverse_analysis_path / patient_name
                target_patient_path.mkdir(exist_ok=True)
                
                # Get all calculated folders for this date (sorted by time, newest first)
                calc_folders = []
                if date_path.exists():
                    for item in date_path.iterdir():
                        if item.is_dir() and item.name.endswith('_calculated'):
                            # Extract timestamp for sorting
                            folder_time_str = item.name.replace('_calculated', '')
                            try:
                                folder_time = datetime.strptime(folder_time_str, '%Y_%m_%d_%H_%M')
                                calc_folders.append((folder_time, item))
                            except ValueError:
                                # If parsing fails, add with current time
                                calc_folders.append((datetime.now(), item))
                
                # Sort by time, newest first
                calc_folders.sort(key=lambda x: x[0], reverse=True)
                
                if not calc_folders:
                    print(f"No calculated folders found for {date_path}")
                    continue
                
                # Determine which tasks to process
                if required_tasks is None:
                    # All tasks mode - get all available tasks from latest calculation
                    latest_calc_folder = calc_folders[0][1]
                    available_tasks = [f.name for f in latest_calc_folder.iterdir() 
                                     if f.is_dir() and f.name not in ['Apose']]
                    tasks_to_process = available_tasks
                else:
                    # Individual tasks mode
                    tasks_to_process = required_tasks
                
                print(f"Processing tasks for {patient_name}/{date_name}: {tasks_to_process}")
                
                # For each required task, find it in calculation folders (newest first)
                for task_name in tasks_to_process:
                    task_found = False
                    
                    for calc_time, calc_folder in calc_folders:
                        task_folder = calc_folder / task_name
                        if task_folder.exists():
                            # Found the task, copy it to inverse analysis
                            target_task_path = target_patient_path / task_name
                            
                            # Remove existing task folder if it exists
                            if target_task_path.exists():
                                shutil.rmtree(target_task_path)
                            target_task_path.mkdir(exist_ok=True)
                            
                            files_copied = False
                            
                            # Copy required folders
                            for folder_name in ["opensim", "videos", "EMG", "EEG"]:
                                src_folder = task_folder / folder_name
                                if src_folder.exists():
                                    dst_folder = target_task_path / folder_name
                                    shutil.copytree(src_folder, dst_folder)
                                    files_copied = True
                            
                            if files_copied:
                                copied_count += 1
                                calc_time_str = calc_time.strftime('%Y_%m_%d_%H_%M')
                                print(f"✅ Copied {task_name} from {calc_time_str}_calculated")
                                
                                # If this task was auto-completed from older calculation
                                if calc_folder != calc_folders[0][1]:
                                    completed_count += 1
                                    print(f"🔄 Auto-completed {task_name} from older calculation")
                                
                            task_found = True
                            break
                    
                    if not task_found:
                        print(f"❌ Task '{task_name}' not found in any calculation folders for {patient_name}/{date_name}")
            
            if copied_count > 0:
                completion_msg = ""
                if completed_count > 0:
                    completion_msg = f"\n🔄 Auto-completed {completed_count} task(s) from older calculations"
                
                QMessageBox.information(self, "Success", 
                    f"Successfully arranged {copied_count} task(s) to Inverse_analysis_data folder.\n\n"
                    f"✅ Tasks processed: {copied_count}\n"
                    f"📁 Structure: patient_name/task_name (with intelligent completion){completion_msg}\n\n"
                    f"📍 Location: {inverse_analysis_path}")
            else:
                QMessageBox.information(self, "Info", 
                    "No calculated results found to arrange.\n"
                    "Please run calculations first or check selected tasks.")
                    
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error arranging files: {str(e)}")
    
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
        # Update slider position based on video player progress
        slider_value = int(self.video_player.progress * 100)
        self.result_video_slider.blockSignals(True)  # Prevent signal loop
        self.result_video_slider.setValue(slider_value)
        self.result_video_slider.blockSignals(False)
        self.frame_label.setText(str(slider_value))

    def select_result_cal_task(self, patient_path, result_select_patient_id, result_select_date, result_select_cal_time, result_select_task):
        self.show_result_path = os.path.join(patient_path, result_select_patient_id, result_select_date, result_select_cal_time, result_select_task)
        print(f"[DataViewer] Loading result from: {self.show_result_path}")
        
        # Load video frames
        self.video_player.load_video(self.show_result_path)
        
        # Load gait analysis data
        self.video_player.result_load_gait_figures(self.show_result_path)
        
        # Force plot update - setCurrentIndex(0) won't fire signal if already at 0
        # So we explicitly call the plot function after setting the index
        self.combobox_result_select_gait_figures.blockSignals(True)
        self.combobox_result_select_gait_figures.setCurrentIndex(0)
        self.combobox_result_select_gait_figures.blockSignals(False)
        
        # Explicitly trigger the plot (since signal might not fire)
        self.result_select_gait_figures(0)
        
        # Load 3D model
        self.video_player.load_gltf_file_in_viewer(f"./Patient_data/{result_select_patient_id}/{result_select_date}/{result_select_cal_time}/{result_select_task}/model.gltf")
        
        self.playButton.setEnabled(True)
        self.result_video_slider.setEnabled(True)
        print(f"[DataViewer] Data loaded successfully")

    def result_disconnet_gait_figures(self):
        try:
            if self.result_current_gait_figures_index in (0, 1): # Hip flexion
                self.result_video_slider.valueChanged.disconnect(self.video_player.update_hip_flexion)
            elif self.result_current_gait_figures_index == 2: # Knee flexion
                self.result_video_slider.valueChanged.disconnect(self.video_player.update_knee_flexion)          
            elif self.result_current_gait_figures_index == 3: # Ankle flexion
                self.result_video_slider.valueChanged.disconnect(self.video_player.update_ankle_flexion)
            elif self.result_current_gait_figures_index == 4: # Speed
                self.result_video_slider.valueChanged.disconnect(self.video_player.update_speed)
            elif self.result_current_gait_figures_index == 5: # Stride
                self.result_video_slider.valueChanged.disconnect(self.video_player.update_stride)
        except TypeError:
            # Signal was not connected, ignore
            pass

    def result_select_gait_figures(self, index):
        if index == 0: # Default - show Hip flexion (same as index 1)
            self.video_player.hip_flexion_plot()
            self.result_video_slider.valueChanged.connect(self.video_player.update_hip_flexion)
            self.result_current_gait_figures_index = 1
        elif index == 1: # Hip flexion
            self.video_player.hip_flexion_plot()
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
            # Stop both 3D animation and video timer
            self.result_web_view_widget.page().runJavaScript("stopAnimation();")
            self.video_player.timer.stop()
            self.is_playing = False
            print("Playback stopped")
        else:
            # Start both 3D animation and video timer
            self.result_web_view_widget.page().runJavaScript("startAnimation();")
            self.video_player.timer.start(33)  # ~30 FPS for smoother sync
            self.is_playing = True
            print("Playback started")

    def slider_changed(self, value):
        # Pause playback when user drags slider
        was_playing = self.is_playing
        if self.is_playing:
            self.play_stop()
            
        # Update progress and sync all components
        self.video_player.progress = value / 100
        self.video_player.slider_changed()
        self.result_video_slider.setValue(value)
        self.frame_label.setText(str(value))
        
        # Resume playback if it was playing before
        # Note: Don't auto-resume to give user control
        print(f"Slider changed to: {value}%")

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