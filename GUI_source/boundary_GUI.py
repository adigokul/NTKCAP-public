import os
import sys
from PyQt6 import uic
from PyQt6.QtWidgets import QLabel, QPushButton, QListWidget, QMainWindow, QApplication, QMessageBox
from PyQt6.QtGui import QIcon, QPixmap, QImage
from PyQt6.QtCore import Qt
import cv2

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        uic.loadUi('GUI_source/NTKCAP_GUI_Boundary.ui', self)
        self.setWindowTitle("Boundary")
        self.patient_id_path = os.path.join(os.getcwd(), "Patient_data", "multi_person")

        self.listWidget_record_boundary_tasks = self.findChild(QListWidget, "listWidget_record_boundary_tasks")
        self.label_camera = [self.findChild(QLabel, "label_camera1"),
                                self.findChild(QLabel, "label_camera2"),
                                self.findChild(QLabel, "label_camera3"),
                                self.findChild(QLabel, "label_camera4")]
        self.camera_selected = []
        self.btn_camera1 = self.findChild(QPushButton, "btn_camera1")
        self.btn_camera2 = self.findChild(QPushButton, "btn_camera2")
        self.btn_camera3 = self.findChild(QPushButton, "btn_camera3")
        self.btn_camera4 = self.findChild(QPushButton, "btn_camera4")
        self.btn_camera1.clicked.connect(self.select_camera1)
        self.btn_camera2.clicked.connect(self.select_camera2)
        self.btn_camera3.clicked.connect(self.select_camera3)
        self.btn_camera4.clicked.connect(self.select_camera4)
        self.btn_camera1.setEnabled(False)
        self.btn_camera2.setEnabled(False)
        self.btn_camera3.setEnabled(False)
        self.btn_camera4.setEnabled(False)

        self.btn_save_boundary = self.findChild(QPushButton, "btn_save_boundary")
        self.btn_save_boundary.clicked.connect(self.save)
        self.btn_save_boundary.setEnabled(False)
        self.listWidget_record_boundary_tasks = self.findChild(QListWidget, "listWidget_record_boundary_tasks")
        self.listWidget_record_boundary_tasks.itemClicked.connect(self.listWidget_record_boundary_tasks_select)
        self.task_boundary_needed_list = []
        self.boundary_task_selected = None
        self.camera_selected = []
        self.task_boundary_needed()
        self.listWidget_record_boundary_tasks_show()

        self.btn_reset = self.findChild(QPushButton, "btn_reset")
        self.btn_reset.clicked.connect(self.Reset)
    def Reset(self):
        for idx, label in enumerate(self.label_camera):
            label.clear()
            label.setText(f"Camera{idx+1}")
        self.btn_camera1.setChecked(False)
        self.btn_camera2.setChecked(False)
        self.btn_camera3.setChecked(False)
        self.btn_camera4.setChecked(False)
        self.btn_camera1.setEnabled(False)
        self.btn_camera2.setEnabled(False)
        self.btn_camera3.setEnabled(False)
        self.btn_camera4.setEnabled(False)
        self.btn_save_boundary.setEnabled(False)
        self.boundary_task_selected = None
        self.camera_selected = []
        self.task_boundary_needed()
        self.listWidget_record_boundary_tasks_show()

    def save(self):
        if len(self.camera_selected) != 2:
            QMessageBox.information(self, "Have not finished selecting", "The number of selected cameras must be two!")
            return
        self.btn_camera1.setChecked(False)
        self.btn_camera2.setChecked(False)
        self.btn_camera3.setChecked(False)
        self.btn_camera4.setChecked(False)
        self.btn_camera1.setEnabled(False)
        self.btn_camera1.setCheckable(False)
        self.btn_camera2.setEnabled(False)
        self.btn_camera2.setCheckable(False)
        self.btn_camera3.setEnabled(False)
        self.btn_camera3.setCheckable(False)
        self.btn_camera4.setEnabled(False)
        self.btn_camera4.setCheckable(False)
        self.btn_save_boundary.setEnabled(False)
        with open(os.path.join(self.patient_id_path, self.boundary_task_selected, 'name', 'Boundary.txt'), "w", encoding="utf-8") as f:
                for cam_id in self.camera_selected:
                    f.write(f"{int(cam_id)}\n")
        self.camera_selected = []
        self.boundary_task_selected = None
        self.task_boundary_needed()
        self.listWidget_record_boundary_tasks_show()
        for idx, label in enumerate(self.label_camera):
            label.clear()
            label.setText(f"Camera{idx+1}")

    def select_camera1(self):
        if '1' in self.camera_selected:
            self.btn_checkable_controller()
            return
        if len(self.camera_selected) < 2:
            self.camera_selected.append('1')
        else:
            self.camera_selected.pop(0)
            self.camera_selected.append('1')
        self.btn_checkable_controller()

    def select_camera2(self):
        if '2' in self.camera_selected:
            self.btn_checkable_controller()
            return
        if len(self.camera_selected) < 2:
            self.camera_selected.append('2')
        else:
            self.camera_selected.pop(0)
            self.camera_selected.append('2')
        self.btn_checkable_controller()

    def select_camera3(self):
        if '3' in self.camera_selected:
            self.btn_checkable_controller()
            return
        if len(self.camera_selected) < 2:
            self.camera_selected.append('3')
        else:
            self.camera_selected.pop(0)
            self.camera_selected.append('3')
        self.btn_checkable_controller()

    def select_camera4(self):        
        if '4' in self.camera_selected:
            self.btn_checkable_controller()
            return
        if len(self.camera_selected) < 2:
            self.camera_selected.append('4')
        else:
            self.camera_selected.pop(0)
            self.camera_selected.append('4')
        self.btn_checkable_controller()

    def btn_checkable_controller(self):
        self.btn_camera1.setChecked(False)
        self.btn_camera2.setChecked(False)
        self.btn_camera3.setChecked(False)
        self.btn_camera4.setChecked(False)
        if '1' in self.camera_selected:
            self.btn_camera1.setChecked(True)
        if '2' in self.camera_selected:
            self.btn_camera2.setChecked(True)
        if '3' in self.camera_selected:
            self.btn_camera3.setChecked(True)
        if '4' in self.camera_selected:
            self.btn_camera4.setChecked(True)

    def listWidget_record_boundary_tasks_select(self, item):
        path_selected = item.text()
        self.btn_camera1.setEnabled(True)
        self.btn_camera1.setCheckable(True)
        self.btn_camera2.setEnabled(True)
        self.btn_camera2.setCheckable(True)
        self.btn_camera3.setEnabled(True)
        self.btn_camera3.setCheckable(True)
        self.btn_camera4.setEnabled(True)
        self.btn_camera4.setCheckable(True)
        self.btn_save_boundary.setEnabled(True)
        self.boundary_task_selected = path_selected
        self.boundary_tasks_video_frame_label_show(path_selected)

    def task_boundary_needed(self):
        self.task_boundary_needed_list.clear()
        for date in os.listdir(self.patient_id_path):
            raw_data_path = os.path.join(self.patient_id_path, date, "raw_data")
            for task in os.listdir(raw_data_path):
                if task == 'calibration':
                    continue
                else:
                    boundary_path = os.path.join(raw_data_path, task, 'name', 'Boundary.txt')
                    if not os.path.exists(boundary_path):
                        self.task_boundary_needed_list.append(os.path.join(date, "raw_data", task))
    
    def listWidget_record_boundary_tasks_dclick(self, item):
        task = item.text()
        self.boundary_tasks_video_frame_label_show(task)
    
    def boundary_tasks_video_frame_label_show(self, task_path):
        videos_path = os.path.join(self.patient_id_path, task_path, 'videos')
        video_path = [os.path.join(videos_path, '1.mp4'), os.path.join(videos_path, '2.mp4'), os.path.join(videos_path, '3.mp4'), os.path.join(videos_path, '4.mp4')]
        for (video, label) in zip(video_path, self.label_camera):
            label.clear()
            cap = cv2.VideoCapture(video)
            _, frame = cap.read()
            cap.release()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            
            bytes_per_line = ch * w
            qimg = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            pixmap_resized = pixmap.scaled(
                self.label_camera[0].size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            label.setPixmap(pixmap_resized)
            cap.release()

    def listWidget_record_boundary_tasks_show(self):
        self.listWidget_record_boundary_tasks.clear()
        if self.task_boundary_needed_list == []: return        
        for task in self.task_boundary_needed_list:
            self.listWidget_record_boundary_tasks.addItem(task)

if __name__ == "__main__":
    App = QApplication(sys.argv)
    App.setWindowIcon(QIcon("GUI_source/Team_logo.jpg"))
    window = MainWindow()
    window.show()
    sys.exit(App.exec())