import os
import sys
from PyQt6 import uic
from PyQt6.QtWidgets import QLabel, QPushButton, QListWidget, QMainWindow, QApplication
from PyQt6.QtGui import QIcon
import cv2
class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        uic.loadUi('GUI_source/NTKCAP_GUI_Boundary.ui', self)
        self.setWindowTitle("Boundary")
        self.patient_id_path = os.path.join(os.getcwd(), "Patient_data", "multi_person")

        self.listWidget_record_boundary_tasks = self.findChild(QListWidget, "listWidget_record_boundary_tasks")
        self.label_camera1 = self.findChild(QLabel, "label_camera1")
        self.label_camera2 = self.findChild(QLabel, "label_camera2")
        self.label_camera3 = self.findChild(QLabel, "label_camera3")
        self.label_camera4 = self.findChild(QLabel, "label_camera4")
        self.btn_camera1 = self.findChild(QPushButton, "btn_camera1")
        self.btn_camera2 = self.findChild(QPushButton, "btn_camera2")
        self.btn_camera3 = self.findChild(QPushButton, "btn_camera3")
        self.btn_camera4 = self.findChild(QPushButton, "btn_camera4")
        self.btn_save_boundary = self.findChild(QPushButton, "btn_save_boundary")
        self.task_boundary_needed_list = []
        self.task_boundary_needed()
        self.listWidget_record_boundary_tasks_show()
    def task_boundary_needed(self):
        for date in os.listdir(self.patient_id_path):
            raw_data_path = os.path.join(self.patient_id_path, date, "raw_data")
            for task in os.listdir(raw_data_path):
                if task == 'calibration':
                    continue
                else:
                    boundary_path = os.path.join(raw_data_path, task, 'name', 'Boundary.txt')
                    if not os.path.exists(boundary_path):
                        self.task_boundary_needed_list.append(boundary_path)
    def listWidget_record_boundary_tasks_show(self):
        if self.task_boundary_needed_list == []: return
        self.listWidget_record_boundary_tasks.clear()
        for task in self.task_boundary_needed_list:
            self.listWidget_record_boundary_tasks.addItems(task)
if __name__ == "__main__":
    App = QApplication(sys.argv)
    App.setWindowIcon(QIcon("GUI_source/Team_logo.jpg"))
    window = MainWindow()
    window.show()
    sys.exit(App.exec())