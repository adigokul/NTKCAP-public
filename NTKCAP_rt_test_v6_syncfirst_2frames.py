from NTK_CAP.script_py.NTK_Cap import *
from check_extrinsic import *
from PyQt6.QtWidgets import *
from PyQt6 import uic
from PyQt6.QtGui import *
from PyQt6.QtCore import *
from multiprocessing import Event, shared_memory, Queue, Array
import sys
import os
import time
import numpy as np
from RT.Tracker_Process import Tracker_Process
from RT.PreTri_Process import PreTri_Process
from RT.SyncProcess import SyncProcess
from RT.CameraProcess import CameraProcess
from RT.UpdateThread import UpdateThread

# os.environ["CUDA_LAUNCH_BLOCKING"] = "10"
'''
找問題 : # !!!!
version : cap.read -> sync -> tracker -> update label + triangulation
'''
            
class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        uic.loadUi("RT/NTKCAP_rt_test.ui", self)
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
        self.sync_frame_queue = [Queue() for _ in range(2)]
        self.tracker_queue = [Queue() for _ in range(4)]
        self.draw_frame_queue = [Queue() for _ in range(4)]
        self.sync_tracker_queue = [Queue() for _ in range(2)]
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
            tracker_sync_shm = shared_memory.SharedMemory(create=True, size=int(np.prod(self.tracker_sync_shm_shape) * np.dtype(np.float32).itemsize * self.buffer_length))
            self.tracker_sync_shm_lst.append(tracker_sync_shm)
            frame_show_shm = shared_memory.SharedMemory(create=True, size=int(np.prod(self.camera_shm_shape) * np.dtype(np.uint8).itemsize * self.buffer_length))
            self.sync_frame_show_shm_lst.append(frame_show_shm)
            sync_frame_shm = shared_memory.SharedMemory(create=True, size=int(np.prod(self.camera_shm_shape) * np.dtype(np.uint8).itemsize * self.buffer_length))
            self.sync_frame_shm_lst.append(sync_frame_shm)
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

        tracker1 = Tracker_Process(
            0,
            self.sync_frame_shm_lst[0].name,            
            self.sync_frame_shm_lst[1].name,
            self.sync_frame_queue[0],
            self.tracker_sync_shm_lst[0].name,
            self.tracker_sync_shm_lst[1].name,
            self.tracker_queue[0],
            self.sync_frame_show_shm_lst[0].name,
            self.draw_frame_queue[0],
            self.sync_frame_show_shm_lst[1].name,
            self.draw_frame_queue[1],
            self.start_camera_evt,
            self.stop_camera_evt
        )
        tracker2 = Tracker_Process(
            1,
            self.sync_frame_shm_lst[2].name,            
            self.sync_frame_shm_lst[3].name,
            self.sync_frame_queue[1],
            self.tracker_sync_shm_lst[2].name,
            self.tracker_sync_shm_lst[3].name,
            self.tracker_queue[1],
            self.sync_frame_show_shm_lst[2].name,
            self.draw_frame_queue[2],
            self.sync_frame_show_shm_lst[3].name,
            self.draw_frame_queue[3],
            self.start_camera_evt,
            self.stop_camera_evt
        )
        self.tracker_sync_proc_lst.append(tracker1)
        self.tracker_sync_proc_lst.append(tracker2)

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
        for process in self.tracker_sync_proc_lst:
            process.start()
        self.pre_tri_process.start()
        for thread in self.threads:
            thread.start()
        self.start_camera_evt.set()
    def CloseCamera(self):
        if not self.camera_opened: return
        self.start_camera_evt.clear()
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
        for queue in self.sync_tracker_queue:
            queue.close() # 1 send to triangulation queue
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
        for process in self.tracker_sync_proc_lst:
            if process.is_alive():
                process.terminate()
                process.join()
        
        self.time_sync_process.terminate() # 1 time stamp sync process
        self.pre_tri_process.terminate() # 1 preparation for triangulation process
        # self.trangulation.terminate()
        # clear list for 4 camera processes & 4 tracker processes
        self.camera_proc_lst.clear() # 4 camera processes
        if hasattr(self, 'manager'):
            self.manager.shutdown()
    def closeEvent(self, event):
        self.CloseCamera()
if __name__ == "__main__":
    App = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(App.exec())