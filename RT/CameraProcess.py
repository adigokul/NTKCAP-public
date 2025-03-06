from multiprocessing import Event, shared_memory, Manager, Queue, Array, Lock, Process
import cv2
import time
import numpy as np

class CameraProcess(Process):
    def __init__(self, cam_id, camera_shm, time_stamp_array, camera_queue, unify_start_time, start_cam_evt, stop_cam_evt, *args, **kwargs):
        super().__init__()

        self.cam_id = cam_id
        self.cam_shm_name = camera_shm
        self.time_stamp_array = time_stamp_array

        self.cam_q = camera_queue
        self.start_evt = start_cam_evt
        self.stop_evt = stop_cam_evt
        self.buffer_length = 20
        self.start_time = unify_start_time
        # Camera
        self.frame_width = 1920
        self.frame_height = 1080
        self.fps = 30
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.cam_shm_shape = (1080, 1920, 3)
        self.frame_id = 0

    def run(self):
        idx = 0
        existing_shm_frame = shared_memory.SharedMemory(name=self.cam_shm_name)
        shared_array_frame = np.ndarray((self.buffer_length,) + self.cam_shm_shape, dtype=np.uint8, buffer=existing_shm_frame.buf)
        
        cap = cv2.VideoCapture(self.cam_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        self.start_evt.wait()
        _, _ = cap.read()
        _, _ = cap.read()

        while self.start_evt.is_set():
            cap_s = time.time() - self.start_time
            _, frame = cap.read()
            cap_e = time.time() - self.start_time
            cal_time = (cap_e + cap_s) * 500

            np.copyto(shared_array_frame[idx,:], frame)
            self.time_stamp_array[idx] = cal_time
            self.cam_q.put(idx)
            idx = (idx+1) % self.buffer_length