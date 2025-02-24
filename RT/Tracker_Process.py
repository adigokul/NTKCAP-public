from check_extrinsic import *
from multiprocessing import Event, shared_memory, Manager, Queue, Array, Lock, Process
from mmdeploy_runtime import PoseTracker
import os
import cv2
import time
import numpy as np
import copy
VISUALIZATION_CFG = dict(
    halpe26=dict(
        sigmas=[
            0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072,
            0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089, 0.026,
            0.026, 0.066, 0.079, 0.079, 0.079, 0.079, 0.079, 0.079
        ]))

det_model_path = os.path.join(os.getcwd(),"NTK_CAP", "ThirdParty", "mmdeploy", "rtmpose-trt", "rtmdet-nano")
pose_model_path = os.path.join(os.getcwd(),"NTK_CAP", "ThirdParty", "mmdeploy", "rtmpose-trt", "rtmpose-t")
device ="cuda"

sigmas = VISUALIZATION_CFG['halpe26']['sigmas']

class Tracker_Process(Process):
    def __init__(self, group_id, sync_frame_shm_name, sync_frame_queue, sync_tracker_shm_name1, sync_tracker_queue1, sync_tracker_shm_name2, sync_tracker_queue2, sync_frame_show_shm_name1, draw_frame_queue1, sync_frame_show_shm_name2, draw_frame_queue2, sync_tracker_show_shm_name1, sync_tracker_show_shm_name2, start_evt, stop_evt, *args, **kwargs):
        super().__init__()
        self.group_id = group_id

        self.sync_frame_shm_name = sync_frame_shm_name
        self.sync_frame_queue = sync_frame_queue

        self.sync_tracker_shm_name1 = sync_tracker_shm_name1
        self.sync_tracker_queue1 = sync_tracker_queue1
        self.sync_tracker_shm_name2 = sync_tracker_shm_name2
        self.sync_tracker_queue2 = sync_tracker_queue2

        self.sync_frame_show_shm_name1 = sync_frame_show_shm_name1
        self.draw_frame_queue1 = draw_frame_queue1
        self.sync_frame_show_shm_name2 = sync_frame_show_shm_name2
        self.draw_frame_queue2 = draw_frame_queue2

        self.sync_tracker_show_shm_name1 = sync_tracker_show_shm_name1
        self.sync_tracker_show_shm_name2 = sync_tracker_show_shm_name2

        self.start_evt = start_evt
        self.stop_evt = stop_evt

        self.frame_shm_shape = (1080, 1920, 3)
        self.sync_2frames_shm_shape = (2, 1080, 1920, 3)
        self.buffer_length = 20
        self.tracker_shm_shape = (1, 26, 3)
    def run(self):
        # RTM model
        tracker = PoseTracker(det_model=det_model_path,pose_model=pose_model_path,device_name=device)
        state = tracker.create_state(det_interval=1, det_min_bbox_size=100, keypoint_sigmas=sigmas, pose_max_num_bboxes=1)

        existing_shm_frame = shared_memory.SharedMemory(name=self.sync_frame_shm_name)
        shared_array_frame = np.ndarray((self.buffer_length,) + self.sync_2frames_shm_shape, dtype=np.uint8, buffer=existing_shm_frame.buf)
        
        existing_shm_tracker1 = shared_memory.SharedMemory(name=self.sync_tracker_shm_name1)
        shared_array_tracker1 = np.ndarray((self.buffer_length,) + self.tracker_shm_shape, dtype=np.float32, buffer=existing_shm_tracker1.buf)
        existing_shm_tracker2 = shared_memory.SharedMemory(name=self.sync_tracker_shm_name2)
        shared_array_tracker2 = np.ndarray((self.buffer_length,) + self.tracker_shm_shape, dtype=np.float32, buffer=existing_shm_tracker2.buf)

        existing_shm_frame_show1 = shared_memory.SharedMemory(name=self.sync_frame_show_shm_name1)
        shared_array_frame_show1 = np.ndarray((self.buffer_length,) + self.frame_shm_shape, dtype=np.uint8, buffer=existing_shm_frame_show1.buf)
        existing_shm_frame_show2 = shared_memory.SharedMemory(name=self.sync_frame_show_shm_name2)
        shared_array_frame_show2 = np.ndarray((self.buffer_length,) + self.frame_shm_shape, dtype=np.uint8, buffer=existing_shm_frame_show2.buf)
        keypoints_template = np.full((1, 26, 3), np.nan, dtype=np.float32)
        
        existing_shm_tracker_show1 = shared_memory.SharedMemory(name=self.sync_tracker_show_shm_name1)
        shared_array_tracker_show1 = np.ndarray((self.buffer_length,) + self.tracker_shm_shape, dtype=np.float32, buffer=existing_shm_tracker_show1.buf)
        existing_shm_tracker_show2 = shared_memory.SharedMemory(name=self.sync_tracker_show_shm_name2)
        shared_array_tracker_show2 = np.ndarray((self.buffer_length,) + self.tracker_shm_shape, dtype=np.float32, buffer=existing_shm_tracker_show2.buf)
        
        idx, idx_show = 0, 0
        self.start_evt.wait()
        while self.start_evt.is_set():
            try:
                idx_get = self.sync_frame_queue.get(timeout=0.01)
            except:
                continue
            t1 = time.time()
            frames = shared_array_frame[idx_get, : ]
            frame1 = frames[0]
            frame2 = frames[1]
            keypoints1, _ = tracker(state, frame1, detect=-1)[:2]
            keypoints2, _ = tracker(state, frame2, detect=-1)[:2]
            if keypoints1.shape == (0, 0, 3):
                keypoints1 = copy.deepcopy(keypoints_template)
            if keypoints2.shape == (0, 0, 3):
                keypoints2 = copy.deepcopy(keypoints_template)
            np.copyto(shared_array_tracker1[idx, :], keypoints1)
            np.copyto(shared_array_tracker2[idx, :], keypoints2)
            # print("Tracker", self.cam_id, count)
            self.sync_tracker_queue1.put(idx)
            self.sync_tracker_queue2.put(idx)
            idx = (idx+1) % self.buffer_length
            
            np.copyto(shared_array_frame_show1[idx_show,:], frame1)
            np.copyto(shared_array_frame_show2[idx_show,:], frame2)
            np.copyto(shared_array_tracker_show1[idx_show, :], keypoints1)
            np.copyto(shared_array_tracker_show2[idx_show, :], keypoints2)
    
            self.draw_frame_queue1.put(idx_show)
            self.draw_frame_queue2.put(idx_show)
            
            idx_show = (idx_show+1) % self.buffer_length
