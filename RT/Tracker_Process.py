from multiprocessing import Event, shared_memory, Manager, Queue, Array, Lock, Process
from mmdeploy_runtime import PoseTracker
import os
import cv2
import time
import numpy as np
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

det_model_path = os.path.join(os.getcwd(),"NTK_CAP", "ThirdParty", "mmdeploy", "rtmpose-trt", "rtmdet-nano")
pose_model_path = os.path.join(os.getcwd(),"NTK_CAP", "ThirdParty", "mmdeploy", "rtmpose-trt", "rtmpose-m")
device ="cuda"

sigmas = VISUALIZATION_CFG['halpe26']['sigmas']
palette = VISUALIZATION_CFG['halpe26']['palette']
skeleton = VISUALIZATION_CFG['halpe26']['skeleton']
link_color = VISUALIZATION_CFG['halpe26']['link_color']
point_color = VISUALIZATION_CFG['halpe26']['point_color']
class Tracker_Process(Process):
    def __init__(self, cam_id, sync_frame_shm_name1, sync_frame_shm_name2, sync_frame_queue, tracker_sync_shm_name1, tracker_sync_shm_name2, tracker_queue, sync_frame_show_shm_name1, draw_frame_queue1, sync_frame_show_shm_name2, draw_frame_queue2, start_camera_evt, stop_camera_evt, *args, **kwargs):
        super().__init__()
        self.cam_id = cam_id

        self.sync_frame_shm_name1 = sync_frame_shm_name1
        self.sync_frame_shm_name2 = sync_frame_shm_name2
        self.sync_frame_queue = sync_frame_queue
        self.tracker_sync_shm_name1 = tracker_sync_shm_name1
        self.tracker_sync_shm_name2 = tracker_sync_shm_name2
        self.tracker_queue = tracker_queue
        self.sync_frame_show_shm_name1 = sync_frame_show_shm_name1
        self.draw_frame_queue1 = draw_frame_queue1
        self.sync_frame_show_shm_name2 = sync_frame_show_shm_name2
        self.draw_frame_queue2 = draw_frame_queue2
        self.frame_shm_shape = (1080, 1920, 3)
        self.buffer_length = 20
        self.tracker_shm_shape = (1, 26, 3)
    def run(self):
        # RTM model
        tracker = PoseTracker(det_model=det_model_path,pose_model=pose_model_path,device_name=device)
        state = tracker.create_state(det_interval=1, det_min_bbox_size=100, keypoint_sigmas=sigmas, pose_max_num_bboxes=1)
    
        existing_shm_frame1 = shared_memory.SharedMemory(name=self.sync_frame_shm_name1)
        shared_array_frame1 = np.ndarray((self.buffer_length,) + self.frame_shm_shape, dtype=np.uint8, buffer=existing_shm_frame1.buf)
        existing_shm_frame2 = shared_memory.SharedMemory(name=self.sync_frame_shm_name2)
        shared_array_frame2 = np.ndarray((self.buffer_length,) + self.frame_shm_shape, dtype=np.uint8, buffer=existing_shm_frame2.buf)
        existing_shm_tracker1 = shared_memory.SharedMemory(name=self.tracker_sync_shm_name1)
        shared_array_tracker1 = np.ndarray((self.buffer_length,) + self.tracker_shm_shape, dtype=np.float32, buffer=existing_shm_tracker1.buf)
        existing_shm_tracker2 = shared_memory.SharedMemory(name=self.tracker_sync_shm_name2)
        shared_array_tracker2 = np.ndarray((self.buffer_length,) + self.tracker_shm_shape, dtype=np.float32, buffer=existing_shm_tracker2.buf)
        keypoints_template = np.full((1, 26, 3), np.nan, dtype=np.float32)
        existing_shm_draw_frame1 = shared_memory.SharedMemory(name=self.sync_frame_show_shm_name1)
        shared_array_draw_frame1 = np.ndarray((self.buffer_length,) + self.frame_shm_shape, dtype=np.uint8, buffer=existing_shm_draw_frame1.buf)
        existing_shm_draw_frame2 = shared_memory.SharedMemory(name=self.sync_frame_show_shm_name2)
        shared_array_draw_frame2 = np.ndarray((self.buffer_length,) + self.frame_shm_shape, dtype=np.uint8, buffer=existing_shm_draw_frame2.buf)
        count = 0
        idx, idx_show = 0, 0

        while True:
            t1 = time.time()
            try:
                idx_get = self.sync_frame_queue.get(timeout=0.01)
            except:
                continue
            
            frame1 = shared_array_frame1[idx_get, : ]
            frame2 = shared_array_frame2[idx_get, : ]
            keypoints1, _ = tracker(state, frame1, detect=-1)[:2]
            keypoints2, _ = tracker(state, frame2, detect=-1)[:2]
            if keypoints1.shape == (0, 0, 3):
                np.copyto(shared_array_tracker1[idx,:], keypoints_template)
            else:
                np.copyto(shared_array_tracker1[idx,:], keypoints1)
            if keypoints2.shape == (0, 0, 3):
                np.copyto(shared_array_tracker2[idx,:], keypoints_template)
            else:
                np.copyto(shared_array_tracker2[idx,:], keypoints2)
            self.tracker_queue.put(idx)
            
            idx = (idx+1) % self.buffer_length
            
            t2 = time.time()
            # print("tracker", self.cam_id, count)
            scores1 = keypoints1[..., 2]
            keypoints1 = np.round(keypoints1[..., :2], 3)
            # self.draw_frame(frame1, keypoints1, scores1, palette, skeleton, link_color, point_color)
            np.copyto(shared_array_draw_frame1[idx_show,:], frame1)
            self.draw_frame_queue1.put(idx_show)
            scores2 = keypoints2[..., 2]
            keypoints2 = np.round(keypoints2[..., :2], 3)
            # self.draw_frame(frame2, keypoints2, scores2, palette, skeleton, link_color, point_color)
            np.copyto(shared_array_draw_frame2[idx_show,:], frame2)
            self.draw_frame_queue2.put(idx_show)
            
            # scores3 = keypoints3[..., 2]
            # keypoints3 = np.round(keypoints3[..., :2], 3)
            # self.draw_frame(frame3, keypoints3, scores3, palette, skeleton, link_color, point_color)
            # np.copyto(shared_array_frame_show3[idx_show,:], frame3)

            # scores4 = keypoints4[..., 2]
            # keypoints4 = np.round(keypoints4[..., :2], 3)
            # self.draw_frame(frame4, keypoints4, scores4, palette, skeleton, link_color, point_color)
            # np.copyto(shared_array_frame_show4[idx_show,:], frame4)

            # self.draw_frame_queue1.put(idx_show)
            # self.draw_frame_queue2.put(idx_show)
            # self.draw_frame_queue3.put(idx_show)
            # self.draw_frame_queue4.put(idx_show)
            
            idx_show = (idx_show+1) % self.buffer_length
            # if keypoints1.shape != (0, 0, 3) and keypoints2.shape != (0, 0, 3) and keypoints3.shape != (0, 0, 3) and keypoints4.shape != (0, 0, 3):
            #     print(time.time()-t1)
            
            count += 1
    
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