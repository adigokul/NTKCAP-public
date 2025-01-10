from multiprocessing import Process, shared_memory
import numpy as np
import cv2, time, os
from mmdeploy_runtime import PoseTracker

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
# !!!!
det_model_path = os.path.join(os.getcwd(),"NTK_CAP", "ThirdParty", "mmdeploy", "rtmpose-trt", "rtmdet-m")
pose_model_path = os.path.join(os.getcwd(),"NTK_CAP", "ThirdParty", "mmdeploy", "rtmpose-trt", "rtmpose-m")
device ="cuda"

sigmas = VISUALIZATION_CFG['halpe26']['sigmas']
palette = VISUALIZATION_CFG['halpe26']['palette']
skeleton = VISUALIZATION_CFG['halpe26']['skeleton']
link_color = VISUALIZATION_CFG['halpe26']['link_color']
point_color = VISUALIZATION_CFG['halpe26']['point_color']



class CameraProcess(Process):
    def __init__(self, shared_name, shared_dict, cam_id, start_evt, rec_evt, stop_evt, stop_rec_evt, queue, *args, **kwargs):
        super().__init__()
        ### define
        self.shared_name = shared_name
        self.shared_dict = shared_dict
        self.cam_id = cam_id
        self.start_evt = start_evt
        self.rec_evt = rec_evt
        self.stop_evt = stop_evt
        self.stop_rec_evt = stop_rec_evt
        self.queue = queue
        self.frame_count = 0
        self.buffer_length = 4
        self.delay_time = 0.2

        ### initialization
        self.recording = False
        self.start_time = None
        
        ### calculate parameter
        # intrinsic
        # mtx = np.array([[774.534, 0.000, 956.895], [0.000, 773.439, 542.879], [0.000, 0.000, 1.000]])
        # dist = np.array([0.015, -0.014, 0.006, -0.003, 0.000])
        width = 1920
        height = 1080
        # newcameramtx, self.roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (width,height), 1, (width,height))
        # self.mapx, self.mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (width,height), 5)

    def run(self):
        # frame shape, 3 for rgb
        shape = (1080, 1920, 3)
        # shape = (449, 611, 3)
        # mapx = self.mapx
        # mapy = self.mapy
        # x, y, w, h = self.roi

        # shared memory setup
        idx = 0 # the current index of frames in shared memory
        existing_shm = shared_memory.SharedMemory(name=self.shared_name)
        shared_array = np.ndarray((self.buffer_length,) + shape, dtype=np.uint8, buffer=existing_shm.buf)
        # shared_array = np.ndarray(shape, dtype=np.uint8, buffer=existing_shm.buf)
        
        np.set_printoptions(precision=4, suppress=True)
        tracker = PoseTracker(det_model=det_model_path,pose_model=pose_model_path,device_name=device)
        state = tracker.create_state(det_interval=1, det_min_bbox_size=100, keypoint_sigmas=sigmas)
        
        cap = cv2.VideoCapture(self.cam_id)
        (width, height) = (1920, 1080)
        # (width, height) = (640, 480)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.start_evt.wait()
        
        while True:
            cap_s = time.perf_counter()
            ret, frame = cap.read()
            cap_e = time.perf_counter()
            if not ret or self.stop_evt.is_set():
                break
            
            # dst = self.process_frame(frame, mapx, mapy, x, y, w, h)
            keypoints, bboxes = tracker(state, frame, detect=-1)[:2]
            # results = tracker(state, dst, detect=-1)
            # keypoints, bboxes, _ = results
            scores = keypoints[..., 2]
            keypoints = np.round(keypoints[..., :2], 3)

            self.draw_frame(frame, keypoints, scores, palette, skeleton, link_color, point_color)
            np.copyto(shared_array[idx,:], frame)
            self.queue.put(idx)
            idx = (idx+1) % self.buffer_length
            
            self.process_recording(cap_s, cap_e)

        cap.release()

    def process_recording(self, cap_s, cap_e):

        if self.stop_rec_evt.is_set():
            self.recording = False
            self.start_time = None
            self.frame_count = 0
            
        if self.rec_evt.is_set():
            self.start_time = self.start_time or cap_s
            cap_s -= self.start_time
            cap_e -= self.start_time
            average_cap_time = round((cap_e + cap_s) / 2, 4)

            self.shared_dict[self.frame_count] = {
                'time': average_cap_time
            }
            self.frame_count += 1
    # intrinsic
    # def process_frame(self, frame, mapx, mapy, x, y, w, h):
    #     dst = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
    #     return dst[y:y+h, x:x+w]
    
    def draw_frame(self, frame, keypoints, scores, palette, skeleton, link_color, point_color):
        keypoints = keypoints.astype(int)
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