import os
import cv2
import time
import math
import numpy as np
from multiprocessing import Event, shared_memory, Manager, Queue, Array, Lock, Process

# Try to import backends with fallback
MMDEPLOY_AVAILABLE = False
RTMLIB_AVAILABLE = False

try:
    from mmdeploy_runtime import PoseTracker as MMDeployPoseTracker
    MMDEPLOY_AVAILABLE = True
except ImportError:
    pass

try:
    from rtmlib import BodyWithFeet
    RTMLIB_AVAILABLE = True
except ImportError:
    try:
        from rtmlib import Body
        RTMLIB_AVAILABLE = True
    except ImportError:
        pass

# Optional cupy (not required)
try:
    import cupy as cp
except ImportError:
    cp = None
VISUALIZATION_CFG = dict(
    coco=dict(
        skeleton=[(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11),
                  (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2),
                  (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)],
        palette=[(255, 128, 0), (255, 153, 51), (255, 178, 102), (230, 230, 0),
                 (255, 153, 255), (153, 204, 255), (255, 102, 255),
                 (255, 51, 255), (102, 178, 255), (51, 153, 255),
                 (255, 153, 153), (255, 102, 102), (255, 51, 51),
                 (153, 255, 153), (102, 255, 102), (51, 255, 51), (0, 255, 0),
                 (0, 0, 255), (255, 0, 0), (255, 255, 255)],
        link_color=[
            0, 0, 0, 0, 7, 7, 7, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 16, 16
        ],
        point_color=[16, 16, 16, 16, 16, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0],
        sigmas=[
            0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072,
            0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089
        ]),
    coco_wholebody=dict(
        skeleton=[(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11),
                  (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2),
                  (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (15, 17),
                  (15, 18), (15, 19), (16, 20), (16, 21), (16, 22), (91, 92),
                  (92, 93), (93, 94), (94, 95), (91, 96), (96, 97), (97, 98),
                  (98, 99), (91, 100), (100, 101), (101, 102), (102, 103),
                  (91, 104), (104, 105), (105, 106), (106, 107), (91, 108),
                  (108, 109), (109, 110), (110, 111), (112, 113), (113, 114),
                  (114, 115), (115, 116), (112, 117), (117, 118), (118, 119),
                  (119, 120), (112, 121), (121, 122), (122, 123), (123, 124),
                  (112, 125), (125, 126), (126, 127), (127, 128), (112, 129),
                  (129, 130), (130, 131), (131, 132)],
        palette=[(51, 153, 255), (0, 255, 0), (255, 128, 0), (255, 255, 255),
                 (255, 153, 255), (102, 178, 255), (255, 51, 51)],
        link_color=[
            1, 1, 2, 2, 0, 0, 0, 0, 1, 2, 1, 2, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
            2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 1, 1, 1,
            1, 2, 2, 2, 2, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 1, 1, 1, 1
        ],
        point_color=[
            0, 0, 0, 0, 0, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2,
            2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
            3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
            3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
            3, 3, 3, 3, 2, 2, 2, 2, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 1, 1,
            1, 1, 3, 2, 2, 2, 2, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 1, 1, 1, 1
        ],
        sigmas=[
            0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072,
            0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089, 0.068,
            0.066, 0.066, 0.092, 0.094, 0.094, 0.042, 0.043, 0.044, 0.043,
            0.040, 0.035, 0.031, 0.025, 0.020, 0.023, 0.029, 0.032, 0.037,
            0.038, 0.043, 0.041, 0.045, 0.013, 0.012, 0.011, 0.011, 0.012,
            0.012, 0.011, 0.011, 0.013, 0.015, 0.009, 0.007, 0.007, 0.007,
            0.012, 0.009, 0.008, 0.016, 0.010, 0.017, 0.011, 0.009, 0.011,
            0.009, 0.007, 0.013, 0.008, 0.011, 0.012, 0.010, 0.034, 0.008,
            0.008, 0.009, 0.008, 0.008, 0.007, 0.010, 0.008, 0.009, 0.009,
            0.009, 0.007, 0.007, 0.008, 0.011, 0.008, 0.008, 0.008, 0.01,
            0.008, 0.029, 0.022, 0.035, 0.037, 0.047, 0.026, 0.025, 0.024,
            0.035, 0.018, 0.024, 0.022, 0.026, 0.017, 0.021, 0.021, 0.032,
            0.02, 0.019, 0.022, 0.031, 0.029, 0.022, 0.035, 0.037, 0.047,
            0.026, 0.025, 0.024, 0.035, 0.018, 0.024, 0.022, 0.026, 0.017,
            0.021, 0.021, 0.032, 0.02, 0.019, 0.022, 0.031
        ]),
    halpe26=dict(
        skeleton=[(15, 13), (13, 11), (11,19),(16, 14), (14, 12), (12,19),
                  (17,18), (18,19), (18,5), (5,7), (7,9), (18,6), (6,8),
                  (8,10), (1,2), (0,1), (0,2), (1,3), (2,4), (3,5), (4,6),
                  (15,20), (15,22), (15,24),(16,21),(16,23), (16,25)],
        palette=[(51, 153, 255), (0, 255, 0), (255, 128, 0)],
        link_color=[
            1, 1, 1, 2, 2, 2, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2
        ],
        point_color=[
            0, 0, 0, 0, 0, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2
        ],
        sigmas=[
            0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072,
            0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089, 0.026,
            0.026, 0.066, 0.079, 0.079, 0.079, 0.079, 0.079, 0.079
        ]
    )
)
det_model_path = os.path.join(os.getcwd(),"NTK_CAP", "ThirdParty", "mmdeploy", "rtmpose-trt", "rtmdet-nano")
pose_model_path = os.path.join(os.getcwd(),"NTK_CAP", "ThirdParty", "mmdeploy", "rtmpose-trt", "rtmpose-m")
device ="cuda"

sigmas = VISUALIZATION_CFG['halpe26']['sigmas']
palette = VISUALIZATION_CFG['halpe26']['palette']
skeleton = VISUALIZATION_CFG['halpe26']['skeleton']
link_color = VISUALIZATION_CFG['halpe26']['link_color']
point_color = VISUALIZATION_CFG['halpe26']['point_color']
class TrackerProcess(Process):
    def __init__(self, start_evt, cam_id, stop_evt, queue_cam, queue_kp, shm, shm_kp, enable_pose_detection=False, *args, **kwargs):
        super().__init__()
        self.start_evt = start_evt
        self.cam_id = cam_id
        self.stop_evt = stop_evt
        self.queue_cam = queue_cam
        self.queue_kp = queue_kp
        self.shm = shm
        self.buffer_length = 4
        self.shm_kp = shm_kp
        self.recording = False
        self.frame_width = 1920
        self.frame_height = 1080
        self.enable_pose_detection = enable_pose_detection
        
    def run(self):
        shape = (1080, 1920, 3)
        existing_shm = shared_memory.SharedMemory(name=self.shm)
        shared_array = np.ndarray((self.buffer_length,) + shape, dtype=np.uint8, buffer=existing_shm.buf)
        
        # Only initialize pose tracker if pose detection is enabled
        tracker = None
        state = None
        use_rtmlib = False
        rtmlib_body = None
        prev_keypoints = None  # For temporal smoothing
        
        if self.enable_pose_detection:
            # Try mmdeploy first (TensorRT - fastest)
            if MMDEPLOY_AVAILABLE:
                try:
                    tracker = MMDeployPoseTracker(det_model=det_model_path, pose_model=pose_model_path, device_name=device)
                    state = tracker.create_state(det_interval=1, det_min_bbox_size=100, keypoint_sigmas=sigmas, pose_max_num_bboxes=1)
                    print(f"✅ [Cam {self.cam_id}] mmdeploy TensorRT tracker initialized")
                except Exception as e:
                    print(f"⚠️ [Cam {self.cam_id}] mmdeploy failed: {e}")
                    tracker = None
            
            # Fallback to rtmlib (ONNX Runtime)
            if tracker is None and RTMLIB_AVAILABLE:
                try:
                    from rtmlib import BodyWithFeet
                    rtmlib_body = BodyWithFeet(device='cuda', backend='onnxruntime')
                    use_rtmlib = True
                    print(f"✅ [Cam {self.cam_id}] rtmlib BodyWithFeet (ONNX) initialized")
                except Exception as e:
                    print(f"⚠️ [Cam {self.cam_id}] rtmlib BodyWithFeet failed: {e}")
                    try:
                        from rtmlib import Body
                        rtmlib_body = Body(device='cuda', backend='onnxruntime')
                        use_rtmlib = True
                        print(f"✅ [Cam {self.cam_id}] rtmlib Body (ONNX) initialized")
                    except Exception as e2:
                        print(f"❌ [Cam {self.cam_id}] All pose backends failed: {e2}")
                        self.enable_pose_detection = False
            
            if tracker is None and rtmlib_body is None:
                print(f"📋 [Cam {self.cam_id}] Running without pose detection")
                self.enable_pose_detection = False
        
        existing_shm_kp = shared_memory.SharedMemory(name=self.shm_kp)
        shared_array_kp = np.ndarray((self.buffer_length,) + shape, dtype=np.uint8, buffer=existing_shm_kp.buf)
        idx = 0
        
        while self.start_evt.is_set():
            try:
                idx_get = self.queue_cam.get(timeout=1)
            except:
                time.sleep(0.01)
                continue
            frame = shared_array[idx_get, : ]
            
            # Only run pose detection if enabled
            if self.enable_pose_detection:
                try:
                    if use_rtmlib and rtmlib_body is not None:
                        # rtmlib inference
                        kpts, scores_raw = rtmlib_body(frame)
                        if len(kpts) > 0:
                            num_people = len(kpts)
                            num_kpts = kpts.shape[1]
                            
                            # Build keypoints array (N, K, 3)
                            keypoints = np.zeros((num_people, num_kpts, 3))
                            for i in range(num_people):
                                for j in range(num_kpts):
                                    keypoints[i, j, 0] = kpts[i, j, 0]
                                    keypoints[i, j, 1] = kpts[i, j, 1]
                                    keypoints[i, j, 2] = scores_raw[i, j]
                            
                            # Apply temporal smoothing
                            alpha = 0.6  # Smoothing factor
                            if prev_keypoints is not None and len(prev_keypoints) > 0:
                                for i in range(min(num_people, len(prev_keypoints))):
                                    for j in range(min(num_kpts, prev_keypoints.shape[1])):
                                        if keypoints[i, j, 2] > 0.3 and prev_keypoints[i, j, 2] > 0.3:
                                            keypoints[i, j, 0] = alpha * keypoints[i, j, 0] + (1 - alpha) * prev_keypoints[i, j, 0]
                                            keypoints[i, j, 1] = alpha * keypoints[i, j, 1] + (1 - alpha) * prev_keypoints[i, j, 1]
                            prev_keypoints = keypoints.copy()
                            
                            scores = keypoints[..., 2]
                            kpts_draw = np.round(keypoints[..., :2], 3)
                            self.draw_frame(frame, kpts_draw, scores, palette, skeleton, link_color, point_color)
                    elif tracker is not None and state is not None:
                        # mmdeploy inference
                        keypoints, _ = tracker(state, frame, detect=-1)[:2]
                        scores = keypoints[..., 2]
                        keypoints = np.round(keypoints[..., :2], 3)
                        self.draw_frame(frame, keypoints, scores, palette, skeleton, link_color, point_color)
                except Exception as e:
                    # Silently continue on error (don't spam console)
                    pass
            
            np.copyto(shared_array_kp[idx,:], frame)
            
            # Prevent queue overflow by dropping old frames
            try:
                self.queue_kp.put_nowait(idx)
            except:
                # Queue is full, skip old frames to prevent buffer overflow
                try:
                    while self.queue_kp.qsize() > 2:  # Keep only 2 frames in queue
                        self.queue_kp.get_nowait()
                    self.queue_kp.put_nowait(idx)
                except:
                    pass  # Queue operations failed, continue
                    
            idx = (idx+1) % self.buffer_length
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