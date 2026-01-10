import os
import cupy as cp
from mmdeploy_runtime import PoseTracker
import cv2
import time
from GUI_source.DevelopVersion.real_time_testmod import tri
from multiprocessing import Event, shared_memory, Queue, Array
import numpy as np
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
det_model_path = os.path.join(os.getcwd(),"NTK_CAP", "ThirdParty", "mmdeploy", "rtmpose-trt", "rtmdet-m")
pose_model_path = os.path.join(os.getcwd(),"NTK_CAP", "ThirdParty", "mmdeploy", "rtmpose-trt", "rtmpose-m")
device ="cuda"

sigmas = VISUALIZATION_CFG['halpe26']['sigmas']
palette = VISUALIZATION_CFG['halpe26']['palette']
skeleton = VISUALIZATION_CFG['halpe26']['skeleton']
link_color = VISUALIZATION_CFG['halpe26']['link_color']
point_color = VISUALIZATION_CFG['halpe26']['point_color']
# camera_shm_shape = (1080, 1920, 3)
# buffer_length = 10
# camera_shmtest = shared_memory.SharedMemory(create=True, size=int(np.prod(camera_shm_shape) * np.dtype(np.uint8).itemsize * buffer_length))
# camera_queue = Queue()
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
ret, frame = cap.read()
# idx = 0
# existing_shm_frame = shared_memory.SharedMemory(name=camera_shmtest.name)
# shared_array_frame = np.ndarray((buffer_length,) + camera_shm_shape, dtype=np.uint8, buffer=existing_shm_frame.buf)
# import pdb; pdb.set_trace()
# np.copyto(shared_array_frame[idx,:], frame)
# camera_queue.put(idx)
# idx += 1
# np.copyto(shared_array_frame[idx,:], frame)
# camera_queue.put(idx)
# idx += 1
# np.copyto(shared_array_frame[idx,:], frame)
# camera_queue.put(idx)
# idx += 1
# np.copyto(shared_array_frame[idx,:], frame)
# camera_queue.put(idx)
# idx += 1
# np.copyto(shared_array_frame[idx,:], frame)
# camera_queue.put(idx)
# count = 0
# while True:
#     t1 = time.time()
#     try:
#         idx_get = camera_queue.get(timeout=0.01)
#     except:
#         continue
#     frame = shared_array_frame[idx_get, : ]
#     print(t1)
    
#     count += 1
#     if count > 4:
#         break
# camera_shmtest.close()
# camera_shmtest.unlink()
# camera_queue.close()
tracker = PoseTracker(det_model=det_model_path,pose_model=pose_model_path,device_name=device)
state = tracker.create_state(det_interval=1, det_min_bbox_size=100, keypoint_sigmas=sigmas, pose_max_num_bboxes=1)
stream_4 = cp.cuda.Stream()
stream_3 = cp.cuda.Stream()
stream_2 = cp.cuda.Stream()
stream_1 = cp.cuda.Stream()
stream_0 = cp.cuda.Stream()

while True:
    t1 = time.time()
    with stream_0:
        keypoints, _ = tracker(state, frame, detect=-1)[:2]
    with stream_1:
        keypoints, _ = tracker(state, frame, detect=-1)[:2]
    with stream_2:
        keypoints, _ = tracker(state, frame, detect=-1)[:2]
    with stream_3:
        keypoints, _ = tracker(state, frame, detect=-1)[:2]
    stream_0.synchronize()
    stream_1.synchronize()
    stream_2.synchronize()
    stream_3.synchronize()
    
    
    
    print("tracker", time.time()-t1)
    t2 = time.time()
    with stream_4:
        tri()
    stream_4.synchronize()
    print("tri", time.time()-t2)