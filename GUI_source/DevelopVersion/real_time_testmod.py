import os
import cv2
import json
import toml
import numpy as np
import cupy as cp
from anytree import Node, RenderTree
from mmdeploy_runtime import PoseTracker
import pdb
import time
from Pose2Sim.skeletons import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
from multiprocessing import Event, shared_memory, Queue, Array, Manager
import sys
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
det_model_path = os.path.join(r"C:\Users\MyUser\Desktop\NTKCAP\NTK_CAP\ThirdParty\mmdeploy\rtmpose-trt\rtmdet-nano")
pose_model_path = os.path.join(r"C:\Users\MyUser\Desktop\NTKCAP\NTK_CAP\ThirdParty\mmdeploy\rtmpose-trt\rtmpose-t")
device ="cuda"

sigmas = VISUALIZATION_CFG['halpe26']['sigmas']
palette = VISUALIZATION_CFG['halpe26']['palette']
skeleton = VISUALIZATION_CFG['halpe26']['skeleton']
link_color = VISUALIZATION_CFG['halpe26']['link_color']
point_color = VISUALIZATION_CFG['halpe26']['point_color']
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
ret, frame = cap.read()
tracker = PoseTracker(det_model=det_model_path,pose_model=pose_model_path,device_name=device)
state = tracker.create_state(det_interval=1, det_min_bbox_size=100, keypoint_sigmas=sigmas, pose_max_num_bboxes=1)

def extract_files_frame_f_fast(f,coord, keypoints_ids):
    '''
    Extract data from json files for frame f, 
    in the order of the body model hierarchy.

    INPUTS:
    - json_tracked_files_f: list of str. Paths of json_files for frame f.
    - keypoints_ids: list of int. Keypoints IDs in the order of the hierarchy.

    OUTPUTS:
    - x_files, y_files, likelihood_files: array: 
      n_cams lists of n_keypoints lists of coordinates.
    '''

    n_cams = 4
    
    x_files, y_files, likelihood_files = [], [], []
    for cam_nb in range(n_cams):
        x_files_cam, y_files_cam, likelihood_files_cam = [], [], []
        js = coord[cam_nb][f]
        for keypoint_id in keypoints_ids:
            try:
                x_files_cam.append( js['people'][0]['pose_keypoints_2d'][keypoint_id*3] )
                y_files_cam.append( js['people'][0]['pose_keypoints_2d'][keypoint_id*3+1] )
                likelihood_files_cam.append( js['people'][0]['pose_keypoints_2d'][keypoint_id*3+2] )
            except:
                x_files_cam.append( np.array(0) )
                y_files_cam.append( np.array(0))
                likelihood_files_cam.append( np.array(0))

        x_files.append(x_files_cam)
        y_files.append(y_files_cam)
        likelihood_files.append(likelihood_files_cam)
        
    x_files = np.array(x_files)
    y_files = np.array(y_files)
    likelihood_files = np.array(likelihood_files)
    return x_files, y_files, likelihood_files
def read_config_file(config):
    config_dict = toml.load(config)
    return config_dict
def bilinear_interpolate_cupy(map, x, y):
    
    x0 = cp.clip(cp.floor(x).astype(cp.int32), 0, map.shape[1] - 2)
    y0 = cp.clip(cp.floor(y).astype(cp.int32), 0, map.shape[0] - 2)
    
    x1, y1 = x0 + 1, y0 + 1

    Ia = map[y0, x0]
    Ib = map[y0, x1]
    Ic = map[y1, x0]
    Id = map[y1, x1]

    wa = (x1 - x) * (y1 - y)
    wb = (x - x0) * (y1 - y)
    wc = (x1 - x) * (y - y0)
    wd = (x - x0) * (y - y0)

    return wa * Ia + wb * Ib + wc * Ic + wd * Id

def computemap(calib_file):
    # Only map one camera because four cameras has same intrinsic matrix and dist. coeff
    calib = toml.load(calib_file)
    mappingx = []
    mappingy = []
    for cam in list(calib.keys()):
        #import pdb;pdb.set_trace()
        if cam != 'metadata':
            K = np.array(calib[cam]['matrix'])
            Kh = np.block([K, np.zeros(3).reshape(3,1)])
            R, _ = cv2.Rodrigues(np.array(calib[cam]['rotation']))
            T = np.array(calib[cam]['translation'])
            H = np.block([[R,T.reshape(3,1)], [np.zeros(3), 1 ]])
            mappingx,mappingy = cv2.initUndistortRectifyMap(np.array(calib[cam]['matrix']), np.array(calib[cam]['distortions']), None, np.array(calib[cam]['matrix']),np.array([1920,1080]), cv2.CV_32FC1)
            break
            #import pdb;pdb.set_trace()
   
    return mappingx,mappingy
def computeP(calib_file):
    '''
    Compute projection matrices from toml calibration file.
    
    INPUT:
    - calib_file: calibration .toml file.
    
    OUTPUT:
    - P: projection matrix as list of arrays
    '''
    
    K, R, T, Kh, H = [], [], [], [], []
    P = []
    
    calib = toml.load(calib_file)
    for cam in list(calib.keys()):
        #import pdb;pdb.set_trace()
        if cam != 'metadata':
            K = np.array(calib[cam]['matrix'])
            Kh = np.block([K, np.zeros(3).reshape(3,1)])
            R, _ = cv2.Rodrigues(np.array(calib[cam]['rotation']))
            T = np.array(calib[cam]['translation'])
            H = np.block([[R,T.reshape(3,1)], [np.zeros(3), 1 ]])
            
            P.append(Kh.dot(H))
            #import pdb;pdb.set_trace()
   
    return P

def find_Q(A4,A3,A2):
    f = cp.shape(A4)
    A_flat = A4.reshape(-1, 8, 4)  # Shape: (250 * 22 * 1, 8, 4)

    # Step 2: Perform SVD in batch using CuPy
    U, S, Vt = cp.linalg.svd(A_flat, full_matrices=False)  # Batched SVD

    # Step 3: Compute Q
    # Transpose Vt to get V
    V = Vt.transpose(0, 2, 1)  # Shape: (batch_size, 4, 4)

    # Extract and compute Q
    Q = cp.array([
        V[:, 0, 3] / V[:, 3, 3],
        V[:, 1, 3] / V[:, 3, 3],
        V[:, 2, 3] / V[:, 3, 3],
        cp.ones(V.shape[0])  # Add 1 as the last element of Q
    ]).T  # Shape: (batch_size, 4)

    # Step 4: Reshape Q back to (250, 22, 1, 4)
    Q4 = Q.reshape(f[0], f[1], f[2], 4)
    f = cp.shape(A3)
    A_flat = A3.reshape(-1, 6, 4)  # Shape: (250 * 22 * 1, 8, 4)

    # Step 2: Perform SVD in batch using CuPy
    U, S, Vt = cp.linalg.svd(A_flat, full_matrices=False)  # Batched SVD

    # Step 3: Compute Q
    # Transpose Vt to get V
    V = Vt.transpose(0, 2, 1)  # Shape: (batch_size, 4, 4)

    # Extract and compute Q
    Q = cp.array([
        V[:, 0, 3] / V[:, 3, 3],
        V[:, 1, 3] / V[:, 3, 3],
        V[:, 2, 3] / V[:, 3, 3],
        cp.ones(V.shape[0])  # Add 1 as the last element of Q
    ]).T  # Shape: (batch_size, 4)

    # Step 4: Reshape Q back to (250, 22, 1, 4)
    Q3 = Q.reshape(f[0], f[1], f[2], 4)
    
    f = cp.shape(A2)
    A_flat = A2.reshape(-1, 4, 4)  # Shape: (250 * 22 * 1, 8, 4)

    # Step 2: Perform SVD in batch using CuPy
    U, S, Vt = cp.linalg.svd(A_flat, full_matrices=False)  # Batched SVD
    # Step 3: Compute Q
    # Transpose Vt to get V
    V = Vt.transpose(0, 2, 1)  # Shape: (batch_size, 4, 4)

    # Extract and compute Q
    Q = cp.array([
        V[:, 0, 3] / V[:, 3, 3],
        V[:, 1, 3] / V[:, 3, 3],
        V[:, 2, 3] / V[:, 3, 3],
        cp.ones(V.shape[0])  # Add 1 as the last element of Q
    ]).T  # Shape: (batch_size, 4)

    # Step 4: Reshape Q back to (250, 22, 1, 4)
    Q2 = Q.reshape(f[0], f[1], f[2], 4)

    pdb.set_trace()
    return Q4,Q3,Q2
def create_QBug(likelihood_threshold,prep_3like,Q3,prep_2like,Q2):
    loc = cp.where(prep_3like < likelihood_threshold)
    prep_3like[loc] = cp.inf
    non_inf_mask3 = ~cp.isinf(prep_3like)
    min_locations_nan3 = cp.argmax(non_inf_mask3, axis=2)
    batch_indices = cp.arange(Q3.shape[0])[:, None]
    time_indices = cp.arange(Q3.shape[1])[None, :]

    selected_slices = Q3[batch_indices, time_indices, min_locations_nan3, :]
    

    Q3_bug = selected_slices[:, :, cp.newaxis, :]

    loc = cp.where(prep_2like < likelihood_threshold)
    prep_2like[loc] = cp.inf

    non_inf_mask2 = ~cp.isinf(prep_2like)
    min_locations_nan2 = cp.argmax(non_inf_mask2, axis=2)
    batch_indices = cp.arange(Q2.shape[0])[:, None]  
    time_indices = cp.arange(Q2.shape[1])[None, :] 

    selected_slices = Q2[batch_indices, time_indices, min_locations_nan2, :]    
    Q2_bug = selected_slices[:, :, cp.newaxis, :]
    return Q3_bug,Q2_bug

def compute_real_dist(P, cam_coord, Q, Q_bug, prep, camera_combinations):
    """
    計算 real_dist，適用於 Q4, Q3, Q2 這三種不同數據
    """
    final_results_x = []
    final_results_y = []
    final_distances = []

    for i, cameras in enumerate(camera_combinations):
        x, y, dist = [], [], []

        for d in cameras:
            P_cam_0 = P[d][0].reshape(1, -1)  # (1, 4)
            P_cam_1 = P[d][1].reshape(1, -1)  # (1, 4)
            P_cam_2 = P[d][2].reshape(1, -1)  # (1, 4)

            P0_dot_Q = cp.einsum('ik,btk->bt', P_cam_0, Q[:, :, i, :])  # (batch, 22, num_cameras)
            P1_dot_Q = cp.einsum('ik,btk->bt', P_cam_1, Q[:, :, i, :])
            P2_dot_Q = cp.einsum('ik,btk->bt', P_cam_2, Q[:, :, i, :])

            x.append(P0_dot_Q / P2_dot_Q)
            y.append(P1_dot_Q / P2_dot_Q)
            dist.append(cp.sqrt(cp.sum((Q_bug[:, :, 0, 0:3] - cam_coord[d]) ** 2, axis=-1)))

        X = cp.stack(x, axis=2)
        Y = cp.stack(y, axis=2)
        distances = cp.stack(dist, axis=2)

        final_results_x.append(X)
        final_results_y.append(Y)
        final_distances.append(distances)

    X_stacked = cp.stack(final_results_x, axis=2)
    Y_stacked = cp.stack(final_results_y, axis=2)
    final_distances_stacked = cp.stack(final_distances, axis=2)

    rpj_coor = cp.stack((X_stacked, Y_stacked), axis=-1)

    x1, y1 = prep[:, :, :, :, 0], prep[:, :, :, :, 1]
    x2, y2 = rpj_coor[:, :, :, :, 0], rpj_coor[:, :, :, :, 1]

    # 計算歐式距離
    rpj = cp.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    real_dist = final_distances_stacked * rpj

    return cp.max(real_dist, axis=-1)

def find_real_dist_error(cam_coord, prep_4, Q4, prep_3, Q3, prep_2, Q2, Q3_bug, Q2_bug, P_cam_comb4, P_cam_comb3, comb_3, P_cam_comb2, comb_2):
    
    result_c4 = cp.einsum('cik,btk->cit', P_cam_comb4, Q4[:,:,0,:])
    
    x_c4 = result_c4[0] / result_c4[2]
    y_c4 = result_c4[1] / result_c4[2]
    dist_c4 = cp.sqrt(cp.sum((Q4[:, :, 0, 0:3] - cp.stack(cam_coord, axis=0)[:, None, :]) ** 2, axis=-1))
    
    X_c4 = cp.expand_dims(x_c4.T, axis=(0, 2))
    Y_c4 = cp.expand_dims(y_c4.T, axis=(0, 2))
    final_dist_c4 = cp.expand_dims(dist_c4.T, axis=(0, 2))

    rpj_coor_4 = cp.stack((X_c4,Y_c4), axis = -1)
    
    x1, y1 = prep_4[:, :, :, :, 0], prep_4[:, :, :, :, 1]
    x2, y2 = rpj_coor_4[:, :, :, :, 0], rpj_coor_4[:, :, :, :, 1]
    # Compute the Euclidean distance
    rpj = cp.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    real_dist  = final_dist_c4*rpj
    real_dist4 = cp.max(real_dist, axis=-1)

    # real_dist3
    

    result_c3 = cp.einsum('ncik,nbtk->ncit', P_cam_comb3, Q3.transpose(2, 0, 1, 3))
    x_c3 = result_c3[:, 0] / result_c3[:, 2]
    y_c3 = result_c3[:, 1] / result_c3[:, 2]

    dist_c3 = cp.sqrt(cp.sum((Q3_bug.transpose(2, 0, 1, 3)[:, :, :, 0:3] - cam_coord[comb_3][:, :, None, :]) ** 2, axis=-1))
    X_c3 = cp.expand_dims(x_c3.transpose(2, 0, 1), axis=0)
    Y_c3 = cp.expand_dims(y_c3.transpose(2, 0, 1), axis=0)
    final_dist_c3 = cp.expand_dims(dist_c3.transpose(2, 0, 1), axis=0)

    rpj_coor_3 = cp.stack((X_c3,Y_c3),axis = -1)
    
    x1, y1 = prep_3[:, :, :, :, 0], prep_3[:, :, :, :, 1]
    x2, y2 = rpj_coor_3[:, :, :, :, 0], rpj_coor_3[:, :, :, :, 1]
    # Compute the Euclidean distance
    rpj = cp.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    real_dist  = final_dist_c3*rpj
    real_dist3 = cp.max(real_dist, axis=-1)
    
    result_c2 = cp.einsum('ncik,nbtk->ncit', P_cam_comb2, Q2.transpose(2, 0, 1, 3))

    x_c2 = result_c2[:, 0] / result_c2[:, 2]
    y_c2 = result_c2[:, 1] / result_c2[:, 2]

    dist_c2 = cp.sqrt(cp.sum((Q2_bug.transpose(2, 0, 1, 3)[:, :, :, 0:3] - cam_coord[comb_2][:, :, None, :]) ** 2, axis=-1))
    X_c2 = cp.expand_dims(x_c2.transpose(2, 0, 1), axis=0)
    Y_c2 = cp.expand_dims(y_c2.transpose(2, 0, 1), axis=0)
    final_dist_c2 = cp.expand_dims(dist_c2.transpose(2, 0, 1), axis=0)  
    
    
    #[754.2223150392452, 1165.0827450392762]
    rpj_coor_2 = cp.stack((X_c2,Y_c2),axis = -1)
    
    x1, y1 = prep_2[:, :, :, :, 0], prep_2[:, :, :, :, 1]
    x2, y2 = rpj_coor_2[:, :, :, :, 0], rpj_coor_2[:, :, :, :, 1]
    # Compute the Euclidean distance
    rpj = cp.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    real_dist  = final_dist_c2*rpj
    real_dist2 = cp.max(real_dist, axis=-1)

    return real_dist4, real_dist3, real_dist2
def find_camera_coordinate(calib_file):
    calib_file['rotation']
    R, _ = cv2.Rodrigues(np.array(calib_file['rotation']))
    T = np.array(calib_file['translation'])
    H = np.block([[R,T.reshape(3,1)], [np.zeros(3), 1 ]])
    R_t = np.transpose(R)
    C = -R_t.dot(T)
    A = R_t.dot(np.array([[0],[0],[1]]))
    return C
def map_to_listdynamic(value):
    if value == 4:
        return [1,2,3,4,5,6,7,8,9,10]  # Map 4 to [0]
    elif value == 3:
        return [5,6,7,8,9,10]  # Map 3 to [1, 2, 3, 4]
    elif value == 2:
        return []  # Map 2 to [5, 6, 7, 8, 9, 10]
def tri():
    
    cal_path = r"C:\Users\MyUser\Desktop\NTKCAP\Patient_data\HealthCare020\2024_12_06\2025_02_18_00_02_calculated\1005_1\pose-2d-tracked"
    template = ["pose_cam1_json", "pose_cam2_json", "pose_cam3_json", "pose_cam4_json"]
    config=r"C:\Users\MyUser\Desktop\NTKCAP\NTK_CAP\template\Empty_project\User\Config.toml"
    
    config_dict = read_config_file(config)
    calib_file = r"C:\Users\MyUser\Desktop\NTKCAP\Patient_data\HealthCare020\2024_12_06\2025_02_18_00_02_calculated\1005_1\calib-2d\Calib_easymocap.toml"
    likelihood_threshold = config_dict.get('triangulation').get('likelihood_threshold')
    mappingx,mappingy = computemap(calib_file)
    P = computeP(calib_file)
    calib = toml.load(calib_file)
    cam_coord=cp.array([find_camera_coordinate(calib[list(calib.keys())[i]]) for i in range(4)])
    coord = []
    path1 = os.path.join(cal_path, template[0])
    path2 = os.path.join(cal_path, template[1])
    path3 = os.path.join(cal_path, template[2])
    path4 = os.path.join(cal_path, template[3])
    coord1 = []
    coord2 = []
    coord3 = []
    coord4 = []
    combinations_3 = [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]
    combinations_2 = [[2, 3], [1, 3], [1, 2], [0, 3], [0, 2], [0, 1]]
    prep = []
    
    for i in os.listdir(path1):
        path = os.path.join(path1, i)
        with open(path, 'r') as f:
            data1 = json.load(f)
            coord1.append(data1)
    for i in os.listdir(path2):
        path = os.path.join(path2, i)
        with open(path, 'r') as f:
            data2 = json.load(f)
            coord2.append(data2)
    for i in os.listdir(path3):
        path = os.path.join(path3, i)
        with open(path, 'r') as f:
            data3 = json.load(f)
            coord3.append(data3)
    for i in os.listdir(path4):
        path = os.path.join(path4, i)
        with open(path, 'r') as f:
            data4 = json.load(f)
            coord4.append(data4)
    coord.append(coord1)
    coord.append(coord2)
    coord.append(coord3)
    coord.append(coord4)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # plt.ion()
    keypoints_ids = [19, 12, 14, 16, 21, 23, 25, 11, 13, 15, 20, 22, 24, 18, 17, 0, 6, 8, 10, 5, 7, 9]
    # plt.show()
    mappingx = cp.array(mappingx)
    mappingy = cp.array(mappingy)
    P = cp.array(P)
    P_cam_0_comb4 = cp.concatenate([P[0][0].reshape(1, -1), P[1][0].reshape(1, -1), P[2][0].reshape(1, -1), P[3][0].reshape(1, -1)], axis=0)
    P_cam_1_comb4 = cp.concatenate([P[0][1].reshape(1, -1), P[1][1].reshape(1, -1), P[2][1].reshape(1, -1), P[3][1].reshape(1, -1)], axis=0)
    P_cam_2_comb4 = cp.concatenate([P[0][2].reshape(1, -1), P[1][2].reshape(1, -1), P[2][2].reshape(1, -1), P[3][2].reshape(1, -1)], axis=0)
    P_cam_comb4 = cp.stack([P_cam_0_comb4, P_cam_1_comb4, P_cam_2_comb4], axis=0)
    comb_3 = cp.array(combinations_3)
    P_cam_comb3 = cp.stack([
        cp.stack([cp.concatenate([P[d][j].reshape(1, -1) for d in combinations_3[i]], axis=0) for j in range(3)], axis=0)
        for i in range(4)
    ], axis=0)

    combinations_2 = [[2, 3], [1, 3], [1, 2], [0, 3], [0, 2], [0, 1]] 
    comb_2 = cp.array(combinations_2)
    P_cam_comb2 = cp.stack([
        cp.stack([cp.concatenate([P[d][j].reshape(1, -1) for d in combinations_2[i]], axis=0) for j in range(3)], axis=0)
        for i in range(6)
    ], axis=0)
    
    # create_A
    P0, P1, P2 = P[:, 0, :].reshape(1, 1, 1, 4, 4), P[:, 1, :].reshape(1, 1, 1, 4, 4), P[:, 2, :].reshape(1, 1, 1, 4, 4)
    P0_c3, P1_c3, P2_c3 = P[comb_3, 0, :][None, None, ...], P[comb_3, 1, :][None, None, ...], P[comb_3, 2, :][None, None, ...]
    P0_c2, P1_c2, P2_c2 = P[comb_2, 0, :][None, None, ...], P[comb_2, 1, :][None, None, ...], P[comb_2, 2, :][None, None, ...]
    
    
    c1 = np.array(coord1[1]['people'][0]['pose_keypoints_2d'])
    c2 = np.array(coord2[1]['people'][0]['pose_keypoints_2d'])
    c3 = np.array(coord3[1]['people'][0]['pose_keypoints_2d'])
    c4 = np.array(coord4[1]['people'][0]['pose_keypoints_2d'])
    arr1 = c1.reshape(26, 3)
    arr2 = c2.reshape(26, 3)
    arr3 = c3.reshape(26, 3)
    arr4 = c4.reshape(26, 3)
    stacked = np.stack([arr1, arr2, arr3, arr4], axis=0)
    stream_4 = cp.cuda.Stream()
    stream_3 = cp.cuda.Stream()
    stream_2 = cp.cuda.Stream()
    stream_41 = cp.cuda.Stream()
    stream_31 = cp.cuda.Stream()
    stream_21 = cp.cuda.Stream()
    stream_t1 = cp.cuda.Stream()
    stream_t2 = cp.cuda.Stream()
    stream_t3 = cp.cuda.Stream()
    stream_t4 = cp.cuda.Stream()
    result = cp.empty((22, 4, 3), dtype=cp.float32)
    prep_4 = cp.empty((1, 22, 1, 4, 3), dtype=cp.float32)
    prep_3 = cp.empty((1, 22, 4, 3, 3), dtype=cp.float32)
    prep_2 = cp.empty((1, 22, 6, 2, 3), dtype=cp.float32)

    prep_4like = cp.empty((1, 22, 1), dtype=cp.float32)
    prep_3like = cp.empty((1, 22, 4), dtype=cp.float32)
    prep_2like = cp.empty((1, 22, 6), dtype=cp.float32)

    A4 = cp.empty((1, 22, 1, 8, 4), dtype=cp.float32)
    A3 = cp.empty((1, 22, 4, 6, 4), dtype=cp.float32)
    A2 = cp.empty((1, 22, 6, 4, 4), dtype=cp.float32)

    Q4 = cp.empty((1, 22, 1, 4), dtype=cp.float32)
    Q3 = cp.empty((1, 22, 4, 4), dtype=cp.float32)
    Q2 = cp.empty((1, 22, 6, 4), dtype=cp.float32)

    real_dist4 = cp.empty((1, 22, 1), dtype=cp.float32)
    real_dist3 = cp.empty((1, 22, 4), dtype=cp.float32)
    real_dist2 = cp.empty((1, 22, 6), dtype=cp.float32)
    # queue = Manager().Queue()
    # shm = shared_memory.SharedMemory(name="test_shm")
    # shared_array = np.ndarray((30, *IMG_SHAPE), dtype=DTYPE, buffer=shm.buf)
    while True:
    # for f in range(133):
        
        t4 = time.time()
        # with torch.inference_mode():
        with stream_t1:
            keypoints, _ = tracker(state, frame, detect=-1)[:2]
        
        with stream_t2:
            keypoints, _ = tracker(state, frame, detect=-1)[:2]
        with stream_t3:
            keypoints, _ = tracker(state, frame, detect=-1)[:2]
        with stream_t4:
            keypoints, _ = tracker(state, frame, detect=-1)[:2]
        
        stream_t1.synchronize()
        stream_t2.synchronize()
        stream_t3.synchronize()
        stream_t4.synchronize()
        
        prep = None
        t5 = time.time()
        prep = cp.array(stacked[:, keypoints_ids, :])
        # print(time.time()-t5)
        
        result[:] = cp.transpose(prep, (1, 0, 2)) # (22, 4, 3)
        prep_4[:] = result[None, :, None, :] # (1, 22, 1, 4, 3)
        
        prep_3[:] = result[:, combinations_3, :][None, :] # (1, 22, 4, 3, 3)
        prep_2[:] = result[:, combinations_2, :][None, :] # (1, 22, 6, 2, 3)
        
        with stream_4:
            t1 = time.time()
            prep_4[:] = cp.stack([bilinear_interpolate_cupy(mappingx, prep_4[:, :, :, :, 0], prep_4[:, :, :, :, 1]), bilinear_interpolate_cupy(mappingy, prep_4[:, :, :, :, 0], prep_4[:, :, :, :, 1]), prep_4[:, :, :, :, 2]], axis=-1)  # Shape: (155, 22, 4, 3, 3)
        
            prep_4like[:] = cp.min(prep_4[:,:,:,:,2],axis=3) # (1, 22, 1)
            
            A4[:] = cp.stack([((P0 - prep_4[..., 0:1] * P2) * prep_4[..., 2:3]), ((P1 - prep_4[..., 1:2] * P2) * prep_4[..., 2:3])], axis=-2).reshape(1, 22, 1, 8, 4) # (1, 22, 1, 8, 4)
            f4 = cp.shape(A4)
            A4_flat = A4.reshape(-1, 8, 4)
        stream_4.synchronize()
        with stream_41:
            _, _, Vt4 = cp.linalg.svd(A4_flat, full_matrices=False)
            
            V4 = Vt4.transpose(0, 2, 1)
            
            Q4[:] = cp.array([V4[:, 0, 3] / V4[:, 3, 3], V4[:, 1, 3] / V4[:, 3, 3], V4[:, 2, 3] / V4[:, 3, 3], cp.ones(V4.shape[0])]).T.reshape(f4[0], f4[1], f4[2], 4) # (1, 22, 1, 4)
            
            # real_dist4
            result_c4 = cp.einsum('cik,btk->cit', P_cam_comb4, Q4[:,:,0,:], optimize=True)
            rpj_coor_4 = cp.stack((cp.expand_dims((result_c4[0] / result_c4[2]).T, axis=(0, 2)), cp.expand_dims((result_c4[1] / result_c4[2]).T, axis=(0, 2))), axis=-1)
            rpj = cp.sqrt((rpj_coor_4[:, :, :, :, 0] - prep_4[:, :, :, :, 0]) ** 2 + (rpj_coor_4[:, :, :, :, 1] - prep_4[:, :, :, :, 1]) ** 2)
            real_dist4[:] = cp.max(cp.expand_dims(cp.sqrt(cp.sum((Q4[:, :, 0, 0:3] - cp.stack(cam_coord, axis=0)[:, None, :]) ** 2, axis=-1)).T, axis=(0, 2)) * rpj, axis=-1) # (1, 22, 1)
            # print(time.time()-t1)
        with stream_3:
            t2 = time.time()
            prep_3[:] = cp.stack([bilinear_interpolate_cupy(mappingx, prep_3[:, :, :, :, 0], prep_3[:, :, :, :, 1]), bilinear_interpolate_cupy(mappingy, prep_3[:, :, :, :, 0], prep_3[:, :, :, :, 1]), prep_3[:, :, :, :, 2]], axis=-1)  # Shape: (155, 22, 4, 3, 3)
            
            prep_3like[:] = cp.min(prep_3[:,:,:,:,2],axis=3) # (1, 22, 4)
            
            A3[:] = cp.stack([((P0_c3 - prep_3[..., 0:1] * P2_c3) * prep_3[..., 2:3]), ((P1_c3 - prep_3[..., 1:2] * P2_c3) * prep_3[..., 2:3])], axis=-2).reshape(1, 22, 4, 6, 4) # (1, 22, 4, 6, 4)
            f3 = cp.shape(A3)
            A3_flat = A3.reshape(-1, 6, 4)
        
        stream_3.synchronize()
        with stream_31:
            _, _, Vt3 = cp.linalg.svd(A3_flat, full_matrices=False)
            
            V3 = Vt3.transpose(0, 2, 1)
            Q3[:] = cp.array([V3[:, 0, 3] / V3[:, 3, 3], V3[:, 1, 3] / V3[:, 3, 3], V3[:, 2, 3] / V3[:, 3, 3], cp.ones(V3.shape[0])]).T.reshape(f3[0], f3[1], f3[2], 4) # (1, 22, 4, 4)
            prep_3like[:] = cp.clip(prep_3like, likelihood_threshold, cp.inf)
            # print(time.time()-t2)
            Q3_bug = Q3[cp.arange(Q3.shape[0])[:, None], cp.arange(Q3.shape[1])[None, :], cp.argmax(~cp.isinf(prep_3like), axis=2), :][:, :, cp.newaxis, :]
            # real_dist3
            result_c3 = cp.einsum('ncik,nbtk->ncit', P_cam_comb3, Q3.transpose(2, 0, 1, 3), optimize=True)
            rpj_coor_3 = cp.stack((cp.expand_dims((result_c3[:, 0] / result_c3[:, 2]).transpose(2, 0, 1), axis=0), cp.expand_dims((result_c3[:, 1] / result_c3[:, 2]).transpose(2, 0, 1), axis=0)),axis = -1)
            rpj = cp.sqrt((rpj_coor_3[:, :, :, :, 0] - prep_3[:, :, :, :, 0]) ** 2 + (rpj_coor_3[:, :, :, :, 1] - prep_3[:, :, :, :, 1]) ** 2)
            real_dist3[:] = cp.max(cp.expand_dims(cp.sqrt(cp.sum((Q3_bug.transpose(2, 0, 1, 3)[:, :, :, 0:3] - cam_coord[comb_3][:, :, None, :]) ** 2, axis=-1)).transpose(2, 0, 1), axis=0) * rpj, axis=-1) # (1, 22, 4)
        
        with stream_2:
            t3 = time.time()
            prep_2[:] = cp.stack([bilinear_interpolate_cupy(mappingx, prep_2[:, :, :, :, 0], prep_2[:, :, :, :, 1]), bilinear_interpolate_cupy(mappingy, prep_2[:, :, :, :, 0], prep_2[:, :, :, :, 1]), prep_2[:, :, :, :, 2]], axis=-1)  # Shape: (155, 22, 4, 3, 3)            
            prep_2like[:] = cp.min(prep_2[:,:,:,:,2],axis=3) # (1, 22, 6)
            A2[:] = cp.stack([((P0_c2 - prep_2[..., 0:1] * P2_c2) * prep_2[..., 2:3]), ((P1_c2 - prep_2[..., 1:2] * P2_c2) * prep_2[..., 2:3])], axis=-2).reshape(1, 22, 6, 4, 4) # (1, 22, 6, 4, 4)
            f2 = cp.shape(A2)
            A2_flat = A2.reshape(-1, 4, 4)
        
        stream_2.synchronize()
        with stream_21:
            _, _, Vt2 = cp.linalg.svd(A2_flat, full_matrices=False)
            
            V2 = Vt2.transpose(0, 2, 1)
            Q2[:] = cp.array([V2[:, 0, 3] / V2[:, 3, 3], V2[:, 1, 3] / V2[:, 3, 3], V2[:, 2, 3] / V2[:, 3, 3], cp.ones(V2.shape[0])]).T.reshape(f2[0], f2[1], f2[2], 4) # (1, 22, 6, 4)
            prep_2like[:] = cp.clip(prep_2like, likelihood_threshold, cp.inf)
            # print(time.time()-t3)
            Q2_bug = Q2[cp.arange(Q2.shape[0])[:, None], cp.arange(Q2.shape[1])[None, :], cp.argmax(~cp.isinf(prep_2like), axis=2), :][:, :, cp.newaxis, :]
            # real_dist2
            result_c2 = cp.einsum('ncik,nbtk->ncit', P_cam_comb2, Q2.transpose(2, 0, 1, 3), optimize=True)

            rpj_coor_2 = cp.stack((cp.expand_dims((result_c2[:, 0] / result_c2[:, 2]).transpose(2, 0, 1), axis=0), cp.expand_dims((result_c2[:, 1] / result_c2[:, 2]).transpose(2, 0, 1), axis=0)),axis = -1)
            rpj = cp.sqrt((rpj_coor_2[:, :, :, :, 0] - prep_2[:, :, :, :, 0]) ** 2 + (rpj_coor_2[:, :, :, :, 1] - prep_2[:, :, :, :, 1]) ** 2)
            real_dist2[:] = cp.max(cp.expand_dims(cp.sqrt(cp.sum((Q2_bug.transpose(2, 0, 1, 3)[:, :, :, 0:3] - cam_coord[comb_2][:, :, None, :]) ** 2, axis=-1)).transpose(2, 0, 1), axis=0) * rpj, axis=-1) # (1, 22, 6)
        stream_41.synchronize()
        stream_31.synchronize()
        stream_21.synchronize()
        
        # delete the liklelihoood vlue which is too low
        prep_like = cp.concatenate((prep_4like, prep_3like, prep_2like),axis =2)
        real_dist = cp.concatenate((real_dist4, real_dist3, real_dist2), axis=2)
        real_dist[cp.where(prep_like < likelihood_threshold)] = cp.inf
        non_inf_mask = ~cp.isinf(real_dist)
        min_locations_nan = cp.argmax(non_inf_mask, axis=2)
        real_dist_dynamic = cp.copy(real_dist)
        list_dynamic_mincam=  {'Hip':4,'RHip':4,'RKnee':3,'RAnkle':3,'RBigToe':3,'RSmallToe':3,'RHeel':3,'LHip':4,'LKnee':3,'LAnkle':3,'LBigToe':3,'LSmallToe':3,'LHeel':3,'Neck':3,'Head':2,'Nose':2,'RShoulder':3,'RElbow':3,'RWrist':3,'LShoulder':3,'LElbow':3,'LWrist':3}
        list_dynamic_mincam_prep = [map_to_listdynamic(value) for value in list_dynamic_mincam.values()]
        ## setting the list dynamic

        for i in range(22):
            real_dist_dynamic[:,i,list_dynamic_mincam_prep[i]] = cp.inf
        # indices = cp.arange(22)
        # real_dist_dynamic[:, indices, list_dynamic_mincam_prep] = cp.inf

        ## find the minimum combination
        Q = cp.concatenate((Q4, Q3, Q2), axis=2)
        temp_shape = cp.shape(Q)
        checkinf = cp.min(real_dist_dynamic,axis =2)
        min_locations = cp.argmin(real_dist_dynamic, axis=2)
        loc =cp.where(checkinf==cp.inf)
        min_locations[loc] = min_locations_nan[loc]
        batch_indices, time_indices = cp.meshgrid(cp.arange(temp_shape[0]), cp.arange(temp_shape[1]), indexing='ij')
        
        Q_selected = Q[batch_indices, time_indices, min_locations][:,:,0:3]
        Q_tot_gpu = Q_selected.reshape(Q_selected.shape[0], -1)
        if keypoints.shape!=(0, 0, 3): print(time.time()-t4)
    #         ax.cla()  # 清除當前圖表
    #         points = Q_tot_gpu[0].reshape(22, 3)
    #         x, y, z = points[:, 0], points[:, 1], points[:, 2]


    #         ax.scatter(x, y, z, c='blue', marker='o')
            
 
    #         ax.set_xlabel('X')
    #         ax.set_ylabel('Y')
    #         ax.set_zlabel('Z')
    #         ax.set_title(f'3D Scatter Plot - Frame {i+1}')
           
    #         ax.set_xlim(-2, 2)
    #         ax.set_ylim(-2, 2)
    #         ax.set_zlim(-2, 2)

            
    #         plt.draw()
    #         fig.canvas.flush_events()
            
    # plt.ioff()
# if __name__ == "__main__":
tri()