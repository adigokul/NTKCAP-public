#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
    ###########################################################################
    ## TRACKING OF PERSON OF INTEREST                                        ##
    ###########################################################################
    
    Openpose detects all people in the field of view.
    - multi_person = false: Triangulates the most prominent person
    - multi_person = true: Triangulates persons across views
                           Tracking them across time frames is done in the triangulation stage.
    If multi_person = false, this module tries all possible triangulations of a chosen
    anatomical point, and chooses the person for whom the reprojection error is smallest. 
    
    If multi_person = true, it computes the distance between epipolar lines (camera to 
    keypoint lines) for all persons detected in all views, and selects the best correspondences. 
    The computation of the affinity matrix from the distance is inspired from the EasyMocap approach.
    
    INPUTS: 
    - a calibration file (.toml extension)
    - json files from each camera folders with several detected persons
    - a Config.toml file
    - a skeleton model
    
    OUTPUTS: 
    - json files for each camera with only one person of interest
    
'''


## INIT
import os
import glob
import fnmatch
import numpy as np
import json
import itertools as it
import toml
from tqdm import tqdm
import cv2
from anytree import RenderTree
from anytree.importer import DictImporter
import logging
from scipy.spatial import ConvexHull, Delaunay

from Pose2Sim.common_multi import retrieve_calib_params, computeP, weighted_triangulation, \
    reprojection, euclidean_distance, natural_sort
from Pose2Sim.skeletons import *


## AUTHORSHIP INFORMATION
__author__ = "David Pagnon"
__copyright__ = "Copyright 2021, Pose2Sim"
__credits__ = ["David Pagnon"]
__license__ = "BSD 3-Clause License"
__version__ = "0.8.2"
__maintainer__ = "David Pagnon"
__email__ = "contact@david-pagnon.com"
__status__ = "Development"


## FUNCTIONS
def persons_combinations(json_files_framef):
    '''
    Find all possible combinations of detected persons' ids. 
    Person's id when no person detected is set to -1.
    
    INPUT:
    - json_files_framef: list of strings

    OUTPUT:
    - personsIDs_comb: array, list of lists of int
    '''
    
    n_cams = len(json_files_framef)
    
    # amount of persons detected for each cam
    nb_persons_per_cam = []
    for c in range(n_cams): # 0-3
        with open(json_files_framef[c], 'r') as js:
            nb_persons_per_cam += [len(json.load(js)['people'])] # 長度為4的list，分別為每台相機分別偵測到幾個人Ex:[1, 1, 1, 1]
    
    # persons_combinations
    id_no_detect = [i for i, x in enumerate(nb_persons_per_cam) if x == 0]  # 沒有偵測到人的相機編號，Ex[0, 1]，若無則為[]
    nb_persons_per_cam = [x if x != 0 else 1 for x in nb_persons_per_cam] # 把偵測到為0的替換為1 Ex:[1, 1, 2, 3]
    range_persons_per_cam = [range(nb_persons_per_cam[c]) for c in range(n_cams)] # Ex:[range(0, 1), range(0, 1), range(0, 2), range(0, 3)]
    personsIDs_comb = np.array(list(it.product(*range_persons_per_cam)), float) # 所有組合(從各自range去推導)
    personsIDs_comb[:,id_no_detect] = np.nan # 把沒偵測到人的相機組合都設為Nan
    
    return personsIDs_comb # Ex:[[nan nan  0.  0.], [nan nan  0.  1.], [nan nan  0.  2.], [nan nan  1.  0.], [nan nan  1.  1.], [nan nan  1.  2.]]


def best_persons_and_cameras_combination(config, json_files_framef, personsIDs_combinations, projection_matrices, tracked_keypoint_id):
    '''
    At the same time, chooses the right person among the multiple ones found by
    OpenPose & excludes cameras with wrong 2d-pose estimation.
    
    1. triangulate the tracked keypoint for all possible combinations of people,
    2. compute difference between reprojection & original openpose detection,
    3. take combination with smallest difference.
    If error is too big, take off one or several of the cameras until err is 
    lower than "max_err_px".
    
    INPUTS:
    - a Config.toml file
    - json_files_framef: list of strings
    - personsIDs_combinations: array, list of lists of int
    - projection_matrices: list of arrays
    - tracked_keypoint_id: int

    OUTPUTS:
    - error_min: float
    - persons_and_cameras_combination: array of ints
    '''
    
    error_threshold_tracking = config.get('personAssociation').get('reproj_error_threshold_association') # 20 px
    min_cameras_for_triangulation = config.get('triangulation').get('min_cameras_for_triangulation') # 2
    likelihood_threshold = config.get('triangulation').get('likelihood_threshold') # 0.2

    n_cams = len(json_files_framef) # 4
    error_min = np.inf # infinity
    nb_cams_off = 0 # cameras will be taken-off until the reprojection error is under threshold
    
    while error_min > error_threshold_tracking and n_cams - nb_cams_off >= min_cameras_for_triangulation:
        # Try all persons combinations
        for combination in personsIDs_combinations: # 所有組合中一組一組看
            # Get x,y,likelihood values from files
            x_files, y_files,likelihood_files = [], [], []
            for index_cam, person_nb in enumerate(combination): # 逐一組合檢查:長度是相機數 Ex:[nan nan  1.  1.] ->長度還是4
                with open(json_files_framef[index_cam], 'r') as json_f: # load json file
                    js = json.load(json_f)
                    try: # 如果不是nan則讀點座標
                        x_files.append( js['people'][int(person_nb)]['pose_keypoints_2d'][tracked_keypoint_id*3] ) # Neck的x座標 (因為一組有三種，間隔要*3)
                        y_files.append( js['people'][int(person_nb)]['pose_keypoints_2d'][tracked_keypoint_id*3+1] ) # Neck的y座標
                        likelihood_files.append( js['people'][int(person_nb)]['pose_keypoints_2d'][tracked_keypoint_id*3+2] ) # Neck的信心
                    except: # 如果是nan則為座標為nan
                        x_files.append(np.nan)
                        y_files.append(np.nan)
                        likelihood_files.append(np.nan)
            # x_files, y_files, likelihood_files 分別包含四台相機的Neck參數(座標+信心)
            # Replace likelihood by 0. if under likelihood_threshold
            likelihood_files = [0. if lik < likelihood_threshold else lik for lik in likelihood_files]
            
            # For each persons combination, create subsets with "nb_cams_off" cameras excluded
            id_cams_off = list(it.combinations(range(len(combination)), nb_cams_off)) # it.combinations(range(), 每組包含的元素數量)
            combinations_with_cams_off = np.array([combination.copy()]*len(id_cams_off)) # 重複複製組合數
            for i, id in enumerate(id_cams_off): # 關相機
                combinations_with_cams_off[i,id] = np.nan

            # Try all subsets
            error_comb = []
            for comb in combinations_with_cams_off:
                # Filter x, y, likelihood, projection_matrices, with subset
                # comb Ex:[nan nan  0.  1.]
                # 只保留非Nan即有偵測到人的相機
                x_files_filt = [x_files[i] for i in range(len(comb)) if not np.isnan(comb[i])]
                y_files_filt = [y_files[i] for i in range(len(comb)) if not np.isnan(comb[i])]
                likelihood_files_filt = [likelihood_files[i] for i in range(len(comb)) if not np.isnan(comb[i])]
                projection_matrices_filt = [projection_matrices[i] for i in range(len(comb)) if not np.isnan(comb[i])]
                
                # Triangulate 2D points
                Q_comb = weighted_triangulation(projection_matrices_filt, x_files_filt, y_files_filt, likelihood_files_filt)
                
                # Reprojection
                x_calc, y_calc = reprojection(projection_matrices_filt, Q_comb)
                                
                # Reprojection error
                error_comb_per_cam = []
                for cam in range(len(x_calc)):
                    q_file = (x_files_filt[cam], y_files_filt[cam])
                    q_calc = (x_calc[cam], y_calc[cam])
                    error_comb_per_cam.append( euclidean_distance(q_file, q_calc) )
                error_comb.append( np.mean(error_comb_per_cam) )
            
            error_min = min(error_comb) # 找誤差最小的
            persons_and_cameras_combination = combinations_with_cams_off[np.argmin(error_comb)] # np.argmin() 最小值index
            
            if error_min < error_threshold_tracking: # 如果小於20px則OK
                break

        nb_cams_off += 1
    
    return error_min, persons_and_cameras_combination

def triangulate_comb_multi(comb, coords, P_all, calib_params, config):
    '''
    Triangulate 2D points and compute reprojection error for a combination of cameras.
    INPUTS:
    - comb: list of ints: combination of persons' ids for each camera
    - coords: array: x, y, likelihood for each camera
    - P_all: list of arrays: projection matrices for each camera
    - calib_params: dict: calibration parameters
    - config: dictionary from Config.toml file
    OUTPUTS:
    - error_comb: float: reprojection error
    - comb: list of ints: combination of persons' ids for each camera
    - Q_comb: array: 3D coordinates of the triangulated point
    ''' 

    undistort_points = config.get('triangulation').get('undistort_points')
    likelihood_threshold = config.get('personAssociation').get('likelihood_threshold_association')

    # Replace likelihood by 0. if under likelihood_threshold
    coords[:,2][coords[:,2] < likelihood_threshold] = 0.
    comb[coords[:,2] == 0.] = np.nan

    # Filter coords and projection_matrices containing nans
    coords_filt = [coords[i] for i in range(len(comb)) if not np.isnan(comb[i])]
    projection_matrices_filt = [P_all[i] for i in range(len(comb)) if not np.isnan(comb[i])]
    if undistort_points:
        calib_params_R_filt = [calib_params['R'][i] for i in range(len(comb)) if not np.isnan(comb[i])]
        calib_params_T_filt = [calib_params['T'][i] for i in range(len(comb)) if not np.isnan(comb[i])]
        calib_params_K_filt = [calib_params['K'][i] for i in range(len(comb)) if not np.isnan(comb[i])]
        calib_params_dist_filt = [calib_params['dist'][i] for i in range(len(comb)) if not np.isnan(comb[i])]

    # Triangulate 2D points
    try:
        x_files_filt, y_files_filt, likelihood_files_filt = np.array(coords_filt).T
        Q_comb = weighted_triangulation(projection_matrices_filt, x_files_filt, y_files_filt, likelihood_files_filt)
    except:
        Q_comb = [np.nan, np.nan, np.nan, 1.]

    # Reprojection
    if undistort_points:
        coords_2D_kpt_calc_filt = [cv2.projectPoints(np.array(Q_comb[:-1]), calib_params_R_filt[i], calib_params_T_filt[i], calib_params_K_filt[i], calib_params_dist_filt[i])[0] for i in range(len(Q_comb))]
        x_calc = [coords_2D_kpt_calc_filt[i][0,0,0] for i in range(len(Q_comb))]
        y_calc = [coords_2D_kpt_calc_filt[i][0,0,1] for i in range(len(Q_comb))]
    else:
        x_calc, y_calc = reprojection(projection_matrices_filt, Q_comb)

    # Reprojection error
    error_comb_per_cam = []
    for cam in range(len(x_calc)):
        q_file = (x_files_filt[cam], y_files_filt[cam])
        q_calc = (x_calc[cam], y_calc[cam])
        error_comb_per_cam.append( euclidean_distance(q_file, q_calc) )
    error_comb = np.mean(error_comb_per_cam)

    return error_comb, comb, Q_comb


def read_json(js_file):
    '''
    Read OpenPose json file
    '''
    with open(js_file, 'r') as json_f:
        js = json.load(json_f)
        json_data = []
        for people in range(len(js['people'])):
            # if len(js['people'][people]['pose_keypoints_2d']) < 3: continue
            # else:
            json_data.append(js['people'][people]['pose_keypoints_2d'])
    return json_data


def compute_rays(json_coord, calib_params, cam_id):
    '''
    Plucker coordinates of rays from camera to each joint of a person
    Plucker coordinates: camera to keypoint line direction (size 3) 
                         moment: origin ^ line (size 3)
                         additionally, confidence

    INPUTS:
    - json_coord: x, y, likelihood for a person seen from a camera (list of 3*joint_nb)
    - calib_params: calibration parameters from retrieve_calib_params('calib.toml')
    - cam_id: camera id (int)

    OUTPUT:
    - plucker: array. nb joints * (6 plucker coordinates + 1 likelihood)
    '''

    x = json_coord[0::3]
    y = json_coord[1::3]
    likelihood = json_coord[2::3]
    # print(json_coord)
    # print(x)
    # print(y)
    
    inv_K = calib_params['inv_K'][cam_id]
    R_mat = calib_params['R_mat'][cam_id]
    T = calib_params['T'][cam_id]

    cam_center = -R_mat.T @ T
    plucker = []
    for i in range(len(x)):
        q = np.array([x[i], y[i], 1])
        norm_Q = R_mat.T @ (inv_K @ q -T) #計算從相機中心到轉換後點的向量
        
        line = norm_Q - cam_center
        norm_line = line/np.linalg.norm(line)
        moment = np.cross(cam_center, norm_line)
        plucker.append(np.concatenate([norm_line, moment, [likelihood[i]]]))

    return np.array(plucker)


def broadcast_line_to_line_distance(p0, p1):
    '''
    Compute the distance between two lines in 3D space.

    see: https://faculty.sites.iastate.edu/jia/files/inline-files/plucker-coordinates.pdf
    p0 = (l0,m0), p1 = (l1,m1)
    dist = | (l0,m0) * (l1,m1) | / || l0 x l1 ||
    (l0,m0) * (l1,m1) = l0 @ m1 + m0 @ l1 (reciprocal product)
    
    No need to divide by the norm of the cross product of the directions, since we
    don't need the actual distance but whether the lines are close to intersecting or not
    => dist = | (l0,m0) * (l1,m1) |

    INPUTS:
    - p0: array(nb_persons_detected * 1 * nb_joints * 7 coordinates)
    - p1: array(1 * nb_persons_detected * nb_joints * 7 coordinates)

    OUTPUT:
    - dist: distances between the two lines (not normalized). 
            array(nb_persons_0 * nb_persons_1 * nb_joints)
    '''

    product = np.sum(p0[..., :3] * p1[..., 3:6], axis=-1) + np.sum(p1[..., :3] * p0[..., 3:6], axis=-1)
    dist = np.abs(product) # dist[i, j, k]表示p0中第i個人的第k個關節與p1中第j個人的第k個關節之間的距離。

    return dist


def compute_affinity(all_json_data_f, calib_params, cum_persons_per_view, reconstruction_error_threshold=0.1):
    '''
    Compute the affinity between all the people in the different views.

    The affinity is defined as 1 - distance/max_distance, with distance the
    distance between epipolar lines in each view (reciprocal product of Plucker 
    coordinates).

    Another approach would be to project one epipolar line onto the other camera
    plane and compute the line to point distance, but it is more computationally 
    intensive (simple dot product vs. projection and distance calculation). 
    
    INPUTS:
    - all_json_data_f: list of json data. For frame f, nb_views*nb_persons*(x,y,likelihood)*nb_joints
    - calib_params: calibration parameters from retrieve_calib_params('calib.toml')
    - cum_persons_per_view: cumulative number of persons per view
    - reconstruction_error_threshold: maximum distance between epipolar lines to consider a match

    OUTPUT:
    - affinity: affinity matrix between all the people in the different views. 
                (nb_views*nb_persons_per_view * nb_views*nb_persons_per_view)
    '''

    # Compute plucker coordinates for all keypoints for each person in each view
    # pluckers_f: dims=(camera, person, joint, 7 coordinates)
    pluckers_f = []
    for cam_id, json_cam  in enumerate(all_json_data_f): # 分相機處理
        pluckers = []
        for json_coord in json_cam:
            # num_kp * 7
            plucker = compute_rays(json_coord, calib_params, cam_id) # LIMIT TO 15 JOINTS? json_coord[:15*3]
            pluckers.append(plucker) # num_detthiscam * num_kp * 7
        pluckers = np.array(pluckers) 
        pluckers_f.append(pluckers)# num_cam * num_detthiscam * num_kp * 7

    # Compute affinity matrix
    distance = np.zeros((cum_persons_per_view[-1], cum_persons_per_view[-1])) + 2*reconstruction_error_threshold # 建立矩陣大小為最大可能偵測人數*最大可能偵測人數(每台相機偵測人數總和)
    for compared_cam0, compared_cam1 in it.combinations(range(len(all_json_data_f)), 2): # 每次選2台出來計算
        # skip when no detection for a camera
        if cum_persons_per_view[compared_cam0] == cum_persons_per_view[compared_cam0+1] \
            or cum_persons_per_view[compared_cam1] == cum_persons_per_view[compared_cam1 +1]:
            continue

        # compute distance
        p0 = pluckers_f[compared_cam0][:,None] # add coordinate on second dimension num_kp * 1 * 7
        p1 = pluckers_f[compared_cam1][None,:] # add coordinate on first dimension  1 * num_kp * 7
        dist = broadcast_line_to_line_distance(p0, p1)  # dist[i, j, k]表示p0中第i個人的第k個關節與p1中第j個人的第k個關節之間的距離。
        likelihood = np.sqrt(p0[..., -1] * p1[..., -1])
        mean_weighted_dist = np.sum(dist*likelihood, axis=-1)/(1e-5 + likelihood.sum(axis=-1)) # array(nb_persons_0 * nb_persons_1)
        # mean_weighted_dist[i, j]表示A集合中第i個人和B集合中第j個人之間的加權平均距離。
        # populate distance matrix
        distance[cum_persons_per_view[compared_cam0]:cum_persons_per_view[compared_cam0+1], \
                 cum_persons_per_view[compared_cam1]:cum_persons_per_view[compared_cam1+1]] \
                 = mean_weighted_dist
        distance[cum_persons_per_view[compared_cam1]:cum_persons_per_view[compared_cam1+1], \
                 cum_persons_per_view[compared_cam0]:cum_persons_per_view[compared_cam0+1]] \
                 = mean_weighted_dist.T

    # compute affinity matrix and clamp it to zero when distance > reconstruction_error_threshold
    distance[distance > reconstruction_error_threshold] = reconstruction_error_threshold
    affinity = 1 - distance / reconstruction_error_threshold

    return affinity


def circular_constraint(cum_persons_per_view):
    '''
    A person can be matched only with themselves in the same view, and with any 
    person from other views

    INPUT:
    - cum_persons_per_view: cumulative number of persons per view

    OUTPUT:
    - circ_constraint: circular constraint matrix
    '''

    circ_constraint = np.identity(cum_persons_per_view[-1])
    for i in range(len(cum_persons_per_view)-1):
        circ_constraint[cum_persons_per_view[i]:cum_persons_per_view[i+1], cum_persons_per_view[i+1]:cum_persons_per_view[-1]] = 1
        circ_constraint[cum_persons_per_view[i+1]:cum_persons_per_view[-1], cum_persons_per_view[i]:cum_persons_per_view[i+1]] = 1
    
    return circ_constraint


def SVT(matrix, threshold):
    '''
    Find a low-rank approximation of the matrix using Singular Value Thresholding.

    INPUTS:
    - matrix: matrix to decompose
    - threshold: threshold for singular values

    OUTPUT:
    - matrix_thresh: low-rank approximation of the matrix
    '''
    
    U, s, Vt = np.linalg.svd(matrix) # decompose matrix
    s_thresh = np.maximum(s - threshold, 0) # set smallest singular values to zero
    matrix_thresh = U @ np.diag(s_thresh) @ Vt # recompose matrix

    return matrix_thresh


def matchSVT(affinity, cum_persons_per_view, circ_constraint, max_iter = 20, w_rank = 50, tol = 1e-4, w_sparse=0.1):
    '''
    Find low-rank approximation of 'affinity' while satisfying the circular constraint.

    INPUTS:
    - affinity: affinity matrix between all the people in the different views
    - cum_persons_per_view: cumulative number of persons per view
    - circ_constraint: circular constraint matrix
    - max_iter: maximum number of iterations
    - w_rank: threshold for singular values
    - tol: tolerance for convergence
    - w_sparse: regularization parameter

    OUTPUT:
    - new_aff: low-rank approximation of the affinity matrix
    '''

    new_aff = affinity.copy()
    N = new_aff.shape[0]
    index_diag = np.arange(N)
    new_aff[index_diag, index_diag] = 0.
    # new_aff = (new_aff + new_aff.T)/2 # symmetric by construction

    Y = np.zeros_like(new_aff) # Initial deviation matrix / residual ()
    W = w_sparse - new_aff # Initial sparse matrix / regularization (prevent overfitting)
    mu = 64 # initial step size

    for iter in range(max_iter):
        new_aff0 = new_aff.copy()
        
        Q = new_aff + Y*1.0/mu
        Q = SVT(Q,w_rank/mu)
        new_aff = Q - (W + Y)/mu

        # Project X onto dimGroups
        for i in range(len(cum_persons_per_view) - 1):
            ind1, ind2 = cum_persons_per_view[i], cum_persons_per_view[i + 1]
            new_aff[ind1:ind2, ind1:ind2] = 0
            
        # Reset diagonal elements to one and ensure X is within valid range [0, 1]
        new_aff[index_diag, index_diag] = 1.
        new_aff[new_aff < 0] = 0
        new_aff[new_aff > 1] = 1
        
        # Enforce circular constraint
        new_aff = new_aff * circ_constraint
        new_aff = (new_aff + new_aff.T) / 2 # kept just in case X loses its symmetry during optimization 
        Y = Y + mu * (new_aff - Q)
        
        # Compute convergence criteria: break if new_aff is close enough to Q and no evolution anymore
        pRes = np.linalg.norm(new_aff - Q) / N # primal residual (diff between new_aff and SVT result)
        dRes = mu * np.linalg.norm(new_aff - new_aff0) / N # dual residual (diff between new_aff and previous new_aff)
        if pRes < tol and dRes < tol:
            break
        if pRes > 10 * dRes: mu = 2 * mu
        elif dRes > 10 * pRes: mu = mu / 2

        iter +=1

    return new_aff


def person_index_per_cam(affinity, cum_persons_per_view, min_cameras_for_triangulation):
    '''
    For each detected person, gives their index for each camera

    INPUTS:
    - affinity: affinity matrix between all the people in the different views
    - min_cameras_for_triangulation: exclude proposals if less than N cameras see them

    OUTPUT:
    - proposals: 2D array: n_persons * n_cams
    '''

    # index of the max affinity for each group (-1 if no detection)
    proposals = []
    for row in range(affinity.shape[0]):
        proposal_row = []
        for cam in range(len(cum_persons_per_view)-1): # 除去起始0，四台相機
            id_persons_per_view = affinity[row, cum_persons_per_view[cam]:cum_persons_per_view[cam+1]]
            # argmax 最大值index
            proposal_row += [np.argmax(id_persons_per_view) if (len(id_persons_per_view)>0 and max(id_persons_per_view)>0) else -1]
        proposals.append(proposal_row)
    proposals = np.array(proposals, dtype=float)

    # remove duplicates and order
    proposals, nb_detections = np.unique(proposals, axis=0, return_counts=True)
    proposals = proposals[np.argsort(nb_detections)[::-1]]

    # remove row if any value is the same in previous rows at same index (nan!=nan so nan ignored)
    proposals[proposals==-1] = np.nan
    mask = np.ones(proposals.shape[0], dtype=bool)
    for i in range(1, len(proposals)):
        mask[i] = ~np.any(proposals[i] == proposals[:i], axis=0).any()
    proposals = proposals[mask]

    # remove identifications if less than N cameras see them
    nb_cams_per_person = [np.count_nonzero(~np.isnan(p)) for p in proposals]
    proposals = np.array([p for (n,p) in zip(nb_cams_per_person, proposals) if n >= min_cameras_for_triangulation])

    return proposals


def rotation_vector_to_matrix(rotation_vector):
    theta = np.linalg.norm(rotation_vector)
    if theta > 0:
        k = rotation_vector / theta
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    else:
        R = np.eye(3)
    return R
    
def compute_normal_vector(p1, p2, p3):
    v1 = p2 - p1
    v2 = p3 - p1
    normal = np.cross(v1, v2)
    return normal / np.linalg.norm(normal)

def is_point_in_hull(point, delaunay):
    return delaunay.find_simplex(point) >= 0

import numpy as np
import toml

def calculate_camera_position(calib_file):
    calib = toml.load(calib_file)
    camera_positions = []
    
    for cam in calib.keys():
        if cam != 'metadata':
            K = np.array(calib[cam]['matrix'])
            dist = np.array(calib[cam]['distortions'])
            rotation_vector = np.array(calib[cam]['rotation'])
            translation_vector = np.array(calib[cam]['translation'])

            R, _ = cv2.Rodrigues(rotation_vector)
            cam_center = -np.dot(R.T, translation_vector)
            camera_positions.append(cam_center)
    
    return camera_positions

def projection_matrix(idx, calib_file):
    calib = toml.load(calib_file)
    projection_matrices = []
    for cam in calib:
        if cam != 'metadata':
            K = np.array(calib[cam]['matrix'])
            R_vec = np.array(calib[cam]['rotation'])
            T = np.array(calib[cam]['translation']).reshape(3, 1)
            R, _ = cv2.Rodrigues(R_vec)  # 旋轉向量轉換為旋轉矩陣

            # 構建投影矩陣 P = K [R|T]
            RT = np.hstack((R, T))
            P = np.dot(K, RT)
            projection_matrices.append(P)
    
    return projection_matrices[idx]

def rewrite_js_file(n_cams, json_tracked_files_f, js_allin_range):
    for cam in range(n_cams):
        with open(json_tracked_files_f[cam], 'w') as json_tracked_f:
            json_tracked_f.write(json.dumps(js_allin_range[cam]))

from Pose2Sim.common import weighted_triangulation
import re

def outsider(js, calib_file, frame, P, frame_range, json_tracked_files_f, state):
    
    kp_idx = 18 # Neck
    nb_det_max_p = max([len(js[i]['people']) for i in range(len(js))]) # maximum num of people detected
    person_to_remove = [] # index for outsiders
    for p in range(nb_det_max_p): # check if the person is outsid the area
        
        count = 0 # count for how many cameras have not detected the person
        for i in js:
            if i["people"][p] == {}:
                count += 1
        if count >= len(js) - 2: # if there are more than 2 cameras didn't detect the person
            person_to_remove.append(p) # remove
            continue
        
        kp_2D = []
        cam_indices = []
        for cam in range(len(js)):
            if js[cam]['people'][p] != {}:
                # record 2D keypoints and the according camera indices
                cam_indices.append(cam)
                kp_2D.append(js[cam]['people'][p]['pose_keypoints_2d'][kp_idx*3:(kp_idx+1)*3] )
        
        P_all = []
        x_all = []
        y_all = []
        lik_all = []
        for idx in range(len(kp_2D)):
            P_all.append(P[cam_indices[idx]]) # projection matrix
            x_all.append(kp_2D[idx][0]) # x
            y_all.append(kp_2D[idx][1]) # y
            lik_all.append(kp_2D[idx][2]) # z
        
        Q = weighted_triangulation(P_all, x_all, y_all, lik_all) # triangulate 3D coordinate for Neck

        pos = []
        pos = calculate_camera_position(calib_file) # compute camera position
        
        pos_2d = [d[:2] for d in pos]
        delaunay = Delaunay(pos_2d) # create Convex hull
        
        if not is_point_in_hull(Q[:2], delaunay): # the person is inside or outside the area defined by four cameras using convex hull
            person_to_remove.append(p)
        
    
    if nb_det_max_p - len(person_to_remove) == 0: # there is no person in the area
        # state = None # whether the previous frame has person in the area : None means no, True means yes 
        if (state) and (frame != 0): # previous frame has person in the area. The situation happens in which the person walks out the area.
            
            all_range_fill = frame_range - frame # the frames that are needed to be filled up
            pattern = re.compile(r'(\d+)\.json') # create json file name template
            kp_f = []
            template_all = []
            
            for js_tr_file in json_tracked_files_f:
                start, end = re.search(pattern, js_tr_file).span(1)
                temp_js_tr_file = js_tr_file[:start] + '{:05d}' + js_tr_file[end:]
                template_all.append(temp_js_tr_file)
                pre_kp_js_file = temp_js_tr_file.format(frame-1)
                with open(pre_kp_js_file, 'r') as f:
                    data = json.load(f)
                kp_f.append(data)
            
            for i in range(all_range_fill):
                all_path = []
                for js_tr_file in template_all:
                    
                    kp_js_file = js_tr_file.format(frame+i)
                    all_path.append(kp_js_file)
                rewrite_js_file(len(kp_f), all_path, kp_f)
            
            state = 'terminate'
            return state

        else:
            js_empty_template = {
                "version": 1.3,
                "people": [
                    {
                        "person_id": [-1],
                        "pose_keypoints_2d": [],
                        "face_keypoints_2d": [],
                        "hand_left_keypoints_2d": [],
                        "hand_right_keypoints_2d": [],
                        "pose_keypoints_3d": [],
                        "face_keypoints_3d": [],
                        "hand_left_keypoints_3d": [],
                        "hand_right_keypoints_3d": []
                    }
                ]
            }
            js_new = [js_empty_template] * len(js)
            state = None
           
            rewrite_js_file(len(js), json_tracked_files_f, js_new)
            return state
    else:
        for index in sorted(person_to_remove, reverse=True):
            for cam in range(len(js)):
                if index <= len(js[cam]['people']):
                    del js[cam]['people'][index]
        if state is None:

            template_all = []
            pattern = re.compile(r'(\d+)\.json')
            for tracked_cam in json_tracked_files_f:
                tracked_cam_temp = tracked_cam
                match = pattern.search(tracked_cam_temp)
                if match:
                    num_str = match.group(1)
                    start, end = match.span(1)
                    template = tracked_cam_temp[:start] + '{:05d}' + tracked_cam_temp[end:]
                template_all.append(template)
                
            for idx in range(frame+1):
                # print(idx)
                all_path = []
                for tracked_cam in template_all:
                    path = tracked_cam.format(idx)
                    all_path.append(path)
    
                rewrite_js_file(len(js), all_path, js)
            state = True
            return state
        else:
            
            rewrite_js_file(len(js), json_tracked_files_f, js)
            state = True
            return state
    


def prepare_rewrite_json_files(json_tracked_files_f, json_files_f, proposals, n_cams, calib_file, f, P, frame_range, state):
    
    '''
    Write new json files with correct association of people across cameras.

    INPUTS:
    - json_tracked_files_f: list of strings: json files to write
    - json_files_f: list of strings: json files to read
    - proposals: 2D array: n_persons * n_cams
    - n_cams: int: number of cameras

    OUTPUT:
    - json files with correct association of people across cameras
    '''
    js_new_all = []
    for cam in range(n_cams):
        json_files_f
        with open(json_files_f[cam], 'r') as json_f:
            js = json.load(json_f)
            js_new = js.copy()
            js_new['people'] = []
            for new_comb in proposals:
                if not np.isnan(new_comb[cam]):

                    js_new['people'] += [js['people'][int(new_comb[cam])]]
                else:
                    js_new['people'] += [{}]
        js_new_all.append(js_new)
    
    state = outsider(js_new_all, calib_file, f, P, frame_range, json_tracked_files_f, state)
    if state == 'terminate':
        return False
    return state
    

def recap_tracking(config, error, nb_cams_excluded):
    '''
    Print a message giving statistics on reprojection errors (in pixel and in m)
    as well as the number of cameras that had to be excluded to reach threshold
    conditions. Also stored in User/logs.txt.

    INPUT:
    - a Config.toml file
    - error: dataframe 
    - nb_cams_excluded: dataframe

    OUTPUT:
    - Message in console
    '''
    
    # Read config
    project_dir = config.get('project').get('project_dir')
    if project_dir == '': project_dir = os.getcwd()
    poseTracked_folder_name = config.get('project').get('poseAssociated_folder_name')
    calib_folder_name = config.get('project').get('calib_folder_name')
    tracked_keypoint = config.get('personAssociation').get('tracked_keypoint')
    error_threshold_tracking = config.get('personAssociation').get('error_threshold_tracking')
    poseTracked_dir = os.path.join(project_dir, poseTracked_folder_name)
    calib_dir = os.path.join(project_dir, calib_folder_name)
    calib_file = glob.glob(os.path.join(calib_dir, '*.toml'))[0]
    
    # Error
    mean_error_px = np.around(np.mean(error), decimals=1)
    
    calib = toml.load(calib_file)
    calib_cam1 = calib[list(calib.keys())[0]]
    fm = calib_cam1['matrix'][0][0]
    Dm = euclidean_distance(calib_cam1['translation'], [0,0,0])
    mean_error_mm = np.around(mean_error_px * Dm / fm * 1000, decimals=1)
    
    # Excluded cameras
    mean_cam_off_count = np.around(np.mean(nb_cams_excluded), decimals=2)

    # Recap
    logging.info(f'\n--> Mean reprojection error for {tracked_keypoint} point on all frames is {mean_error_px} px, which roughly corresponds to {mean_error_mm} mm. ')
    logging.info(f'--> In average, {mean_cam_off_count} cameras had to be excluded to reach the demanded {error_threshold_tracking} px error threshold.')
    logging.info(f'\nTracked json files are stored in {poseTracked_dir}.')
    
from datetime import datetime
def track_2d_all(config):
    '''
    For each frame,
    - Find all possible combinations of detected persons
    - Triangulate 'tracked_keypoint' for all combinations
    - Reproject the point on all cameras
    - Take combination with smallest reprojection error
    - Write json file with only one detected person
    Print recap message
    
    INPUTS: 
    - a calibration file (.toml extension)
    - json files from each camera folders with several detected persons
    - a Config.toml file
    - a skeleton model
    
    OUTPUTS: 
    - json files for each camera with only one person of interest    
    '''
    
    # Read config
    
    project_dir = config.get('project').get('project_dir') 
    if project_dir == '': project_dir = os.getcwd() # walk1
    
    multi_person = config.get('project').get('multi_person') # warning:not finish yet
    
    calib_folder_name = config.get('project').get('calib_folder_name') # calib-2d
    poseTracked_folder_name = config.get('project').get('poseAssociated_folder_name') # pose-2d-tracked
    pose_folder_name = config.get('project').get('pose_folder_name') # pose-2d
    pose_model = config.get('pose').get('pose_model') # Halpe26
    tracked_keypoint = config.get('personAssociation').get('tracked_keypoint') # Neck
    min_cameras_for_triangulation = config.get('triangulation').get('min_cameras_for_triangulation')
    reconstruction_error_threshold = config.get('personAssociation').get('multi_person').get('reconstruction_error_threshold') # warning:not finish yet
    error_threshold_tracking = config.get('personAssociation').get('reproj_error_threshold_association')
    json_folder_extension =  config.get('project').get('pose_json_folder_extension') # img
    min_affinity = config.get('personAssociation').get('multi_person').get('min_affinity') # warning:not finish yet
    frame_range = config.get('project').get('frame_range') # []
    
    calib_dir = os.path.join(project_dir, calib_folder_name) # walk1/calib-2d
    calib_file = glob.glob(os.path.join(calib_dir, '*.toml'))[0] # walk1/calib-2d 中的.toml
    pose_dir = os.path.join(project_dir, pose_folder_name) # walk1/pose-2d
    poseTracked_dir = os.path.join(project_dir, poseTracked_folder_name) # walk1/pose-2d-tracked

    # projection matrix from toml calibration file
    P = computeP(calib_file) # 內外參矩陣，用於2d轉3d
    calib_params = retrieve_calib_params(calib_file) # size, intri_matrix, distortions, optim_K, inv_K, rotation_v, rotation_m, translation_v
    
    # selection of tracked keypoint id
    model = eval(pose_model)
    tracked_keypoint_id = [node.id for _, _, node in RenderTree(model) if node.name==tracked_keypoint][0]
    
    # 2d-pose files selection
    pose_listdirs_names = next(os.walk(pose_dir))[1] # [pose_cam1_json, ...]
    pose_listdirs_names = natural_sort(pose_listdirs_names) # pose_cam1_json 從小排到大
    json_dirs_names = [k for k in pose_listdirs_names if json_folder_extension in k] # [pose_cam1_json, ...]
    json_files_names = [fnmatch.filter(os.listdir(os.path.join(pose_dir, js_dir)), '*.json') for js_dir in json_dirs_names] # 長度為4的list，pose_cam1_json..中個別的json檔名
    json_files_names = [natural_sort(j) for j in json_files_names] # json檔名從小排到大
    json_files = [[os.path.join(pose_dir, j_dir, j_file) for j_file in json_files_names[j]] for j, j_dir in enumerate(json_dirs_names)] # list長為4，json的絕對路徑
    
    # 2d-pose-associated files creation
    if not os.path.exists(poseTracked_dir): os.mkdir(poseTracked_dir)   
    try: [os.mkdir(os.path.join(poseTracked_dir,k)) for k in json_dirs_names]
    except: pass
    json_tracked_files = [[os.path.join(poseTracked_dir, j_dir, j_file) for j_file in json_files_names[j]] for j, j_dir in enumerate(json_dirs_names)]
    
    # person's tracking
    f_range = [[min([len(j) for j in json_files])] if frame_range==[] else frame_range][0] # 155
    n_cams = len(json_dirs_names) # 4
    error_min_tot, cameras_off_tot = [], []



    Q_kpt = [np.array([0., 0., 0., 1.])]
    state = True
    for f in tqdm(range(*f_range)): # 所有幀數
        
        json_files_f = [json_files[c][f] for c in range(n_cams)] # 不同相機的同一幀，是檔案路徑
        json_tracked_files_f = [json_tracked_files[c][f] for c in range(n_cams)] # 不同相機的同一幀，儲存personassopciation後的keypoints
        
        Q_kpt_old = Q_kpt
        if not multi_person:
            # all possible combinations of persons
            personsIDs_comb = persons_combinations(json_files_f) 
            
            # choose person of interest and exclude cameras with bad pose estimation
            error_min, persons_and_cameras_combination = best_persons_and_cameras_combination(config, json_files_f, personsIDs_comb, P, tracked_keypoint_id)
            error_min_tot.append(error_min)
            cameras_off_count = np.count_nonzero(np.isnan(persons_and_cameras_combination))
            cameras_off_tot.append(cameras_off_count)
            
            # rewrite json files with only one person of interest
            for cam_nb, person_id in enumerate(persons_and_cameras_combination):
                with open(json_tracked_files_f[cam_nb], 'w') as json_tracked_f:
                    with open(json_files_f[cam_nb], 'r') as json_f:
                        js = json.load(json_f)
                        if not np.isnan(person_id):
                            js['people'] = [js['people'][int(person_id)]]
                        else: 
                            js['people'] = []
                    json_tracked_f.write(json.dumps(js))
        else:
            all_json_data_f = []

            for js_file in json_files_f:
                all_json_data_f.append(read_json(js_file)) # len=4, 每一個包含偵測到的點座標+信心
            
            persons_per_view = [0] + [len(j) for j in all_json_data_f] # [0, num_peo_cam1.... ]
            cum_persons_per_view = np.cumsum(persons_per_view) # [0, numpeocam1, numpeocam1+2....]
            affinity = compute_affinity(all_json_data_f, calib_params, cum_persons_per_view, reconstruction_error_threshold=reconstruction_error_threshold)    
            circ_constraint = circular_constraint(cum_persons_per_view)
            affinity = affinity * circ_constraint
            affinity = matchSVT(affinity, cum_persons_per_view, circ_constraint, max_iter = 20, w_rank = 50, tol = 1e-4, w_sparse=0.1)
            affinity[affinity<min_affinity] = 0
            proposals = person_index_per_cam(affinity, cum_persons_per_view, min_cameras_for_triangulation)
            
            
            
            state = prepare_rewrite_json_files(json_tracked_files_f, json_files_f, proposals, n_cams, calib_file, f, P, f_range[0], state)
            
            if (not state) and (state is not None):
                break
            
    # recap message
    recap_tracking(config, error_min_tot, cameras_off_tot)
    print('成功結束personAssociation')
    