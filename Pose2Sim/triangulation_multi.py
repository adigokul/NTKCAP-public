from scipy.optimize import linear_sum_assignment
import subprocess
import os
import glob
import fnmatch
import numpy as np
import json
import itertools as it
import toml
from tqdm import tqdm
import cv2
import pandas as pd
import re
from scipy import interpolate
from anytree import RenderTree
import logging
from scipy.spatial import Delaunay
from Pose2Sim.common import retrieve_calib_params, computeP, weighted_triangulation, reprojection, euclidean_distance, natural_sort
from Pose2Sim.skeletons import *

from pathlib import Path
import copy
from scipy.io import savemat
import shutil

# PersonAssociation Functions
def zup2yup(Q):
    '''
    Turns Z-up system coordinates into Y-up coordinates

    INPUT:
    - Q: pandas dataframe
    N 3D points as columns, ie 3*N columns in Z-up system coordinates
    and frame number as rows

    OUTPUT:
    - Q: pandas dataframe with N 3D points in Y-up system coordinates
    '''
    
    # X->Y, Y->Z, Z->X
    cols = list(Q.columns)
    cols = np.array([[cols[i*3+1],cols[i*3+2],cols[i*3]] for i in range(int(len(cols)/3))]).flatten()
    Q = Q[cols]

    return Q

def make_trc(config, Q, keypoints_names, f_range, name, multi_p_path): # triangulation-tag
    '''
    Make Opensim compatible trc file from a dataframe with 3D coordinates

    INPUT:
    - config: dictionary of configuration parameters
    - Q: pandas dataframe with 3D coordinates as columns, frame number as rows
    - keypoints_names: list of strings
    - f_range: list of two numbers. Range of frames

    OUTPUT:
    - trc file
    '''

    # Read config
    project_dir = config.get('project').get('project_dir')
    if project_dir == '': project_dir = os.getcwd()
    frame_rate = config.get('project').get('frame_rate')
    seq_name = os.path.basename(project_dir)
    pose3d_folder_name = config.get('project').get('pose3d_folder_name')
    pose3d_dir = os.path.join(project_dir, pose3d_folder_name)
    multi_person = config.get('project').get('multi_person')
    if multi_person:
        seq_name = f'{os.path.basename(os.path.realpath(project_dir))}_{name}'
    else:
        seq_name = f'{os.path.basename(os.path.realpath(project_dir))}'
    pose3d_dir = os.path.join(project_dir, 'pose-3d')

    trc_f = f'{seq_name}_{f_range[0]}-{f_range[1]}.trc'

    #Header
    DataRate = CameraRate = OrigDataRate = frame_rate
    NumFrames = len(Q)
    NumMarkers = len(keypoints_names)
    header_trc = ['PathFileType\t4\t(X/Y/Z)\t' + trc_f, 
            'DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames', 
            '\t'.join(map(str,[DataRate, CameraRate, NumFrames, NumMarkers, 'm', OrigDataRate, f_range[0], f_range[1]])),
            'Frame#\tTime\t' + '\t\t\t'.join(keypoints_names) + '\t\t',
            '\t\t'+'\t'.join([f'X{i+1}\tY{i+1}\tZ{i+1}' for i in range(len(keypoints_names))])]
    
    # Zup to Yup coordinate system
    Q = zup2yup(Q)
    
    #Add Frame# and Time columns
    Q.index = np.array(range(0, f_range[1]-f_range[0])) + 1
    Q.insert(0, 't', Q.index / frame_rate)

    #Write file
    if not os.path.exists(pose3d_dir): os.mkdir(pose3d_dir)
    trc_path = os.path.join(pose3d_dir, trc_f)
    with open(trc_path, 'w') as trc_o:
        [trc_o.write(line+'\n') for line in header_trc]
        Q.to_csv(trc_o, sep='\t', index=True, header=None, line_terminator='\n')

    return trc_path

def interpolate_zeros_nans(col, *args):
    '''
    Interpolate missing points (of value zero),
    unless more than N contiguous values are missing.

    INPUTS:
    - col: pandas column of coordinates
    - args[0] = N: max number of contiguous bad values, above which they won't be interpolated
    - args[1] = kind: 'linear', 'slinear', 'quadratic', 'cubic'. Default: 'cubic'

    OUTPUT:
    - col_interp: interpolated pandas column
    '''
    if len(args)==2:
        N, kind = args
    if len(args)==1:
        N = np.inf
        kind = args[0]
    if not args:
        N = np.inf
    
    # Interpolate nans
    mask = ~(np.isnan(col) | col.eq(0)) # true where nans or zeros
    idx_good = np.where(mask)[0]
    # import ipdb; ipdb.set_trace()
    if 'kind' not in locals(): # 'linear', 'slinear', 'quadratic', 'cubic'
        f_interp = interpolate.interp1d(idx_good, col[idx_good], kind="linear", bounds_error=False)
    else:
        f_interp = interpolate.interp1d(idx_good, col[idx_good], kind=kind, fill_value='extrapolate', bounds_error=False)
    col_interp = np.where(mask, col, f_interp(col.index)) #replace at false index with interpolated values
    
    # Reintroduce nans if lenght of sequence > N
    idx_notgood = np.where(~mask)[0]
    gaps = np.where(np.diff(idx_notgood) > 1)[0] + 1 # where the indices of true are not contiguous
    sequences = np.split(idx_notgood, gaps)
    if sequences[0].size>0:
        for seq in sequences:
            if len(seq) > N: # values to exclude from interpolation are set to false when they are too long 
                col_interp[seq] = np.nan
    return col_interp

def read_json(js_file):
    '''
    Read OpenPose json file
    '''
    with open(js_file, 'r') as json_f:
        js = json.load(json_f)
        json_data = []
        for people in range(len(js['people'])):
            
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
    
    inv_K = calib_params['inv_K'][cam_id]
    R_mat = calib_params['R_mat'][cam_id]
    T = calib_params['T'][cam_id]

    cam_center = -R_mat.T @ T
    plucker = []
    for i in range(len(x)):
        q = np.array([x[i], y[i], 1])
        norm_Q = R_mat.T @ (inv_K @ q -T) # 計算從相機中心到轉換後點的向量 The vector of the camera center to the transformed 2d coords
        
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
def find_camera_coordinate(calib_file):
    calib_file['rotation']
    R, _ = cv2.Rodrigues(np.array(calib_file['rotation']))
    T = np.array(calib_file['translation'])
    H = np.block([[R,T.reshape(3,1)], [np.zeros(3), 1 ]])
    R_t = np.transpose(R)
    C = -R_t.dot(T)
    A = R_t.dot(np.array([[0],[0],[1]]))
    return C
def euclidean_dist_with_multiplication(q1,q2,Q,calib_file):
    '''
    Euclidean distance between 2 points (N-dim) with focal point and dist calibration
    
    INPUTS:
    - q1: list of N_dimensional coordinates of point
    - q2: idem

    OUTPUTS:
    - euc_dist: float. Euclidian distance between q1 and q2
    
    '''

    
    q1 = np.array(q1)
    q2 = np.array(q2)
    dist = q2 - q1
    C = find_camera_coordinate(calib_file)
    D = euclidean_distance(C, Q)

    fm = calib_file['matrix'][0][0]
    index = D
    euc_dist = np.sqrt(np.sum( [d**2 for d in dist]))
    euc_dist = euc_dist*index

    return euc_dist
def camera2point_dist(Q,calib_file):

    C = find_camera_coordinate(calib_file)
    C_new = [C[1],C[0],C[2]]
    D = euclidean_distance(C, Q)
    return D
def triangulation_from_best_cameras_ver_dynamic(config, x_files, y_files, likelihood_files, projection_matrices,body_name):
    '''
    Triangulates 2D keypoint coordinates, only choosing the cameras for which 
    reprojection error is under threshold.

    1. Creates subset with N cameras excluded 
    2. Tries all possible triangulations
    3. Chooses the one with smallest reprojection error
    If error too big, take off one more camera.
        If then below threshold, retain result.
        If better but still too big, take off one more camera.
    4. Add the strongness of exclusion
    5. counting the camera off due to tracking or likelihood too low
    6. dynamic of min cam num
    7. good luck
    
    INPUTS:
    - a Config.toml file
    - coords_2D_kpt: 
    - projection_matrices: list of arrays

    OUTPUTS:
    - Q: array of triangulated point (x,y,z,1.)
    - error_min: float
    - nb_cams_excluded: int
    '''
    list_dynamic_mincam=  {'Hip':4,'RHip':4,'RKnee':3,'RAnkle':3,'RBigToe':3,'RSmallToe':3,'RHeel':3,'LHip':4,'LKnee':3,'LAnkle':3,'LBigToe':3,'LSmallToe':3,'LHeel':3,'Neck':3,'Head':2,'Nose':2,'RShoulder':3,'RElbow':3,'RWrist':3,'LShoulder':3,'LElbow':3,'LWrist':3}
    
    # Read config
    error_threshold_triangulation = config.get('triangulation').get('reproj_error_threshold_triangulation')
    min_cameras_for_triangulation = config.get('triangulation').get('min_cameras_for_triangulation')
    min_cameras_for_triangulation = list_dynamic_mincam[body_name]
    # Initialize
    
    n_cams = len(x_files)
    error_min = np.inf 
    ###### Get intrisic paramter
    project_dir = config.get('project').get('project_dir')
    if project_dir == '': project_dir = os.getcwd()
    calib_folder_name = config.get('project').get('calib_folder_name')
    calib_dir = os.path.join(project_dir, calib_folder_name)
    calib_file = glob.glob(os.path.join(calib_dir, '*.toml'))[0]
    calib = toml.load(calib_file)

    #######End
    nb_cams_off = 0
    cam_initially_off = np.where( (np.isnan(likelihood_files))| (likelihood_files ==0))
    nb_cam_initially_off =np.shape((cam_initially_off))[1]
    if len(cam_initially_off)>0:
         # cameras will be taken-off until the reprojection error is under threshold

        left_combination  = np.array([range(n_cams)])
        left_combination = np.delete(left_combination,cam_initially_off,None)     
        ini_del =True
    else:
        left_combination = np.array(range(n_cams))
        ini_del =False

    error = []
    id_excluded_cams_temp = []
    error_min_temp = []
    nb_cams_excluded_temp = []
    Q_temp =[]
    exclude_record =[]
    error_record=[]
    count_all_com = 0
    first_tri = 1
    
    #################error_min > error_threshold_triangulation and 
    while n_cams - nb_cams_off-nb_cam_initially_off >= min_cameras_for_triangulation or first_tri == 1:
        # Create subsets with "nb_cams_off" cameras excluded
        id_cams_off = np.array(list(it.combinations(left_combination, nb_cams_off)))##All comnination of exclude cam num
        id_cam_off_exclusion = id_cams_off 

        ## combine initial and exclude 
        if id_cams_off.size == 0 and ini_del ==True:
            id_cams_off = cam_initially_off
        else:
            id_cams_off = np.append(id_cams_off,np.repeat(np.array(cam_initially_off),np.shape(id_cams_off)[0] ,axis = 0),axis=1)
        
        projection_matrices_filt = [projection_matrices]*len(id_cams_off)
        x_files_filt = np.vstack([list(x_files).copy()]*len(id_cams_off))
        y_files_filt = np.vstack([y_files.copy()]*len(id_cams_off))
        likelihood_files_filt = np.vstack([likelihood_files.copy()]*len(id_cams_off))

        if nb_cams_off+nb_cam_initially_off > 0:
            for i in range(len(id_cams_off)):
                x_files_filt[i][id_cams_off[i]] = np.nan
                y_files_filt[i][id_cams_off[i]] = np.nan
                likelihood_files_filt[i][id_cams_off[i]] = np.nan
        nb_cams_excluded_filt = [np.count_nonzero(np.nan_to_num(x)==0) for x in likelihood_files_filt] # count nans and zeros
        
        projection_matrices_filt = [ [ p[i] for i in range(n_cams) if not np.isnan(x_files_filt[j][i]) ] for j, p in enumerate(projection_matrices_filt) ]
        x_files_filt = [ [ xx for ii, xx in enumerate(x) if not np.isnan(xx) ] for x in x_files_filt ]
        y_files_filt = [ [ xx for ii, xx in enumerate(x) if not np.isnan(xx) ] for x in y_files_filt ]
        likelihood_files_filt = [ [ xx for ii, xx in enumerate(x) if not np.isnan(xx) ] for x in likelihood_files_filt ]            

        # Triangulate 2D points

        Q_filt = [weighted_triangulation(projection_matrices_filt[i], x_files_filt[i], y_files_filt[i], likelihood_files_filt[i]) for i in range(len(id_cams_off))]
        
        # Reprojection
        coords_2D_kpt_calc_filt = [reprojection(projection_matrices_filt[i], Q_filt[i])  for i in range(len(id_cams_off))]
        coords_2D_kpt_calc_filt = np.array(coords_2D_kpt_calc_filt, dtype=object)
        x_calc_filt = coords_2D_kpt_calc_filt[:,0]
        y_calc_filt = coords_2D_kpt_calc_filt[:,1]
        # Reprojection error
        error = []
        error1 = []

        for config_id in range(len(x_calc_filt)):
            q_file = [(x_files_filt[config_id][i], y_files_filt[config_id][i]) for i in range(len(x_files_filt[config_id]))]
            q_calc = [(x_calc_filt[config_id][i], y_calc_filt[config_id][i]) for i in range(len(x_calc_filt[config_id]))]
        
            cam_used = np.array(range(n_cams))

            if nb_cams_off>0:
                cam_used = np.delete(cam_used,id_cams_off[config_id])
            # record the exclusion ones not the whole
            exclude_record.append(id_cam_off_exclusion[config_id])
            ##max without dist.
        
            if len(q_file)>0:
                error.append( np.max( [euclidean_dist_with_multiplication(q_file[i], q_calc[i],Q_filt[0][0:3],calib[list(calib.keys())[cam_used[i]]]) for i in range(len(q_file))] ) )
                error_record.append( np.max( [euclidean_dist_with_multiplication(q_file[i], q_calc[i],Q_filt[0][0:3],calib[list(calib.keys())[cam_used[i]]]) for i in range(len(q_file))] ))
            ######max without dist.
                error1.append( np.max( [euclidean_distance(q_file[i], q_calc[i]) for i in range(len(q_file))] ) )
            else:
                error.append(float('inf'))
                error1.append(float('inf'))

        # Choosing best triangulation (with min reprojection error)
        error_min = min(error)
        best_cams = np.argmin(error)
        
        nb_cams_excluded = nb_cams_excluded_filt[best_cams]
        Q = Q_filt[best_cams][:-1]
        
        nb_cams_off += 1
        
        id_excluded_cams_temp.append(id_cam_off_exclusion[best_cams])
        error_min_temp.append(error_min)
        nb_cams_excluded_temp.append(nb_cams_excluded)   
        count_all_com =count_all_com+1
        Q_temp.append(Q)
        first_tri = 0   

    
    # Index of excluded cams for this keypoint
    error_min_final = min(error_min_temp)
    best_cams_final = np.argmin(error_min_temp)
    nb_cams_excluded_final = nb_cams_excluded_temp[best_cams_final]
    id_excluded_cams_final = id_excluded_cams_temp[best_cams_final]
    Q_final = Q_temp[best_cams_final]

    if len(id_excluded_cams_final)>0:
        strongness_of_exclusion = error_min_temp[0]-error_min_final
    else:
        strongness_of_exclusion = 0

    dist_camera2point =np.array([camera2point_dist(Q_final,calib[list(calib.keys())[camera]]) for camera in range(4)])

    # If triangulation not successful, error = 0,  and 3D coordinates as missing values
    if error_min > error_threshold_triangulation:
        error_min = np.nan
        Q = np.array([0.,0.,0.])
        Q = np.array([np.nan, np.nan, np.nan])

    return Q_final, error_min_final, nb_cams_excluded_final, id_excluded_cams_final,exclude_record,error_record,dist_camera2point,strongness_of_exclusion
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

def is_point_in_hull(point, delaunay):
    return delaunay.find_simplex(point) >= 0

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
    
def min_with_single_indices(L, T):
    '''
    Let L be a list (size s) with T associated tuple indices (size s).
    Select the smallest values of L, considering that 
    the next smallest value cannot have the same numbers 
    in the associated tuple as any of the previous ones.

    Example:
    L = [  20,   27,  51,    33,   43,   23,   37,   24,   4,   68,   84,    3  ]
    T = list(it.product(range(2),range(3)))
      = [(0,0),(0,1),(0,2),(0,3),(1,0),(1,1),(1,2),(1,3),(2,0),(2,1),(2,2),(2,3)]

    - 1st smallest value: 3 with tuple (2,3), index 11
    - 2nd smallest value when excluding indices (2,.) and (.,3), i.e. [(0,0),(0,1),(0,2),X,(1,0),(1,1),(1,2),X,X,X,X,X]:
    20 with tuple (0,0), index 0
    - 3rd smallest value when excluding [X,X,X,X,X,(1,1),(1,2),X,X,X,X,X]:
    23 with tuple (1,1), index 5
    
    INPUTS:
    - L: list (size s)
    - T: T associated tuple indices (size s)

    OUTPUTS: 
    - minL: list of smallest values of L, considering constraints on tuple indices
    - argminL: list of indices of smallest values of L
    - T_minL: list of tuples associated with smallest values of L
    '''
    
    minL = [np.nanmin(L)]
    argminL = [np.nanargmin(L)]
    T_minL = [T[argminL[0]]]
    mask_tokeep = np.array([True for t in T])
    i=0
    while mask_tokeep.any()==True:
        mask_tokeep = mask_tokeep & np.array([t[0]!=T_minL[i][0] and t[1]!=T_minL[i][1] for t in T])
        if mask_tokeep.any()==True:
            indicesL_tokeep = np.where(mask_tokeep)[0]
            minL += [np.nanmin(np.array(L)[indicesL_tokeep]) if not np.isnan(np.array(L)[indicesL_tokeep]).all() else np.nan]
            argminL += [indicesL_tokeep[np.nanargmin(np.array(L)[indicesL_tokeep])] if not np.isnan(minL[-1]) else indicesL_tokeep[0]]
            T_minL += (T[argminL[i+1]],)
            i+=1
    
    return np.array(minL), np.array(argminL), np.array(T_minL)
def sort_people_new(Q_kpt_old, Q_kpt, f):
    
    nb_p_each_frame = len(Q_kpt)
    
    personsIDs_comb = sorted(list(it.product(range(len(Q_kpt_old)),range(len(Q_kpt)))))
    
    dist = []
    frame_by_frame_dist = []
    for comb in personsIDs_comb:
        frame_by_frame_dist += [euclidean_distance(Q_kpt_old[comb[0]][:3],Q_kpt[comb[1]][:3])]
    dist += [euclidean_distance(Q_kpt_old[comb[0]], Q_kpt[comb[1]]) for comb in personsIDs_comb]
    personsIDs_comb_new, dist_new = [], []
    for i in range(0, len(personsIDs_comb), int(len(personsIDs_comb) / nb_p_each_frame)):
        
        dist_uncheck = dist[i:i+int(len(personsIDs_comb) / nb_p_each_frame)]        
        if np.all(np.isnan(dist_uncheck)):
            continue        
        personsIDs_comb_new.append(personsIDs_comb[dist_uncheck.index(min(dist_uncheck))+i])
    _, _, min_dist_comb = min_with_single_indices(dist, personsIDs_comb)
    
    return min_dist_comb

def save_data_to_json(Apose_cropped_frame, task_cropped_frame, json_path):
    data_dict = {
        "data1": [[[frame.tolist() for frame in cam] for cam in person] for person in Apose_cropped_frame],
        "data2": [[[frame.tolist() for frame in cam] for cam in person] for person in task_cropped_frame]
    }
    with open(json_path, 'w') as f:
        json.dump(data_dict, f)
def extra_person_num(person):
    match = re.search(r'\d+', person)
    return int(match.group())-1
def build_exclude_dict(groups):
    exclude_dict = {}    
    for group in groups:
        group_indices = [extra_person_num(person) for person in group]
        for person in group_indices:
            if person not in exclude_dict:
                exclude_dict[person] = set()
            exclude_dict[person].update(other for other in group_indices if other != person)
    exclude_dict = {k: list(v) for k, v in exclude_dict.items()}    
    return exclude_dict

def find_clean_group(task_exclusions, remain_unmatch, exclude_match):    
    for object in remain_unmatch:
        if exclude_match[object] is None:
            remain_unmatch.remove(object)
            for i in task_exclusions[object]:
                remain_unmatch.remove(i)
            result_list = [object] + task_exclusions[object]
            return result_list, remain_unmatch
 
def find_best_match(similarity_matrix, clean_group):
    
    sub_matrix = similarity_matrix[np.ix_(range(similarity_matrix.shape[0]), clean_group)]
    
    row_ind, col_ind = linear_sum_assignment(-sub_matrix)
    
    best_matches = [(clean_group[row], col) for row, col in zip(row_ind, col_ind)]
    return best_matches

def cross_product(line1, line2, p):
    return (line2[0] - line1[0]) * (p[1] - line1[1]) - (line2[1] - line1[1]) * (p[0] - line1[0])

def triangulation_from_best_cameras_ver_realdistdynamic_RANSAC(config, coords_2D_kpt, projection_matrices,body_name):
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! undistort out of bound error

    '''
    Triangulates 2D keypoint coordinates, only choosing the cameras for which 
    reprojection error is under threshold.

    1. Creates subset with N cameras excluded 
    2. Tries all possible triangulations
    3. Chooses the one with smallest reprojection error
    If error too big, take off one more camera.
        If then below threshold, retain result.
        If better but still too big, take off one more camera.
    4. Add the strongness of exclusion
    5. counting the camera off due to tracking or likelihood too low
    6. dynamic of min cam num
    7. good luck
    
    INPUTS:
    - a Config.toml file
    - coords_2D_kpt: 
    - projection_matrices: list of arrays

    OUTPUTS:
    - Q: array of triangulated point (x,y,z,1.)
    - error_min: float
    - nb_cams_excluded: int
    '''    
    list_dynamic_mincam=  {'Hip':4,'RHip':4,'RKnee':3,'RAnkle':3,'RBigToe':3,'RSmallToe':3,'RHeel':3,'LHip':4,'LKnee':3,'LAnkle':3,'LBigToe':3,'LSmallToe':3,'LHeel':3,'Neck':3,'Head':2,'Nose':2,'RShoulder':3,'RElbow':3,'RWrist':3,'LShoulder':3,'LElbow':3,'LWrist':3}
    list_TR_realdist = {'Hip':np.inf,'RHip':np.inf,'RKnee':100,'RAnkle':np.inf,'RBigToe':np.inf,'RSmallToe':np.inf,'RHeel':np.inf,'LHip':np.inf,'LKnee':100,'LAnkle':np.inf,'LBigToe':np.inf,'LSmallToe':np.inf,'LHeel':np.inf,'Neck':np.inf,'Head':np.inf,'Nose':np.inf,'RShoulder':np.inf,'RElbow':100,'RWrist':np.inf,'LShoulder':np.inf,'LElbow':100,'LWrist':np.inf}

    
    # Read config
    error_threshold_triangulation = config.get('triangulation').get('reproj_error_threshold_triangulation')
    min_cameras_for_triangulation = config.get('triangulation').get('min_cameras_for_triangulation')
    min_cameras_for_triangulation = list_dynamic_mincam[body_name]
    TR_2cam = list_TR_realdist[body_name]
    # Initialize
    x_files, y_files, likelihood_files = coords_2D_kpt
    
    n_cams = len(x_files)
    error_min = np.inf 
    
    ###### Get intrisic paramter
    project_dir = config.get('project').get('project_dir')
    if project_dir == '': project_dir = os.getcwd()
    calib_folder_name = config.get('project').get('calib_folder_name')
    calib_dir = os.path.join(project_dir, calib_folder_name)
    calib_file = glob.glob(os.path.join(calib_dir, '*.toml'))[0]
    calib = toml.load(calib_file)
    #######End
    nb_cams_off = 0
    cam_initially_off = np.where( (np.isnan(likelihood_files))| (likelihood_files ==0))
    nb_cam_initially_off =np.shape((cam_initially_off))[1]
    if len(cam_initially_off)>0:
         # cameras will be taken-off until the reprojection error is under threshold

        left_combination  = np.array([range(n_cams)])
        left_combination = np.delete(left_combination,cam_initially_off,None)     
        ini_del =True
    else:
        left_combination = np.array(range(n_cams))
        ini_del =False

    error, all_error, all_error4 = [], [], []
    cam_correspond = []
    exclude_record =[]
    error_record=[]
    error_record1 =[]
    count_all_com = 0
    first_tri = 1
    Q_all = []
    voting = []
    #################error_min > error_threshold_triangulation and 
    while n_cams - nb_cams_off-nb_cam_initially_off >= min_cameras_for_triangulation or first_tri == 1:
        # Create subsets with "nb_cams_off" cameras excluded
        id_cams_off = np.array(list(it.combinations(left_combination, nb_cams_off)))##All comnination of exclude cam num
        id_cam_off_exclusion = id_cams_off 

        ## combine initial and exclude 
        if id_cams_off.size == 0 and ini_del ==True:
            id_cams_off = cam_initially_off
        else:
            id_cams_off = np.append(id_cams_off,np.repeat(np.array(cam_initially_off),np.shape(id_cams_off)[0] ,axis = 0),axis=1)

        projection_matrices_filt = [projection_matrices]*len(id_cams_off)
        x_files_filt = np.vstack([list(x_files).copy()]*len(id_cams_off))
        y_files_filt = np.vstack([y_files.copy()]*len(id_cams_off))
        likelihood_files_filt = np.vstack([likelihood_files.copy()]*len(id_cams_off))

        if nb_cams_off+nb_cam_initially_off > 0:
            for i in range(len(id_cams_off)):
                x_files_filt[i][id_cams_off[i]] = np.nan
                y_files_filt[i][id_cams_off[i]] = np.nan
                likelihood_files_filt[i][id_cams_off[i]] = np.nan
        nb_cams_excluded_filt = [np.count_nonzero(np.nan_to_num(x)==0) for x in likelihood_files_filt] # count nans and zeros
        
        projection_matrices_filt = [ [ p[i] for i in range(n_cams) if not np.isnan(x_files_filt[j][i]) ] for j, p in enumerate(projection_matrices_filt) ]

        x_files_filt = [ [ xx for ii, xx in enumerate(x) if not np.isnan(xx) ] for x in x_files_filt ]
        y_files_filt = [ [ xx for ii, xx in enumerate(x) if not np.isnan(xx) ] for x in y_files_filt ]
        likelihood_files_filt = [ [ xx for ii, xx in enumerate(x) if not np.isnan(xx) ] for x in likelihood_files_filt ]   
        cols = len(likelihood_files_filt[0])   # Number of inner lists
        rows = len(likelihood_files_filt)    # Length of each inner list   
        likelihood_files_ones = [[1 for _ in range(cols)] for _ in range(rows)]      

        # Triangulate 2D points
        if not (body_name =='Hip'or body_name =='RHip' or body_name =='LHip'):
            
            likelihood_files_filt = [[x**2 for x in sublist] for sublist in likelihood_files_filt]
        if first_tri==1:
            p4 = projection_matrices_filt[0]
            x4 =x_files_filt[0]
            y4 = y_files_filt[0]
            like4 =likelihood_files_filt[0]
            like4_1s = likelihood_files_ones[0]
            cam_used4 = np.array(range(n_cams))
            cam_used4 = np.delete(cam_used4,id_cams_off[0])

        Q_filt = [weighted_triangulation(projection_matrices_filt[i], x_files_filt[i], y_files_filt[i], likelihood_files_filt[i]) for i in range(len(id_cams_off))]
        # Reprojection
        coords_2D_kpt_calc_filt4 = [reprojection(p4, Q_filt[i])  for i in range(len(id_cams_off))]
        coords_2D_kpt_calc_filt4 = np.array(coords_2D_kpt_calc_filt4, dtype=object)
        x_calc_filt4= coords_2D_kpt_calc_filt4[:,0]
        y_calc_filt4= coords_2D_kpt_calc_filt4[:,1]
        coords_2D_kpt_calc_filt = [reprojection(projection_matrices_filt[i], Q_filt[i])  for i in range(len(id_cams_off))]
        coords_2D_kpt_calc_filt = np.array(coords_2D_kpt_calc_filt, dtype=object)
        x_calc_filt = coords_2D_kpt_calc_filt[:,0]
        y_calc_filt = coords_2D_kpt_calc_filt[:,1]
        # Reprojection error
        error = []
        error1 = []

        for config_id in range(len(x_calc_filt)):
            q_file4 = [(x4[i],y4[i]) for i in range(len(x4))]
            q_calc4 = [(x_calc_filt4[config_id][i], y_calc_filt4[config_id][i]) for i in range(len(x_calc_filt4[config_id]))]
            q_file = [(x_files_filt[config_id][i], y_files_filt[config_id][i]) for i in range(len(x_files_filt[config_id]))]
            q_calc = [(x_calc_filt[config_id][i], y_calc_filt[config_id][i]) for i in range(len(x_calc_filt[config_id]))]

            cam_used = np.array(range(n_cams))
           
            cam_used = np.delete(cam_used,id_cams_off[config_id])
            # record the exclusion ones not the whole
            exclude_record.append(id_cam_off_exclusion[config_id])
            ##max without dist.
            
            if len(q_file)>0:
                error.append( np.max( [euclidean_dist_with_multiplication(q_file[i], q_calc[i],Q_filt[config_id][0:3],calib[list(calib.keys())[cam_used4[i]]]) for i in range(len(q_file))] ) )
                error_record.append( np.max( [euclidean_dist_with_multiplication(q_file[i], q_calc[i],Q_filt[config_id][0:3],calib[list(calib.keys())[cam_used4[i]]]) for i in range(len(q_file))] ))
                all_error.append([euclidean_dist_with_multiplication(q_file[i], q_calc[i],Q_filt[config_id][0:3],calib[list(calib.keys())[cam_used4[i]]]) for i in range(len(q_file))])
                all_error4.append([euclidean_dist_with_multiplication(q_file4[i], q_calc4[i],Q_filt[config_id][0:3],calib[list(calib.keys())[cam_used4[i]]]) for i in range(len(q_file4))])
                cam_correspond.append(cam_used)
                Q_all.append(Q_filt[config_id]) 
                max_index, max_value = max(enumerate(all_error4[-1]), key=lambda x: x[1])
                voting.append([len(cam_used),max_index,max_value])
            ######max without dist.
                error1.append( np.max( [euclidean_distance(q_file[i], q_calc[i]) for i in range(len(q_file))] ) )
            else:
                error.append(float('inf'))
                error1.append(float('inf'))
        
        nb_cams_off += 1
        count_all_com =count_all_com+1
        first_tri = 0   
    
    # Index of excluded cams for this keypoint

    result_voting = []
    for item in voting:
        first_num = item[0]
        middle_num = cam_used4[item[1]]
        result_voting.append([middle_num] * first_num)
        
    result_voting =[item for sublist in result_voting for item in sublist]
    most_common = max(set(result_voting), key=result_voting.count)
    
    indices_to_remove = np.where(cam_used4 == most_common)
    final_comb = np.delete(cam_used4,indices_to_remove)
    index = next((i for i, arr in enumerate(cam_correspond) if np.array_equal(arr, final_comb)), 0)
    Q_final = Q_all[index][:-1]
    error_min_final = np.max(all_error[index])
    if index!=0:
        id_excluded_cams_final= np.array([most_common])
    else:
        id_excluded_cams_final = np.array([])

    nb_cams_excluded_final = 4-len(final_comb)
    if error_min_final>TR_2cam and len(final_comb)==3:
        max_index, max_value = max(enumerate(all_error[index]), key=lambda x: x[1])
        cam_close2 = final_comb[max_index]
        indices_to_remove = np.where(final_comb == cam_close2)
        cam_used_2 = np.delete(final_comb,indices_to_remove)
        p2 =[p4[i] for i in cam_used_2]
        x2 = [x4[i] for i in cam_used_2]
        y2 =[y4[i] for i in cam_used_2]
        like2 =[like4[i] for i in cam_used_2] 
        Q_final = weighted_triangulation(p2, x2, y2,like2)[:-1]
        id_excluded_cams_final =np.array([most_common,cam_close2])
        nb_cams_excluded_final = 2
    if len(final_comb)<4:
        #strongness_of_exclusion =np.mean(all_error[0])-np.mean(all_error[index])
        strongness_of_exclusion =np.mean(all_error[0])-np.mean(all_error[index])
    else:
        strongness_of_exclusion = 0

    dist_camera2point =np.array([camera2point_dist(Q_final,calib[list(calib.keys())[camera]]) for camera in range(4)])
    
    all_error_noweight=[]
    residuals_SVD=[]
    return Q_final, error_min_final, nb_cams_excluded_final, id_excluded_cams_final,exclude_record,error_record,dist_camera2point,strongness_of_exclusion
    # return Q_final, error_min_final, nb_cams_excluded_final, id_excluded_cams_final,exclude_record,error_record,dist_camera2point,strongness_of_exclusion,all_error,all_error_noweight,cam_correspond,residuals_SVD

def track_2d_all(config, c_project_path, pk = True):
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
    PWD = project_dir
    while os.path.basename(PWD) != 'NTKCAP': 
        PWD = os.path.dirname(PWD)

    empty_project_path = os.path.join(PWD, "NTK_CAP", "template", "Empty_project")
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
    interp_gap_smaller_than = config.get('triangulation').get('interp_if_gap_smaller_than')
    calib_dir = os.path.join(project_dir, calib_folder_name) # walk1/calib-2d
    calib_file = glob.glob(os.path.join(calib_dir, '*.toml'))[0] # .toml in walk1/calib-2d
    pose_dir = os.path.join(project_dir, pose_folder_name) # walk1/pose-2d
    poseTracked_dir = os.path.join(project_dir, poseTracked_folder_name) # walk1/pose-2d-tracked
    likelihood_threshold = config.get('triangulation').get('likelihood_threshold')
    # projection matrix from toml calibration file
    P = computeP(calib_file)
    calib_params = retrieve_calib_params(calib_file) # size, intri_matrix, distortions, optim_K, inv_K, rotation_v, rotation_m, translation_v
    # selection of tracked keypoint id
    model = eval(pose_model)
    keypoints_ids = [node.id for _, _, node in RenderTree(model) if node.id!=None]
    keypoints_names = [node.name for _, _, node in RenderTree(model) if node.id!=None]
    keypoints_idx = list(range(len(keypoints_ids)))
    keypoints_nb = len(keypoints_ids)
    tracked_keypoint_id = [node.id for _, _, node in RenderTree(model) if node.name==tracked_keypoint][0]
    
    # 2d-pose files selection
    pose_listdirs_names = next(os.walk(pose_dir))[1] # [pose_cam1_json, ...]
    pose_listdirs_names = natural_sort(pose_listdirs_names) # pose_cam1_json in ascending order
    json_dirs_names = [k for k in pose_listdirs_names if json_folder_extension in k] # [pose_cam1_json, ...]
    json_files_names = [fnmatch.filter(os.listdir(os.path.join(pose_dir, js_dir)), '*.json') for js_dir in json_dirs_names] # len=num_cam list，json filename in each pose_camn_json
    json_files_names = [natural_sort(j) for j in json_files_names] # json filename in ascending order
    json_files = [[os.path.join(pose_dir, j_dir, j_file) for j_file in json_files_names[j]] for j, j_dir in enumerate(json_dirs_names)] # len = 4 list，absolute path of json    
    # 2d-pose-associated files creation
    if not os.path.exists(poseTracked_dir): os.mkdir(poseTracked_dir)   
    try: [os.mkdir(os.path.join(poseTracked_dir,k)) for k in json_dirs_names]
    except: pass
    json_tracked_files = [[os.path.join(poseTracked_dir, j_dir, j_file) for j_file in json_files_names[j]] for j, j_dir in enumerate(json_dirs_names)]
    interpolation_kind = config.get('triangulation').get('interpolation')
    # person's tracking
    f_range = [[0,min([len(j) for j in json_files_names])] if frame_range==[] else frame_range][0]
    frames_nb =  f_range[1]-f_range[0]
    n_cams = len(json_dirs_names) # 4
    keypoints_2d_template, Q_kpt = [np.nan] * 78, [np.array([0., 0., 0., 1.])]
    max_people, Tracker, Q_tot = 0, {}, []
    pos = calculate_camera_position(calib_file)
    pos_2d = [d[:2] for d in pos]
    delaunay = Delaunay(pos_2d)
    camera_coordinates = {
        "1": pos[0],
        "2": pos[1],
        "3": pos[2],
        "4": pos[3]
    }
    matrix_not_same_p = []
    camera_wall = [1, 2]
    camera_coords_with_index = [(int(key), coord) for key, coord in camera_coordinates.items()]
    sorted_coords_with_index = sorted(
        camera_coords_with_index,
        key=lambda item: (-item[1][1])
    )
    top_two = sorted_coords_with_index[:2]
    bottom_two = sorted_coords_with_index[2:]
    top_two_sorted = sorted(top_two, key=lambda item: -item[1][0])
    bottom_two_sorted = sorted(bottom_two, key=lambda item: item[1][0])
    final_sorted_coords = top_two_sorted + bottom_two_sorted

    sorted_indices = [item[0] for item in final_sorted_coords]
    sorted_coords = [item[1] for item in final_sorted_coords]
    
    id1 = sorted_indices.index(camera_wall[0])
    id2 = sorted_indices.index(camera_wall[1])
    if set(camera_wall) == set([sorted_indices[0], sorted_indices[1]]) or set(camera_wall) == set([sorted_indices[2], sorted_indices[3]]):
        base = 'x'
    elif set(camera_wall) == set([sorted_indices[1], sorted_indices[2]]) or set(camera_wall) == set([sorted_indices[0], sorted_indices[3]]):
        base = 'y'
    center_p = np.mean(sorted_coords, axis=0)
    for f in tqdm(range(*f_range)): # all frames     
        json_files_f = [json_files[c][f] for c in range(n_cams)] # same frame from different cameras，file location
        json_tracked_files_f = [json_tracked_files[c][f] for c in range(n_cams)] # same frame from different, keypoints
        matrix_not_same_p.append([])
        all_json_data_f, js_new_all = [], []
        for js_file in json_files_f:
            all_json_data_f.append(read_json(js_file)) # len=4, each one contains coordinate and confidence of keypoint
        persons_per_view = [0] + [len(j) for j in all_json_data_f] # [0, num_peo_cam1.... ]
        cum_persons_per_view = np.cumsum(persons_per_view) # [0, numpeocam1, numpeocam1+2....]
        affinity = compute_affinity(all_json_data_f, calib_params, cum_persons_per_view, reconstruction_error_threshold=reconstruction_error_threshold)    
        circ_constraint = circular_constraint(cum_persons_per_view)
        affinity = affinity * circ_constraint
        affinity = matchSVT(affinity, cum_persons_per_view, circ_constraint, max_iter = 20, w_rank = 50, tol = 1e-4, w_sparse=0.1)
        affinity[affinity<min_affinity] = 0
        proposals = person_index_per_cam(affinity, cum_persons_per_view, min_cameras_for_triangulation)   
        # Load keypoints_2d
        keypoints_2d_template_tri = [np.nan] * 22
        for comb in proposals: # if not in cameras would be Nan
            js_new = []
            for cam in range(n_cams):
                with open(json_files_f[cam], 'r') as json_f:
                    js = json.load(json_f)                        
                    if not np.isnan(comb[cam]):                            
                        js_new.append(js['people'][int(comb[cam])]['pose_keypoints_2d'])                        
                    else:
                        js_new.append([])                    
            js_new_all.append(js_new)            
        kp_idx = 18 # Neck
        Q, index_remove, camera_exclude, error, nb_cams_excluded, id_excluded_cams,exclude_record,error_record,cam_dist,strongness_exclusion, current_num = [], [], [], [], [], [], [], [], [], [], 0
        # Check whether the person is inside the area
        
        for p in range(len(js_new_all)):
            x_all, y_all, lik_all, X_all, Y_all, L_all, P_all = [], [], [], [], [], [], []   
            camera_exclude.append([])         
            for cam in range(len(js_new_all[p])):
                if js_new_all[p][cam] != []:
                    P_all.append(P[cam])
                    # Use the neck to determine
                    x_all.append(js_new_all[p][cam][kp_idx * 3])
                    y_all.append(js_new_all[p][cam][kp_idx * 3 + 1])
                    lik_all.append(js_new_all[p][cam][kp_idx * 3 + 2])
                    x, y, lik = [], [], []
                    for kp_id in keypoints_ids:                            
                        x.append(js_new_all[p][cam][kp_id * 3])
                        y.append(js_new_all[p][cam][kp_id * 3+1])
                        lik.append(js_new_all[p][cam][kp_id * 3+2])
                    X_all.append(x)
                    Y_all.append(y)
                    L_all.append(lik)                        
                else:
                    camera_exclude[p].append(cam)
                    X_all.append(keypoints_2d_template_tri)
                    Y_all.append(keypoints_2d_template_tri)
                    L_all.append(keypoints_2d_template_tri)  
            X_all, Y_all, L_all = np.array(X_all), np.array(Y_all), np.array(L_all)
            
            with np.errstate(invalid='ignore'):
                L_all[L_all<likelihood_threshold] = 0.
            
            if pk: # open one wall if needed
                q = weighted_triangulation(P_all, x_all, y_all, lik_all) 
                cross1, cross2 = cross_product(sorted_coords[id1], sorted_coords[id2], q[:2]), cross_product(sorted_coords[id1], sorted_coords[id2], center_p)
                if not is_point_in_hull(q[:2], delaunay):
                    if cross1 * cross2 < 0:
                        axis = 0 if base == 'x' else 1
                        lower, upper = min(sorted_coords[id1][axis], sorted_coords[id2][axis]), max(sorted_coords[id1][axis], sorted_coords[id2][axis])
                        if not (q[axis] > lower and q[axis] < upper):                            
                            index_remove.append(p)
                            continue
                    else:
                        index_remove.append(p)
                        continue
            else:
                q = weighted_triangulation(P_all, x_all, y_all, lik_all) 
                if not is_point_in_hull(q[:2], delaunay):
                    index_remove.append(p)
                    continue
            
            Q.append([]), error.append([]), nb_cams_excluded.append([]), id_excluded_cams.append([]), exclude_record.append([]), error_record.append([]), cam_dist.append([]), strongness_exclusion.append([])
            # if the person does than directly do the triangulation
            
            for keypoint_idx in keypoints_idx:   
                coords_2D_kpt = (np.array([item[keypoint_idx] for item in X_all]), np.array([item[keypoint_idx] for item in Y_all]), np.array([item[keypoint_idx] for item in L_all]))
                
                Q_kpt, error_kpt, nb_cams_excluded_kpt, id_excluded_cams_kpt, exclude_record_kpt, error_record_kpt, cam_dist_kpt, strongness_of_exclusion_kpt = triangulation_from_best_cameras_ver_realdistdynamic_RANSAC(config, coords_2D_kpt, P, keypoints_names[keypoint_idx])
                
                # Q_kpt, error_kpt, nb_cams_excluded_kpt, id_excluded_cams_kpt, exclude_record_kpt, error_record_kpt, cam_dist_kpt, strongness_of_exclusion_kpt = triangulation_from_best_cameras_ver_realdistdynamic_RANSAC(config, np.array([item[keypoint_idx] for item in X_all]), np.array([item[keypoint_idx] for item in Y_all]), np.array([item[keypoint_idx] for item in L_all]), P, keypoints_names[keypoint_idx])
                Q[current_num].append(Q_kpt), error[current_num].append(error_kpt), nb_cams_excluded[current_num].append(nb_cams_excluded_kpt), id_excluded_cams[current_num].append(id_excluded_cams_kpt), exclude_record[current_num].append(exclude_record_kpt), error_record[current_num].append(error_record_kpt), cam_dist[current_num].append(cam_dist_kpt), strongness_exclusion[current_num].append(strongness_of_exclusion_kpt)                     
            current_num += 1 
        # error num_person*num_keypoints  nb_cams_excluded num_person*num_keypoints  exclude_record num_person*num_keypoints  error_record num_person*num_keypoints  cam_dist num_person*num_keypoints
        js_new_all = [item for idx, item in enumerate(js_new_all) if idx not in index_remove]
        camera_exclude = [item for idx, item in enumerate(camera_exclude) if idx not in index_remove]
        max_people = max(max_people, len(js_new_all))
        for idx in range(len(js_new_all)):                           
            for each_cam in range(len(js_new_all[idx])):
                if js_new_all[idx][each_cam] == []:
                    js_new_all[idx][each_cam] = keypoints_2d_template
        if f != 0:
            final_comb = sort_people_new(Q_old, Q, f)                                                            
            for idx, c in enumerate(final_comb): # (pre, cur)                         
                if f == 1:                                                                     
                    Tracker[f"person{idx+1}"] = {
                        'id' : c[1],
                        'Apose matching status' : False,
                        'Apose frame' : None,
                        'pose_keypoints_2d': np.full((frames_nb, *np.array(js_new_all_old[c[0]]).shape), np.nan),
                        'name' : None,
                        'keypoints_3d': np.full((frames_nb, *np.array(Q_old[c[0]]).shape), np.nan),
                        'Apose matching confidence' : None,
                        'matching status' : False,
                        'id_excluded_cams' : [],
                        'exclude_record' : [],
                        'error_record' : [],
                        'cam_dist' : [],
                        'strongness_exclusion' : [],
                        'first_frame' : 0,
                        'latest_frame' : None                                               
                    }
                    # check whether four views have caught the person
                    Tracker[f"person{idx+1}"]['pose_keypoints_2d'][0] = np.array(js_new_all_old[c[0]])
                    Tracker[f"person{idx+1}"]['keypoints_3d'][0] = Q_old[c[0]]
                    Tracker[f"person{idx+1}"]['id_excluded_cams'].append(id_excluded_cams[c[0]])                        
                    Tracker[f"person{idx+1}"]['exclude_record'].append(exclude_record[c[0]])
                    Tracker[f"person{idx+1}"]['error_record'].append(error_record[c[0]])
                    Tracker[f"person{idx+1}"]['cam_dist'].append(cam_dist[c[0]])
                    Tracker[f"person{idx+1}"]['strongness_exclusion'].append(strongness_exclusion[c[0]])
                    if (not np.isnan(js_new_all[c[1]]).any()) and (not Tracker[f"person{idx+1}"]['Apose matching status']) and all(avg > 0.3 for avg in [np.mean(arr[2::3]) for arr in js_new_all[c[1]]]):
                        Tracker[f"person{idx+1}"]['Apose matching status'] = True
                        Tracker[f"person{idx+1}"]['Apose frame'] = f
                        Tracker[f"person{idx+1}"]['Apose matching confidence'] = [np.mean(arr[2::3]) for arr in js_new_all[c[1]]]
                    Tracker[f"person{idx+1}"]['pose_keypoints_2d'][f] = np.array(js_new_all[c[1]])
                    Tracker[f"person{idx+1}"]['keypoints_3d'][f] = Q[c[1]]
                    Tracker[f"person{idx+1}"]['id_excluded_cams'].append(id_excluded_cams[c[1]])
                    Tracker[f"person{idx+1}"]['exclude_record'].append(exclude_record[c[1]])
                    Tracker[f"person{idx+1}"]['error_record'].append(error_record[c[1]])
                    Tracker[f"person{idx+1}"]['cam_dist'].append(cam_dist[c[1]])
                    Tracker[f"person{idx+1}"]['strongness_exclusion'].append(strongness_exclusion[c[1]])
                    Tracker[f"person{idx+1}"]['latest_frame'] = 1
                    matrix_not_same_p[0].append(f"person{idx+1}")
                    matrix_not_same_p[1].append(f"person{idx+1}")                    
                else:                        
                    matched = False                                           
                    for person, person_info in Tracker.items():
                        if c[0] == person_info['id'] and person_info['matching status'] == True and person_info['latest_frame'] == f-1:
                            # check whether four views have caught the person
                            if (not np.isnan(js_new_all[c[1]]).any()) and (not Tracker[person]['Apose matching status']) and all(avg > 0.3 for avg in [np.mean(arr[2::3]) for arr in js_new_all[c[1]]]):
                                Tracker[person]['Apose matching status'] = True
                                Tracker[person]['Apose frame'] = f
                                Tracker[person]["Apose matching confidence"] = [np.mean(arr[2::3]) for arr in js_new_all[c[1]]]
                            matched = True
                            person_info['matching status'] = False
                            Tracker[person]['id'] = c[1]
                            Tracker[person]['pose_keypoints_2d'][f] = np.array(js_new_all[c[1]])
                            Tracker[person]['keypoints_3d'][f] = Q[c[1]]
                            Tracker[person]['id_excluded_cams'].append(id_excluded_cams[c[1]])                             
                            Tracker[person]['exclude_record'].append(exclude_record[c[1]])
                            Tracker[person]['error_record'].append(error_record[c[1]])
                            Tracker[person]['cam_dist'].append(cam_dist[c[1]])
                            Tracker[person]['strongness_exclusion'].append(strongness_exclusion[c[1]])
                            Tracker[person]['latest_frame'] = f

                            matrix_not_same_p[f].append(person)
                    if not matched: # new person
                        cur_num_p = len(Tracker)                            
                        Tracker[f"person{cur_num_p+1}"] = {
                            'id' : c[1],
                            'Apose matching status' : False,
                            'Apose frame' : None,
                            'pose_keypoints_2d': np.full((frames_nb, *np.array(js_new_all_old[c[0]]).shape), np.nan),
                            'name' : None,
                            'keypoints_3d': np.full((frames_nb, *np.array(Q_old[c[0]]).shape), np.nan),
                            'Apose matching confidence' : None,
                            'matching status' : False,
                            'id_excluded_cams' : [],
                            'exclude_record' : [],
                            'error_record' : [],
                            'cam_dist' : [],
                            'strongness_exclusion' : [],  
                            'first_frame' : f-1,
                            'latest_frame' : None
                        }                            
                        # check whether four views have caught the person
                        if (not np.isnan(js_new_all[c[1]]).any()) and (not Tracker[f"person{cur_num_p+1}"]['Apose matching status']) and all(avg > 0.3 for avg in [np.mean(arr[2::3]) for arr in js_new_all[c[1]]]):
                            Tracker[f"person{cur_num_p+1}"]['Apose matching status'] = True
                            Tracker[f"person{cur_num_p+1}"]['Apose frame'] = f
                            Tracker[f"person{cur_num_p+1}"]['Apose matching confidence'] = [np.mean(arr[2::3]) for arr in js_new_all[c[1]]]                                                               
                        Tracker[f"person{cur_num_p+1}"]['pose_keypoints_2d'][f-1] = np.array(js_new_all[c[0]])
                        Tracker[f"person{cur_num_p+1}"]['keypoints_3d'][f-1] = Q[c[0]]
                        Tracker[f"person{cur_num_p+1}"]['id_excluded_cams'].append(id_excluded_cams[c[0]])                     
                        Tracker[f"person{cur_num_p+1}"]['exclude_record'].append(exclude_record[c[0]])
                        Tracker[f"person{cur_num_p+1}"]['error_record'].append(error_record[c[0]])
                        Tracker[f"person{cur_num_p+1}"]['cam_dist'].append(cam_dist[c[0]])
                        Tracker[f"person{cur_num_p+1}"]['strongness_exclusion'].append(strongness_exclusion[c[0]])
                        Tracker[f"person{cur_num_p+1}"]['pose_keypoints_2d'][f] = np.array(js_new_all[c[1]])
                        Tracker[f"person{cur_num_p+1}"]['keypoints_3d'][f] = Q[c[1]]
                        Tracker[f"person{cur_num_p+1}"]['id_excluded_cams'].append(id_excluded_cams[c[1]])                     
                        Tracker[f"person{cur_num_p+1}"]['exclude_record'].append(exclude_record[c[1]])
                        Tracker[f"person{cur_num_p+1}"]['error_record'].append(error_record[c[1]])
                        Tracker[f"person{cur_num_p+1}"]['cam_dist'].append(cam_dist[c[1]])
                        Tracker[f"person{cur_num_p+1}"]['strongness_exclusion'].append(strongness_exclusion[c[1]])
                        Tracker[f"person{cur_num_p+1}"]['latest_frame'] = f
                        max_people = max(max_people, len(Tracker))
                        matrix_not_same_p[f].append(f"person{cur_num_p+1}")    
            for _, pro in Tracker.items():
                pro['matching status'] = True
        Q_old, js_new_all_old = copy.deepcopy(Q), copy.deepcopy(js_new_all)
    # If the num of valid frames of a subject are less than 5, it would be removed
    p_to_remove = [per for per, pro in Tracker.items() if ((pro['latest_frame'] - pro['first_frame'] <= 5) or (not pro['Apose matching status']))]
    for p_name in p_to_remove:
        Tracker.pop(p_name)
    # Paths of the task -> template : C:\NTKCAP\Patient_data\multi_person\2024_08_22\2024_10_30_23_58_calculated\side2side_walk
    calculate_project_path = Path(c_project_path)
    task_date_path = calculate_project_path.parents[1]
    raw_data_path = os.path.join(task_date_path, 'raw_data')
    raw_data_task_path = os.path.join(raw_data_path, calculate_project_path.name)
    name_folder_path = os.path.join(raw_data_task_path, 'name')
    name_list_path = os.path.join(name_folder_path, 'name.txt')
    name_checked_list_path = os.path.join(raw_data_task_path, 'name_checked', 'name.txt')
    name_cal_list_path = os.path.join(name_folder_path, 'name_cal.txt')
    Patient_data_path = calculate_project_path.parents[3]
    # Task related information tag
    task_name = calculate_project_path.name
    task_category_date = calculate_project_path.parents[1].name
    calculate_time = calculate_project_path.parents[0].name

    # read the txt file with all person names in this task
    final_Tracker = {}
    if os.path.exists(name_folder_path):
        if not os.path.exists(name_cal_list_path): # exist name/name.txt -> has not been calculated yet
            name_list = []  
            with open(name_list_path, 'r') as file: 
                lines = file.readlines()
                for line in lines:
                    name_list.append(line.strip())
            Apose_cropped_frame = []
            task_cropped_frame = []
            Apose_match_video_path = []
            task_match_video_path = []
            pose_cam_json_template = ['pose_cam1_json', 'pose_cam2_json', 'pose_cam3_json', 'pose_cam4_json']
            video_name_template = ['1.mp4', '2.mp4', '3.mp4', '4.mp4']
            for id_t, (tracker_person, property) in enumerate(Tracker.items()):
                task_cropped_frame.append([])
                task_match_video_path.append([])
                # Paths of single person                 
                for cam_id in range(n_cams):
                    task_match_num = Tracker[tracker_person]['Apose frame']
                    task_match_keypoints_2d = Tracker[tracker_person]['pose_keypoints_2d'][task_match_num][cam_id]
                    task_raw_video_path = os.path.join(raw_data_task_path, 'videos', video_name_template[cam_id])
                    cap_task = cv2.VideoCapture(task_raw_video_path)
                    cap_task.set(cv2.CAP_PROP_POS_FRAMES, task_match_num)
                    _, frame_task = cap_task.read()            
                    x_task = task_match_keypoints_2d[0::3]
                    y_task = task_match_keypoints_2d[1::3]        
                    x_task_min, x_task_max, y_task_min, y_task_max = max(int(min(x_task)) - 1, 0), min(int(max(x_task)) + 1, frame_task.shape[1]), max(int(min(y_task)) - 1, 0), min(int(max(y_task)) + 1, frame_task.shape[0])
                    cropped_task_frame = frame_task[y_task_min:y_task_max, x_task_min:x_task_max]
                    task_cropped_frame[id_t].append(cropped_task_frame)
                    cap_task.release()       
            for id_p, name in enumerate(name_list):
                Apose_cropped_frame.append([])            
                Apose_match_video_path.append([])
                tracker_person = f"person{id_p+1}"
                # Paths of single person            
                calculate_apose_path_p = os.path.join(Patient_data_path, name, task_category_date, calculate_time, 'Apose')   
                calculate_task_path_p = os.path.join(Patient_data_path, name, task_category_date, calculate_time, task_name)
                if not os.path.exists(calculate_task_path_p): 
                    shutil.copytree(empty_project_path, calculate_task_path_p)
                    if not os.path.exists(os.path.join(calculate_task_path_p, 'pose-2d-tracked', 'pose_cam1_json')):
                        os.mkdir(os.path.join(calculate_task_path_p, 'pose-2d-tracked', 'pose_cam1_json'))  
                        os.mkdir(os.path.join(calculate_task_path_p, 'pose-2d-tracked', 'pose_cam2_json'))
                        os.mkdir(os.path.join(calculate_task_path_p, 'pose-2d-tracked', 'pose_cam3_json'))
                        os.mkdir(os.path.join(calculate_task_path_p, 'pose-2d-tracked', 'pose_cam4_json'))  
                for cam_id in range(n_cams):                
                    apose_cal_path_p = os.path.join(calculate_apose_path_p, 'pose-2d-tracked', pose_cam_json_template[cam_id], '00000.json')
                    with open(apose_cal_path_p, 'r') as file:
                        apose_kps_2d = json.load(file)                               
                    apose_raw_video_path = os.path.join(Patient_data_path, name, task_category_date, 'raw_data', 'Apose', 'videos', video_name_template[cam_id])
                    cap_apose = cv2.VideoCapture(apose_raw_video_path)
                    cap_apose.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    _, frame_apose = cap_apose.read()
                    x_apose = apose_kps_2d['people'][0]['pose_keypoints_2d'][0::3]
                    y_apose = apose_kps_2d['people'][0]['pose_keypoints_2d'][1::3]
                    x_apose_min, x_apose_max, y_apose_min, y_apose_max = max(int(min(x_apose)) - 1, 0), min(int(max(x_apose)) + 1, frame_apose.shape[1]), max(int(min(y_apose)) - 1, 0), min(int(max(y_apose)) + 1, frame_apose.shape[0])
                    cropped_apose_frame = frame_apose[y_apose_min:y_apose_max, x_apose_min:x_apose_max]
                    Apose_cropped_frame[id_p].append(cropped_apose_frame)
                    cap_apose.release()        
            save_data_to_json(Apose_cropped_frame, task_cropped_frame, os.path.join(calculate_project_path, 'cropped_frame.json'))
            # Customize
            subprocess.run(
                ["C:/Users/MyUser/anaconda3/envs/torchreid/python.exe", "C:/Users/MyUser/Desktop/NTKCAP/Pose2Sim/reid.py", os.path.join(calculate_project_path, 'cropped_frame.json')]
            )
            with open(os.path.join(calculate_project_path, 'cropped_frame.json'), 'r') as file:
                similarities = json.load(file)
            num_tasks, num_apose = len(task_cropped_frame), len(Apose_cropped_frame)
            similarity_matrix = [[0] * num_tasks for _ in range(num_apose)]
            for p_task in range(num_tasks):
                for p_apose in range(num_apose):
                    avg_similarity = sum(similarities[p_task][p_apose]) / len(similarities[p_task][p_apose])
                    similarity_matrix[p_apose][p_task] = avg_similarity            
            if num_tasks == num_apose:
                exclude_dict = build_exclude_dict(matrix_not_same_p)
                exclude_matched = {key: None for key in exclude_dict}
                remain_unmatch = list(range(num_tasks)) 
                while remain_unmatch:        
                    clean_group, remain_unmatch = find_clean_group(exclude_dict, remain_unmatch, exclude_matched)
                    best_matches = find_best_match(np.array(similarity_matrix), clean_group)
                    for match in best_matches:
                        Tracker[list(Tracker.keys())[match[0]]]['name'] = name_list[match[1]]
                        exclude_matched[match[0]] = match[1]
            else:
                try:
                    rep_times, cal_match_record, succ_match_id = [0]*num_tasks, [[] for _ in range(num_tasks)], [np.nan]*num_tasks
                    unique_matrix_not_same_p = list(set(tuple(sorted(group)) for group in matrix_not_same_p))
                    exclude_dict = build_exclude_dict(unique_matrix_not_same_p)
                    unique_matrix_not_same_p = [tuple(int(''.join(filter(str.isdigit, s)))-1 for s in t) for t in unique_matrix_not_same_p if len(t) > 1]
                    for exclude_comb in unique_matrix_not_same_p:                 
                        rep_times[exclude_comb[0]] += 1
                        rep_times[exclude_comb[1]] += 1
                    sorted_indices = sorted(range(len(rep_times)), key=lambda i: rep_times[i], reverse=True)
                    while np.nan in succ_match_id:
                        similarity_matrix_cal, matched_record, succ_match_id = copy.deepcopy(similarity_matrix), [False]*num_tasks, [np.nan]*num_tasks
                        for task_id in sorted_indices:
                            if matched_record[task_id]: continue
                            else:
                                sim_try_id = max(range(len(similarity_matrix_cal)), key=lambda row: similarity_matrix_cal[row][task_id])
                                for idx in range(num_apose):
                                    similarity_matrix_cal[idx][task_id] = 1 if idx == sim_try_id else 0
                                for exclude_id in exclude_dict[task_id]: similarity_matrix_cal[sim_try_id][exclude_id] = 0
                                matched_record[task_id] = True
                                succ_match_id[task_id] = sim_try_id
                                cal_match_record[task_id].append(sim_try_id)
                                for exclude_id in exclude_dict[task_id]:                                            
                                    if not matched_record[exclude_id]:
                                        sim_try_id = max(range(len(similarity_matrix_cal)), key=lambda row: similarity_matrix_cal[row][exclude_id])
                                        if similarity_matrix_cal[sim_try_id][exclude_id] == 0: 
                                            matched_record = [True] * num_tasks
                                            break                                
                                        for idx in range(num_apose):
                                            similarity_matrix_cal[idx][exclude_id] = 1 if idx == sim_try_id else 0
                                        matched_record[exclude_id] = True
                                        succ_match_id[exclude_id] = sim_try_id
                        for sim_fail_task_id, sim_fail_list in enumerate(cal_match_record):
                            for sim_fail_id in sim_fail_list:
                                similarity_matrix[sim_fail_id][sim_fail_task_id] = 0
                    for person_id, best_match_id in enumerate(succ_match_id):
                        Tracker[list(Tracker.keys())[person_id]]['name'] = name_list[best_match_id]
                except:
                    def has_conflict(area, block_key):
                        block = Tracker[block_key]
                        for existing_block_key in area:
                            existing_block = Tracker[existing_block_key]
                            if not (block['latest_frame'] <= existing_block['first_frame'] or block['first_frame'] >= existing_block['latest_frame']):
                                return True
                        return False
                    areas = [[] for _ in range(len(name_list))]
                    for block_key in sorted(Tracker.keys(), key=lambda x: Tracker[x]['first_frame']):
                        for area in areas:
                            if not has_conflict(area, block_key):
                                area.append(block_key)
                                break
                    for area_index, area in enumerate(areas):
                        area_name = name_list[area_index]
                        for person_key in area:
                            Tracker[person_key]['name'] = area_name
            os.makedirs(os.path.join(name_folder_path, 'Apose'))
            os.makedirs(os.path.join(name_folder_path, 'task'))            
            for i_t in range(len(Tracker)):
                os.makedirs(os.path.join(name_folder_path, 'task', f"{i_t}"))                
                for n_c in range(n_cams):
                    cropped_img_save_path = os.path.join(name_folder_path, 'task', f"{i_t}", f"{n_c}.jpg")
                    cv2.imwrite(cropped_img_save_path, task_cropped_frame[i_t][n_c])
            for i_a in range(len(Apose_cropped_frame)):
                os.makedirs(os.path.join(name_folder_path, 'Apose', f"{name_list[i_a]}"))
                for n_c in range(n_cams):
                    cropped_img_save_path = os.path.join(raw_data_task_path, 'name', 'Apose', f"{name_list[i_a]}", f"{n_c}.jpg")
                    cv2.imwrite(cropped_img_save_path, Apose_cropped_frame[i_a][n_c])
            os.remove(os.path.join(name_folder_path, 'name.txt'))
            cal_name_path = os.path.join(name_folder_path, 'name_cal.txt')
            with open(cal_name_path, "w") as f:
                for person in Tracker.keys():                
                    f.write(Tracker[person]['name'] + "\n")
        else: # exists name/name_cal.txt
            num_tasks = len(Tracker)
            name_cal = []
            with open(name_cal_list_path, 'r') as file: 
                lines = file.readlines()
                for line, key in zip(lines, Tracker.keys()):
                    name = line.strip()
                    Tracker[key]['name'] = name
                    name_cal.append(name)
                    calculate_apose_path_p = os.path.join(Patient_data_path, name, task_category_date, calculate_time, 'Apose')   
                    calculate_task_path_p = os.path.join(Patient_data_path, name, task_category_date, calculate_time, task_name)
                    if not os.path.exists(calculate_task_path_p): 
                        shutil.copytree(empty_project_path, calculate_task_path_p)
                        if not os.path.exists(os.path.join(calculate_task_path_p, 'pose-2d-tracked', 'pose_cam1_json')):
                            os.mkdir(os.path.join(calculate_task_path_p, 'pose-2d-tracked', 'pose_cam1_json'))  
                            os.mkdir(os.path.join(calculate_task_path_p, 'pose-2d-tracked', 'pose_cam2_json'))
                            os.mkdir(os.path.join(calculate_task_path_p, 'pose-2d-tracked', 'pose_cam3_json'))
                            os.mkdir(os.path.join(calculate_task_path_p, 'pose-2d-tracked', 'pose_cam4_json'))  
            num_apose = len(list(set(name_cal)))
    else:
        num_tasks = len(Tracker)
        name_checked_list = []        
        with open(name_checked_list_path, 'r') as file: 
            lines = file.readlines()
            for line in lines:
                name = line.strip()
                name_checked_list.append(name)
                calculate_apose_path_p = os.path.join(Patient_data_path, name, task_category_date, calculate_time, 'Apose')   
                calculate_task_path_p = os.path.join(Patient_data_path, name, task_category_date, calculate_time, task_name)
                if not os.path.exists(calculate_task_path_p): 
                    shutil.copytree(empty_project_path, calculate_task_path_p)
                    if not os.path.exists(os.path.join(calculate_task_path_p, 'pose-2d-tracked', 'pose_cam1_json')):
                        os.mkdir(os.path.join(calculate_task_path_p, 'pose-2d-tracked', 'pose_cam1_json'))  
                        os.mkdir(os.path.join(calculate_task_path_p, 'pose-2d-tracked', 'pose_cam2_json'))
                        os.mkdir(os.path.join(calculate_task_path_p, 'pose-2d-tracked', 'pose_cam3_json'))
                        os.mkdir(os.path.join(calculate_task_path_p, 'pose-2d-tracked', 'pose_cam4_json'))                  
        for idx, (person, properties) in enumerate(Tracker.items()):
            Tracker[person]['name'] = name_checked_list[idx]
        num_apose = len(list(set(name_checked_list)))
    if num_tasks != num_apose:
        for person, properties in Tracker.items():
            if properties['name'] in final_Tracker:            
                final_Tracker[properties['name']]['pose_keypoints_2d'] = np.where(~np.isnan(properties['pose_keypoints_2d']), properties['pose_keypoints_2d'], final_Tracker[properties['name']]['pose_keypoints_2d'])
                final_Tracker[properties['name']]['keypoints_3d'] = np.where(~np.isnan(properties['keypoints_3d']), properties['keypoints_3d'], final_Tracker[properties['name']]['keypoints_3d'])   
                for frame, frame_subjuct in zip(range(properties['first_frame'], properties['latest_frame']+1), range(properties['latest_frame']-properties['first_frame']+1)):
                        final_Tracker[properties['name']]['id_excluded_cams'][frame] = properties['id_excluded_cams'][frame_subjuct]
                        final_Tracker[properties['name']]['exclude_record'][frame] = properties['exclude_record'][frame_subjuct]
                        final_Tracker[properties['name']]['error_record'][frame] = properties['error_record'][frame_subjuct]
                        final_Tracker[properties['name']]['cam_dist'][frame] = properties['cam_dist'][frame_subjuct]                    
                        final_Tracker[properties['name']]['strongness_exclusion'][frame] = properties['strongness_exclusion'][frame_subjuct]
            else:
                final_Tracker[properties['name']] = {
                    'pose_keypoints_2d': properties['pose_keypoints_2d'],
                    'id_excluded_cams' : [[np.array([], dtype=np.float64) for _ in range(22)] for _ in range(frames_nb)],             
                    'exclude_record' : [[np.array([], dtype=np.float64) for _ in range(22)] for _ in range(frames_nb)],
                    'error_record' : [[[] for _ in range(22)] for _ in range(frames_nb)],
                    'cam_dist' : [[np.array([], dtype=np.float64) for _ in range(22)] for _ in range(frames_nb)],
                    'strongness_exclusion' : [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for _ in range(frames_nb)],
                    'keypoints_3d': properties['keypoints_3d']
                }
                for frame, frame_subjuct in zip(range(properties['first_frame'], properties['latest_frame']+1), range(properties['latest_frame']-properties['first_frame']+1)):
                    final_Tracker[properties['name']]['id_excluded_cams'][frame] = properties['id_excluded_cams'][frame_subjuct]
                    final_Tracker[properties['name']]['exclude_record'][frame] = properties['exclude_record'][frame_subjuct]
                    final_Tracker[properties['name']]['error_record'][frame] = properties['error_record'][frame_subjuct]
                    final_Tracker[properties['name']]['cam_dist'][frame] = properties['cam_dist'][frame_subjuct]               
                    final_Tracker[properties['name']]['strongness_exclusion'][frame] = properties['strongness_exclusion'][frame_subjuct]
    else:
        for person, properties in Tracker.items():
            final_Tracker[properties['name']] = {
                'pose_keypoints_2d': properties['pose_keypoints_2d'],
                'keypoints_3d': properties['keypoints_3d'],
                'id_excluded_cams' : properties['id_excluded_cams'],                
                'exclude_record' : properties['exclude_record'],
                'error_record' : properties['error_record'],
                'cam_dist' : properties['cam_dist'],
                'strongness_exclusion' : properties['strongness_exclusion']
            }
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
    js_people_empty_template = js_empty_template['people'][0]
    for frame in range(*f_range):                    
        for cam_id in range(len(json_tracked_files)):
            json_tracked_files_f = json_tracked_files[cam_id][frame]
            json_tracked_template = copy.deepcopy(js_empty_template)              
            
            for person, properties in final_Tracker.items():
                json_people_tamplate = copy.deepcopy(js_people_empty_template)
                json_people_tamplate['person_id'] = person
                json_people_tamplate['pose_keypoints_2d'] = properties['pose_keypoints_2d'][frame][cam_id].tolist()
                json_tracked_template['people'].append(json_people_tamplate)
                del json_tracked_template['people'][0]
                parts = json_tracked_files_f.split(os.sep)
                parts = [person if part == 'multi_person' else part for part in parts]
                json_tracked_files_p_f = os.sep.join(parts)                    
                with open(json_tracked_files_p_f, 'w') as file:                    
                    json.dump(json_tracked_template, file)
        Q_tot.append([])
        for person, item in final_Tracker.items():            
            Q_tot[frame].append(np.array(item['keypoints_3d'][frame].reshape(-1)))    

    for person, item in final_Tracker.items(): 
        p_project_dir = os.path.join(Patient_data_path, person, task_category_date, calculate_time, task_name)   
        id_excluded_cams_record_tot, exclude_record_tot, error_record_tot , cam_dist_tot, strongness_exclusion_tot = item['id_excluded_cams'], item['exclude_record'], item['error_record'], item['cam_dist'], item['strongness_exclusion']        
        mdic = {'exclude':exclude_record_tot,'error':error_record_tot,'keypoints_name':keypoints_names,'cam_dist':cam_dist_tot,'cam_choose':id_excluded_cams_record_tot,'strongness_of_exclusion':strongness_exclusion_tot}
        savemat(os.path.join(p_project_dir,'rpj.mat'), mdic) 
        np.savez(os.path.join(p_project_dir,'User','reprojection_record.npz'),exclude= exclude_record_tot,error=error_record_tot,keypoints_name=keypoints_names,cam_dist=cam_dist_tot,cam_choose=id_excluded_cams_record_tot,strongness_of_exclusion =strongness_exclusion_tot)
        np.shape(error_record_tot)
              
    Q_tot = [pd.DataFrame([Q_tot[f][n] for f in range(*f_range)]) for n in range(num_apose)]
    # Delete participants with less than 4 valid triangulated frames
    # for each person, for each keypoint, frames to interpolate
    zero_nan_frames = [np.where( Q_tot[n].iloc[:,::3].T.eq(0) | ~np.isfinite(Q_tot[n].iloc[:,::3].T) ) for n in range(num_apose)]
    zero_nan_frames_per_kpt = [[zero_nan_frames[n][1][np.where(zero_nan_frames[n][0]==k)[0]] for k in range(keypoints_nb)] for n in range(num_apose)]
    non_nan_nb_first_kpt = [frames_nb - len(zero_nan_frames_per_kpt[n][0]) for n in range(num_apose)]
    deleted_person_id = [n for n in range(len(non_nan_nb_first_kpt)) if non_nan_nb_first_kpt[n]<4]
    Q_tot = [Q_tot[n] for n in range(len(Q_tot)) if n not in deleted_person_id]

    nb_persons_to_detect = len(Q_tot)
    # Interpolate missing values
    if interpolation_kind != 'none':
        for n in range(nb_persons_to_detect):
            try:
                Q_tot[n] = Q_tot[n].apply(interpolate_zeros_nans, axis=0, args = [interp_gap_smaller_than, interpolation_kind])
            except:
                logging.info(f'Interpolation was not possible for person {n}. This means that not enough points are available, which is often due to a bad calibration.')
    # Fill non-interpolated values with last valid one
    for n in range(nb_persons_to_detect): 
        Q_tot[n] = Q_tot[n].ffill(axis=0).bfill(axis=0)
        Q_tot[n].replace(np.nan, 0, inplace=True)
    # Create TRC file
    _ = [make_trc(config, Q_tot[n], keypoints_names, f_range, m, os.path.join(Patient_data_path, m, task_category_date, calculate_time, task_name)) for n, m in enumerate(final_Tracker.keys())]    
    
    print('Seccussfully finished Triangulation of multiple person')