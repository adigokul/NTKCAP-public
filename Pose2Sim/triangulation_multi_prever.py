#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
    ###########################################################################
    ## ROBUST TRIANGULATION  OF 2D COORDINATES                               ##
    ###########################################################################
    
    This module triangulates 2D json coordinates and builds a .trc file readable 
    by OpenSim.
    
    The triangulation is weighted by the likelihood of each detected 2D keypoint,
    strives to meet the reprojection error threshold and the likelihood threshold.
    Missing values are then interpolated.

    In case of multiple subjects detection, make sure you first run the track_2d 
    module.

    INPUTS: 
    - a calibration file (.toml extension)
    - json files for each camera with only one person of interest
    - a Config.toml file
    - a skeleton model
    
    OUTPUTS: 
    - a .trc file with 3D coordinates in Y-up system coordinates
    
'''


## INIT
import os
import glob
import fnmatch
import numpy as np
import json
import itertools as it
import pandas as pd
import toml
from tqdm import tqdm
from scipy import interpolate
from scipy.spatial import ConvexHull, Delaunay
from collections import Counter
import logging
import cv2

from Pose2Sim.common_multi import retrieve_calib_params, computeP, weighted_triangulation, \
    reprojection, euclidean_distance, natural_sort, euclidean_dist_with_multiplication
# from pose2Sim.common_multi import convert_to_c3d
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

tracker = {}

def sort_people_new(Q_kpt_old, Q_kpt, f, total_frame_num):
    global tracker
    nb_p_each_frame = len(Q_kpt)
    nan_array = np.full((22,3), np.nan)

    if len(Q_kpt_old) < len(Q_kpt):
        Q_kpt_old = np.concatenate((Q_kpt_old, nan_array*(len(Q_kpt)-len(Q_kpt_old))))
    
    personsIDs_comb = sorted(list(it.product(range(len(Q_kpt_old)),range(len(Q_kpt)))))
    dist = []
    dist += [euclidean_distance(Q_kpt_old[comb[0]], Q_kpt[comb[1]]) for comb in personsIDs_comb]
    
    personsIDs_comb_new = []
    for i in range(0, len(personsIDs_comb), int(len(personsIDs_comb) / nb_p_each_frame)):
        dist_uncheck = dist[i:i+4]
        if np.all(np.isnan(dist_uncheck)):
            continue
        
        personsIDs_comb_new.append(personsIDs_comb[dist_uncheck.index(min(dist_uncheck))+i])
    _, _, min_dist_comb = min_with_single_indices(dist, personsIDs_comb)
    
    
    for idx, c in enumerate(min_dist_comb):
        if f == 1:
            tracker[f"person{idx+1}"] = {
                'id' : c[1],
                'matching status' : False,
                'keypoints' : np.expand_dims(Q_kpt_old[c[0]], axis=0)
            }
            tracker[f"person{idx+1}"]['keypoints'] = np.append(tracker[f"person{idx+1}"]['keypoints'], np.expand_dims(Q_kpt[c[1]], axis=0), axis=0)
        else:
            matched = False
            for idx_pre, (person, p_pre) in enumerate(tracker.items()):
                if p_pre['id'] == -1 : continue
                    
                if c[0] == p_pre['id'] and tracker[person]['matching status'] == True:
                    
                    matched = True
                    tracker[person]['id'] = c[1]
                    tracker[person]['matching status'] = False
                    tracker[person]['keypoints'] = np.append(tracker[person]['keypoints'], np.expand_dims(Q_kpt[c[1]], axis=0), axis=0)
                    break
            if not matched: # new person
                nan_fill = np.full((f-1, 22, 3), np.nan)
                new_name = f"person{len(tracker)+1}"

                tracker[new_name] = {
                    'id':c[1],
                    'matching status' : False,
                    'keypoints':np.append(nan_fill, np.expand_dims(Q_kpt_old[c[0]], axis=0), axis=0)
                }
                tracker[new_name]['keypoints'] = np.append(tracker[new_name]['keypoints'], np.expand_dims(Q_kpt[c[1]], axis=0), axis=0)    
    for i in range(len(tracker)):
        if tracker[f"person{i+1}"]['matching status'] == True:
            nan_fill = np.full((total_frame_num - len(tracker[f"person{i+1}"]['keypoints']), 22, 3), np.nan)
            tracker[f"person{i+1}"]['id'] = -1
            tracker[f"person{i+1}"]['keypoints'] = np.append(tracker[f"person{i+1}"]['keypoints'], nan_fill, axis=0)
        tracker[f"person{i+1}"]['matching status'] = True
    

def sort_people(Q_kpt_old, Q_kpt):
    '''
    Associate persons across frames
    Persons' indices are sometimes swapped when changing frame
    A person is associated to another in the next frame when they are at a small distance
    
    INPUTS:
    - Q_kpt_old: list of arrays of 3D coordinates [X, Y, Z, 1.] for the previous frame
    - Q_kpt: idem Q_kpt_old, for current frame
    
    OUTPUT:
    - Q_kpt_new: array with reordered persons
    - personsIDs_sorted: index of reordered persons
    '''
    
    # Generate possible person correspondences across frames
    if len(Q_kpt_old) < len(Q_kpt):
        Q_kpt_old = np.concatenate((Q_kpt_old, [[0., 0., 0., 1.]]*(len(Q_kpt)-len(Q_kpt_old))))
    
    personsIDs_comb = sorted(list(it.product(range(len(Q_kpt_old)),range(len(Q_kpt)))))
    
    # Compute distance between persons from one frame to another
    frame_by_frame_dist = []
    # print(f)
    for comb in personsIDs_comb:
        # print(Q_kpt_old[comb[0]])
        # print(Q_kpt[comb[1]])
        # print(comb)
        frame_by_frame_dist += [euclidean_distance(Q_kpt_old[comb[0]],Q_kpt[comb[1]])]
        # print(euclidean_distance(Q_kpt_old[comb[0]],Q_kpt[comb[1]]))
        
    # sort correspondences by distance
    
    minL, _, associated_tuples = min_with_single_indices(frame_by_frame_dist, personsIDs_comb)
    
    # associate 3D points to same index across frames, nan if no correspondence
    Q_kpt_new, personsIDs_sorted = [], []
    for i in range(len(Q_kpt_old)):
        id_in_old =  associated_tuples[:,1][associated_tuples[:,0] == i].tolist()
        
        if len(id_in_old) > 0:
            personsIDs_sorted += id_in_old
            Q_kpt_new += [Q_kpt[id_in_old[0]]]
        else:
            personsIDs_sorted += [-1]
            Q_kpt_new += [Q_kpt_old[i]]
    
    return Q_kpt_new, personsIDs_sorted, associated_tuples


def make_trc(config, Q, keypoints_names, f_range, id_person=-1):
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
        seq_name = f'{os.path.basename(os.path.realpath(project_dir))}_P{id_person+1}'
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


def retrieve_right_trc_order(trc_paths):
    '''
    Lets the user input which static file correspond to each generated trc file.
    
    INPUT:
    - trc_paths: list of strings
    
    OUTPUT:
    - trc_id: list of integers
    '''
    
    logging.info('\n\nReordering trc file IDs:')
    logging.info(f'\nPlease visualize the generated trc files in Blender or OpenSim.\nTrc files are stored in {os.path.dirname(trc_paths[0])}.\n')
    retry = True
    while retry:
        retry = False
        logging.info('List of trc files:')
        [logging.info(f'#{t_list}: {os.path.basename(trc_list)}') for t_list, trc_list in enumerate(trc_paths)]
        trc_id = []
        for t, trc_p in enumerate(trc_paths):
            logging.info(f'\nStatic trial #{t} corresponds to trc number:')
            trc_id += [input('Enter ID:')]
        
        # Check non int and duplicates
        try:
            trc_id = [int(t) for t in trc_id]
            duplicates_in_input = (len(trc_id) != len(set(trc_id)))
            if duplicates_in_input:
                retry = True
                print('\n\nWARNING: Same ID entered twice: please check IDs again.\n')
        except:
            print('\n\nWARNING: The ID must be an integer: please check IDs again.\n')
            retry = True
    
    return trc_id


# def recap_triangulate(config, error, nb_cams_excluded, keypoints_names, cam_excluded_count, interp_frames, non_interp_frames, trc_path):
#     '''
#     Print a message giving statistics on reprojection errors (in pixel and in m)
#     as well as the number of cameras that had to be excluded to reach threshold 
#     conditions. Also stored in User/logs.txt.

#     INPUT:
#     - a Config.toml file
#     - error: dataframe 
#     - nb_cams_excluded: dataframe
#     - keypoints_names: list of strings

#     OUTPUT:
#     - Message in console
#     '''

    # Read config
    # project_dir = config.get('project').get('project_dir')
    # if batch
    # session_dir = os.path.realpath(os.path.join(project_dir, '..', '..'))
    # if single trial
    # session_dir = session_dir if 'Config.toml' in os.listdir(session_dir) else os.getcwd()
    # calib_dir = [os.path.join(session_dir, c) for c in os.listdir(session_dir) if 'calib' in c.lower()][0]
    # calib_file = glob.glob(os.path.join(calib_dir, '*.toml'))[0] # lastly created calibration file
    # calib = toml.load(calib_file)
    # cam_names = np.array([calib[c].get('name') for c in list(calib.keys())])
    # cam_names = cam_names[list(cam_excluded_count[0].keys())] # bugs
    # error_threshold_triangulation = config.get('triangulation').get('reproj_error_threshold_triangulation')
    # likelihood_threshold = config.get('triangulation').get('likelihood_threshold_triangulation')
    # show_interp_indices = config.get('triangulation').get('show_interp_indices')
    # interpolation_kind = config.get('triangulation').get('interpolation')
    # interp_gap_smaller_than = config.get('triangulation').get('interp_if_gap_smaller_than')
    # make_c3d = config.get('triangulation').get('make_c3d')
    # handle_LR_swap = config.get('triangulation').get('handle_LR_swap')
    # undistort_points = config.get('triangulation').get('undistort_points')
    
    # Recap
    # calib_cam1 = calib[list(calib.keys())[0]]
    # fm = calib_cam1['matrix'][0][0]
    # Dm = euclidean_distance(calib_cam1['translation'], [0,0,0])

    # logging.info('')
    # nb_persons_to_detect = len(error)
    # for n in range(nb_persons_to_detect):
    #     if nb_persons_to_detect > 1:
    #         logging.info(f'\n\nPARTICIPANT {n+1}\n')
        
    #     for idx, name in enumerate(keypoints_names):
    #         mean_error_keypoint_px = np.around(error[n].iloc[:,idx].mean(), decimals=1) # RMS à la place?
    #         mean_error_keypoint_m = np.around(mean_error_keypoint_px * Dm / fm, decimals=3)
    #         mean_cam_excluded_keypoint = np.around(nb_cams_excluded[n].iloc[:,idx].mean(), decimals=2)
    #         logging.info(f'Mean reprojection error for {name} is {mean_error_keypoint_px} px (~ {mean_error_keypoint_m} m), reached with {mean_cam_excluded_keypoint} excluded cameras. ')
    #         if show_interp_indices:
    #             if interpolation_kind != 'none':
    #                 if len(list(interp_frames[n][idx])) == 0 and len(list(non_interp_frames[n][idx])) == 0:
    #                     logging.info(f'  No frames needed to be interpolated.')
    #                 if len(list(interp_frames[n][idx]))>0: 
    #                     interp_str = str(interp_frames[n][idx]).replace(":", " to ").replace("'", "").replace("]", "").replace("[", "")
    #                     logging.info(f'  Frames {interp_str} were interpolated.')
    #                 if len(list(non_interp_frames[n][idx]))>0:
    #                     noninterp_str = str(non_interp_frames[n][idx]).replace(":", " to ").replace("'", "").replace("]", "").replace("[", "")
    #                     logging.info(f'  Frames {noninterp_str} were not interpolated.')
    #             else:
    #                 logging.info(f'  No frames were interpolated because \'interpolation_kind\' was set to none. ')
        
    #     mean_error_px = np.around(error[n]['mean'].mean(), decimals=1)
    #     mean_error_mm = np.around(mean_error_px * Dm / fm *1000, decimals=1)
    #     mean_cam_excluded = np.around(nb_cams_excluded[n]['mean'].mean(), decimals=2)

    #     logging.info(f'\n--> Mean reprojection error for all points on all frames is {mean_error_px} px, which roughly corresponds to {mean_error_mm} mm. ')
    #     logging.info(f'Cameras were excluded if likelihood was below {likelihood_threshold} and if the reprojection error was above {error_threshold_triangulation} px.') 
    #     if interpolation_kind != 'none':
    #         logging.info(f'Gaps were interpolated with {interpolation_kind} method if smaller than {interp_gap_smaller_than} frames.') 
    #     logging.info(f'In average, {mean_cam_excluded} cameras had to be excluded to reach these thresholds.')
        
        # cam_excluded_count[n] = {i: v for i, v in zip(cam_names, cam_excluded_count[n].values())}
        # cam_excluded_count[n] = {k: v for k, v in sorted(cam_excluded_count[n].items(), key=lambda item: item[1])[::-1]}
        # str_cam_excluded_count = ''
        # for i, (k, v) in enumerate(cam_excluded_count[n].items()):
        #     if i ==0:
        #          str_cam_excluded_count += f'Camera {k} was excluded {int(np.round(v*100))}% of the time, '
        #     elif i == len(cam_excluded_count[n])-1:
        #         str_cam_excluded_count += f'and Camera {k}: {int(np.round(v*100))}%.'
        #     else:
        #         str_cam_excluded_count += f'Camera {k}: {int(np.round(v*100))}%, '
        # logging.info(str_cam_excluded_count)
    #     logging.info(f'\n3D coordinates are stored at {trc_path[n]}.')
        
    # logging.info('\n\n')
    # if make_c3d:
    #     logging.info('All trc files have been converted to c3d.')
    # logging.info(f'Limb swapping was {"handled" if handle_LR_swap else "not handled"}.')
    # logging.info(f'Lens distortions were {"taken into account" if undistort_points else "not taken into account"}.')

def triangulation_from_best_cameras(config, coords_2D_kpt, coords_2D_kpt_swapped, projection_matrices, body_name):
    '''
    Triangulates 2D keypoint coordinates. If reprojection error is above threshold,
    tries swapping left and right sides. If still above, removes a camera until error
    is below threshold unless the number of remaining cameras is below a predefined number.

    1. Creates subset with N cameras excluded 
    2. Tries all possible triangulations
    3. Chooses the one with smallest reprojection error
    If error too big, take off one more camera.
        If then below threshold, retain result.
        If better but still too big, take off one more camera.
    
    INPUTS:
    - a Config.toml file
    - coords_2D_kpt: (x,y,likelihood) * ncams array
    - coords_2D_kpt_swapped: (x,y,likelihood) * ncams array  with left/right swap
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
    handle_LR_swap = config.get('triangulation').get('handle_LR_swap')

    # Initialize
    x_files, y_files, likelihood_files = coords_2D_kpt
    x_files_swapped, y_files_swapped, likelihood_files_swapped = coords_2D_kpt_swapped
    n_cams = len(x_files)
    error_min = np.inf 
    project_dir = config.get('project').get('project_dir')
    if project_dir == '': project_dir = os.getcwd()
    calib_folder_name = config.get('project').get('calib_folder_name')
    calib_dir = os.path.join(project_dir, calib_folder_name)
    calib_file = glob.glob(os.path.join(calib_dir, '*.toml'))[0]
    calib = toml.load(calib_file)
    nb_cams_off = 0 # cameras will be taken-off until reprojection error is under threshold
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
    error_record1 =[]
    count_all_com = 0
    first_tri = 1
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
        #import pdb;pdb.set_trace()
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
            
            #import pdb;pdb.set_trace()
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

        # idxs = np.argsort(error)
        # Q = Q_filt[idxs[:3]].mean(axis=0)
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

    # If triangulation not successful, error = 0,  and 3D coordinates as missing values
    if error_min > error_threshold_triangulation:
        error_min = np.nan
        Q = np.array([0.,0.,0.])
        Q = np.array([np.nan, np.nan, np.nan])

    return Q_final, error_min_final, nb_cams_excluded_final, id_excluded_cams_final

def extract_files_frame_f(json_tracked_files_f, keypoints_ids, nb_persons_to_detect):
    '''
    Extract data from json files for frame f, 
    in the order of the body model hierarchy.

    INPUTS:
    - json_tracked_files_f: list of str. Paths of json_files for frame f.
    - keypoints_ids: list of int. Keypoints IDs in the order of the hierarchy.
    - nb_persons_to_detect: int

    OUTPUTS:
    - x_files, y_files, likelihood_files: [[[list of coordinates] * n_cams ] * nb_persons_to_detect]
    '''

    n_cams = len(json_tracked_files_f)
    
    x_files = [[] for n in range(nb_persons_to_detect)]
    y_files = [[] for n in range(nb_persons_to_detect)]
    likelihood_files = [[] for n in range(nb_persons_to_detect)]
    for n in range(nb_persons_to_detect):
        for cam_nb in range(n_cams):
            x_files_cam, y_files_cam, likelihood_files_cam = [], [], []
            with open(json_tracked_files_f[cam_nb], 'r') as json_f:
                js = json.load(json_f)
                for keypoint_id in keypoints_ids:
                    try:
                        x_files_cam.append( js['people'][n]['pose_keypoints_2d'][keypoint_id*3] )
                        y_files_cam.append( js['people'][n]['pose_keypoints_2d'][keypoint_id*3+1] )
                        likelihood_files_cam.append( js['people'][n]['pose_keypoints_2d'][keypoint_id*3+2] )
                    except:
                        x_files_cam.append( np.nan )
                        y_files_cam.append( np.nan )
                        likelihood_files_cam.append( np.nan )
            x_files[n].append(x_files_cam)
            y_files[n].append(y_files_cam)
            likelihood_files[n].append(likelihood_files_cam)
        
    x_files = np.array(x_files)
    y_files = np.array(y_files)
    likelihood_files = np.array(likelihood_files)

    return x_files, y_files, likelihood_files


def triangulate_all(config):
    '''
    For each frame
    For each keypoint
    - Triangulate keypoint
    - Reproject it on all cameras
    - Take off cameras until requirements are met
    Interpolate missing values
    Create trc file
    Print recap message
    
     INPUTS: 
    - a calibration file (.toml extension)
    - json files for each camera with indices matching the detected persons
    - a Config.toml file
    - a skeleton model
    
    OUTPUTS: 
    - a .trc file with 3D coordinates in Y-up system coordinates 
    '''
    global tracker
    # Read config
    project_dir = config.get('project').get('project_dir')
    if project_dir == '': project_dir = os.getcwd()
    
    calib_folder_name = config.get('project').get('calib_folder_name')
    pose_model = config.get('pose').get('pose_model')
    pose_folder_name = config.get('project').get('pose_folder_name')
    json_folder_extension =  config.get('project').get('pose_json_folder_extension')
    frame_range = config.get('project').get('frame_range')
    likelihood_threshold = config.get('triangulation').get('likelihood_threshold')
    interpolation_kind = config.get('triangulation').get('interpolation')
    interp_gap_smaller_than = config.get('triangulation').get('interp_if_gap_smaller_than')
    show_interp_indices = config.get('triangulation').get('show_interp_indices')
    pose_dir = os.path.join(project_dir, pose_folder_name)
    poseTracked_folder_name = config.get('project').get('poseAssociated_folder_name')
    calib_dir = os.path.join(project_dir, calib_folder_name)
    calib_file = glob.glob(os.path.join(calib_dir, '*.toml'))[0]
    poseTracked_dir = os.path.join(project_dir, poseTracked_folder_name)
    make_c3d = config.get('triangulation').get('make_c3d')

    undistort_points = config.get('triangulation').get('undistort_points')
    multi_person = config.get('project').get('multi_person')
    reorder_trc = config.get('triangulation').get('reorder_trc')
    
    # Projection matrix from toml calibration file
    P = computeP(calib_file, undistort=undistort_points)
    calib_params = retrieve_calib_params(calib_file)
    
    
    
    # Retrieve keypoints from model
    model = eval(pose_model)
    keypoints_ids = [node.id for _, _, node in RenderTree(model) if node.id!=None]
    keypoints_names = [node.name for _, _, node in RenderTree(model) if node.id!=None]
    keypoints_idx = list(range(len(keypoints_ids)))
    keypoints_nb = len(keypoints_ids)
    
    keypoints_names_swapped = [keypoint_name.replace('R', 'L') if keypoint_name.startswith('R') else keypoint_name.replace('L', 'R') if keypoint_name.startswith('L') else keypoint_name for keypoint_name in keypoints_names]
    keypoints_names_swapped = [keypoint_name_swapped.replace('right', 'left') if keypoint_name_swapped.startswith('right') else keypoint_name_swapped.replace('left', 'right') if keypoint_name_swapped.startswith('left') else keypoint_name_swapped for keypoint_name_swapped in keypoints_names_swapped]
    keypoints_idx_swapped = [keypoints_names.index(keypoint_name_swapped) for keypoint_name_swapped in keypoints_names_swapped]
    
    # 2d-pose files selection
    pose_listdirs_names = next(os.walk(pose_dir))[1]
    pose_listdirs_names = natural_sort(pose_listdirs_names)
    json_dirs_names = [k for k in pose_listdirs_names if json_folder_extension in k]
    try: 
        json_files_names = [fnmatch.filter(os.listdir(os.path.join(poseTracked_dir, js_dir)), '*.json') for js_dir in json_dirs_names]
        json_files_names = [natural_sort(j) for j in json_files_names]
        json_tracked_files = [[os.path.join(poseTracked_dir, j_dir, j_file) for j_file in json_files_names[j]] for j, j_dir in enumerate(json_dirs_names)]
    except:
        json_files_names = [fnmatch.filter(os.listdir(os.path.join(pose_dir, js_dir)), '*.json') for js_dir in json_dirs_names]
        json_files_names = [natural_sort(j) for j in json_files_names]
        json_tracked_files = [[os.path.join(pose_dir, j_dir, j_file) for j_file in json_files_names[j]] for j, j_dir in enumerate(json_dirs_names)]
    
    # Triangulation
    f_range = [[0,min([len(j) for j in json_files_names])] if frame_range==[] else frame_range][0]
    frames_nb = f_range[1]-f_range[0]
    nb_persons_to_detect = max([len(json.load(open(json_fname))['people']) for json_fname in json_tracked_files[0]])
    
    n_cams = len(json_dirs_names)
    
    Q = [[[np.nan]*3]*keypoints_nb for n in range(nb_persons_to_detect)]
    Q_old = [[[np.nan]*3]*keypoints_nb for n in range(nb_persons_to_detect)]
    error = [[] for n in range(nb_persons_to_detect)]
    nb_cams_excluded = [[] for n in range(nb_persons_to_detect)]
    id_excluded_cams = [[] for n in range(nb_persons_to_detect)]
    Q_tot, error_tot, nb_cams_excluded_tot,id_excluded_cams_tot = [], [], [], []
    
    for f in tqdm(range(*f_range)):
        # Get x,y,likelihood values from files
        json_tracked_files_f = [json_tracked_files[c][f] for c in range(n_cams)]
        
        x_files, y_files, likelihood_files = extract_files_frame_f(json_tracked_files_f, keypoints_ids, nb_persons_to_detect)
        # undistort points
        if undistort_points:
            for n in range(nb_persons_to_detect):
                points = [np.array(tuple(zip(x_files[n][i],y_files[n][i]))).reshape(-1, 1, 2).astype('float32') for i in range(n_cams)]
                undistorted_points = [cv2.undistortPoints(points[i], calib_params['K'][i], calib_params['dist'][i], None, calib_params['optim_K'][i]) for i in range(n_cams)]
                x_files[n] =  np.array([[u[i][0][0] for i in range(len(u))] for u in undistorted_points])
                y_files[n] =  np.array([[u[i][0][1] for i in range(len(u))] for u in undistorted_points])
                # This is good for slight distortion. For fisheye camera, the model does not work anymore. See there for an example https://github.com/lambdaloop/aniposelib/blob/d03b485c4e178d7cff076e9fe1ac36837db49158/aniposelib/cameras.py#L301
                
        # Replace likelihood by 0 if under likelihood_threshold
        with np.errstate(invalid='ignore'):
            for n in range(nb_persons_to_detect):
                x_files[n][likelihood_files[n] < likelihood_threshold] = np.nan
                y_files[n][likelihood_files[n] < likelihood_threshold] = np.nan
                likelihood_files[n][likelihood_files[n] < likelihood_threshold] = 0.
        
        # Q_old = Q except when it has nan, otherwise it takes the Q_old value
        nan_mask = np.isnan(Q)
        Q_old = np.where(nan_mask, Q_old, Q)
        Q = [[] for n in range(nb_persons_to_detect)]
        error = [[] for n in range(nb_persons_to_detect)]
        nb_cams_excluded = [[] for n in range(nb_persons_to_detect)]
        id_excluded_cams = [[] for n in range(nb_persons_to_detect)]
        
        for n in range(nb_persons_to_detect):
            for keypoint_idx in keypoints_idx:
            # Triangulate cameras with min reprojection error
            # 同一個人所有相機同一幀的x,y,信心
                coords_2D_kpt = np.array( (x_files[n][:, keypoint_idx], y_files[n][:, keypoint_idx], likelihood_files[n][:, keypoint_idx]) )
                coords_2D_kpt_swapped = np.array(( x_files[n][:, keypoints_idx_swapped[keypoint_idx]], y_files[n][:, keypoints_idx_swapped[keypoint_idx]], likelihood_files[n][:, keypoints_idx_swapped[keypoint_idx]] ))
                Q_kpt, error_kpt, nb_cams_excluded_kpt, id_excluded_cams_kpt = triangulation_from_best_cameras(config, coords_2D_kpt, coords_2D_kpt_swapped, P, keypoints_names[keypoint_idx])
                
                
                Q[n].append(Q_kpt) # Q_kpt:same kp(3D), all cams, same person, same frame
                error[n].append(error_kpt)
                nb_cams_excluded[n].append(nb_cams_excluded_kpt)
                id_excluded_cams[n].append(id_excluded_cams_kpt)
        
        # import pdb; pdb.set_trace()
        if multi_person:
            # reID persons across frames by checking the distance from one frame to another
            if f !=0:
                
                # Q, personsIDs_sorted, associated_tuples = sort_people_new(Q_old, Q, f, frames_nb)
                
                sort_people_new(Q_old, Q, f, frames_nb)
                # Q, personsIDs_sorted, associated_tuples = sort_people(Q_old, Q)
                
                # error_sorted, nb_cams_excluded_sorted, id_excluded_cams_sorted = [], [], []
                # for i in range(len(Q)):
                #     id_in_old =  associated_tuples[:,1][associated_tuples[:,0] == i].tolist()
                #     if len(id_in_old) > 0:
                #         personsIDs_sorted += id_in_old
                #         error_sorted += [error[id_in_old[0]]]
                #         nb_cams_excluded_sorted += [nb_cams_excluded[id_in_old[0]]]
                #         id_excluded_cams_sorted += [id_excluded_cams[id_in_old[0]]]
                #     else:
                #         personsIDs_sorted += [-1]
                #         error_sorted += [error[i]]
                #         nb_cams_excluded_sorted += [nb_cams_excluded[i]]
                #         id_excluded_cams_sorted += [id_excluded_cams[i]]
                # error, nb_cams_excluded, id_excluded_cams = error_sorted, nb_cams_excluded_sorted, id_excluded_cams_sorted
            
        # Add triangulated points, errors and excluded cameras to pandas dataframes
        # Q_tot.append([np.concatenate(Q[n]) for n in range(nb_persons_to_detect)])
        # error_tot.append(error)
        # nb_cams_excluded_tot.append(nb_cams_excluded)
        # id_excluded_cams = [item for sublist in id_excluded_cams for item in sublist]
        # id_excluded_cams_tot.append(id_excluded_cams)
    
    
    # person1_data = []
    # person2_data = []

    # for i in range(len(Q_tot)):
    #     person1_data.append(Q_tot[i][:22].tolist())
    #     person2_data.append(Q_tot[i][22:].tolist())
    # kp_json_file = [person1_data, person2_data]
    # with open('kp_data.json', 'w') as json_file:
    #     json.dump(kp_json_file, json_file)

    # skeletons = [
    #     (0, 1), (1, 2), (2, 3), (3, 4), (3, 5), (3, 6), (0, 7), (7, 8),
    #     (8, 9), (9, 10), (9, 11), (9, 12), (13, 0),(13, 14), (14, 15), (13, 16),
    #     (16, 17), (17, 18), (13, 19), (19, 20), (20, 21)]
    # list_dynamic_mincam=  {'Hip':4,'RHip':4,'RKnee':3,'RAnkle':3,'RBigToe':3,'RSmallToe':3,'RHeel':3,'LHip':4,'LKnee':3,'LAnkle':3,'LBigToe':3,'LSmallToe':3,'LHeel':3,'Neck':3,'Head':2,'Nose':2,'RShoulder':3,'RElbow':3,'RWrist':3,'LShoulder':3,'LElbow':3,'LWrist':3}
    # check_outside_kp = [0, 13, 14, 15]
    # check_outside_frnum = 5
    
    
    for frame in range(*f_range):
        Q_tot.append([])
        for person in tracker.items():
            Q_tot[frame].append(np.array(person[1]['keypoints'][frame].reshape(-1)))
    
    # fill values for if a person that was not initially detected has entered the frame 
    Q_tot = [list(tpl) for tpl in zip(*it.zip_longest(*Q_tot, fillvalue=[np.nan]*keypoints_nb*3))]
    
    # error_tot = [list(tpl) for tpl in zip(*it.zip_longest(*error_tot, fillvalue=[np.nan]*keypoints_nb*3))]
    # nb_cams_excluded_tot = [list(tpl) for tpl in zip(*it.zip_longest(*nb_cams_excluded_tot, fillvalue=[np.nan]*keypoints_nb*3))]
    # id_excluded_cams_tot = [list(tpl) for tpl in zip(*it.zip_longest(*id_excluded_cams_tot, fillvalue=[np.nan]*keypoints_nb*3))]
    
    # dataframes for each person
    
    Q_tot = [pd.DataFrame([Q_tot[f][n] for f in range(frames_nb)]) for n in range(nb_persons_to_detect)]
    
    # error_tot = [pd.DataFrame([error_tot[f][n] for f in range(frames_nb)]) for n in range(nb_persons_to_detect)]
    # nb_cams_excluded_tot = [pd.DataFrame([nb_cams_excluded_tot[f][n] for f in range(frames_nb)]) for n in range(nb_persons_to_detect)]
    # id_excluded_cams_tot = [pd.DataFrame([id_excluded_cams_tot[f][n] for f in range(frames_nb)]) for n in range(nb_persons_to_detect)]
    
    # for n in range(nb_persons_to_detect):
    #     error_tot[n]['mean'] = error_tot[n].mean(axis = 1)
    #     nb_cams_excluded_tot[n]['mean'] = nb_cams_excluded_tot[n].mean(axis = 1)
        
    # Delete participants with less than 4 valid triangulated frames
    # for each person, for each keypoint, frames to interpolate
    zero_nan_frames = [np.where( Q_tot[n].iloc[:,::3].T.eq(0) | ~np.isfinite(Q_tot[n].iloc[:,::3].T) ) for n in range(nb_persons_to_detect)]
    zero_nan_frames_per_kpt = [[zero_nan_frames[n][1][np.where(zero_nan_frames[n][0]==k)[0]] for k in range(keypoints_nb)] for n in range(nb_persons_to_detect)]
    non_nan_nb_first_kpt = [frames_nb - len(zero_nan_frames_per_kpt[n][0]) for n in range(nb_persons_to_detect)]
    deleted_person_id = [n for n in range(len(non_nan_nb_first_kpt)) if non_nan_nb_first_kpt[n]<4]

    Q_tot = [Q_tot[n] for n in range(len(Q_tot)) if n not in deleted_person_id]

    # error_tot = [error_tot[n] for n in range(len(error_tot)) if n not in deleted_person_id]
    # nb_cams_excluded_tot = [nb_cams_excluded_tot[n] for n in range(len(nb_cams_excluded_tot)) if n not in deleted_person_id]
    # id_excluded_cams_tot = [id_excluded_cams_tot[n] for n in range(len(id_excluded_cams_tot)) if n not in deleted_person_id]
    nb_persons_to_detect = len(Q_tot)
    
    # IDs of excluded cameras
    # id_excluded_cams_tot = [np.concatenate([id_excluded_cams_tot[f][k] for f in range(frames_nb)]) for k in range(keypoints_nb)]
    # id_excluded_cams_tot = [np.hstack(np.hstack(np.array(id_excluded_cams_tot[n]))) for n in range(nb_persons_to_detect)]
    # cam_excluded_count = [dict(Counter(k)) for k in id_excluded_cams_tot]
    # [cam_excluded_count[n].update((x, y/frames_nb/keypoints_nb) for x, y in cam_excluded_count[n].items()) for n in range(nb_persons_to_detect)]
    
    # Optionally, for each keypoint, show indices of frames that should be interpolated
    # if show_interp_indices:
    #     gaps = [[np.where(np.diff(zero_nan_frames_per_kpt[n][k]) > 1)[0] + 1 for k in range(keypoints_nb)] for n in range(nb_persons_to_detect)]
    #     sequences = [[np.split(zero_nan_frames_per_kpt[n][k], gaps[n][k]) for k in range(keypoints_nb)] for n in range(nb_persons_to_detect)]
    #     interp_frames = [[[f'{seq[0]}:{seq[-1]}' for seq in seq_kpt if len(seq)<=interp_gap_smaller_than and len(seq)>0] for seq_kpt in sequences[n]] for n in range(nb_persons_to_detect)]
    #     non_interp_frames = [[[f'{seq[0]}:{seq[-1]}' for seq in seq_kpt if len(seq)>interp_gap_smaller_than] for seq_kpt in sequences[n]] for n in range(nb_persons_to_detect)]
    # else:
    #     interp_frames = None
    #     non_interp_frames = []

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
        # Q_tot[n].replace(np.nan, 0, inplace=True)
    
    
    # Create TRC file
    trc_paths = [make_trc(config, Q_tot[n], keypoints_names, f_range, id_person=n) for n in range(len(Q_tot))]
    # if make_c3d:
    #     c3d_paths = [convert_to_c3d(t) for t in trc_paths]
    # Reorder TRC files
    if multi_person and reorder_trc and len(trc_paths)>1:
        trc_id = retrieve_right_trc_order(trc_paths)
        [os.rename(t, t+'.old') for t in trc_paths]
        [os.rename(t+'.old', trc_paths[i]) for i, t in zip(trc_id,trc_paths)]
        # if make_c3d:
        #     [os.rename(c, c+'.old') for c in c3d_paths]
        #     [os.rename(c+'.old', c3d_paths[i]) for i, c in zip(trc_id,c3d_paths)]
        error_tot = [error_tot[i] for i in trc_id]
        nb_cams_excluded_tot = [nb_cams_excluded_tot[i] for i in trc_id]
        cam_excluded_count = [cam_excluded_count[i] for i in trc_id]
        interp_frames = [interp_frames[i] for i in trc_id]
        non_interp_frames = [non_interp_frames[i] for i in trc_id]
        
        logging.info('\nThe trc and c3d files have been renamed to match the order of the static sequences.')
    
    
    # Recap message
    # recap_triangulate(config, error_tot, nb_cams_excluded_tot, keypoints_names, cam_excluded_count, interp_frames, non_interp_frames, trc_paths)
    