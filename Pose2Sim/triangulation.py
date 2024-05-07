#!/usr/bin/env python
# -*- coding: utf-8 -*-
###version2

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
from collections import Counter
import logging

from Pose2Sim.common import computeP, weighted_triangulation, reprojection, \
    euclidean_distance, natural_sort, euclidean_dist_with_multiplication,camera2point_dist
from Pose2Sim.skeletons import *


## AUTHORSHIP INFORMATION
__author__ = "David Pagnon"
__copyright__ = "Copyright 2021, Pose2Sim"
__credits__ = ["David Pagnon"]
__license__ = "BSD 3-Clause License"
__version__ = '0.4'
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


def make_trc(config, Q, keypoints_names, f_range):
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
    #import pdb;pdb.set_trace()
    #Write file
    if not os.path.exists(pose3d_dir): os.mkdir(pose3d_dir)
    trc_path = os.path.join(pose3d_dir, trc_f)
    with open(trc_path, 'w') as trc_o:
        [trc_o.write(line+'\n') for line in header_trc]
        Q.to_csv(trc_o, sep='\t', index=True, header=None, line_terminator='\n')

    return trc_path

def make_trc_toe_mean(config, Q, keypoints_names, f_range):
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

    Q[14] =(Q[14]+Q[17])/2
    Q[17] =(Q[14]+Q[17])/2
    Q[32] = (Q[32]+Q[35])/2
    Q[35]= (Q[32]+Q[35])/2
    import pdb;pdb.set_trace()
    with open(trc_path, 'w') as trc_o:
        [trc_o.write(line+'\n') for line in header_trc]
        Q.to_csv(trc_o, sep='\t', index=True, header=None, line_terminator='\n')
    
    return trc_path
def recap_triangulate(config, error, nb_cams_excluded, keypoints_names, cam_excluded_count, interp_frames, non_interp_frames, trc_path):
    '''
    Print a message giving statistics on reprojection errors (in pixel and in m)
    as well as the number of cameras that had to be excluded to reach threshold 
    conditions. Also stored in User/logs.txt.

    INPUT:
    - a Config.toml file
    - error: dataframe 
    - nb_cams_excluded: dataframe
    - keypoints_names: list of strings

    OUTPUT:
    - Message in console
    '''

    # Read config
    project_dir = config.get('project').get('project_dir')
    if project_dir == '': project_dir = os.getcwd()
    calib_folder_name = config.get('project').get('calib_folder_name')
    calib_dir = os.path.join(project_dir, calib_folder_name)
    calib_file = glob.glob(os.path.join(calib_dir, '*.toml'))[0]
    calib = toml.load(calib_file)
    cam_names = np.array([calib[c].get('name') for c in list(calib.keys())])
    cam_names = cam_names[list(cam_excluded_count.keys())]
    error_threshold_triangulation = config.get('triangulation').get('reproj_error_threshold_triangulation')
    likelihood_threshold = config.get('triangulation').get('likelihood_threshold')
    show_interp_indices = config.get('triangulation').get('show_interp_indices')
    interpolation_kind = config.get('triangulation').get('interpolation')
    
    # Recap
    calib_cam1 = calib[list(calib.keys())[0]]
    fm = calib_cam1['matrix'][0][0]
    Dm = euclidean_distance(calib_cam1['translation'], [0,0,0])

    logging.info('')
    for idx, name in enumerate(keypoints_names):
        mean_error_keypoint_px = np.around(error.iloc[:,idx].mean(), decimals=1) # RMS à la place?
        mean_error_keypoint_m = np.around(mean_error_keypoint_px * Dm / fm, decimals=3)
        mean_cam_excluded_keypoint = np.around(nb_cams_excluded.iloc[:,idx].mean(), decimals=2)
        logging.info(f'Mean reprojection error for {name} is {mean_error_keypoint_px} px (~ {mean_error_keypoint_m} m), reached with {mean_cam_excluded_keypoint} excluded cameras. ')
        if show_interp_indices:
            if interpolation_kind != 'none':
                if len(list(interp_frames[idx])) ==0:
                    logging.info(f'  No frames needed to be interpolated.')
                else: 
                    logging.info(f'  Frames {list(interp_frames[idx])} were interpolated.')
                if len(list(non_interp_frames[idx]))>0:
                    logging.info(f'  Frames {list(non_interp_frames[idx])} could not be interpolated: consider adjusting thresholds.')
            else:
                logging.info(f'  No frames were interpolated because \'interpolation_kind\' was set to none. ')
    
    mean_error_px = np.around(error['mean'].mean(), decimals=1)
    mean_error_mm = np.around(mean_error_px * Dm / fm *1000, decimals=1)
    mean_cam_excluded = np.around(nb_cams_excluded['mean'].mean(), decimals=2)

    logging.info(f'\n--> Mean reprojection error for all points on all frames is {mean_error_px} px, which roughly corresponds to {mean_error_mm} mm. ')
    logging.info(f'Cameras were excluded if likelihood was below {likelihood_threshold} and if the reprojection error was above {error_threshold_triangulation} px.')
    logging.info(f'In average, {mean_cam_excluded} cameras had to be excluded to reach these thresholds.')
    cam_excluded_count = {i: v for i, v in zip(cam_names, cam_excluded_count.values())}
    str_cam_excluded_count = ''
    for i, (k, v) in enumerate(cam_excluded_count.items()):
        if i ==0:
             str_cam_excluded_count += f'Camera {k} was excluded {int(np.round(v*100))}% of the time, '
        elif i == len(cam_excluded_count)-1:
            str_cam_excluded_count += f'and Camera {k}: {int(np.round(v*100))}%.'
        else:
            str_cam_excluded_count += f'Camera {k}: {int(np.round(v*100))}%, '
    logging.info(str_cam_excluded_count)

    logging.info(f'\n3D coordinates are stored at {trc_path}.')

def triangulation_from_best_cameras_ver1(config, coords_2D_kpt, projection_matrices):
    '''
    Triangulates 2D keypoint coordinates, only choosing the cameras for which 
    reprojection error is under threshold.

    1. Creates subset with N cameras excluded 
    2. Tries all possible triangulations
    3. Chooses the one with smallest reprojection error
    If error too big, take off one more camera.
        If then below threshold, retain result.
        If better but still too big, take off one more camera.
    
    INPUTS:
    - a Config.toml file
    - coords_2D_kpt: 
    - projection_matrices: list of arrays

    OUTPUTS:
    - Q: array of triangulated point (x,y,z,1.)
    - error_min: float
    - nb_cams_excluded: int
    '''
    
    # Read config
    error_threshold_triangulation = config.get('triangulation').get('reproj_error_threshold_triangulation')
    min_cameras_for_triangulation = config.get('triangulation').get('min_cameras_for_triangulation')

    # Initialize
    x_files, y_files, likelihood_files = coords_2D_kpt
    n_cams = len(x_files)
    error_min = np.inf 
    nb_cams_off = 0 # cameras will be taken-off until the reprojection error is under threshold
    
    while error_min > error_threshold_triangulation and n_cams - nb_cams_off >= min_cameras_for_triangulation:
        # Create subsets with "nb_cams_off" cameras excluded
        id_cams_off = np.array(list(it.combinations(range(n_cams), nb_cams_off)))##All comnination of exclude cam num
        projection_matrices_filt = [projection_matrices]*len(id_cams_off)
        
        x_files_filt = np.vstack([list(x_files).copy()]*len(id_cams_off))
        y_files_filt = np.vstack([y_files.copy()]*len(id_cams_off))
        likelihood_files_filt = np.vstack([likelihood_files.copy()]*len(id_cams_off))
        
        if nb_cams_off > 0:
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
        
        for config_id in range(len(x_calc_filt)):
            q_file = [(x_files_filt[config_id][i], y_files_filt[config_id][i]) for i in range(len(x_files_filt[config_id]))]
            q_calc = [(x_calc_filt[config_id][i], y_calc_filt[config_id][i]) for i in range(len(x_calc_filt[config_id]))]
            #import pdb
            #pdb.set_trace()
            error.append( np.mean( [euclidean_distance(q_file[i], q_calc[i]) for i in range(len(q_file))] ) )
            
        # Choosing best triangulation (with min reprojection error)
        #import ipdb; ipdb.set_trace()
        error_min = min(error)
        best_cams = np.argmin(error)
        nb_cams_excluded = nb_cams_excluded_filt[best_cams]
        Q = Q_filt[best_cams][:-1]

        # idxs = np.argsort(error)
        # Q = Q_filt[idxs[:3]].mean(axis=0)
        
        nb_cams_off += 1
    
    # Index of excluded cams for this keypoint
    id_excluded_cams = id_cams_off[best_cams]
    
    # If triangulation not successful, error = 0,  and 3D coordinates as missing values
    # if error_min > error_threshold_triangulation:
    #     error_min = np.nan
    #     # Q = np.array([0.,0.,0.])
    #     Q = np.array([np.nan, np.nan, np.nan])
        
    return Q, error_min, nb_cams_excluded, id_excluded_cams


def triangulation_from_best_cameras_verNTK1(config, coords_2D_kpt, projection_matrices):
    '''
    Triangulates 2D keypoint coordinates, only choosing the cameras for which 
    reprojection error is under threshold.

    1. Creates subset with N cameras excluded 
    2. Tries all possible triangulations
    3. Chooses the one with smallest reprojection error
    If error too big, take off one more camera.
        If then below threshold, retain result.
        If better but still too big, take off one more camera.
    
    INPUTS:
    - a Config.toml file
    - coords_2D_kpt: 
    - projection_matrices: list of arrays

    OUTPUTS:
    - Q: array of triangulated point (x,y,z,1.)
    - error_min: float
    - nb_cams_excluded: int
    '''
    
    # Read config
    error_threshold_triangulation = config.get('triangulation').get('reproj_error_threshold_triangulation')
    min_cameras_for_triangulation = config.get('triangulation').get('min_cameras_for_triangulation')

    # Initialize
    x_files, y_files, likelihood_files = coords_2D_kpt
    n_cams = len(x_files)
    error_min = np.inf 
    nb_cams_off = 0 # cameras will be taken-off until the reprojection error is under threshold
    
    while error_min > error_threshold_triangulation and n_cams - nb_cams_off >= min_cameras_for_triangulation:
        # Create subsets with "nb_cams_off" cameras excluded
        id_cams_off = np.array(list(it.combinations(range(n_cams), nb_cams_off)))##All comnination of exclude cam num
        projection_matrices_filt = [projection_matrices]*len(id_cams_off)
        
        x_files_filt = np.vstack([list(x_files).copy()]*len(id_cams_off))
        y_files_filt = np.vstack([y_files.copy()]*len(id_cams_off))
        likelihood_files_filt = np.vstack([likelihood_files.copy()]*len(id_cams_off))
        
        if nb_cams_off > 0:
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
        
        for config_id in range(len(x_calc_filt)):
            q_file = [(x_files_filt[config_id][i], y_files_filt[config_id][i]) for i in range(len(x_files_filt[config_id]))]
            q_calc = [(x_calc_filt[config_id][i], y_calc_filt[config_id][i]) for i in range(len(x_calc_filt[config_id]))]
            #import pdb
            #pdb.set_trace()
            error.append( np.mean( [euclidean_distance(q_file[i], q_calc[i]) for i in range(len(q_file))] ) )
            
        # Choosing best triangulation (with min reprojection error)
        #import ipdb; ipdb.set_trace()
        error_min = min(error)
        best_cams = np.argmin(error)
        nb_cams_excluded = nb_cams_excluded_filt[best_cams]
        Q = Q_filt[best_cams][:-1]

        # idxs = np.argsort(error)
        # Q = Q_filt[idxs[:3]].mean(axis=0)
         
        # Choosing best triangulation (with min reprojection error)
        error_min = min(error)
        best_cams = np.argmin(error)
        nb_cams_excluded = nb_cams_excluded_filt[best_cams]
        
        # Q = Q_filt[best_cams][:-1]
        top_k_cam = np.argsort(error)[:2]
        Q = np.stack(Q_filt)[top_k_cam, :-1].mean(axis=0)
        
        nb_cams_off += 1
        #import pdb;pdb.set_trace()
    
    # Index of excluded cams for this keypoint
    id_excluded_cams = id_cams_off[best_cams]
    
    # If triangulation not successful, error = 0,  and 3D coordinates as missing values
    # if error_min > error_threshold_triangulation:
    #     error_min = np.nan
    #     # Q = np.array([0.,0.,0.])
    #     Q = np.array([np.nan, np.nan, np.nan])
        
    return Q, error_min, nb_cams_excluded, id_excluded_cams
def triangulation_from_best_cameras(config, coords_2D_kpt, projection_matrices):
    '''
    Triangulates 2D keypoint coordinates, only choosing the cameras for which 
    reprojection error is under threshold.

    1. Creates subset with N cameras excluded 
    2. Tries all possible triangulations
    3. Chooses the one with smallest reprojection error
    If error too big, take off one more camera.
        If then below threshold, retain result.
        If better but still too big, take off one more camera.
    
    INPUTS:
    - a Config.toml file
    - coords_2D_kpt: 
    - projection_matrices: list of arrays

    OUTPUTS:
    - Q: array of triangulated point (x,y,z,1.)
    - error_min: float
    - nb_cams_excluded: int
    '''
    
    # Read config
    error_threshold_triangulation = config.get('triangulation').get('reproj_error_threshold_triangulation')
    min_cameras_for_triangulation = config.get('triangulation').get('min_cameras_for_triangulation')

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
    calib_cam1 = calib[list(calib.keys())[0]]
    #import pdb
    #pdb.set_trace()
    #######End
    nb_cams_off = 0
    

    error = []
    id_excluded_cams_temp = []
    error_min_temp = []
    nb_cams_excluded_temp = []
    Q_temp =[]
    exclude_record =[]
    error_record=[]
    error_record1 =[]
    count_all_com = 0
    
    #################error_min > error_threshold_triangulation and 
    while n_cams - nb_cams_off >= min_cameras_for_triangulation:
        # Create subsets with "nb_cams_off" cameras excluded
        id_cams_off = np.array(list(it.combinations(range(n_cams), nb_cams_off)))##All comnination of exclude cam num
        #import pdb;pdb.set_trace()
        
        
        projection_matrices_filt = [projection_matrices]*len(id_cams_off)
        
        x_files_filt = np.vstack([list(x_files).copy()]*len(id_cams_off))
        y_files_filt = np.vstack([y_files.copy()]*len(id_cams_off))
        likelihood_files_filt = np.vstack([likelihood_files.copy()]*len(id_cams_off))
        
        #import pdb
        #pdb.set_trace()
        if nb_cams_off > 0:
            for i in range(len(id_cams_off)):
                x_files_filt[i][id_cams_off[i]] = np.nan
                y_files_filt[i][id_cams_off[i]] = np.nan
                likelihood_files_filt[i][id_cams_off[i]] = np.nan
        nb_cams_excluded_filt = [np.count_nonzero(np.nan_to_num(x)==0) for x in likelihood_files_filt] # count nans and zeros
        
        
        #import pdb
        #pdb.set_trace()
        projection_matrices_filt = [ [ p[i] for i in range(n_cams) if not np.isnan(x_files_filt[j][i]) ] for j, p in enumerate(projection_matrices_filt) ]
        #import pdb;pdb.set_trace()
        x_files_filt = [ [ xx for ii, xx in enumerate(x) if not np.isnan(xx) ] for x in x_files_filt ]
        y_files_filt = [ [ xx for ii, xx in enumerate(x) if not np.isnan(xx) ] for x in y_files_filt ]
        likelihood_files_filt = [ [ xx for ii, xx in enumerate(x) if not np.isnan(xx) ] for x in likelihood_files_filt ]            
        #import pdb
        #pdb.set_trace()
        # Triangulate 2D points

        Q_filt = [weighted_triangulation(projection_matrices_filt[i], x_files_filt[i], y_files_filt[i], likelihood_files_filt[i]) for i in range(len(id_cams_off))]
        
        #import pdb;pdb.set_trace()
        # Reprojection
        coords_2D_kpt_calc_filt = [reprojection(projection_matrices_filt[i], Q_filt[i])  for i in range(len(id_cams_off))]
        coords_2D_kpt_calc_filt = np.array(coords_2D_kpt_calc_filt, dtype=object)
        x_calc_filt = coords_2D_kpt_calc_filt[:,0]
        y_calc_filt = coords_2D_kpt_calc_filt[:,1]
        #import pdb
        #pdb.set_trace()
        # Reprojection error
        error = []
        error1 = []

        for config_id in range(len(x_calc_filt)):
            q_file = [(x_files_filt[config_id][i], y_files_filt[config_id][i]) for i in range(len(x_files_filt[config_id]))]
            q_calc = [(x_calc_filt[config_id][i], y_calc_filt[config_id][i]) for i in range(len(x_calc_filt[config_id]))]
        
            #import pdb
            #pdb.set_trace()
            

            cam_used = np.array(range(n_cams))

            if nb_cams_off>0:
                cam_used = np.delete(cam_used,id_cams_off[config_id])
           
            exclude_record.append(id_cams_off[config_id])
            ######mean with dist.
            #error.append( np.mean( [euclidean_dist_with_multiplication(q_file[i], q_calc[i],Q_filt[0][0:3],calib[list(calib.keys())[cam_used[i]]]) for i in range(len(q_file))] ) )
            #error_record.append( np.mean( [euclidean_dist_with_multiplication(q_file[i], q_calc[i],Q_filt[0][0:3],calib[list(calib.keys())[cam_used[i]]]) for i in range(len(q_file))] ) )
            ######mean without dist.
            #error.append( np.mean( [euclidean_distance(q_file[i], q_calc[i]) for i in range(len(q_file))] ) )
            #error_record.append( np.mean( [euclidean_distance(q_file[i], q_calc[i]) for i in range(len(q_file))] ) )
            ######max with dist.
            error.append( np.max( [euclidean_dist_with_multiplication(q_file[i], q_calc[i],Q_filt[0][0:3],calib[list(calib.keys())[cam_used[i]]]) for i in range(len(q_file))] ) )
            error_record.append( np.max( [euclidean_dist_with_multiplication(q_file[i], q_calc[i],Q_filt[0][0:3],calib[list(calib.keys())[cam_used[i]]]) for i in range(len(q_file))] ) )
            #print([euclidean_dist_with_multiplication(q_file[i], q_calc[i],Q_filt[0][0:3],calib[list(calib.keys())[cam_used[i]]]) for i in range(len(q_file))] )
            ######max without dist.
            error1.append( np.max( [euclidean_distance(q_file[i], q_calc[i]) for i in range(len(q_file))] ) )
            error_record1.append( np.max( [euclidean_distance(q_file[i], q_calc[i]) for i in range(len(q_file))] ) )
            #print([euclidean_distance(q_file[i], q_calc[i]) for i in range(len(q_file))])
            #import pdb;pdb.set_trace()
        # Choosing best triangulation (with min reprojection error)
        #import pdb
        #pdb.set_trace()
        error_min = min(error)
        best_cams = np.argmin(error)
        
        nb_cams_excluded = nb_cams_excluded_filt[best_cams]
        #import pdb;pdb.set_trace()
        Q = Q_filt[best_cams][:-1]

        # idxs = np.argsort(error)
        # Q = Q_filt[idxs[:3]].mean(axis=0)
        
        nb_cams_off += 1
        
        id_excluded_cams_temp.append(id_cams_off[best_cams])
        error_min_temp.append(error_min)
        nb_cams_excluded_temp.append(nb_cams_excluded)   
        count_all_com =count_all_com+1
        Q_temp.append(Q)
        

    
    # Index of excluded cams for this keypoint
    error_min_final = min(error_min_temp)
    best_cams_final = np.argmin(error_min_temp)
    nb_cams_excluded_final = nb_cams_excluded_temp[best_cams_final]
    id_excluded_cams_final = id_excluded_cams_temp[best_cams_final]
    Q_final = Q_temp[best_cams_final]
    
    dist_camera2point =np.array([camera2point_dist(Q_final,calib[list(calib.keys())[camera]]) for camera in range(4)])
    
   



    # If triangulation not successful, error = 0,  and 3D coordinates as missing values
    if error_min > error_threshold_triangulation:
        error_min = np.nan
        Q = np.array([0.,0.,0.])
        Q = np.array([np.nan, np.nan, np.nan])
    
    #import pdb;pdb.set_trace()    
    return Q_final, error_min_final, nb_cams_excluded_final, id_excluded_cams_final,exclude_record,error_record,dist_camera2point
def triangulation_from_best_cameras_ver_dynamic(config, coords_2D_kpt, projection_matrices,body_name):
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
    list_dynamic_mincam_ver6=  {'Hip':4,'RHip':4,'RKnee':3,'RAnkle':4,'RBigToe':4,'RSmallToe':4,'RHeel':4,'LHip':4,'LKnee':3,'LAnkle':4,'LBigToe':4,'LSmallToe':4,'LHeel':4,'Neck':2,'Head':3,'Nose':3,'RShoulder':3,'RElbow':3,'RWrist':3,'LShoulder':3,'LElbow':3,'LWrist':3}

    list_dynamic_mincam_ver5=  {'Hip':4,'RHip':4,'RKnee':3,'RAnkle':4,'RBigToe':3,'RSmallToe':3,'RHeel':3,'LHip':4,'LKnee':3,'LAnkle':4,'LBigToe':3,'LSmallToe':3,'LHeel':3,'Neck':2,'Head':3,'Nose':3,'RShoulder':3,'RElbow':3,'RWrist':3,'LShoulder':3,'LElbow':3,'LWrist':3}
    
    list_dynamic_mincam=  {'Hip':4,'RHip':4,'RKnee':3,'RAnkle':3,'RBigToe':3,'RSmallToe':3,'RHeel':3,'LHip':4,'LKnee':3,'LAnkle':3,'LBigToe':3,'LSmallToe':3,'LHeel':3,'Neck':3,'Head':2,'Nose':2,'RShoulder':3,'RElbow':3,'RWrist':3,'LShoulder':3,'LElbow':3,'LWrist':3}
    
    list_dynamic_mincam_ver3=  {'Hip':4,'RHip':4,'RKnee':3,'RAnkle':2,'RBigToe':2,'RSmallToe':2,'RHeel':3,'LHip':4,'LKnee':3,'LAnkle':3,'LBigToe':2,'LSmallToe':2,'LHeel':3,'Neck':2,'Head':3,'Nose':3,'RShoulder':3,'RElbow':3,'RWrist':3,'LShoulder':3,'LElbow':3,'LWrist':3}
    list_dynamic_mincam_ver2 = {'Hip':4,'RHip':4,'RKnee':3,'RAnkle':2,'RBigToe':2,'RSmallToe':2,'RHeel':3,'LHip':4,'LKnee':3,'LAnkle':3,'LBigToe':2,'LSmallToe':2,'LHeel':2,'Neck':2,'Head':3,'Nose':3,'RShoulder':3,'RElbow':3,'RWrist':2,'LShoulder':3,'LElbow':3,'LWrist':2}
    list_dynamic_mincam_ver1 = {'Hip':4,'RHip':4,'RKnee':3,'RAnkle':2,'RBigToe':2,'RSmallToe':2,'RHeel':2,'LHip':4,'LKnee':3,'LAnkle':2,'LBigToe':2,'LSmallToe':2,'LHeel':2,'Neck':2,'Head':3,'Nose':3,'RShoulder':3,'RElbow':2,'RWrist':2,'LShoulder':3,'LElbow':2,'LWrist':2}
    
    # Read config
    error_threshold_triangulation = config.get('triangulation').get('reproj_error_threshold_triangulation')
    min_cameras_for_triangulation = config.get('triangulation').get('min_cameras_for_triangulation')
    min_cameras_for_triangulation = list_dynamic_mincam[body_name]
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

    #import pdb
    #pdb.set_trace()
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
    error_record1 =[]
    count_all_com = 0
    first_tri = 1
    
    #################error_min > error_threshold_triangulation and 
    while n_cams - nb_cams_off-nb_cam_initially_off >= min_cameras_for_triangulation or first_tri == 1:
        # Create subsets with "nb_cams_off" cameras excluded
        id_cams_off = np.array(list(it.combinations(left_combination, nb_cams_off)))##All comnination of exclude cam num
        id_cam_off_exclusion = id_cams_off 
        #import pdb;pdb.set_trace()

        ## combine initial and exclude 
        if id_cams_off.size == 0 and ini_del ==True:
            id_cams_off = cam_initially_off
        else:
            id_cams_off = np.append(id_cams_off,np.repeat(np.array(cam_initially_off),np.shape(id_cams_off)[0] ,axis = 0),axis=1)
        
        projection_matrices_filt = [projection_matrices]*len(id_cams_off)
        x_files_filt = np.vstack([list(x_files).copy()]*len(id_cams_off))
        y_files_filt = np.vstack([y_files.copy()]*len(id_cams_off))
        likelihood_files_filt = np.vstack([likelihood_files.copy()]*len(id_cams_off))
        
        #import pdb
        #pdb.set_trace()
        if nb_cams_off+nb_cam_initially_off > 0:
            for i in range(len(id_cams_off)):
                x_files_filt[i][id_cams_off[i]] = np.nan
                y_files_filt[i][id_cams_off[i]] = np.nan
                likelihood_files_filt[i][id_cams_off[i]] = np.nan
        nb_cams_excluded_filt = [np.count_nonzero(np.nan_to_num(x)==0) for x in likelihood_files_filt] # count nans and zeros
        
        
        #import pdb
        #pdb.set_trace()
        projection_matrices_filt = [ [ p[i] for i in range(n_cams) if not np.isnan(x_files_filt[j][i]) ] for j, p in enumerate(projection_matrices_filt) ]
        #import pdb;pdb.set_trace()
        x_files_filt = [ [ xx for ii, xx in enumerate(x) if not np.isnan(xx) ] for x in x_files_filt ]
        y_files_filt = [ [ xx for ii, xx in enumerate(x) if not np.isnan(xx) ] for x in y_files_filt ]
        likelihood_files_filt = [ [ xx for ii, xx in enumerate(x) if not np.isnan(xx) ] for x in likelihood_files_filt ]            
        #import pdb
        #pdb.set_trace()
        # Triangulate 2D points

        Q_filt = [weighted_triangulation(projection_matrices_filt[i], x_files_filt[i], y_files_filt[i], likelihood_files_filt[i]) for i in range(len(id_cams_off))]
        
        #import pdb;pdb.set_trace()
        # Reprojection
        coords_2D_kpt_calc_filt = [reprojection(projection_matrices_filt[i], Q_filt[i])  for i in range(len(id_cams_off))]
        coords_2D_kpt_calc_filt = np.array(coords_2D_kpt_calc_filt, dtype=object)
        x_calc_filt = coords_2D_kpt_calc_filt[:,0]
        y_calc_filt = coords_2D_kpt_calc_filt[:,1]
        #import pdb
        #pdb.set_trace()
        # Reprojection error
        error = []
        error1 = []

        for config_id in range(len(x_calc_filt)):
            q_file = [(x_files_filt[config_id][i], y_files_filt[config_id][i]) for i in range(len(x_files_filt[config_id]))]
            q_calc = [(x_calc_filt[config_id][i], y_calc_filt[config_id][i]) for i in range(len(x_calc_filt[config_id]))]
        
            #import pdb
            #pdb.set_trace()
            

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


            #import pdb;pdb.set_trace()
        # Choosing best triangulation (with min reprojection error)
        #import pdb
        #pdb.set_trace()
        error_min = min(error)
        best_cams = np.argmin(error)
        
        nb_cams_excluded = nb_cams_excluded_filt[best_cams]
        #import pdb;pdb.set_trace()
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

    
    dist_camera2point =np.array([camera2point_dist(Q_final,calib[list(calib.keys())[camera]]) for camera in range(4)])
    
   



    # If triangulation not successful, error = 0,  and 3D coordinates as missing values
    if error_min > error_threshold_triangulation:
        error_min = np.nan
        Q = np.array([0.,0.,0.])
        Q = np.array([np.nan, np.nan, np.nan])
    
    #import pdb;pdb.set_trace() 

    return Q_final, error_min_final, nb_cams_excluded_final, id_excluded_cams_final,exclude_record,error_record,dist_camera2point,strongness_of_exclusion
                               

def extract_files_frame_f(json_tracked_files_f, keypoints_ids):
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

    n_cams = len(json_tracked_files_f)
    
    x_files, y_files, likelihood_files = [], [], []
    for cam_nb in range(n_cams):
        x_files_cam, y_files_cam, likelihood_files_cam = [], [], []
        with open(json_tracked_files_f[cam_nb], 'r') as json_f:
            js = json.load(json_f)
            for keypoint_id in keypoints_ids:
                try:
                    x_files_cam.append( js['people'][0]['pose_keypoints_2d'][keypoint_id*3] )
                    y_files_cam.append( js['people'][0]['pose_keypoints_2d'][keypoint_id*3+1] )
                    likelihood_files_cam.append( js['people'][0]['pose_keypoints_2d'][keypoint_id*3+2] )
                except:
                    x_files_cam.append( np.nan )
                    y_files_cam.append( np.nan )
                    likelihood_files_cam.append( np.nan )

        x_files.append(x_files_cam)
        y_files.append(y_files_cam)
        likelihood_files.append(likelihood_files_cam)
        
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
    - json files for each camera with only one person of interest
    - a Config.toml file
    - a skeleton model
    
    OUTPUTS: 
    - a .trc file with 3D coordinates in Y-up system coordinates 
    '''
    
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
    #import pdb;pdb.set_trace()
    # Projection matrix from toml calibration file
    P = computeP(calib_file)
    
    # Retrieve keypoints from model
    model = eval(pose_model)
    keypoints_ids = [node.id for _, _, node in RenderTree(model) if node.id!=None]
    keypoints_names = [node.name for _, _, node in RenderTree(model) if node.id!=None]
    keypoints_idx = list(range(len(keypoints_ids)))
    keypoints_nb = len(keypoints_ids)
    
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
    
    n_cams = len(json_dirs_names)
    Q_tot, error_tot, nb_cams_excluded_tot,id_excluded_cams_tot,exclude_record_tot,error_record_tot,cam_dist_tot,id_excluded_cams_record_tot,strongness_exclusion_tot  = [], [], [], [],[],[],[],[],[]
    for f in tqdm(range(*f_range)):
        # Get x,y,likelihood values from files
        json_tracked_files_f = [json_tracked_files[c][f] for c in range(n_cams)]
        x_files, y_files, likelihood_files = extract_files_frame_f(json_tracked_files_f, keypoints_ids)
        
        # Replace likelihood by 0 if under likelihood_threshold
        with np.errstate(invalid='ignore'):
            likelihood_files[likelihood_files<likelihood_threshold] = 0.
        
        Q, error, nb_cams_excluded, id_excluded_cams,exclude_record,error_record,cam_dist,strongness_exclusion = [], [], [], [],[],[],[],[]
        for keypoint_idx in keypoints_idx:
            #import pdb;pdb.set_trace()
        # Triangulate cameras with min reprojection error
            coords_2D_kpt = ( x_files[:, keypoint_idx], y_files[:, keypoint_idx], likelihood_files[:, keypoint_idx] )
            id_excluded_cams_kpt,exclude_record_kpt,error_record_kpt,cam_dist_kpt = -1,-1,-1,-1
            #Q_kpt, error_kpt, nb_cams_excluded_kpt, id_excluded_cams_kpt,exclude_record_kpt,error_record_kpt,cam_dist_kpt = triangulation_from_best_cameras(config, coords_2D_kpt, P)
            Q_kpt, error_kpt, nb_cams_excluded_kpt, id_excluded_cams_kpt,exclude_record_kpt,error_record_kpt,cam_dist_kpt,strongness_of_exclusion_kpt = triangulation_from_best_cameras_ver_dynamic(config, coords_2D_kpt, P,keypoints_names[keypoint_idx])
            #Q_kpt, error_kpt, nb_cams_excluded_kpt, id_excluded_cams_kpt = triangulation_from_best_cameras_ver1(config, coords_2D_kpt, P)
            #Q_kpt, error_kpt, nb_cams_excluded_kpt, id_excluded_cams_kpt = triangulation_from_best_cameras_verNTK1(config, coords_2D_kpt, P)
            #if f==90:
                #import pdb;pdb.set_trace()
            Q.append(Q_kpt)
            error.append(error_kpt)
            nb_cams_excluded.append(nb_cams_excluded_kpt)
            id_excluded_cams.append(id_excluded_cams_kpt)
            exclude_record.append(exclude_record_kpt)
            error_record.append(error_record_kpt)
            cam_dist.append(cam_dist_kpt)
            strongness_exclusion.append(strongness_of_exclusion_kpt)
        # Add triangulated points, errors and excluded cameras to pandas dataframes
        Q_tot.append(np.concatenate(Q))
        error_tot.append(error)
        nb_cams_excluded_tot.append(nb_cams_excluded)
        id_excluded_cams_record_tot.append(id_excluded_cams)
        id_excluded_cams = [item for sublist in id_excluded_cams for item in sublist]
        
        id_excluded_cams_tot.append(id_excluded_cams)
        exclude_record_tot.append(exclude_record)
        error_record_tot.append(error_record)
        cam_dist_tot.append(cam_dist)
        strongness_exclusion_tot.append(strongness_exclusion)
 
            
    Q_tot = pd.DataFrame(Q_tot)
    error_tot = pd.DataFrame(error_tot)
    nb_cams_excluded_tot = pd.DataFrame(nb_cams_excluded_tot)
    
    id_excluded_cams_tot = [item for sublist in id_excluded_cams_tot for item in sublist]
    cam_excluded_count = dict(Counter(id_excluded_cams_tot))
    cam_excluded_count.update((x, y/keypoints_nb/frames_nb) for x, y in cam_excluded_count.items())
    
    error_tot['mean'] = error_tot.mean(axis = 1)
    nb_cams_excluded_tot['mean'] = nb_cams_excluded_tot.mean(axis = 1)

    # Optionally, for each keypoint, show indices of frames that should be interpolated
    if show_interp_indices:
        zero_nan_frames = np.where( Q_tot.iloc[:,::3].T.eq(0) | ~np.isfinite(Q_tot.iloc[:,::3].T) )
        zero_nan_frames_per_kpt = [zero_nan_frames[1][np.where(zero_nan_frames[0]==k)[0]] for k in range(keypoints_nb)]
        gaps = [np.where(np.diff(zero_nan_frames_per_kpt[k]) > 1)[0] + 1 for k in range(keypoints_nb)]
        sequences = [np.split(zero_nan_frames_per_kpt[k], gaps[k]) for k in range(keypoints_nb)]
        interp_frames = [[f'{seq[0]}:{seq[-1]+1}' for seq in seq_kpt if len(seq)<=interp_gap_smaller_than and len(seq)>0] for seq_kpt in sequences]
        non_interp_frames = [[f'{seq[0]}:{seq[-1]+1}' for seq in seq_kpt if len(seq)>interp_gap_smaller_than] for seq_kpt in sequences]
    else:
        interp_frames = None
        non_interp_frames = []

    # Interpolate missing values
    if interpolation_kind != 'none':
        #import ipdb; ipdb.set_trace()
        Q_tot = Q_tot.apply(interpolate_zeros_nans, axis=0, args = [interp_gap_smaller_than, interpolation_kind])
    Q_tot.replace(np.nan, 0, inplace=True)

    from scipy.io import savemat
    mdic = {'exclude':exclude_record_tot,'error':error_record_tot,'keypoints_name':keypoints_names,'cam_dist':cam_dist_tot,'cam_choose':id_excluded_cams_record_tot,'strongness_of_exclusion':strongness_exclusion_tot}
    savemat(os.path.join(project_dir,'rpj.mat'), mdic)
    
    
    np.savez(os.path.join(project_dir,'User','reprojection_record.npz'),exclude=exclude_record_tot,error=error_record_tot,keypoints_name=keypoints_names,cam_dist=cam_dist_tot,cam_choose=id_excluded_cams_record_tot,strongness_of_exclusion =strongness_exclusion_tot)
    np.shape(error_record_tot)
    #pdb.set_trace()
    # Create TRC file
    trc_path = make_trc(config, Q_tot, keypoints_names, f_range)
    
    # Recap message
    recap_triangulate(config, error_tot, nb_cams_excluded_tot, keypoints_names, cam_excluded_count, interp_frames, non_interp_frames, trc_path)
