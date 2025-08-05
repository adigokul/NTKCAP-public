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
from scipy.io import savemat
import json
import itertools as it
import pandas as pd
import toml
from tqdm import tqdm
from scipy import interpolate
from collections import Counter
import logging
import cv2
from Pose2Sim.common import computeP, weighted_triangulation, reprojection, \
    euclidean_distance, natural_sort, euclidean_dist_with_multiplication,camera2point_dist, computemap,undistort_points1
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
                               
def triangulation_from_best_cameras_ThreeCamRowv(config, coords_2D_kpt, projection_matrices,body_name):
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
    
    
    list_dynamic_mincam = {'Hip':4,'RHip':4,'RKnee':3,'RAnkle':3,'RBigToe':3,'RSmallToe':3,'RHeel':3,'LHip':4,'LKnee':3,'LAnkle':3,'LBigToe':3,'LSmallToe':3,'LHeel':3,'Neck':3,'Head':2,'Nose':2,'RShoulder':3,'RElbow':3,'RWrist':3,'LShoulder':3,'LElbow':3,'LWrist':3}
    list_TR_realdist = {'Hip':np.inf,'RHip':np.inf,'RKnee':np.inf,'RAnkle':np.inf,'RBigToe':np.inf,'RSmallToe':np.inf,'RHeel':np.inf,'LHip':np.inf,'LKnee':np.inf,'LAnkle':np.inf,'LBigToe':np.inf,'LSmallToe':np.inf,'LHeel':np.inf,'Neck':np.inf,'Head':np.inf,'Nose':np.inf,'RShoulder':np.inf,'RElbow':np.inf,'RWrist':np.inf,'LShoulder':np.inf,'LElbow':np.inf,'LWrist':np.inf}
    list_dynamic_mincam3 = {'Hip':3,'RHip':3,'RKnee':3,'RAnkle':3,'RBigToe':3,'RSmallToe':3,'RHeel':3,'LHip':3,'LKnee':3,'LAnkle':3,'LBigToe':3,'LSmallToe':3,'LHeel':3,'Neck':3,'Head':3,'Nose':3,'RShoulder':3,'RElbow':3,'RWrist':3,'LShoulder':3,'LElbow':3,'LWrist':3}
    
    # Read config
    min_cameras_for_triangulation = config.get('triangulation').get('min_cameras_for_triangulation')
    min_cameras_for_triangulation = list_dynamic_mincam3[body_name]
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
    all_error = []
    all_error4 = []
    cam_correspond = []
    exclude_record =[]
    error_record=[]
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
            cam_used4 = np.array(range(n_cams))
            cam_used4 = np.delete(cam_used4,id_cams_off[0])
        


        Q_filt = [weighted_triangulation(projection_matrices_filt[i], x_files_filt[i], y_files_filt[i],likelihood_files_filt[i]) for i in range(len(id_cams_off))]

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
        import pdb; pdb.set_trace()
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
                try:
                    error.append( np.max( [euclidean_dist_with_multiplication(q_file[i], q_calc[i],Q_filt[config_id][0:3],calib[list(calib.keys())[cam_used4[i]]]) for i in range(len(q_file))] ) )
                    error_record.append( np.max( [euclidean_dist_with_multiplication(q_file[i], q_calc[i],Q_filt[config_id][0:3],calib[list(calib.keys())[cam_used4[i]]]) for i in range(len(q_file))] ))
                    all_error.append([euclidean_dist_with_multiplication(q_file[i], q_calc[i],Q_filt[config_id][0:3],calib[list(calib.keys())[cam_used4[i]]]) for i in range(len(q_file))])
                    all_error4.append([euclidean_dist_with_multiplication(q_file4[i], q_calc4[i],Q_filt[config_id][0:3],calib[list(calib.keys())[cam_used4[i]]]) for i in range(len(q_file4))])
                    import pdb; pdb.set_trace()
                except:
                    import pdb;pdb.set_trace()
                cam_correspond.append(cam_used)
                Q_all.append(Q_filt[config_id]) 
                max_index, max_value = max(enumerate(all_error4[-1]), key=lambda x: x[1])
                voting.append([len(cam_used),max_index,max_value])

            # max without dist.
                error1.append( np.max( [euclidean_distance(q_file[i], q_calc[i]) for i in range(len(q_file))] ) )
            else:
                error.append(float('inf'))
                error1.append(float('inf'))

        import pdb; pdb.set_trace()
        nb_cams_off += 1
        count_all_com =count_all_com+1
        first_tri = 0   
    import pdb; pdb.set_trace()
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
    import pdb; pdb.set_trace()
    if len(final_comb)<4:
        #strongness_of_exclusion =np.mean(all_error[0])-np.mean(all_error[index])
        strongness_of_exclusion =np.mean(all_error[0])-np.mean(all_error[index])
    else:
        strongness_of_exclusion = 0

    import pdb; pdb.set_trace()
    
    dist_camera2point =np.array([camera2point_dist(Q_final,calib[list(calib.keys())[camera]]) for camera in range(4)])

    all_error_noweight=[]
    residuals_SVD=[]
    return Q_final, error_min_final, nb_cams_excluded_final, id_excluded_cams_final, exclude_record,error_record,dist_camera2point,strongness_of_exclusion,all_error,all_error_noweight,cam_correspond,residuals_SVD
                               
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

def undistort_points(x_map,y_map,x_coor,y_coor):
    print(undistort_points)
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
    # Projection matrix from toml calibration file
    P = computeP(calib_file)
    mappingx,mappingy =computemap(calib_file)
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
    all_error_tot,all_error_noweight_tot,cam_correspond_tot,residual_SVD_tot=[] ,[],[],[]
    for f in tqdm(range(*f_range)):
        # Get x,y,likelihood values from files
        json_tracked_files_f = [json_tracked_files[c][f] for c in range(n_cams)]
        x_files, y_files, likelihood_files = extract_files_frame_f(json_tracked_files_f, keypoints_ids)
        
        # Replace likelihood by 0 if under likelihood_threshold
        with np.errstate(invalid='ignore'):
            likelihood_files[likelihood_files<likelihood_threshold] = 0.
        
        Q, error, nb_cams_excluded, id_excluded_cams,exclude_record,error_record,cam_dist,strongness_exclusion = [], [], [], [],[],[],[],[]
        all_error,all_error_noweight,cam_correspond,residual_SVD=[] ,[] ,[],[]
        for keypoint_idx in keypoints_idx:
        # Triangulate cameras with min reprojection error
            coords_2D_kpt = ( x_files[:, keypoint_idx], y_files[:, keypoint_idx], likelihood_files[:, keypoint_idx] )
            coords_2D_kpt =undistort_points1(mappingx,mappingy,coords_2D_kpt)
            id_excluded_cams_kpt,exclude_record_kpt,error_record_kpt,cam_dist_kpt = -1,-1,-1,-1
            all_error_noweight_kpt = []
            residual_SVD_kpt = []
            
            Q_kpt, error_kpt, nb_cams_excluded_kpt, id_excluded_cams_kpt,exclude_record_kpt,error_record_kpt,cam_dist_kpt,strongness_of_exclusion_kpt,all_error_kpt,all_error_noweight_kpt,cam_correspond_kpt,residual_SVD_kpt = triangulation_from_best_cameras_ThreeCamRowv(config, coords_2D_kpt, P,keypoints_names[keypoint_idx])
            
            Q.append(Q_kpt)
            error.append(error_kpt)
            nb_cams_excluded.append(nb_cams_excluded_kpt)
            id_excluded_cams.append(id_excluded_cams_kpt)
            exclude_record.append(exclude_record_kpt)
            error_record.append(error_record_kpt)
            cam_dist.append(cam_dist_kpt)
            strongness_exclusion.append(strongness_of_exclusion_kpt)
            all_error.append(all_error_kpt)
            all_error_noweight.append(all_error_noweight_kpt)
            cam_correspond.append(cam_correspond_kpt)
            residual_SVD.append(residual_SVD_kpt)
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
        all_error_tot.append(all_error)
        all_error_noweight_tot.append(all_error_noweight)
        cam_correspond_tot.append(cam_correspond)
        residual_SVD_tot.append(residual_SVD)
 
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
        Q_tot = Q_tot.apply(interpolate_zeros_nans, axis=0, args = [interp_gap_smaller_than, interpolation_kind])
    Q_tot.replace(np.nan, 0, inplace=True)
    
    mdic = {'chosen_error':error_tot,'all_error':all_error_tot,'all_error_noweight':all_error_noweight_tot,'residual_SVD':residual_SVD_tot,'corres_camlist':cam_correspond_tot,'exclude':exclude_record_tot,'error':error_record_tot,'keypoints_name':keypoints_names,'cam_dist':cam_dist_tot,'cam_choose':id_excluded_cams_record_tot,'strongness_of_exclusion':strongness_exclusion_tot}
    savemat(os.path.join(project_dir,'rpj.mat'), mdic)
    
    np.savez(os.path.join(project_dir,'User','reprojection_record.npz'),exclude=np.array(exclude_record_tot, dtype=object),error=error_record_tot,keypoints_name=keypoints_names,cam_dist=cam_dist_tot,cam_choose=id_excluded_cams_record_tot,strongness_of_exclusion =strongness_exclusion_tot)
    np.shape(error_record_tot)
    
    # Create TRC file
    trc_path = make_trc(config, Q_tot, keypoints_names, f_range)
    
    # Recap message
    recap_triangulate(config, error_tot, nb_cams_excluded_tot, keypoints_names, cam_excluded_count, interp_frames, non_interp_frames, trc_path)
