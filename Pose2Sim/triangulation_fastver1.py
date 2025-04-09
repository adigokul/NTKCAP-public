#!/usr/bin/env python
# -*- coding: utf-8 -*-
###version2

'''
   GPU version of triangulation base on ver_dynamic
    
'''


## INIT
import os
relative_temp_dir = "./temp"
os.makedirs(relative_temp_dir, exist_ok=True)
os.environ["TEMP"] = relative_temp_dir
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
import cupy as cp
try:
    from Pose2Sim.common import computeP, weighted_triangulation, reprojection, \
    euclidean_distance, natural_sort, euclidean_dist_with_multiplication, camera2point_dist,computemap,undistort_points1,find_camera_coordinate
    from Pose2Sim.skeletons import *
except:
    from common import computeP, weighted_triangulation, reprojection, \
    euclidean_distance, natural_sort, euclidean_dist_with_multiplication, camera2point_dist,computemap,undistort_points1,find_camera_coordinate
    from skeletons import *
from scipy.io import savemat

## AUTHORSHIP INFORMATION
__author__ = "David Pagnon"
__copyright__ = "Copyright 2021, Pose2Sim"
__credits__ = ["David Pagnon"]
__license__ = "BSD 3-Clause License"
__version__ = '0.4'
__maintainer__ = "David Pagnon"
__email__ = "contact@david-pagnon.com"
__status__ = "Development"
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
def create_prep(coord,f_range,n_cams,keypoints_ids,json_tracked_files):

    # Elements
    # elements = [0, 1, 2, 3]

    # Total elements
    # n = len(elements)

    # Generate combinations based on remaining elements
    # combinations_4 = [elements]  # No deletions
    # combinations_3 = [elements[:i] + elements[i+1:] for i in range(len(elements))]  # Delete one element
    # combinations_2 = [elements[:i] + elements[i+1:j] + elements[j+1:] for i in range(len(elements)) for j in range(i+1, len(elements))]  # Delete two elements
    combinations_4 = [[0, 1, 2, 3]]
    combinations_3 = [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]
    combinations_2 = [[2, 3], [1, 3], [1, 2], [0, 3], [0, 2], [0, 1]]
    # Print results
    print("4-Combinations (No deletions):", combinations_4)
    print("3-Combinations (Keep 3 elements):", combinations_3)
    print("2-Combinations (Keep 2 elements):", combinations_2)
    # Combine all combinations
    all_combinations = combinations_4 + combinations_3 + combinations_2
    prep = []
    prep4 =[]
    prep3 = []
    prep2 = []
    
    for f in tqdm(range(*f_range)):
        json_tracked_files_f = [json_tracked_files[c][f] for c in range(n_cams)]
        x_files, y_files, likelihood_files = extract_files_frame_f_fast(f,coord, keypoints_ids)
        #x_files, y_files, likelihood_files = extract_files_frame_f(json_tracked_files_f, keypoints_ids)
        #@import pdb;pdb.set_trace()
        arrays = [x_files, y_files, likelihood_files]
        # Stack and transpose
        stacked = np.stack(arrays, axis=0)  # Shape: (3, 4, 22)
        result = np.transpose(stacked, (2, 1, 0))  # Shape: (22, 4, 3)

        prep.append(result)

    prep = cp.array(prep)
    result_list = []

    # Iterate over combinations
    for comb in combinations_4:
        # Extract the slices corresponding to the combination
        combined = prep[:, :, comb, :] # combined.shape = (133, 22, 4, 3)
        result_list.append(combined) # (1, 133, 22, 4, 3)
    # Stack the combinations into a new dimension
    prep_4 = cp.stack(result_list, axis=2)  # Shape: (133, 22, 1, 4, 3)

    result_list = []
    # Iterate over combinations
    for comb in combinations_3:
        # Extract the slices corresponding to the combination
        combined = prep[:, :, comb, :]
        result_list.append(combined)
    # Stack the combinations into a new dimension
    prep_3 = cp.stack(result_list, axis=2)  # Shape: (184, 22, 6, 4, 3)

    result_list = []

    # Iterate over combinations
    for comb in combinations_2:
        # Extract the slices corresponding to the combination
        combined = prep[:, :, comb, :]
        result_list.append(combined)

    # Stack the combinations into a new dimension
    prep_2 = cp.stack(result_list, axis=2)  # Shape: (184, 22, 6, 4, 3)
    

    return prep_4,prep_3,prep_2

def bilinear_interpolate_cupy(map, x, y):
    """
    Perform bilinear interpolation for CuPy arrays.
    map: The 2D CuPy array on which interpolation is performed.
    x: x-coordinates (float) for interpolation.
    y: y-coordinates (float) for interpolation.
    """
    # Get integer coordinates surrounding the point
    x0 = cp.floor(x).astype(cp.int32)
    x1 = x0 + 1
    y0 = cp.floor(y).astype(cp.int32)
    y1 = y0 + 1

    # Ensure coordinates are within bounds
    x0 = cp.clip(x0,0, map.shape[1] - 2)
    y0 = cp.clip(y0,0, map.shape[0] - 2)
    x1 = x0 + 1
    y1 = y0 + 1

    # Use cp.take_along_axis for advanced indexing
    Ia = map[y0, x0]
    Ib = map[y0, x1]
    Ic = map[y1, x0]
    Id = map[y1, x1]

    # Interpolation weights
    wa = (x1 - x) * (y1 - y)
    wb = (x - x0) * (y1 - y)
    wc = (x1 - x) * (y - y0)
    wd = (x - x0) * (y - y0)

    # Calculate the interpolated value
    return wa * Ia + wb * Ib + wc * Ic + wd * Id

def undistort_points_cupy(mappingx, mappingy, prep_4,prep_3,prep_2):
    """
    Undistort points using CuPy, optimized for batched operations.
    mappingx: The x-mapping 2D CuPy array.
    mappingy: The y-mapping 2D CuPy array.
    prep_3: Input array of shape (155, 22, 4, 3, 3) where:
        - prep_3[..., 0] contains x-coordinates.
        - prep_3[..., 1] contains y-coordinates.
        - prep_3[..., 2] contains likelihood.
    """
    x = prep_4[:, :, :, :, 0]  # Shape: (155, 22, 4, 3)
    y = prep_4[:, :, :, :, 1]  # Shape: (155, 22, 4, 3)
    likelihood = prep_4[:, :, :, :, 2]  # Shape: (155, 22, 4, 3)

    # Perform bilinear interpolation on x and y
    x_undistorted = bilinear_interpolate_cupy(mappingx, x, y)  # Shape: (155, 22, 4, 3)
    y_undistorted = bilinear_interpolate_cupy(mappingy, x, y)  # Shape: (155, 22, 4, 3)

    # Combine results into a single array
    prep_4 = cp.stack([x_undistorted, y_undistorted, likelihood], axis=-1)  # Shape: (155, 22, 4, 3, 3)


    # Extract x, y, and likelihood values
    x = prep_3[:, :, :, :, 0]  # Shape: (155, 22, 4, 3)
    y = prep_3[:, :, :, :, 1]  # Shape: (155, 22, 4, 3)
    likelihood = prep_3[:, :, :, :, 2]  # Shape: (155, 22, 4, 3)

    # Perform bilinear interpolation on x and y
    x_undistorted = bilinear_interpolate_cupy(mappingx, x, y)  # Shape: (155, 22, 4, 3)
    y_undistorted = bilinear_interpolate_cupy(mappingy, x, y)  # Shape: (155, 22, 4, 3)

    # Combine results into a single array
    prep_3 = cp.stack([x_undistorted, y_undistorted, likelihood], axis=-1)  # Shape: (155, 22, 4, 3, 3)


    x = prep_2[:, :, :, :, 0]  # Shape: (155, 22, 4, 3)
    y = prep_2[:, :, :, :, 1]  # Shape: (155, 22, 4, 3)
    likelihood = prep_2[:, :, :, :, 2]  # Shape: (155, 22, 4, 3)

    # Perform bilinear interpolation on x and y
    x_undistorted = bilinear_interpolate_cupy(mappingx, x, y)  # Shape: (155, 22, 4, 3)
    y_undistorted = bilinear_interpolate_cupy(mappingy, x, y)  # Shape: (155, 22, 4, 3)

    # Combine results into a single array
    prep_2 = cp.stack([x_undistorted, y_undistorted, likelihood], axis=-1)  # Shape: (155, 22, 4, 3, 3)
    return prep_4,prep_3,prep_2
    
def create_A(prep_4,prep_3,prep_2,P):
    # Elements
    sh = np.shape(prep_4)
    elements = [0, 1, 2, 3]

    # Total elements
    n = len(elements)
    combinations_4 = [elements]  # No deletions
    combinations_3 = [elements[:i] + elements[i+1:] for i in range(len(elements))]  # Delete one element
    combinations_2 = [elements[:i] + elements[i+1:j] + elements[j+1:] for i in range(len(elements)) for j in range(i+1, len(elements))]  # Delete two elements
    results = []

    # Iterate over the range 4 for c
    for c in range(4):
        # Compute the first part
        part1 = (P[c][0] - prep_4[:, :, :, c, 0:1] * P[c][2]) * prep_4[:, :, :, c, 2:3]
        # Compute the second part
        part2 = (P[c][1] - prep_4[:, :, :, c, 1:2] * P[c][2]) * prep_4[:, :, :, c, 2:3]
        
        # Append the results along the last axis
        results.append(part1)
        temp = cp.array(part1)
        results.append(part2)
    # Concatenate all results along the new axis
    A4 = cp.stack(results, axis=3)  # Concatenate along the 4th dimension

    results1 = []
    final_results =[]
    # Iterate over the range 3 for c
    for i in range(4):
        results1 = []
        for c in range(3):
            camera_index = combinations_3[i]
            d = camera_index[c]
            
            # Compute the first part
            part1 = (P[d][0] - prep_3[:, :, i, c, 0:1] * P[d][2]) * prep_3[:, :, i, c, 2:3]
            # Compute the second part
            part2 = (P[d][1] - prep_3[:, :, i, c, 1:2] * P[d][2]) * prep_3[:, :, i, c, 2:3]
            #import pdb;pdb.set_trace()
            # Append the results along the last axis
            results1.append(part1)           
            results1.append(part2)
        inner_results_stacked = cp.stack(results1, axis=2) 
        final_results.append(inner_results_stacked)
        
    A3 = cp.stack(final_results, axis=2) 
    
    results1 = []
    final_results =[]
    # Iterate over the range 3 for c
    for i in range(6):
        results1 = []
        for c in range(2):
            camera_index = combinations_2[i]
            d = camera_index[c]
            
            # Compute the first part
            part1 = (P[d][0] - prep_2[:, :, i, c, 0:1] * P[d][2]) * prep_2[:, :, i, c, 2:3]
            # Compute the second part
            part2 = (P[d][1] - prep_2[:, :, i, c, 1:2] * P[d][2]) * prep_2[:, :, i, c, 2:3]
            # Append the results along the last axis
            results1.append(part1)           
            results1.append(part2)
        inner_results_stacked = cp.stack(results1, axis=2) 
        final_results.append(inner_results_stacked)
        
    A2 = cp.stack(final_results, axis=2) 
    return A4,A3,A2
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


    return Q4,Q3,Q2
def create_QBug(likelihood_threshold,prep_3like,Q3,prep_2like,Q2):
    loc = cp.where(prep_3like < likelihood_threshold)
    prep_3like[loc] = cp.inf
    # Find the index of the first non-inf element along axis 2
    non_inf_mask3 = ~cp.isinf(prep_3like)
    min_locations_nan3 = cp.argmax(non_inf_mask3, axis=2)
    batch_indices = cp.arange(Q3.shape[0])[:, None]  # Shape: (184, 1)
    time_indices = cp.arange(Q3.shape[1])[None, :]   # Shape: (1, 22)

# Use advanced indexing to extract the desired values
    selected_slices = Q3[batch_indices, time_indices, min_locations_nan3, :]
    
# Add an additional dimension to match the shape (184, 22, 1, 4)
    Q3_bug = selected_slices[:, :, cp.newaxis, :]

    loc = cp.where(prep_2like < likelihood_threshold)
    prep_2like[loc] = cp.inf
    # Find the index of the first non-inf element along axis 2
    non_inf_mask2 = ~cp.isinf(prep_2like)
    min_locations_nan2 = cp.argmax(non_inf_mask2, axis=2)
    batch_indices = cp.arange(Q2.shape[0])[:, None]  # Shape: (184, 1)
    time_indices = cp.arange(Q2.shape[1])[None, :]   # Shape: (1, 22)

# Use advanced indexing to extract the desired values
    selected_slices = Q2[batch_indices, time_indices, min_locations_nan2, :]
    
# Add an additional dimension to match the shape (184, 22, 1, 4)
    Q2_bug = selected_slices[:, :, cp.newaxis, :]
    return Q3_bug,Q2_bug
def find_real_dist_error(P,cam_coord,prep_4,Q4,prep_3,Q3,prep_2,Q2,Q3_bug,Q2_bug):
    elements = [0, 1, 2, 3]

    # Total elements
    n = len(elements)
    combinations_4 = [elements]  # No deletions
    combinations_3 = [elements[:i] + elements[i+1:] for i in range(len(elements))]  # Delete one element
    combinations_2 = [elements[:i] + elements[i+1:j] + elements[j+1:] for i in range(len(elements)) for j in range(i+1, len(elements))]  # Delete two elements       
    # Compute the first part
    
    final_resultsx =[]
    final_resultsy =[]
    final_dist = []
    for i in range(1):
        x = []
        y = []
        dist = []
        for c in range(4):
            camera_index = combinations_4[i]
            d = camera_index[c]
            P_cam_0 = P[d][0].reshape(1, -1)  # Shape: (1, 4)
            P_cam_1 = P[d][1].reshape(1, -1)  # Shape: (1, 4)
            P_cam_2 = P[d][2].reshape(1, -1)  # Shape: (1, 4)
            
            P0_dot_Q = cp.einsum('ik,btk->bt', P_cam_0, Q4[:,:,i,:])  # Shape: (250, 22, 4)
            P1_dot_Q = cp.einsum('ik,btk->bt', P_cam_1, Q4[:,:,i,:])  # Shape: (250, 22, 4)
            P2_dot_Q = cp.einsum('ik,btk->bt', P_cam_2, Q4[:,:,i,:])  # Shape: (250, 22, 4)
            x.append(P0_dot_Q / P2_dot_Q)
            y.append(P1_dot_Q / P2_dot_Q)
            dist.append(cp.sqrt(cp.sum((Q4[:, :, i, 0:3] - cam_coord[d]) ** 2, axis=-1)))

        inner_results_stackedx = cp.stack(x, axis=2) 
        final_resultsx.append(inner_results_stackedx)
        inner_results_stackedy = cp.stack(y, axis=2) 
        final_resultsy.append(inner_results_stackedy)
        final_dist.append(cp.stack(dist,axis=2))

    X = cp.stack(final_resultsx, axis=2)
    Y = cp.stack(final_resultsy, axis=2) 
    final_dist = cp.stack(final_dist,axis = 2)
    #[754.2223150392452, 1165.0827450392762]
    rpj_coor_4 = cp.stack((X,Y),axis = -1)
    
    x1, y1 = prep_4[:, :, :, :, 0], prep_4[:, :, :, :, 1]
    x2, y2 = rpj_coor_4[:, :, :, :, 0], rpj_coor_4[:, :, :, :, 1]
    # Compute the Euclidean distance
    rpj = cp.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    real_dist  = final_dist*rpj
    real_dist4 = cp.max(real_dist, axis=-1)

    final_resultsx =[]
    final_resultsy =[]
    final_dist = []
    #Q3_bug = 
    for i in range(4):
        x = []
        y = []
        dist = []
        for c in range(3):
            camera_index = combinations_3[i]
            d = camera_index[c]
            P_cam_0 = P[d][0].reshape(1, -1)  # Shape: (1, 4)
            P_cam_1 = P[d][1].reshape(1, -1)  # Shape: (1, 4)
            P_cam_2 = P[d][2].reshape(1, -1)  # Shape: (1, 4)
            
            P0_dot_Q = cp.einsum('ik,btk->bt', P_cam_0, Q3[:,:,i,:])  # Shape: (250, 22, 4)
            P1_dot_Q = cp.einsum('ik,btk->bt', P_cam_1, Q3[:,:,i,:])  # Shape: (250, 22, 4)
            P2_dot_Q = cp.einsum('ik,btk->bt', P_cam_2, Q3[:,:,i,:])  # Shape: (250, 22, 4)
            x.append(P0_dot_Q / P2_dot_Q)
            y.append(P1_dot_Q / P2_dot_Q)
            dist.append(cp.sqrt(cp.sum((Q3_bug[:, :, 0, 0:3] - cam_coord[d]) ** 2, axis=-1)))

        inner_results_stackedx = cp.stack(x, axis=2) 
        final_resultsx.append(inner_results_stackedx)
        inner_results_stackedy = cp.stack(y, axis=2) 
        final_resultsy.append(inner_results_stackedy)
        final_dist.append(cp.stack(dist,axis=2))

    X = cp.stack(final_resultsx, axis=2)
    Y = cp.stack(final_resultsy, axis=2) 
    final_dist = cp.stack(final_dist,axis = 2)
    #[754.2223150392452, 1165.0827450392762]
    rpj_coor_3 = cp.stack((X,Y),axis = -1)
    
    x1, y1 = prep_3[:, :, :, :, 0], prep_3[:, :, :, :, 1]
    x2, y2 = rpj_coor_3[:, :, :, :, 0], rpj_coor_3[:, :, :, :, 1]
    # Compute the Euclidean distance
    rpj = cp.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    real_dist  = final_dist*rpj
    real_dist3 = cp.max(real_dist, axis=-1)
    final_resultsx =[]
    final_resultsy =[]
    final_dist = []
    for i in range(6):
        x = []
        y = []
        dist = []
        for c in range(2):
            camera_index = combinations_2[i]
            d = camera_index[c]
            P_cam_0 = P[d][0].reshape(1, -1)  # Shape: (1, 4)
            P_cam_1 = P[d][1].reshape(1, -1)  # Shape: (1, 4)
            P_cam_2 = P[d][2].reshape(1, -1)  # Shape: (1, 4)
            
            P0_dot_Q = cp.einsum('ik,btk->bt', P_cam_0, Q2[:,:,i,:])  # Shape: (250, 22, 4)
            P1_dot_Q = cp.einsum('ik,btk->bt', P_cam_1, Q2[:,:,i,:])  # Shape: (250, 22, 4)
            P2_dot_Q = cp.einsum('ik,btk->bt', P_cam_2, Q2[:,:,i,:])  # Shape: (250, 22, 4)
            x.append(P0_dot_Q / P2_dot_Q)
            y.append(P1_dot_Q / P2_dot_Q)
            dist.append(cp.sqrt(cp.sum((Q2_bug[:, :, 0, 0:3] - cam_coord[d]) ** 2, axis=-1)))

        inner_results_stackedx = cp.stack(x, axis=2) 
        final_resultsx.append(inner_results_stackedx)
        inner_results_stackedy = cp.stack(y, axis=2) 
        final_resultsy.append(inner_results_stackedy)
        final_dist.append(cp.stack(dist,axis=2))
    X = cp.stack(final_resultsx, axis=2)
    Y = cp.stack(final_resultsy, axis=2) 
    final_dist = cp.stack(final_dist,axis = 2)
    #[754.2223150392452, 1165.0827450392762]
    rpj_coor_2 = cp.stack((X,Y),axis = -1)
    
    x1, y1 = prep_2[:, :, :, :, 0], prep_2[:, :, :, :, 1]
    x2, y2 = rpj_coor_2[:, :, :, :, 0], rpj_coor_2[:, :, :, :, 1]
    # Compute the Euclidean distance
    rpj = cp.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    real_dist  = final_dist*rpj
    real_dist2 = cp.max(real_dist, axis=-1)

    return real_dist4,real_dist3,real_dist2

def map_to_listdynamic(value):
    if value == 4:
        return [1,2,3,4,5,6,7,8,9,10]  # Map 4 to [0]
    elif value == 3:
        return [5,6,7,8,9,10]  # Map 3 to [1, 2, 3, 4]
    elif value == 2:
        return []  # Map 2 to [5, 6, 7, 8, 9, 10]
import time
def triangulate_all(coord,config):
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
    #x_files, y_files, likelihood_files = extract_files_frame_f(json_tracked_files_f, keypoints_ids)

###########################################
    list_dynamic_mincam=  {'Hip':4,'RHip':4,'RKnee':3,'RAnkle':3,'RBigToe':3,'RSmallToe':3,'RHeel':3,'LHip':4,'LKnee':3,'LAnkle':3,'LBigToe':3,'LSmallToe':3,'LHeel':3,'Neck':3,'Head':2,'Nose':2,'RShoulder':3,'RElbow':3,'RWrist':3,'LShoulder':3,'LElbow':3,'LWrist':3}
    list_dynamic_mincam_prep = [map_to_listdynamic(value) for value in list_dynamic_mincam.values()]
    calib = toml.load(calib_file)
    cam_coord=cp.array([find_camera_coordinate(calib[list(calib.keys())[i]]) for i in range(4)])
    P = cp.array(P)
    mappingx,mappingy =computemap(calib_file)
    mappingx = cp.array(mappingx)
    mappingy = cp.array(mappingy)
    #coords_2D_kpt =undistort_points1(mappingx,mappingy,coords_2D_kpt)
    prep_4,prep_3,prep_2 = create_prep(coord,f_range,n_cams,keypoints_ids,json_tracked_files)
    prep_4,prep_3,prep_2 =undistort_points_cupy(mappingx, mappingy, prep_4,prep_3,prep_2)
    prep_4like=cp.min(prep_4[:,:,:,:,2],axis=3)
    prep_3like=cp.min(prep_3[:,:,:,:,2],axis=3)
    prep_2like=cp.min(prep_2[:,:,:,:,2],axis=3)

    prep_like = cp.concatenate((prep_4like,prep_3like,prep_2like),axis =2)
    #undistort
    
    #array([[ 0.2907211 , -2.36826029,  0.34837825,  1.        ]])
    A4,A3,A2 = create_A(prep_4,prep_3,prep_2,P)
    Q4,Q3,Q2 =find_Q(A4,A3,A2)
    Q = cp.concatenate((Q4,Q3,Q2),axis = 2)
    Q3_bug,Q2_bug=create_QBug(likelihood_threshold,prep_3like,Q3,prep_2like,Q2)
    t1 = time.time()
    real_dist4,real_dist3,real_dist2=find_real_dist_error(P,cam_coord,prep_4,Q4,prep_3,Q3,prep_2,Q2,Q3_bug,Q2_bug)
    print("time for find_real_dist_error", time.time()-t1)
    ## delete the liklelihoood vlue which is too low
    real_dist = cp.concatenate((real_dist4,real_dist3,real_dist2),axis = 2)
    loc = cp.where(prep_like < likelihood_threshold)
    
    real_dist[loc] = cp.inf
    # Find the index of the first non-inf element along axis 2
    non_inf_mask = ~cp.isinf(real_dist)
    min_locations_nan = cp.argmax(non_inf_mask, axis=2)
    real_dist_dynamic = cp.copy(real_dist)
    
    ## setting the list dynamic
    for i in range(22):    
        real_dist_dynamic[:,i,list_dynamic_mincam_prep[i]] = cp.inf
    
    ## find the minimum combination
    temp_shape = cp.shape(Q)
    checkinf = cp.min(real_dist_dynamic,axis =2)
    min_locations = cp.argmin(real_dist_dynamic, axis=2)
    loc =cp.where(checkinf==cp.inf)
    min_locations[loc] = min_locations_nan[loc]
    batch_indices, time_indices = cp.meshgrid(cp.arange(temp_shape[0]), cp.arange(temp_shape[1]), indexing='ij')
    Q_selected = Q[batch_indices, time_indices, min_locations]
    Q_selected = Q_selected[:,:,0:3]
    Q_selected = cp.asnumpy(Q_selected)
    Q_tot_gpu = [Q_selected[i].ravel() for i in range(Q_selected.shape[0])]
    

    #### interpolatopm
    #import pdb;pdb.set_trace()
    Q_tot_gpu = pd.DataFrame(Q_tot_gpu)
    if show_interp_indices:
        zero_nan_frames = np.where( Q_tot_gpu.iloc[:,::3].T.eq(0) | ~np.isfinite(Q_tot_gpu.iloc[:,::3].T) )
        zero_nan_frames_per_kpt = [zero_nan_frames[1][np.where(zero_nan_frames[0]==k)[0]] for k in range(keypoints_nb)]
        gaps = [np.where(np.diff(zero_nan_frames_per_kpt[k]) > 1)[0] + 1 for k in range(keypoints_nb)]
        sequences = [np.split(zero_nan_frames_per_kpt[k], gaps[k]) for k in range(keypoints_nb)]
        interp_frames = [[f'{seq[0]}:{seq[-1]+1}' for seq in seq_kpt if len(seq)<=interp_gap_smaller_than and len(seq)>0] for seq_kpt in sequences]
        non_interp_frames = [[f'{seq[0]}:{seq[-1]+1}' for seq in seq_kpt if len(seq)>interp_gap_smaller_than] for seq_kpt in sequences]
    else:
        interp_frames = None
        non_interp_frames = []
    if interpolation_kind != 'none':
        #import ipdb; ipdb.set_trace()
        Q_tot_gpu = Q_tot_gpu.apply(interpolate_zeros_nans, axis=0, args = [interp_gap_smaller_than, interpolation_kind])
    Q_tot_gpu.replace(np.nan, 0, inplace=True)

### SAVE strongness of exclusion and cam_id_exclude
    batch_indices = cp.arange(real_dist.shape[0])[:, None]  # Shape: (184, 1)
    time_indices = cp.arange(real_dist.shape[1])[None, :]  # Shape: (1, 22)
    real_dist_1st = real_dist[batch_indices, time_indices, min_locations_nan]
    batch_indices = cp.arange(real_dist.shape[0])[:, None]  # Shape: (184, 1)
    time_indices = cp.arange(real_dist.shape[1])[None, :]  # Shape: (1, 22)
    real_dist_final = real_dist_dynamic[batch_indices, time_indices, min_locations]
    strongness_exclusion_tot = (real_dist_1st-real_dist_final).tolist()
    result_list = [cp.asnumpy(min_locations[i, :]) for i in range(min_locations.shape[0])]
    value_map = {
        0: [],
        1: [0],
        2: [1],
        3: [2],
        4: [3],
        5: [0, 1],
        6: [0, 2],
        7: [0, 3],
        8: [1, 2],
        9: [1, 3],
        10: [2, 3],
    }

    # Transform the result_list
    id_excluded_cams_record_tot = [
        [value_map[val] for val in array]  # Map each value in the 22-element array
        for array in result_list           # Process each 22-element array in the list
    ]
    #import pdb;pdb.set_trace()
    np.savez(os.path.join(project_dir,'User','reprojection_record.npz'),cam_choose=id_excluded_cams_record_tot,strongness_of_exclusion =strongness_exclusion_tot)
    mdic = {'cam_choose':id_excluded_cams_record_tot,'strongness_of_exclusion':strongness_exclusion_tot}
    savemat(os.path.join(project_dir,'rpj.mat'), mdic)

    trc_path = make_trc(config, Q_tot_gpu, keypoints_names, f_range)

# import time
# s = time.time()
# dir_task = r'C:\Users\MyUser\Desktop\NTKCAP\Patient_data\HealthCare020\2024_12_06\2025_02_19_20_29_calculated\1005_1'        
# os.chdir(dir_task)
# config_dict = toml.load(os.path.join(dir_task,'User','Config.toml'))

# triangulate_all(config_dict)
# print(time.time()-s)