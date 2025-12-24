#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
    ########################################################
    ## Convert AlphaPose json file to OpenPose json files ##
    ########################################################
    
    Converts AlphaPose single json file to OpenPose frame-by-frame files.
        
    Usage: 
    python -m AlphaPose_to_OpenPose -i input_alphapose_json_file -o output_openpose_json_folder
    OR python -m AlphaPose_to_OpenPose -i input_alphapose_json_file
    OR from Pose2Sim.Utilities import AlphaPose_to_OpenPose; AlphaPose_to_OpenPose.AlphaPose_to_OpenPose_func(r'input_alphapose_json_file', r'output_openpose_json_folder')
'''


## INIT
import json
import os
import argparse


## AUTHORSHIP INFORMATION
__author__ = "David Pagnon"
__copyright__ = "Copyright 2023, Pose2Sim"
__credits__ = ["David Pagnon"]
__license__ = "BSD 3-Clause License"
__version__ = '0.4'
__maintainer__ = "David Pagnon"
__email__ = "contact@david-pagnon.com"
__status__ = "Development"


# FUNCTIONS
def AlphaPose_to_OpenPose_func(*args):
    '''
    Converts AlphaPose/RTMPose single json file to OpenPose frame-by-frame files.
    
    Supports multiple input formats:
    - AlphaPose format: {"image_id": N, "people": [{"pose_keypoints_2d": [...]}]}
    - Legacy RTMPose format: {"image_id": N, "keypoints": [...]}
        
    Usage: 
    python -m AlphaPose_to_OpenPose -i input_alphapose_json_file -o output_openpose_json_folder
    OR python -m AlphaPose_to_OpenPose -i input_alphapose_json_file
    OR from Pose2Sim.Utilities import AlphaPose_to_OpenPose; AlphaPose_to_OpenPose.AlphaPose_to_OpenPose_func(r'input_alphapose_json_file', r'output_openpose_json_folder')
    '''

    try:
        input_alphapose_json_file = os.path.realpath(args[0]['input_alphapose_json_file']) # invoked with argparse
        if args[0]['output_openpose_json_folder'] == None:
            output_openpose_json_folder = os.path.splitext(input_alphapose_json_file)[0]
        else:
            output_openpose_json_folder = os.path.realpath(args[0]['output_openpose_json_folder'])
    except:
        input_alphapose_json_file = os.path.realpath(args[0]) # invoked as a function
        try:
            output_openpose_json_folder = os.path.realpath(args[1])
        except:
            output_openpose_json_folder = os.path.splitext(input_alphapose_json_file)[0]
        
    if not os.path.exists(output_openpose_json_folder):    
        os.mkdir(output_openpose_json_folder)

    # Open AlphaPose json file
    with open(input_alphapose_json_file, 'r') as alpha_json_f:
        alpha_js = json.load(alpha_json_f)
        
        if len(alpha_js) == 0:
            print(f"[WARNING] Empty JSON file: {input_alphapose_json_file}")
            return
        
        # Detect input format
        first_entry = alpha_js[0]
        has_people_key = 'people' in first_entry
        has_keypoints_key = 'keypoints' in first_entry
        
        coords = []
        
        for i, a in enumerate(alpha_js):
            json_dict = {'version':1.3, 'people':[]}
            frame_id = int(alpha_js[i].get('image_id', i))
            
            if has_people_key:
                # Standard format: {"image_id": N, "people": [{"pose_keypoints_2d": [...]}]}
                people_data = alpha_js[i].get('people', [])
                num_people = len(people_data)
                
                for k in range(num_people):
                    coords = people_data[k].get('pose_keypoints_2d', [])
                    person_data = {'person_id': [k], 
                                    'pose_keypoints_2d': coords, 
                                    'face_keypoints_2d': [],   
                                    'hand_left_keypoints_2d':[], 
                                    'hand_right_keypoints_2d':[], 
                                    'pose_keypoints_3d':[], 
                                    'face_keypoints_3d':[], 
                                    'hand_left_keypoints_3d':[], 
                                    'hand_right_keypoints_3d':[]}
                    json_dict['people'].append(person_data)
                    
            elif has_keypoints_key:
                # Legacy RTMPose format: {"image_id": N, "keypoints": [...]}
                coords = alpha_js[i].get('keypoints', [])
                person_data = {'person_id': [0], 
                                'pose_keypoints_2d': coords, 
                                'face_keypoints_2d': [],   
                                'hand_left_keypoints_2d':[], 
                                'hand_right_keypoints_2d':[], 
                                'pose_keypoints_3d':[], 
                                'face_keypoints_3d':[], 
                                'hand_left_keypoints_3d':[], 
                                'hand_right_keypoints_3d':[]}
                json_dict['people'].append(person_data)
            else:
                print(f"[WARNING] Unknown JSON format at frame {i}, skipping...")
                # Create empty frame entry
                pass
                
            json_file = os.path.join(output_openpose_json_folder, os.path.splitext(os.path.basename(str(frame_id).zfill(5)))[0]+'.json')
            with open(json_file, 'w') as js_f:
                js_f.write(json.dumps(json_dict))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_alphapose_json_file', required = True, help='input AlphaPose single json file')
    parser.add_argument('-o', '--output_openpose_json_folder', required = False, help='output folder for frame-by-frame OpenPose json files')
    args = vars(parser.parse_args())
    
    AlphaPose_to_OpenPose_func(args)