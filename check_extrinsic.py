# Extrinsic calibration once a day
# 
import os
import cv2
import toml
import json
import numpy as np
import subprocess

#size = '4,3'
#temp = False
#error_list = []
# calib_toml = r"C:\Users\陳柏宏\Desktop\testfordraw\Calib.toml"
# json_file_folder_path = r"C:\Users\陳柏宏\Desktop\testfordraw\ExtrinsicCalibration"

def fintune_chessboard_v1(calib_toml_path, extrinsic_folder_path, camera_num = 4, size = '4,3'):
    script_path = 'GUI_test.py'
    with open(calib_toml_path) as file:
        calib_data = toml.load(file)

    intri_matrix = np.array(calib_data['cam_1']['matrix']) # 3*3 Intrinsic matrix
    discoef = np.array(calib_data['cam_1']['distortions']) # distortion coefficients
    rot = np.array(calib_data['cam_1']['rotation']) # rotation vector
    trans = np.array(calib_data['cam_1']['translation']) # translation vector

    ex_chess_folder_path = os.path.join(extrinsic_folder_path, 'chessboard')
    ex_image_folder_path = os.path.join(extrinsic_folder_path, 'images')
    for i in range(camera_num):
        ex_chess_json_folder_path = os.path.join(ex_chess_folder_path, str(i+1))
        ex_image_camera_path = os.path.join(ex_image_folder_path, str(i+1))
        for idx, json_file in enumerate(os.listdir(ex_chess_json_folder_path)):
            json_file_path = os.path.join(ex_chess_json_folder_path, json_file) # each json file path
            filename = json_file.split('.')[0]
            image_file_path = os.path.join(ex_image_camera_path, (filename + '.jpg'))
            with open(json_file_path) as file:
                points = json.load(file) # all data in json
                point_3D = np.array(points['keypoints3d']) # 3d points coordinates
                point_2D = points['keypoints2d'] # 2d points coordinates
                image_points = [point[:2] for point in point_2D] # eliminate confidence
                image_points_pre, _ = cv2.projectPoints(point_3D, rot, trans, intri_matrix, discoef) # reproject 3d points to 2d
                # compare the error 
                error = cv2.norm(np.array(image_points, dtype=np.float64).reshape(-1, 1, 2), np.array(image_points_pre, dtype=np.float64).reshape(-1, 1, 2), cv2.NORM_L2)
                eachframe_avgerror = error / 12
                if eachframe_avgerror > 0.3:
                    subprocess.run(['python', script_path, '--image_path', image_file_path, '--json_file', json_file_path, '--size', size])
