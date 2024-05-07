# get camera number andwrite config
import os
import cv2
import json
import sys
import numpy as np
import multiprocessing
import multiprocessing, threading, logging, sys, traceback
import time
import keyboard
import shutil
from datetime import datetime
import subprocess

######################################################


######################################################
# create calibration folder
def create_calibration_folder(PWD, button_create=False):
    now_calibration = os.path.join(PWD, "calibration")
    now_calibration = os.path.join(now_calibration, "Calibration")
    if os.path.exists(now_calibration):
        with open(os.path.join(now_calibration, "time.txt"), "r") as file:
            get_time = file.read()
        shutil.move(now_calibration, now_calibration + "_" + get_time)
    
    os.makedirs(now_calibration)
    data_path = os.path.join(now_calibration, "data")
    ExtrinsicCalibration_path = os.path.join(now_calibration, "ExtrinsicCalibration")
    ExtrinsicCalibration_path = os.path.join(ExtrinsicCalibration_path, "videos")
    IntrinsicCalibration_path = os.path.join(now_calibration, "IntrinsicCalibration")
    IntrinsicCalibration_path = os.path.join(IntrinsicCalibration_path, "videos")
    os.makedirs(data_path)
    os.makedirs(ExtrinsicCalibration_path)
    os.makedirs(IntrinsicCalibration_path)

    now = datetime.now()
    now_time = str(now.year) + "_" + str(now.month) + "_" + str(now.day) + "_" + str(now.hour) + "_" + str(now.minute) + "_" + str(now.second)
    with open(os.path.join(now_calibration, "time.txt"), "w") as file:
        file.write(now_time)

    pass
######################################################
# update camera ID config
def camera_config_update(save_path, search_num=20):
    config_name = os.path.join(save_path, "config.json")
    with open(config_name, 'r') as f:
        data = json.load(f)
    camera_list = []
    for i in range(search_num):
        print(i)
        if len(camera_list) == data['cam']['number']:
            break
        try:
            cap = cv2.VideoCapture(i)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            ret, frame = cap.read()
            frame_shape = np.shape(frame)
            # import ipdb;ipdb.set_trace()
            for j in range(10):
                ret, frame = cap.read()

            if ret:
                if len(frame_shape) >= 2:
                    if frame_shape[0] == 1080:
                        if frame_shape[1] == 1920:
                            camera_list.append(i)
                            print("HD:" + str(i))
            
        except:
            print("無法開啟USB Port:" + str(i))
    print("============================")
    # import ipdb;ipdb.set_trace()
    for i in camera_list:
        print("Detect FHD Camera : " + str(i))
    print("更新config.json檔案")
    data['cam']['list'] = camera_list

    # 写入更新后的JSON文件
    with open(config_name, 'w') as f:
        json.dump(data, f, indent=4)

######################################################
# create camera ID config
def camera_config_create(save_path):
    data = {
        "cam": {
            "number": 4,
            "list": [],
            "resolution": [1920, 1080],
            "name": "HD camera"
        }
    }
    try:
        os.makedirs(save_path)
        print("創建資料夾成功:" + save_path)
    except FileExistsError:
        print("資料夾已存在")
        print("將更新config.json檔案")
    except Exception as e:
        print(f"創建資料夾失敗 '{save_path}' 失敗：{e}")
    save_name = os.path.join(save_path, "config.json")
    with open(save_name, 'w') as f:
        json.dump(data, f, indent=4)

######################################################
# 檢查相機(imshow)
def camera_show(camera_id, pos, event_start, event_stop):
    cap = cv2.VideoCapture(camera_id)
    width = 1920
    height = 1080
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    # 讀取攝像機影像，並將影像寫入mp4檔案中
    count = 0
    old_time = time.time()
    time.sleep(0.00001)
    while True:
        if event_start.is_set():
            break
    while True:
        now_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        x = "FPS: " + str(1/(now_time - old_time))
        cv2.waitKey(1) 

        if event_stop.is_set():
            break
        cv2.putText(frame, x, (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (30, 144, 255), 1, cv2.LINE_AA)
        frame = cv2.resize(frame, (640, 480))
        cv2.imshow("cam" + str(camera_id), frame)
        cv2.moveWindow("cam" + str(camera_id), pos[0], pos[1])
        count += 1
        old_time = now_time
    cap.release()

def camera_config_open_camera(save_path):
    config_name = os.path.join(save_path, "config.json")
    with open(config_name, 'r') as f:
        data = json.load(f)

    num_cameras = data['cam']['list']
    processes = []
    event_start = multiprocessing.Event()
    event_stop = multiprocessing.Event()

    font = cv2.FONT_HERSHEY_SIMPLEX
    now_cam_num = 0
    position = [[10, 10], [10, 500], [700, 10], [700, 500]]
    for i in num_cameras:
        now_cam_num = now_cam_num + 1
        p = multiprocessing.Process(target=camera_show, args=(i, position[now_cam_num - 1], event_start, event_stop,))
        processes.append(p)
        p.start()
    time.sleep(1)
    event_start.set()
    while True:
        if keyboard.is_pressed('q'):
            print('quit')
            event_stop.set()
            break

    for p in processes:
        p.join()

######################################################
# 拍攝內參
def camera_intrinsic_calibration(config_path, save_path, button_capture=False, button_stop=False, camera_ID=[0]):
    config_name = os.path.join(config_path, "config.json")
    save_path = os.path.join(save_path, "IntrinsicCalibration")
    save_path = os.path.join(save_path, "videos")
    save_path_1 = os.path.join(save_path, "1.mp4")
    with open(config_name, 'r') as f:
        data = json.load(f)

    num_cameras = data['cam']['list']
    select_camera = []
    for i in camera_ID:
        select_camera.append(num_cameras[i])

    mesage = "press q to stop recording"
    mesage2 = "press c to stop capture"

    for i in select_camera:
        number = 0
        cap = cv2.VideoCapture(i)
        width = 1920
        height = 1080
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        fps = 30
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # 創建寫入影像的video writer物件
        video_writers = cv2.VideoWriter(save_path_1, fourcc, fps, (width, height))
        while True:
            mesage3 = "Now capture: " + str(number)
            k = cv2.waitKey(1)
            ret, frame = cap.read()
            if not ret:
                break
            if k == ord("c") | button_capture:
                video_writers.write(frame)
                number += 1
            if k == ord("q") | button_stop:
                break
            cv2.putText(frame, str(mesage), (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (30, 144, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, str(mesage2), (10, 260), cv2.FONT_HERSHEY_SIMPLEX, 2, (30, 144, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, str(mesage3), (10, 320), cv2.FONT_HERSHEY_SIMPLEX, 2, (30, 144, 255), 1, cv2.LINE_AA)
            frame = cv2.resize(frame, (1280, 720))
            cv2.imshow("cam" + str(i), frame)
            cv2.moveWindow("cam" + str(i), 100, 100)
    video_writers.release()
    cv2.destroyAllWindows()
    time.sleep(1)
    for i in range(2,5):
        try:
            os.remove(os.path.join(save_path,  str(i) + ".mp4"))
        except:
            pass
        shutil.copy(save_path_1, os.path.join(save_path,  str(i) + ".mp4"))
    
######################################################
# 拍攝外參
def camera_extrinsicCalibration_calibration(camera_id, now_cam_num, save_path, pos, event_start, event_stop, button_start=False, button_stop=False):
    cap = cv2.VideoCapture(camera_id)
    width = 1920
    height = 1080
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # 創建寫入影像的video writer物件
    video_writers = cv2.VideoWriter(os.path.join(save_path, f"{now_cam_num}.mp4"), fourcc, fps, (width, height))
    # 讀取攝像機影像，並將影像寫入mp4檔案中
    time.sleep(0.00001)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.putText(frame, "press s to start recording", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (30, 144, 255), 1, cv2.LINE_AA)
        frame = cv2.resize(frame, (640, 480))
        cv2.imshow("Cam number:" + str(now_cam_num) + "Cam ID:" + str(camera_id), frame)
        cv2.moveWindow("Cam number:" + str(now_cam_num) + "Cam ID:" + str(camera_id), pos[0], pos[1])
        cv2.waitKey(1) 
        if event_start.is_set() | button_start:
            break

    # while True:
    for i in range(10):
        ret, frame = cap.read()
        if not ret:
            break
        cv2.waitKey(1) 
        if event_stop.is_set() | button_stop:
            break
        video_writers.write(frame)  
        cv2.putText(frame, "press q to stop recording", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (30, 144, 255), 1, cv2.LINE_AA)
        frame = cv2.resize(frame, (640, 480))
        cv2.imshow("Cam number:" + str(now_cam_num) + "Cam ID:" + str(camera_id), frame)
        cv2.moveWindow("Cam number:" + str(now_cam_num) + "Cam ID:" + str(camera_id), pos[0], pos[1])

    video_writers.release()
    cap.release()

def camera_extrinsicCalibration_record(config_path, save_path, button_capture=False, button_stop=False):
    config_name = os.path.join(config_path, "config.json")
    save_path = os.path.join(save_path, "ExtrinsicCalibration")
    save_path = os.path.join(save_path, "videos")
    # save_path_1 = os.path.join(save_path, "1.mp4")
    with open(config_name, 'r') as f:
        data = json.load(f)

    num_cameras = data['cam']['list']

    event_start = multiprocessing.Event()
    event_stop = multiprocessing.Event()
    processes = []
    now_cam_num = 0
    position = [[10, 10], [10, 500], [700, 10], [700, 500]]
    for i in num_cameras:
        now_cam_num = now_cam_num + 1
        p = multiprocessing.Process(target=camera_extrinsicCalibration_calibration, args=(i, now_cam_num, save_path, position[now_cam_num - 1], event_start, event_stop))
        processes.append(p)
        p.start()

    time.sleep(1)
    while True:
        if keyboard.is_pressed('s'):
            event_start.set()
            time.sleep(1)
            break

    for p in processes:
        p.join()

    # print("finshed")

######################################################
# A pose
def camera_Apose(camera_id, now_cam_num, save_path, pos, event_start, event_stop, button_start=False, button_stop=False):
    cap = cv2.VideoCapture(camera_id)
    width = 1920
    height = 1080
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # 創建寫入影像的video writer物件
    video_writers = cv2.VideoWriter(os.path.join(save_path, f"{now_cam_num}.mp4"), fourcc, fps, (width, height))
    # 讀取攝像機影像，並將影像寫入mp4檔案中
    time.sleep(0.00001)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.putText(frame, "press s to start recording", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (30, 144, 255), 1, cv2.LINE_AA)
        frame = cv2.resize(frame, (640, 480))
        cv2.imshow("Cam number:" + str(now_cam_num) + "Cam ID:" + str(camera_id), frame)
        cv2.moveWindow("Cam number:" + str(now_cam_num) + "Cam ID:" + str(camera_id), pos[0], pos[1])
        cv2.waitKey(1) 
        if event_start.is_set() | button_start:
            break

    # while True:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        k = cv2.waitKey(1)
        if keyboard.is_pressed('q'):
            # print('quit')
            event_stop.set()
        if event_stop.is_set() | button_stop:
            break
        video_writers.write(frame)  
        cv2.putText(frame, "press q to stop recording", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (30, 144, 255), 1, cv2.LINE_AA)
        frame = cv2.resize(frame, (640, 480))
        cv2.imshow("Cam number:" + str(now_cam_num) + "Cam ID:" + str(camera_id), frame)
        cv2.moveWindow("Cam number:" + str(now_cam_num) + "Cam ID:" + str(camera_id), pos[0], pos[1])

    video_writers.release()
    cap.release()

def camera_Apose_record(config_path, save_path, patientID, date, button_capture=False, button_stop=False):
    config_name = os.path.join(config_path, "config.json")
    save_path = os.path.join(save_path, "data")
    save_path = os.path.join(save_path, patientID)
    save_path = os.path.join(save_path, date)
    save_path = os.path.join(save_path, "Apose")
    save_path = os.path.join(save_path, "videos")
    os.makedirs(save_path)
    # save_path_1 = os.path.join(save_path, "1.mp4")
    with open(config_name, 'r') as f:
        data = json.load(f)

    num_cameras = data['cam']['list']

    event_start = multiprocessing.Event()
    event_stop = multiprocessing.Event()
    processes = []
    now_cam_num = 0
    position = [[10, 10], [10, 500], [700, 10], [700, 500]]
    for i in num_cameras:
        now_cam_num = now_cam_num + 1
        p = multiprocessing.Process(target=camera_Apose, args=(i, now_cam_num, save_path, position[now_cam_num - 1], event_start, event_stop))
        processes.append(p)
        p.start()

    time.sleep(1)
    while True:
        if keyboard.is_pressed('s'):
            event_start.set()
            time.sleep(1)
            break

    for p in processes:
        p.join()

######################################################
# Motion
def camera_Motion(camera_id, now_cam_num, save_path, pos, event_start, event_stop, button_start=False, button_stop=False):
    cap = cv2.VideoCapture(camera_id)
    width = 1920
    height = 1080
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # 創建寫入影像的video writer物件
    video_writers = cv2.VideoWriter(os.path.join(save_path, f"{now_cam_num}.mp4"), fourcc, fps, (width, height))
    # 讀取攝像機影像，並將影像寫入mp4檔案中
    time.sleep(0.00001)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.putText(frame, "press s to start recording", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (30, 144, 255), 1, cv2.LINE_AA)
        frame = cv2.resize(frame, (640, 480))
        cv2.imshow("Cam number:" + str(now_cam_num) + "Cam ID:" + str(camera_id), frame)
        cv2.moveWindow("Cam number:" + str(now_cam_num) + "Cam ID:" + str(camera_id), pos[0], pos[1])
        cv2.waitKey(1) 
        if event_start.is_set() | button_start:
            break

    # while True:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        k = cv2.waitKey(1)
        if keyboard.is_pressed('q'):
            # print('quit')
            event_stop.set()
        if event_stop.is_set() | button_stop:
            break
        video_writers.write(frame)  
        cv2.putText(frame, "press q to stop recording", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (30, 144, 255), 1, cv2.LINE_AA)
        frame = cv2.resize(frame, (640, 480))
        cv2.imshow("Cam number:" + str(now_cam_num) + "Cam ID:" + str(camera_id), frame)
        cv2.moveWindow("Cam number:" + str(now_cam_num) + "Cam ID:" + str(camera_id), pos[0], pos[1])

    video_writers.release()
    cap.release()

def camera_Motion_record(config_path, save_path, patientID, task, date, button_capture=False, button_stop=False):
    config_name = os.path.join(config_path, "config.json")
    save_path = os.path.join(save_path, "data")
    save_path = os.path.join(save_path, patientID)
    save_path = os.path.join(save_path, date)
    save_path = os.path.join(save_path, task)
    save_path = os.path.join(save_path, "videos")
    os.makedirs(save_path)
    # save_path_1 = os.path.join(save_path, "1.mp4")
    with open(config_name, 'r') as f:
        data = json.load(f)

    num_cameras = data['cam']['list']

    event_start = multiprocessing.Event()
    event_stop = multiprocessing.Event()
    processes = []
    now_cam_num = 0
    position = [[10, 10], [10, 500], [700, 10], [700, 500]]
    for i in num_cameras:
        now_cam_num = now_cam_num + 1
        p = multiprocessing.Process(target=camera_Motion, args=(i, now_cam_num, save_path, position[now_cam_num - 1], event_start, event_stop))
        processes.append(p)
        p.start()

    time.sleep(1)
    while True:
        if keyboard.is_pressed('s'):
            event_start.set()
            time.sleep(1)
            break

    for p in processes:
        p.join()

######################################################
# 計算內外參
import subprocess
def extract_video(PWD, file_path):
    now_path = os.path.join(PWD, "NTK_CAP")
    now_path = os.path.join(now_path, "ThirdParty")
    now_path = os.path.join(now_path, "EasyMocap")
    now_path = os.path.join(now_path, "scripts")
    now_path = os.path.join(now_path, "preprocess")
    now_path = os.path.join(now_path, "extract_video.py")
    subprocess.run(["python", now_path, file_path, "--no2d"])

def detect_chessboard(PWD, file_path):
    now_path = os.path.join(PWD, "NTK_CAP")
    now_path = os.path.join(now_path, "ThirdParty")
    now_path = os.path.join(now_path, "EasyMocap")
    now_path = os.path.join(now_path, "apps")
    now_path = os.path.join(now_path, "calibration")
    now_path = os.path.join(now_path, "detect_chessboard.py")
    subprocess.run(["python", now_path, file_path, "--out", "output", "--pattern", "4,3", "--grid", "0.145"])

def calib_intri(PWD):
    file_path = os.path.join(PWD, "calibration")
    file_path = os.path.join(file_path, "Calibration")
    file_path = os.path.join(file_path, "IntrinsicCalibration")
    extract_video(PWD, file_path)
    detect_chessboard(PWD, file_path)
    now_path = os.path.join(PWD, "NTK_CAP")
    now_path = os.path.join(now_path, "ThirdParty")
    now_path = os.path.join(now_path, "EasyMocap")
    now_path = os.path.join(now_path, "apps")
    now_path = os.path.join(now_path, "calibration")
    now_path = os.path.join(now_path, "calib_intri.py")

    subprocess.run(["python", now_path, file_path, "--num", "300"])

def calib_extri(PWD):
    file_path = os.path.join(PWD, "calibration")
    file_path = os.path.join(file_path, "Calibration")
    output_toml_path = os.path.join(file_path, "Calib.toml")
    file_path = os.path.join(file_path, "ExtrinsicCalibration")

    extract_video(PWD, file_path)
    detect_chessboard(PWD, file_path)
    now_path = os.path.join(PWD, "NTK_CAP")
    now_path = os.path.join(now_path, "ThirdParty")
    now_path = os.path.join(now_path, "EasyMocap")
    now_path = os.path.join(now_path, "apps")
    now_path = os.path.join(now_path, "calibration")
    now_path = os.path.join(now_path, "calib_extri.py")

    intri_path = os.path.join(PWD, "calibration")
    intri_path = os.path.join(intri_path, "Calibration")
    intri_path = os.path.join(intri_path, "IntrinsicCalibration")
    intri_path = os.path.join(intri_path, "output")
    intri_path = os.path.join(intri_path, "intri.yml")

    subprocess.run(["python", now_path, file_path, "--intri", intri_path])

    cali_path = os.path.join(PWD, "calibration")
    cali_path = os.path.join(cali_path, "Calibration")
    cali_path = os.path.join(cali_path, "ExtrinsicCalibration")
    cali_path_in = os.path.join(cali_path, "intri.yml")
    cali_path_ex = os.path.join(cali_path, "extri.yml")

    now_path = os.path.join(PWD, "NTK_CAP")
    now_path = os.path.join(now_path, "script_py")
    now_path = os.path.join(now_path, "calib_yml_to_toml.py")
    subprocess.run(["python", now_path, "-i", cali_path_in, "-e", cali_path_ex, "-t", output_toml_path])



######################################################
# openpose & pose2sim

def marker_caculate(PWD):
    ori_path = PWD
    openpose_path = os.path.join(PWD, "NTK_CAP")
    openpose_path = os.path.join(openpose_path, "ThirdParty")
    openpose_path = os.path.join(openpose_path, "openpose")
    openpose_path = os.path.join(openpose_path, "openpose")
    openpose_exe = os.path.join(openpose_path, "bin")
    openpose_exe = os.path.join(openpose_exe, "OpenPoseDemo.exe")

    posesim_path = os.path.join(PWD, "NTK_CAP")
    posesim_path = os.path.join(posesim_path, "ThirdParty")
    posesim_path = os.path.join(posesim_path, "OpenSim")
    posesim_path = os.path.join(posesim_path, "bin")
    posesim_exe = os.path.join(posesim_path, "opensim-cmd.exe")

    calib_ori_path = os.path.join(PWD, "calibration")
    calib_ori_path = os.path.join(calib_ori_path, "Calibration")
    calib_ori_path = os.path.join(calib_ori_path, "Calib.toml")

    empty_project_path = os.path.join(PWD, "NTK_CAP")
    empty_project_path = os.path.join(empty_project_path, "template")
    empty_project_path = os.path.join(empty_project_path, "Empty_project")
    
    caculate_finshed_path = os.path.join(PWD, "patient")
    
    data_path = os.path.join(PWD, "calibration")
    data_path = os.path.join(data_path, "Calibration")
    data_path = os.path.join(data_path, "data")

    data_patient_list = os.listdir(data_path)

    for i in data_patient_list:
        print("now patient: " + i)
        now_patient = os.path.join(data_path, i)
        record_date = os.listdir(now_patient)
        old_apose_path = os.path.join(caculate_finshed_path, i)
        for j in record_date:
            old_apose_path = os.path.join(old_apose_path, j)
            old_apose_path = os.path.join(old_apose_path, "Apose")
            print("now date: " + j)
            now_date = os.path.join(now_patient, j)
            apose_stage = 0
            apose_using = False
            apose_file = os.path.join(now_date, "Apose")
            
            if os.path.exists(apose_file):
                print("使用最新Apose校正")
                apose_stage = 1
                apose_using = True
                scaling_model = os.path.join(old_apose_path, "Empty_project")
                scaling_model = os.path.join(scaling_model, "opensim")
                scaling_model = os.path.join(scaling_model, "Model_Pose2Sim_Body25_scaled.osim")
            elif os.path.exists(old_apose_path):
                print("使用前次拍攝之Apose")
                apose_using = True
                scaling_model = os.path.join(old_apose_path, "Empty_project")
                scaling_model = os.path.join(scaling_model, "opensim")
                scaling_model = os.path.join(scaling_model, "Model_Pose2Sim_Body25_scaled.osim")
            else:
                print("不使用Apose")

            if apose_stage == 1:
                try:
                    print("嘗試創建資料夾")
                    print("apose_file: " + old_apose_path)
                    os.makedirs(old_apose_path)
                    print("資料夾成功創建")
                except:
                    print("將覆蓋Apose")
                    subprocess.run(["rmdir", "/s", "/q", old_apose_path], check=True, shell=True)
                    print("重新創建資料夾")
                    os.makedirs(old_apose_path)
                try:
                    now_project = os.path.join(old_apose_path, "Empty_project")
                    shutil.copytree(empty_project_path, now_project)
                except:
                    print("複製專案失敗")
                apose_file_videos = os.path.join(apose_file, "videos")
                now_project_videos = os.path.join(now_project, "videos")
                now_project_calib = os.path.join(now_project, "calib-2d")
                now_project_calib = os.path.join(now_project_calib, "Calib.toml")
                shutil.copytree(apose_file_videos, now_project_videos)
                subprocess.run(["rmdir", "/s", "/q", apose_file], check=True, shell=True)
                shutil.copy(calib_ori_path, now_project_calib)

                os.chdir(openpose_path)
                print("切換至" + os.getcwd())
                for l in range(1,5):
                    now_videos = os.path.join(now_project_videos, str(l) + ".mp4")
                    now_json = os.path.join(now_project, "pose-2d")
                    now_json = os.path.join(now_json, "pose_cam" + str(l) + "_json")
                    subprocess.run([openpose_exe, "BODY_25", "--video", now_videos, "--write_json", now_json, "--number_people_max", "1"])
                os.chdir(ori_path)
                print("切換至" + os.getcwd())
                os.chdir(now_project)
                print("切換至" + os.getcwd())
                subprocess.run(["python", "gettrc.py"])
                os.chdir(ori_path)
                print("切換至" + os.getcwd())
                # import ipdb;ipdb.set_trace()
                now_project_3d = os.path.join(now_project, "pose-3d")
                trc_files = os.listdir(now_project_3d)
                for m in trc_files:
                    if "filt" in m:
                        copy_3d = m

                now_project_trc_ori = os.path.join(now_project_3d, copy_3d)
                copy_3d_2 = os.path.join(now_project, "opensim")
                copy_3d_2 = os.path.join(copy_3d_2, "Empty_project_filt_0-30.trc")
                shutil.copy(now_project_trc_ori, copy_3d_2)

                os.chdir(posesim_path)
                print("切換至" + os.getcwd())
                now_project_opensim = os.path.join(now_project, "opensim")
                now_project_opensim_scaling = os.path.join(now_project_opensim, "Scaling_Setup_Pose2Sim_Body25.xml")
                os.chdir(now_project_opensim)
                print("切換至" + os.getcwd())
                subprocess.run([posesim_exe, "run-tool", now_project_opensim_scaling])
                os.chdir(ori_path)
                print("切換至" + os.getcwd())

            tasks = os.listdir(now_date)
            for k in tasks:
                now_task = os.path.join(now_date, k)
                now_project = os.path.join(caculate_finshed_path, i)
                now_project = os.path.join(now_project, j)
                now_project = os.path.join(now_project, k)
                try:
                    os.makedirs(now_project)
                except:
                    print("試驗名稱重複")
                try:
                    now_project = os.path.join(now_project, "Empty_project")
                    shutil.copytree(empty_project_path, now_project)
                except:
                    print("複製專案失敗")

                now_task_videos = os.path.join(now_task, "videos")
                now_project_videos = os.path.join(now_project, "videos")
                now_project_calib = os.path.join(now_project, "calib-2d")
                now_project_calib = os.path.join(now_project_calib, "Calib.toml")
                shutil.copytree(now_task_videos, now_project_videos)
                shutil.copy(calib_ori_path, now_project_calib)

                os.chdir(openpose_path)
                print("切換至" + os.getcwd())
                for l in range(1,5):
                    now_videos = os.path.join(now_task_videos, str(l) + ".mp4")
                    now_json = os.path.join(now_project, "pose-2d")
                    now_json = os.path.join(now_json, "pose_cam" + str(l) + "_json")
                    print(now_videos)
                    subprocess.run([openpose_exe, "BODY_25", "--video", now_videos, "--write_json", now_json, "--number_people_max", "1"])

                os.chdir(ori_path)
                print("切換至" + os.getcwd())
                os.chdir(now_project)
                print("切換至" + os.getcwd())
                subprocess.run(["python", "gettrc.py"])
                os.chdir(ori_path)
                print("切換至" + os.getcwd())
                # import ipdb;ipdb.set_trace()
                now_project_3d = os.path.join(now_project, "pose-3d")
                trc_files = os.listdir(now_project_3d)
                for m in trc_files:
                    if "filt" in m:
                        copy_3d = m

                now_project_trc_ori = os.path.join(now_project_3d, copy_3d)
                copy_3d_2 = os.path.join(now_project, "opensim")
                copy_3d_2 = os.path.join(copy_3d_2, "Empty_project_filt_0-30.trc")
                shutil.copy(now_project_trc_ori, copy_3d_2)

                os.chdir(posesim_path)
                print("切換至" + os.getcwd())
                now_project_opensim = os.path.join(now_project, "opensim")
                if apose_using:
                    now_project_opensim_scaling = os.path.join(now_project_opensim, "Model_Pose2Sim_Body25_scaled.osim")
                    shutil.copy(scaling_model, now_project_opensim_scaling)
                    now_project_opensim_scaling = os.path.join(now_project_opensim, "IK_Setup_Pose2Sim_Body25.xml")
                else:
                    now_project_opensim_scaling = os.path.join(now_project_opensim, "IK_Setup_Pose2Sim_Body25_without_scaling.xml")
                os.chdir(now_project_opensim)
                print("切換至" + os.getcwd())
                subprocess.run([posesim_exe, "run-tool", now_project_opensim_scaling])
                os.chdir(ori_path)

        subprocess.run(["rmdir", "/s", "/q", now_patient], check=True, shell=True)
                


