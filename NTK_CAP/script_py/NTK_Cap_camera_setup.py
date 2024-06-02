# get camera number andwrite config

import os
import cv2
import json
import numpy as np
import multiprocessing
import multiprocessing
import time
import keyboard
import shutil
from datetime import datetime
import subprocess
from ultralytics import YOLO

######################################################

#Version = '1.0.2'

######################################################
# create calibration folder
def create_calibration_folder(PWD, button_create=False):  
    cali_fold = os.path.join(PWD, "calibration")
    os.makedirs(cali_fold, exist_ok=True)

    
    data_path = os.path.join(PWD, "Patient_data")
    ExtrinsicCalibration_path = os.path.join(cali_fold, "ExtrinsicCalibration")
    ExtrinsicCalibration_path = os.path.join(ExtrinsicCalibration_path, "videos")
    IntrinsicCalibration_path = os.path.join(cali_fold, "IntrinsicCalibration")
    IntrinsicCalibration_template_path = os.path.join(PWD, "NTK_CAP", "template","IntrinsicCalibration")
    os.makedirs(data_path, exist_ok=True)
    os.makedirs(ExtrinsicCalibration_path, exist_ok=True)
    os.makedirs(IntrinsicCalibration_path, exist_ok=True)
    if os.path.exists(os.path.join(IntrinsicCalibration_path,'output','intri.yml')):
        # If it exists, delete it first
        os.remove(os.path.join(IntrinsicCalibration_path,'output','intri.yml'))
    # Now copy the source directory to the destination
    shutil.copy(os.path.join(IntrinsicCalibration_template_path,'output','intri.yml'), os.path.join(IntrinsicCalibration_path,'output','intri.yml'))
    # now = datetime.now()
    # now_time = str(now.year) + "_" + str(now.month) + "_" + str(now.day) + "_" + str(now.hour) + "_" + str(now.minute) + "_" + str(now.second)
    # with open(os.path.join(now_calibration, "time.txt"), "w") as file:
    #     file.write(now_time)

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
        cv2.putText(frame, x, (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (30, 144, 255), 4, cv2.LINE_AA)
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
    save_path = os.path.join(save_path, "calibration")
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
    save_path = os.path.join(save_path, "calibration")
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

    cali_time_file_path = os.path.join(save_path, "cali_time.txt")
    cali_file_path = os.path.join(save_path, "calibration")
    Extri_file_path = os.path.join(cali_file_path, "ExtrinsicCalibration")
    toml_file_path = os.path.join(cali_file_path, "Calib.toml")

    save_path = os.path.join(save_path, "Patient_data")
    save_path = os.path.join(save_path, patientID)
    save_path_date = os.path.join(save_path, date)
    save_path_date = os.path.join(save_path_date, 'raw_data')
    save_path_Apose = os.path.join(save_path_date, "Apose")
    time_file_path = os.path.join(save_path_Apose, "recordtime.txt")
    save_path_videos = os.path.join(save_path_Apose, "videos")
    os.makedirs(save_path_videos)

    if os.path.exists(cali_time_file_path):
        shutil.copy(cali_time_file_path, save_path_date)
        print("已成功複製 cali_time.txt")
    else:
        print("cali_time.txt 不存在")

    if os.path.exists(Extri_file_path):
        
        shutil.copytree(Extri_file_path, os.path.join(save_path_date,"ExtrinsicCalibration"))
        print("已成功複製 ExtrinsicCalibration資料夾")
    else:
        print("ExtrinsicCalibration資料夾 不存在")

    if os.path.exists(toml_file_path):
        shutil.copy(toml_file_path, save_path_date)
        print("已成功複製 Calib.toml")
    else:
        print("Calib.toml 不存在")

    if os.path.exists(time_file_path):
        with open(time_file_path, "r") as file:
            formatted_datetime = file.read().strip()
    else:
        now = datetime.now()
        formatted_datetime = now.strftime("%Y_%m_%d_%H%M")
        with open(time_file_path, "w") as file:
            file.write(formatted_datetime)

    
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
        p = multiprocessing.Process(target=camera_Apose, args=(i, now_cam_num, save_path_videos, position[now_cam_num - 1], event_start, event_stop))
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
    save_path = os.path.join(save_path, "Patient_data")
    save_path = os.path.join(save_path, patientID)
    save_path = os.path.join(save_path, date)
    save_path = os.path.join(save_path, 'raw_data')
    save_path = os.path.join(save_path, task)
    
    time_file_path = os.path.join(save_path, "recordtime.txt")
    save_path = os.path.join(save_path, "videos")
    os.makedirs(save_path)

    if os.path.exists(time_file_path):
        with open(time_file_path, "r") as file:
            formatted_datetime = file.read().strip()
    else:
        now = datetime.now()
        formatted_datetime = now.strftime("%Y_%m_%d_%H%M")
        with open(time_file_path, "w") as file:
            file.write(formatted_datetime)


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
    #subprocess.run(["python", now_path, file_path, "--out", "output", "--pattern", "4,3", "--grid", "0.15"])
    subprocess.run(["python", now_path, file_path, "--out", "output", "--pattern", "4,3", "--grid", "0.29"])
def print_yolo_result(PWD,file_path):
    model_trained = YOLO('yolo_model_v1.pt')
    directory_path = os.path.join(file_path, 'yolo_backup')
    source_path = os.path.join(file_path, 'images')
    # Check if the directory exists
    if os.path.exists(directory_path):
    # Remove the directory if it already exists
        shutil.rmtree(directory_path)
    # Create the directory
    os.makedirs(directory_path, exist_ok=True)
    directories = [d for d in os.listdir(source_path) if os.path.isdir(os.path.join(source_path, d))]
    # Iterate over each directory and copy it to the destination directory
    
    for directory in directories:
        source_subdir = os.path.join(source_path, directory)
        destination_subdir = os.path.join(directory_path, directory)
        shutil.copytree(source_subdir, destination_subdir)
        for root, dirs, files in os.walk(destination_subdir):
            for file in files:
                imgname = os.path.join(root, file)
                img = cv2.imread(imgname)
                result_ex = model_trained.predict(source =imgname, save=False, conf=0.5, max_det=1)
                x_min, y_min, x_max, y_max, _, _ = result_ex[0].boxes.data[0]       
                img[:int(y_min), :] = np.array([0, 255, 0])
                img[int(y_max):, :] = np.array([0, 255, 0])
                img[:, :int(x_min)] = np.array([0, 255, 0])
                img[:, int(x_max):] = np.array([0, 255, 0])
                cv2.imwrite(imgname, img)
    
    
def calib_intri(PWD):

    file_path = os.path.join(PWD, "claibration")
    file_path = os.path.join(file_path, "IntrinsicCalibration")
    extract_video(PWD, file_path)
    detect_chessboard(PWD, file_path)
    now_path = os.path.join(PWD, "NTK_CAP")
    now_path = os.path.join(now_path, "ThirdParty")
    now_path = os.path.join(now_path, "EasyMocap")
    now_path = os.path.join(now_path, "apps")
    now_path = os.path.join(now_path, "calibration")
    now_path = os.path.join(now_path, "calib_intri.py")

    subprocess.run(["python", now_path, file_path, "--num", "1000"])

def calib_extri(PWD):

    file_path = os.path.join(PWD, "calibration")
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
    intri_path = os.path.join(intri_path, "IntrinsicCalibration")
    intri_path = os.path.join(intri_path, "output")
    intri_path = os.path.join(intri_path, "intri.yml")

    subprocess.run(["python", now_path, file_path, "--intri", intri_path])

    cali_path = os.path.join(PWD, "calibration")
    cali_path = os.path.join(cali_path, "ExtrinsicCalibration")
    cali_path_in = os.path.join(cali_path, "intri.yml")
    cali_path_ex = os.path.join(cali_path, "extri.yml")

    now_path = os.path.join(PWD, "NTK_CAP")
    now_path = os.path.join(now_path, "script_py")
    now_path = os.path.join(now_path, "calib_yml_to_toml.py")
    subprocess.run(["python", now_path, "-i", cali_path_in, "-e", cali_path_ex, "-t", output_toml_path])
    print_yolo_result(PWD,file_path)



######################################################
