# get camera number andwrite config
import os
import cv2
import json
import numpy as np
import multiprocessing, threading, logging, sys, traceback
import multiprocess
import time
import keyboard
import shutil
from datetime import datetime
import subprocess
#import function_b
import easymocap
import import_ipynb
#from full_process import rtm2json
#from xml_update import *
#from Pose2Sim import Pose2Sim
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

save_path =r'C:\Users\eddlai\Desktop\NTKCAP\config'
camera_config_open_camera(save_path)