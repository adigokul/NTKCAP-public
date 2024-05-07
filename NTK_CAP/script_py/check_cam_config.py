import sys
import multiprocessing
import multiprocessing, threading, logging, sys, traceback

import cv2
import time
import os
import numpy as np
import json

import keyboard

mpl = multiprocessing.log_to_stderr()
mpl.setLevel(logging.INFO)

def camera(camera_id, event_start, event_stop, serial_num):
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
        cv2.imshow("cam" + str(camera_id), frame)
        count += 1
        old_time = now_time

    cap.release()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        print("config path: " + config_path)
        with open(config_path, 'r') as f:
            data = json.load(f)
    with multiprocessing.Manager() as manager:
        serial_num = multiprocessing.Value('i', 0)

        num_cameras = data['cam']['list']
        processes = []

        event_start = multiprocessing.Event()
        event_stop = multiprocessing.Event()

        print("start")
        print("==============================")

        font = cv2.FONT_HERSHEY_SIMPLEX

        now_cam_num = 0
        for i in num_cameras:
            now_cam_num = now_cam_num + 1
            p = multiprocessing.Process(target=camera, args=(i, event_start, event_stop, serial_num,))
            processes.append(p)
            p.start()
        time.sleep(1)
        print("start showing")
        print("press q to stop quite")

        event_start.set()
        while True:
            if keyboard.is_pressed('q'):
                print('quit')
                event_stop.set()
                break

        for p in processes:
            p.join()

        print("finshed")
