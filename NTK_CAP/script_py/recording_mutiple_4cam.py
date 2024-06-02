
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
# cpname = multiprocessing.current_process().name

# def camera(camera_id, save_path, event_start, event_stop, q_sin, serial_num):
def camera(camera_id, now_cam_num, save_path, event_start, event_stop, serial_num):
    # mpl.info("{0} is currently doing...".format(camera_id))
    cap = cv2.VideoCapture(camera_id)
    width = 1920
    height = 1080
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    timestamps = []

    # 創建寫入影像的video writer物件
    video_writers = cv2.VideoWriter(os.path.join(save_path, f"{now_cam_num}.mp4"), fourcc, fps, (width, height))
    
    # 讀取攝像機影像，並將影像寫入mp4檔案中
    count = 0
    old_time = time.time()
    time.sleep(0.00001)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.putText(frame, "press s to start recording", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (30, 144, 255), 1, cv2.LINE_AA)
        cv2.imshow("cam" + str(camera_id), frame)
        cv2.waitKey(1) 
        if event_start.is_set():
            break

    while True:
        now_time = time.time()
        ret, frame = cap.read()
        # serial = serial_num.value
        # timestamps.append([serial, now_time])
        if not ret:
            break

        x = str(1/(now_time - old_time))
        # cv2.putText(frame, x, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (30, 144, 255), 1, cv2.LINE_AA)
        # cv2.putText(frame, str(now_time), (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (30, 144, 255), 1, cv2.LINE_AA)
        # cv2.putText(frame, "serial number: " + str(serial), (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (30, 144, 255), 1, cv2.LINE_AA)
        # sin = q_sin.get()
        # cv2.putText(frame, "sin wave value: " + str(sin[-1]), (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (30, 144, 255), 1, cv2.LINE_AA)

        cv2.waitKey(1) 

        if event_stop.is_set():
            break

        
        video_writers.write(frame)  
        cv2.putText(frame, "press q to stop recording", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (30, 144, 255), 1, cv2.LINE_AA)
        cv2.imshow("cam" + str(camera_id), frame)
        count += 1
        old_time = now_time

    video_writers.release()
    cap.release()

    # json_str = json.dumps(timestamps)
    # with open(save_path + "\\" + 'timestamps_cam_' + str(camera_id) + '.json', 'w') as f:
    #     f.write(json_str)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        save_path = sys.argv[1]
        print("save path: " + save_path)
    if len(sys.argv) > 2:
        config_path = sys.argv[2]
        print("config path: " + config_path)
        with open(config_path, 'r') as f:
            data = json.load(f)
    with multiprocessing.Manager() as manager:
        serial_num = multiprocessing.Value('i', 0)
        print(serial_num)

        # save_path = r"E:\\github\\YYLabOpenCap\\cameras\\mutiple_cam\\data"
        num_cameras = data['cam']['list']
        processes = []

        event_start = multiprocessing.Event()
        event_stop = multiprocessing.Event()

        
        print("==============================")

        font = cv2.FONT_HERSHEY_SIMPLEX
        str_1 = 'Press s start recording'

        recording_ture_false = input("press y to start, n to quit\n")
        if recording_ture_false == "y":
            now_cam_num = 0
            for i in num_cameras:
                now_cam_num = now_cam_num + 1
                p = multiprocessing.Process(target=camera, args=(i, now_cam_num, save_path, event_start, event_stop, serial_num,))
                processes.append(p)
                p.start()
            time.sleep(2)
            print("press s to start")
            while True:
                if keyboard.is_pressed('s'):
                    print('start recording')
                    event_start.set()
                    break
            print("press q to stop recording & save file")
            
            while True:
                if keyboard.is_pressed('q'):
                    print('stop recording')
                    event_stop.set()
                    break

            for p in processes:
                p.join()

            print("finshed")
        elif recording_ture_false == "n":
            print("quit without recording")
