import sys
import cv2
import time
import os
import json

def camera(camera_id, now_cam_num, save_path):
    print(save_path)
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
    count = 0
    time.sleep(0.00001)

    mesage = "press s to start recording"
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.putText(frame, str(mesage), (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (30, 144, 255), 1, cv2.LINE_AA)
        cv2.imshow("cam" + str(camera_id), frame)
        k = cv2.waitKey(1) 
        if k == ord("s"):
            break
    
    mesage = "press q to stop recording"
    mesage2 = "press c to stop capture"
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        k = cv2.waitKey(1)
        if k == ord("q"):
            break

        if k == ord("c"):
            video_writers.write(frame)  
            count += 1

        cv2.putText(frame, str(mesage), (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (30, 144, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, str(mesage2), (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (30, 144, 255), 1, cv2.LINE_AA)
        cv2.imshow("cam" + str(camera_id), frame)

    video_writers.release()
    cap.release()

if __name__ == "__main__":


    if len(sys.argv) > 1:
        save_path = sys.argv[1]
        print("save path: " + save_path)
    if len(sys.argv) > 2:
        config_path = sys.argv[2]
        print("config path: " + config_path)
        with open(config_path, 'r') as f:
            data = json.load(f)

    print(sys.argv[3])
    if len(sys.argv) > 3:
        copy_ini = sys.argv[3]
        print("==========")
        print(copy_ini)
        print("==========")
        if copy_ini == "True":
            num_cameras = data['cam']['list']
        else:
            num_cameras = [data['cam']['list'][0]]
    

    print("start")
    print("==============================")

    now_cam_num = 0
    for i in num_cameras:
        now_cam_num = now_cam_num + 1
        camera(i, now_cam_num, save_path)
        cv2.destroyAllWindows()
