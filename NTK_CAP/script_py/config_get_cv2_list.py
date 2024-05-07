import cv2
import sys
import json
import numpy as np

if len(sys.argv) > 1:
        file_path = sys.argv[1]
        print("config path: " + file_path)

# 读取JSON文件
with open(file_path, 'r') as f:
    data = json.load(f)

check_num = 20
cam_list = []
for i in range(check_num):
    try:
        cap = cv2.VideoCapture(i)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        ret, frame = cap.read()
        frame_shape = np.shape(frame)

        if len(frame_shape) >= 2:
            if frame_shape[0] == 1080:
                if frame_shape[1] == 1920:
                    cam_list.append(i)
                    if len(cam_list) == data['cam']['number']:
                        break
    
    except:
        print("無法開啟相機")

# print(cam_list)

# 更新字段值
data['cam']['list'] = cam_list

# 写入更新后的JSON文件
with open(file_path, 'w') as f:
    json.dump(data, f, indent=4)
