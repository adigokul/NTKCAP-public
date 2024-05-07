# Copyright (c) OpenMMLab. All rights reserved.
import os
import json
import cv2
import numpy as np
from mmdeploy_runtime import PoseTracker
import timer
from mmpose.apis import MMPoseInferencer
import json
import cv2
import numpy as np
import os
import math
import mmpose
#from matplotlib import pyplot as plt
import subprocess
from PIL import Image, ImageOps
from IPython.display import clear_output
import shutil
from timeit import default_timer as timer
import json
import subprocess
import os
from pathlib import Path
import inspect
from mmdeploy_runtime import PoseTracker
import traceback
Video_path=r'C:\\Users\\user\\Desktop\\NTKCAP\\Patient_data\\1\\2024_4_26\\2024_05_02_00_19_calculated\\Apose\\videos\\1.mp4'
out_dir=r'C:\\Users\\user\\Desktop\\NTKCAP\\Patient_data\\1\\2024_4_26\\2024_05_02_00_19_calculated\\Apose\\pose-2d\\pose_cam1_json.json'
out_video=r'C:\\Users\\user\\Desktop\\NTKCAP\\Patient_data\\1\\2024_4_26\\2024_05_02_00_19_calculated\\Apose\\videos_pose_estimation\\1.mp4'
PWD =r'C:\\Users\\user\\Desktop\\NTKCAP'
def rtm2json_gpu(Video_path, out_dir, out_video, PWD):
    VISUALIZATION_CFG = dict(
    halpe26=dict(
        skeleton=[(15, 13), (13, 11), (11,19),(16, 14), (14, 12), (12,19),
                  (17,18), (18,19), (18,5), (5,7), (7,9), (18,6), (6,8),
                  (8,10), (1,2), (0,1), (0,2), (1,3), (2,4), (3,5), (4,6),
                  (15,20), (15,22), (15,24),(16,21),(16,23), (16,25)],
        palette=[(51, 153, 255), (0, 255, 0), (255, 128, 0)],
        link_color=[
            1, 1, 1, 2, 2, 2, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2
        ],
        point_color=[
            0, 0, 0, 0, 0, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2
        ],
        sigmas=[
            0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072,
            0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089, 0.026,
            0.026, 0.066, 0.079, 0.079, 0.079, 0.079, 0.079, 0.079
        ]))
    det_model = os.path.join(PWD, "NTK_CAP", "ThirdParty", "mmdeploy", "rtmpose-trt", "rtmdet-m")#連到轉換過的det_model的資料夾
    pose_model = os.path.join(PWD, "NTK_CAP", "ThirdParty", "mmdeploy", "rtmpose-trt", "rtmpose-m")#連到轉換過的pose_model的資料夾
    device_name = "cuda"
    thr=0.5
    frame_id = 0
    skeleton_type='halpe26'
    np.set_printoptions(precision=4, suppress=True)
    
    video = cv2.VideoCapture(Video_path)
    ###save new video setting
    fps = video.get(cv2.CAP_PROP_FPS)
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_video, fourcc, fps, (width, height))
    ###
    tracker = PoseTracker(det_model, pose_model, device_name)    
    sigmas = VISUALIZATION_CFG[skeleton_type]['sigmas']
    state = tracker.create_state(det_interval=1, det_min_bbox_size=100, keypoint_sigmas=sigmas)
    ###skeleton style
    skeleton = VISUALIZATION_CFG[skeleton_type]['skeleton']
    palette = VISUALIZATION_CFG[skeleton_type]['palette']
    link_color = VISUALIZATION_CFG[skeleton_type]['link_color']
    point_color = VISUALIZATION_CFG[skeleton_type]['point_color']    
    data1 = []
    while True:
        success, frame = video.read()
        if not success:
            break
        results = tracker(state, frame, detect=-1)
        keypoints, bboxes, _ = results
        scores = keypoints[..., 2]
        keypoints = keypoints[..., :2]
        
        for kpts, score, bbox in zip(keypoints, scores, bboxes):
            import pdb;pdb.set_trace
            show = [1] * len(kpts)
            temp = []
            for i in range(26):
                x = float(kpts[i][0])
                y = float(kpts[i][1])
                each_score = float(score[i])
                temp.append(x)
                temp.append(y)
                temp.append(each_score)
        
            data1.append({"image_id" : frame_id,"keypoints" :temp})#存成相同格式
            kpts=kpts.astype(int) #cv2只能畫整數點位 
            ###畫線畫點
            for (u, v), color in zip(skeleton, link_color):
                if score[u] > thr and score[v] > thr:
                    cv2.line(frame, kpts[u], tuple(kpts[v]), palette[color], 1,cv2.LINE_AA)
                else:
                    show[u] = show[v] = 0
            for kpt, show, color in zip(kpts, show, point_color):
                if show:
                    cv2.circle(frame, kpt, 1, palette[color], 2, cv2.LINE_AA)    
            out.write(frame)
        frame_id += 1

    ###將檔案放入json檔案中
    save_file = open(out_dir, "w") 
    json.dump(data1, save_file, indent = 6)  
    save_file.close() 
        
    
    video.release()
    out.release()
    cv2.destroyAllWindows()
    
def rtm2json_cpu(Video_path,out_dir,out_video):
    temp_dir = os.getcwd()

    ###########Enter dir for mmpose and script_p
    AlphaPose_to_OpenPose = "../../script_py"
    dir_save = os.path.join(os.getcwd(),'NTK_CAP','ThirdParty','mmpose')
    #import pdb;pdb.set_trace()
    os.chdir(dir_save)
    inferencer = MMPoseInferencer('body26') #使用body26 model
    output_file = Path(out_video).parent
    output_file = os.path.join(output_file,'temp_folder')
    if not os.path.isdir(output_file):
        os.mkdir(output_file)

    result_generator = inferencer(Video_path,pred_out_dir=output_file)#運算json資料並儲存到mmpose的predictions
    #results = [result for result in result_generator]
    results = []
    start = 0
    for result in result_generator:
        end = timer()
        print('fps'+str(1/(end-start))+ '\n')
        results.append(result)
        start = timer()
        

    # Opening JSON file
    
    json_dir = os.listdir(output_file)
    json_dir = os.path.join(output_file,json_dir[0])

    
    
    
    
    ### video output
    video_full_path = Video_path
    output = out_video
    cap = cv2.VideoCapture(video_full_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    line = [[0,1],[1,3],[0,2],[2,4],[0,17],[0,18],[18,5],[5,7],[7,9],[18,6],[6,8],[8,10],[18,19],[19,11],[11,13],[13,15],[15,24],[15,20],[15,22],[19,12],[12,14],[14,16],[16,25],[16,21],[16,23]]
    fps = cap.get(cv2.CAP_PROP_FPS)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output, fourcc, fps, (width, height))
    count_frame = 0
    while True:
        ret, frame = cap.read()
        if not ret:
                 break
        f = open(json_dir)
        
        temp = json.load(f) #載入同檔名的json file
        f.close()
        #第一個0代表第幾幀，第二個0代表畫面中的第幾+1位辨識體，第三個0代表第幾個節點
        keypoints = temp[count_frame]["instances"][0]['keypoints'] #呼叫該檔案中的keypoints列表
        keypoint_scores = temp[count_frame]["instances"][0]['keypoint_scores'] #呼叫該檔案中的keypoint_scores列表
        count = 0
        for i in range(26):
            p = keypoint_scores[count]*255 #隨score顏色進行變換
            #若keypoint score太低，則標示出來
            if keypoint_scores[count]>=0.3 and keypoint_scores[count]<0.5:
                cv2.circle(frame,(int(keypoints[count][0]), int(keypoints[count][1])), 3, (255-p, p, 0), 3)#frame, (x,y), radius, color, thickness
                font = cv2.FONT_HERSHEY_SIMPLEX # font
                org = (int(keypoints[count][0])+3, int(keypoints[count][1])) # 偏移
                fontScale = 0.5 # fontScale
                color = (255, 255, 255) # Blue color in BGR
                thickness = 1 # Line thickness of 2 px 
                # Using cv2.putText() method 
                image = cv2.putText(frame, str(int(keypoint_scores[count]*100)), org, font,  
                                       fontScale, color, thickness, cv2.LINE_AA) 
            elif keypoint_scores[count]>=0.5:
                cv2.circle(frame,(int(keypoints[count][0]), int(keypoints[count][1])), 2, (0, 255,p ), 3)
                
            count = count+1

        for i in range(25):
            if keypoint_scores[line[i][0]]>0.3 and keypoint_scores[line[i][1]]>0.3: 

                cv2.line(frame, (int(keypoints[line[i][0]][0]), int(keypoints[line[i][0]][1])), (int(keypoints[line[i][1]][0]), int(keypoints[line[i][1]][1])), (0, 0, 255), 1)
        out.write(frame)
        count_frame = count_frame+1
    
    cap.release()
    out.release()


    #### json to openpose perframe
    f = open(json_dir)

    # returns JSON object as 
    # a dictionary
    list = os.listdir(output_file)
    data = json.load(f)

    # Iterating through the json

    # Closing file
    f.close()
    data1 = []
    for i in range(len(data)):
        temp = []
        for k in range(26):
            score = data[i]['instances'][0][ 'keypoint_scores'][k]
            x = data[i]['instances'][0][ 'keypoints'][k][0]
            y =data[i]['instances'][0][ 'keypoints'][k][1]
            temp.append(x)
            temp.append(y)
            temp.append(score)
        data1.append({"image_id" : i,"keypoints" :temp})


    result = data1

    save_file = open(out_dir, "w") 
    json.dump(result, save_file, indent = 6)  
    save_file.close()

#rtm2json_cpu(Video_path, out_dir, out_video)
rtm2json_gpu(Video_path, out_dir, out_video, PWD)