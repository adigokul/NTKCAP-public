##draw normal background

from mmpose.apis import MMPoseInferencer
import json
import cv2
import numpy as np
import os
import math
import mmpose

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
####parameter
def rtm2json_gpu(Video_path, out_dir, out_video):
    AlphaPose_to_OpenPose = "./NTK_CAP/script_py"
    temp_dir = os.getcwd()
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
    det_model = os.path.join(temp_dir , "NTK_CAP", "ThirdParty", "mmdeploy", "rtmpose-trt", "rtmdet-m")#連到轉換過的det_model的資料夾
    pose_model = os.path.join(temp_dir , "NTK_CAP", "ThirdParty", "mmdeploy", "rtmpose-trt", "rtmpose-m")#連到轉換過的pose_model的資料夾
    device_name = "cuda"
    thr=0.5
    frame_id = 0
    skeleton_type='halpe26'
    np.set_printoptions(precision=13, suppress=True)
    
    video = cv2.VideoCapture(Video_path)
    ###save new video setting
    fps = video.get(cv2.CAP_PROP_FPS)
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
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
        if scores.size==0:
            temp = []
            for i in range(26):
                x = float(0)
                y = float(0)
                each_score = float(0)
                temp.append(x)
                temp.append(y)
                temp.append(each_score)
        
            data1.append({"image_id" : frame_id,"keypoints" :temp})#存成相同格式
            
                
        else:
            temp = []
            for i in range(26):
                x = float(keypoints[0][i][0])
                y = float(keypoints[0][i][1])
                each_score = float(scores[0][i])
                temp.append(x)
                temp.append(y)
                temp.append(each_score)
        
            data1.append({"image_id" : frame_id,"keypoints" :temp})#存成相同格式
      
                
        frame_id += 1

    ###將檔案放入json檔案中
    save_file = open(out_dir, "w") 
    json.dump(data1, save_file, indent = 6)  
    save_file.close()   
    video.release()

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
    temp = data1
    while True:
        ret, frame = cap.read()
        if not ret:
                 break
        
        #第一個0代表第幾幀，第二個0代表畫面中的第幾+1位辨識體，第三個0代表第幾個節點
        keypoints =[]
        keypoint_scores =[]
        for i in range(26):
            
            keypoints.append([temp[count_frame]['keypoints'][i*3],temp[count_frame]['keypoints'][i*3+1]])
            keypoint_scores.append(temp[count_frame]['keypoints'][i*3+2]) #呼叫該檔案中的keypoint_scores列表
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
    clear_output(wait=False)

    os.chdir(AlphaPose_to_OpenPose )
    subprocess.run(['python', '-m','AlphaPose_to_OpenPose', '-i', out_dir])
    os.chdir(temp_dir)

    os.remove(out_dir)



def rtm2json_cpu(Video_path,out_dir,out_video):
    temp_dir = os.getcwd()

    ###########Enter dir for mmpose and script_p
    AlphaPose_to_OpenPose = "../../script_py"
    dir_save = os.path.join(os.getcwd(),'NTK_CAP','ThirdParty','mmpose')
    #import pdb;pdb.set_trace()
    os.chdir(dir_save)
    
    output_file = Path(out_video).parent
    output_file = os.path.join(output_file,'temp_folder')
    if not os.path.isdir(output_file):
        os.mkdir(output_file)
    
        
    
    inferencer = MMPoseInferencer('body26') #使用body26 model
    result_generator = inferencer(Video_path,pred_out_dir=output_file)#運算json資料並儲存到mmpose的predictions
    #results = [result for result in result_generator]
    results = []
    start = 0
    for result in result_generator:
        end = timer()
        print('fps'+str(1/(end-start))+ '\n')
        results.append(result)
        start = timer()
        clear_output(wait=False)

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
    clear_output(wait=False)

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
    
    os.chdir(AlphaPose_to_OpenPose )
    subprocess.run(['python', '-m','AlphaPose_to_OpenPose', '-i', out_dir])
    os.chdir(temp_dir)
    os.remove(json_dir)
    os.remove(out_dir)
    shutil.rmtree(output_file)
def rtm2json(Video_path,out_dir,out_video):
    try:
        rtm2json_gpu(Video_path,out_dir,out_video)
    except:
        rtm2json_cpu(Video_path,out_dir,out_video)
    


def rtm2json_rpjerror(Video_path,out_video,rpj_all_dir):
    halpe26_pose2sim_rpj_order = [16,-1,-1,-1,-1,20,17,21,18,22,19,8,2,9,3,10,4,-1,14,1,11,5,12,6,13,7]
    rpj_all = np.load(rpj_all_dir, allow_pickle=True)
    show_tr = 0.2
    camera_num = Video_path[-5:-4]
    cam_exclude = rpj_all['cam_choose']
    error = rpj_all['error']
    strongness = rpj_all['strongness_of_exclusion']

    temp_dir = os.getcwd()
    output_file = Path(out_video).parent.parent
    check_track = os.path.join(output_file,'pose-2d-tracked','pose_cam' + str(camera_num ) +'_json')
    output_file = os.path.join(output_file,'pose-2d','pose_cam' + str(camera_num ) +'_json')
    
    #import pdb;pdb.set_trace()
    output_json = os.listdir(output_file)
    check_json =os.listdir(check_track)
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
    dots = np.array((range(0,26)))*3
    #keypoints
    frame_temp  = 0
    keypointsx= np.empty((1,1,26))
    keypointsy= np.empty((1,1,26))
    keypoint_scores = np.empty((1,26))
    track_state = np.empty((1))
    for i in range(len(output_json)):
        f = open(os.path.join(output_file,output_json[frame_temp]))
        temp = json.load(f) #載入同檔名的json file
        f.close()
        if frame_temp<len(check_json):
            fc = open(os.path.join(check_track,check_json[frame_temp]))
            temp_check = json.load(fc)
            fc.close()
        else:
            fc = open(os.path.join(check_track,check_json[len(check_json)-1]))
            temp_check = json.load(fc)
            fc.close()
        if len(temp_check['people']) ==0:
            track_state = np.append(track_state,[0])
        else:
            track_state = np.append(track_state,[1])
        
        
        #import pdb;pdb.set_trace()
        x = np.array(temp["people"][0]['pose_keypoints_2d'])[dots]
        y = np.array(temp["people"][0]['pose_keypoints_2d'])[dots+1]
        scores =np.array(temp["people"][0]['pose_keypoints_2d'])[dots+2]
        #import pdb;pdb.set_trace()
        #import pdb;pdb.set_trace()
        keypointsx = np.append(keypointsx,[[x]],axis=0)
        keypointsy = np.append(keypointsy,[[y]],axis=0)
        keypoint_scores = np.append(keypoint_scores,[scores],axis=0)
        frame_temp = frame_temp+1

    keypointsx = np.delete(keypointsx,0,0)
    keypointsy = np.delete(keypointsy,0,0)
    keypoint_scores = np.delete(keypoint_scores,0,0)
    keypoints =np.concatenate((keypointsx, keypointsy), axis=1)
    track_state = np.delete(track_state,0,0)
    #import pdb;pdb.set_trace()

    while True:
        ret, frame = cap.read()
        if not ret:
                    break
        
        frame_count = count_frame
        
        #第一個0代表第幾幀，第二個0代表畫面中的第幾+1位辨識體，第三個0代表第幾個節點
        count = 0
        for i in range(26):

            indicator = halpe26_pose2sim_rpj_order[i] 
            #import pdb;pdb.set_trace()
            if count_frame<np.shape(cam_exclude)[0]:
                
                if  any(cam_exclude_temp ==(int(camera_num)-1) for cam_exclude_temp in cam_exclude[count_frame][indicator-1]):  
                    rpj_state = True
                else:
                    rpj_state = False
            else:
                rpj_state = False
            p = keypoint_scores[frame_count][count]*255 #隨score顏色進行變換
            #若keypoint score太低，則標示出來
            if keypoint_scores[frame_count][count]>=show_tr and keypoint_scores[frame_count][count]<0.5 and rpj_state == False and track_state[frame_count]==1:
                cv2.circle(frame,(int(keypoints[frame_count][0][count]), int(keypoints[frame_count][1][count])), 3, (255-p, p, 0), 3)#frame, (x,y), radius, color, thickness
                font = cv2.FONT_HERSHEY_SIMPLEX # font
                org = (int(keypoints[frame_count][0][count])+3, int(keypoints[frame_count][1][count])) # 偏移
                fontScale = 0.5 # fontScale
                color = (255, 255, 255) # Blue color in BGR
                thickness = 1 # Line thickness of 2 px 
                # Using cv2.putText() method 
                image = cv2.putText(frame, str(int(keypoint_scores[frame_count][count]*100)), org, font,  
                                        fontScale, color, thickness, cv2.LINE_AA) 
            elif keypoint_scores[frame_count][count]>=0.5 and rpj_state == False and track_state[frame_count]==1:
                cv2.circle(frame,(int(keypoints[frame_count][0][count]), int(keypoints[frame_count][1][count])), 2, (0, 255, p), 3)

                 
            elif rpj_state == True and track_state[frame_count]==1:
                #error_all21 = error[frame_count][indicator-1][0]-error[frame_count][indicator-1][int(camera_num)]
                #print(error_all21)
                
                cv2.circle(frame,(int(keypoints[frame_count][0][count]), int(keypoints[frame_count][1][count])), 4, (strongness[frame_count][indicator-1]*10, strongness[frame_count][indicator-1]*10,strongness[frame_count][indicator-1]*10), 3)

            count = count+1

            
                

        for i in range(25):
            if keypoint_scores[frame_count][line[i][0]]>show_tr and keypoint_scores[frame_count][line[i][1]]>show_tr and track_state[frame_count]==1: 
                #import pdb;pdb.set_trace()
                cv2.line(frame, (int(keypoints[frame_count][0][line[i][0]]), int(keypoints[frame_count][1][line[i][0]])), (int(keypoints[frame_count][0][line[i][1]]), int(keypoints[frame_count][1][line[i][1]])), (0, 0, 255), 1)
            elif track_state[frame_count]==0:
                cv2.line(frame, (int(keypoints[frame_count][0][line[i][0]]), int(keypoints[frame_count][1][line[i][0]])), (int(keypoints[frame_count][0][line[i][1]]), int(keypoints[frame_count][1][line[i][1]])), (255, 255, 255), 1) 
        out.write(frame)
        count_frame = count_frame+1
        frame_count = frame_count+1

    cap.release()
    out.release()
    clear_output(wait=False)


    os.chdir(temp_dir)
def openpose2json_full(video_full_path,output_video,openpose,json_s_folder):
    ### openpose video
    openpose_exe = os.path.join(openpose,'bin','OpenPoseDemo.exe')
    dir_now = os.getcwd()
    #if not os.path.exists(os.path.join(output_path,'json_temp')):
        #os.mkdir(os.path.join(output_path,'json_temp'))
    os.chdir(openpose)
    subprocess.run([openpose_exe, "BODY_25", "--video", video_full_path, "--write_json", json_s_folder, "--number_people_max", "1"])
    os.chdir(dir_now)
    ### output mp4
    name_j = os.listdir(json_s_folder)
    cap = cv2.VideoCapture(video_full_path)
    line = [[0,15],[15,17],[0,16],[16,18],[0,1],[1,2],[1,5],[2,3],[5,6],[3,4],[6,7],[8,9],[8,12],[9,10],[12,13],[10,11],[13,14],[11,24],[14,21],[11,22],[14,19],[22,23],[19,20],[1,8]]
    fps = cap.get(cv2.CAP_PROP_FPS)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output =output_video
    out = cv2.VideoWriter(output, fourcc, fps, (width, height))
    count_frame = 0
    while True:
        ret, frame = cap.read()
        if not ret:
                break
        f = open(os.path.join(json_s_folder ,name_j[count_frame]))
        temp = json.load(f) 
        keypoints = temp['people'][0]['pose_keypoints_2d']
        count = 0
        for i in range(25):
            print(i)
            if keypoints[count+2]>0.3 and keypoints[count+2]<0.5:
                p = keypoints[count+2]*255
                cv2.circle(frame,(int(keypoints[count]), int(keypoints[count+1])), 2, (255-p, p, 0), 3)
                # font 
                font = cv2.FONT_HERSHEY_SIMPLEX 
                # org 
                org = (int(keypoints[count])+3, int(keypoints[count+1])) 
                # fontScale 
                fontScale = 0.5
                # Blue color in BGR 
                color = (255, 255, 255) 
                # Line thickness of 2 px 
                thickness = 1
                # Using cv2.putText() method 
                image = cv2.putText(frame, str(int(keypoints[count+2]*100)), org, font,  
                                    fontScale, color, thickness, cv2.LINE_AA) 
            else:
                p = keypoints[count+2]*255
                cv2.circle(frame,(int(keypoints[count]), int(keypoints[count+1])), 2, (0, 255, p), 3)
            count = count+3
        for i in range(24):

            if keypoints[line[i][0]*3+2]>0.3 and keypoints[line[i][1]*3+2]>0.3: 
                print(i)
                cv2.line(frame, (int(keypoints[line[i][0]*3]), int(keypoints[line[i][0]*3+1])), (int(keypoints[line[i][1]*3]), int(keypoints[line[i][1]*3+1])), (0, 0, 255), 1)
        out.write(frame)
        count_frame = count_frame+1

    cap.release()
    out.release()
    clear_output(wait=False)



def openpose2json_video(video_full_path,output_video,json_s_folder):

    ### output mp4
    name_j = os.listdir(json_s_folder)
    cap = cv2.VideoCapture(video_full_path)
    line = [[0,15],[15,17],[0,16],[16,18],[0,1],[1,2],[1,5],[2,3],[5,6],[3,4],[6,7],[8,9],[8,12],[9,10],[12,13],[10,11],[13,14],[11,24],[14,21],[11,22],[14,19],[22,23],[19,20],[1,8]]
    fps = cap.get(cv2.CAP_PROP_FPS)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output =output_video
    out = cv2.VideoWriter(output, fourcc, fps, (width, height))
    count_frame = 0
    while True:
        ret, frame = cap.read()
        if not ret:
                break
        f = open(os.path.join(json_s_folder ,name_j[count_frame]))
        temp = json.load(f) 
        keypoints = temp['people'][0]['pose_keypoints_2d']
        count = 0
        for i in range(25):
            print(i)
            if keypoints[count+2]>0.3 and keypoints[count+2]<0.5:
                p = keypoints[count+2]*255
                cv2.circle(frame,(int(keypoints[count]), int(keypoints[count+1])), 2, (255-p, p, 0), 3)
                # font 
                font = cv2.FONT_HERSHEY_SIMPLEX 
                # org 
                org = (int(keypoints[count])+3, int(keypoints[count+1])) 
                # fontScale 
                fontScale = 0.5
                # Blue color in BGR 
                color = (255, 255, 255) 
                # Line thickness of 2 px 
                thickness = 1
                # Using cv2.putText() method 
                image = cv2.putText(frame, str(int(keypoints[count+2]*100)), org, font,  
                                    fontScale, color, thickness, cv2.LINE_AA) 
            else:
                p = keypoints[count+2]*255
                cv2.circle(frame,(int(keypoints[count]), int(keypoints[count+1])), 2, (0, 255, p), 3)
            count = count+3
        for i in range(24):

            if keypoints[line[i][0]*3+2]>0.3 and keypoints[line[i][1]*3+2]>0.3: 
                print(i)
                cv2.line(frame, (int(keypoints[line[i][0]*3]), int(keypoints[line[i][0]*3+1])), (int(keypoints[line[i][1]*3]), int(keypoints[line[i][1]*3+1])), (0, 0, 255), 1)
        out.write(frame)
        count_frame = count_frame+1

    cap.release()
    out.release()
    clear_output(wait=False)


def add_frame_from_video(video_full_path,output_video):
    cap = cv2.VideoCapture(video_full_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output =output_video
    out = cv2.VideoWriter(output, fourcc, fps, (width, height))
    count_frame = 0
    while True:
        ret, frame = cap.read()
        if not ret:
                break
        font = cv2.FONT_HERSHEY_SIMPLEX 
    
        # org 
        org = (50,50) 
        
        # fontScale 
        fontScale = 1
        
        # Blue color in BGR 
        color = (255, 0, 0) 
        
        # Line thickness of 2 px 
        thickness = 2
        
        # Using cv2.putText() method .
        count_frame = count_frame+1    
        image = cv2.putText(frame, 'frame: '+ str(count_frame), org, font,  
                        fontScale, color, thickness, cv2.LINE_AA) 
            
        out.write(frame)
        #window_name = 'Image'
        #cv2.imshow(window_name, image)
        #cv2.waitKey(0)  
                           
    out.release()
    cap.release