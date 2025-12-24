##draw normal background

from mmpose.apis import MMPoseInferencer
import json
import cv2
import numpy as np
import os
import math
import platform
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
MMDEPLOY_AVAILABLE = False
RTMLIB_AVAILABLE = False
# Force skip mmdeploy due to incompatible TensorRT engines (built on Windows)
FORCE_RTMLIB = True

# Try rtmlib first (preferred for cross-platform compatibility)
try:
    from rtmlib import Body as RTMLibBody
    RTMLIB_AVAILABLE = True
    print('rtmlib found (using as primary backend)')
except:
    print('rtmlib not found')

# Only try mmdeploy if rtmlib is not available and we're not forcing rtmlib
if not RTMLIB_AVAILABLE and not FORCE_RTMLIB:
    try:
        from mmdeploy_runtime import PoseTracker as MMDeployPoseTracker
        MMDEPLOY_AVAILABLE = True
        print('mmdeploy SDK found')
    except:
        print('mmdeploy SDK not found')

import traceback


class RTMLibPoseTrackerWrapper:
    """Wrapper to make rtmlib work like mmdeploy's PoseTracker interface."""
    
    def __init__(self, det_model_path, pose_model_path, device_name):
        """Initialize using rtmlib's PoseTracker with BodyWithFeet for stable tracking."""
        backend = 'onnxruntime'
        device = 'cuda' if 'cuda' in device_name.lower() else 'cpu'
        
        # Use rtmlib's BodyWithFeet directly (tracking disabled due to instability with varying people count)
        try:
            from rtmlib import BodyWithFeet
            # Direct model without tracking - more stable when number of people varies
            self.body = BodyWithFeet(device=device, backend=backend)
            self.model_type = 'halpe26'
            self.use_tracker = False
            print(f"RTMLibPoseTrackerWrapper: Using BodyWithFeet (Halpe26, no tracking) on {device}")
        except Exception as e:
            print(f"BodyWithFeet init failed: {e}, falling back to Body")
            # Fallback to Body (COCO 17 keypoints)
            from rtmlib import Body
            self.body = Body(device=device, backend=backend)
            self.model_type = 'coco17'
            self.use_tracker = False
            print(f"RTMLibPoseTrackerWrapper: Fallback to Body (COCO17) on {device}")
        
        self._frame_id = 0
        self._prev_keypoints = None  # For temporal smoothing
    
    def create_state(self, **kwargs):
        """Create a dummy state (rtmlib doesn't need state management)."""
        return {"frame_id": 0}
    
    def __call__(self, state, frame, detect=-1):
        """Run inference on a frame, returning (keypoints, bboxes, track_ids)."""
        # rtmlib returns keypoints and scores
        try:
            keypoints, scores = self.body(frame)
        except Exception as e:
            print(f"[WARNING] Pose estimation failed: {e}")
            return np.zeros((0, 26, 3)), np.zeros((0, 4)), np.zeros(0, dtype=np.uint32)
        
        # Convert to mmdeploy format: keypoints shape should be (N, K, 3) where 3 = x, y, score
        if len(keypoints) == 0:
            # No detections
            return np.zeros((0, 26, 3)), np.zeros((0, 4)), np.zeros(0, dtype=np.uint32)
        
        num_people = len(keypoints)
        num_kpts = keypoints.shape[1] if len(keypoints.shape) > 1 else 17
        
        # Create combined keypoints with scores
        combined = np.zeros((num_people, 26, 3))
        
        if self.model_type == 'halpe26' and num_kpts >= 26:
            # BodyWithFeet outputs Halpe26 format directly (26 keypoints)
            for i in range(num_people):
                for j in range(26):
                    combined[i, j, 0] = keypoints[i, j, 0]  # x
                    combined[i, j, 1] = keypoints[i, j, 1]  # y
                    combined[i, j, 2] = scores[i, j] if j < len(scores[i]) else 0.5  # score
        else:
            # COCO 17 keypoints - need to map to Halpe26
            # COCO: 0=nose, 1=left_eye, 2=right_eye, 3=left_ear, 4=right_ear, 
            #       5=left_shoulder, 6=right_shoulder, 7=left_elbow, 8=right_elbow,
            #       9=left_wrist, 10=right_wrist, 11=left_hip, 12=right_hip,
            #       13=left_knee, 14=right_knee, 15=left_ankle, 16=right_ankle
            for i in range(num_people):
                # Direct mapping for first 17 keypoints
                for j in range(min(num_kpts, 17)):
                    combined[i, j, 0] = keypoints[i, j, 0]
                    combined[i, j, 1] = keypoints[i, j, 1]
                    combined[i, j, 2] = scores[i, j] if j < len(scores[i]) else 0.5
                
                # Synthesize additional keypoints (17-25)
                # 17=head (top of head, above nose)
                if num_kpts >= 3:
                    combined[i, 17, 0] = keypoints[i, 0, 0]  # Same x as nose
                    combined[i, 17, 1] = keypoints[i, 0, 1] - 30  # Above nose
                    combined[i, 17, 2] = scores[i, 0] * 0.8
                
                # 18=neck (midpoint of shoulders)
                if num_kpts >= 7:
                    combined[i, 18, 0] = (keypoints[i, 5, 0] + keypoints[i, 6, 0]) / 2
                    combined[i, 18, 1] = (keypoints[i, 5, 1] + keypoints[i, 6, 1]) / 2
                    combined[i, 18, 2] = (scores[i, 5] + scores[i, 6]) / 2
                
                # 19=hip (midpoint of hips)
                if num_kpts >= 13:
                    combined[i, 19, 0] = (keypoints[i, 11, 0] + keypoints[i, 12, 0]) / 2
                    combined[i, 19, 1] = (keypoints[i, 11, 1] + keypoints[i, 12, 1]) / 2
                    combined[i, 19, 2] = (scores[i, 11] + scores[i, 12]) / 2
                
                # Foot keypoints (20-25) - leave as zeros for COCO fallback
                # The triangulation will handle missing keypoints
        
        # Apply temporal smoothing to reduce jitter
        if self._prev_keypoints is not None and num_people > 0:
            # Simple exponential smoothing for stability
            alpha = 0.7  # Higher = more responsive, lower = smoother
            for i in range(min(num_people, len(self._prev_keypoints))):
                for j in range(26):
                    if combined[i, j, 2] > 0.3 and self._prev_keypoints[i, j, 2] > 0.3:
                        # Only smooth high-confidence keypoints
                        combined[i, j, 0] = alpha * combined[i, j, 0] + (1 - alpha) * self._prev_keypoints[i, j, 0]
                        combined[i, j, 1] = alpha * combined[i, j, 1] + (1 - alpha) * self._prev_keypoints[i, j, 1]
                    elif combined[i, j, 2] < 0.2 and self._prev_keypoints[i, j, 2] > 0.3:
                        # Use previous frame if current detection is poor
                        combined[i, j, :] = self._prev_keypoints[i, j, :]
        
        # Store for next frame
        self._prev_keypoints = combined.copy() if num_people > 0 else None
        
        # Compute bboxes from keypoints
        bboxes = np.zeros((num_people, 4))
        for i in range(num_people):
            valid_kpts = combined[i, :, 2] > 0.2  # Slightly higher threshold
            if valid_kpts.any():
                x_coords = combined[i, valid_kpts, 0]
                y_coords = combined[i, valid_kpts, 1]
                bboxes[i] = [x_coords.min(), y_coords.min(), x_coords.max(), y_coords.max()]
        
        track_ids = np.arange(num_people, dtype=np.uint32)
        
        return combined, bboxes, track_ids


def create_pose_tracker(det_model_path, pose_model_path, device_name):
    """Factory function to create a PoseTracker with fallback support."""
    global MMDEPLOY_AVAILABLE, RTMLIB_AVAILABLE, FORCE_RTMLIB
    
    # Use rtmlib (preferred for cross-platform compatibility)
    if RTMLIB_AVAILABLE:
        try:
            tracker = RTMLibPoseTrackerWrapper(det_model_path, pose_model_path, device_name)
            print("Using rtmlib PoseTracker (ONNX Runtime - cross-platform)")
            return tracker
        except Exception as e:
            print(f"rtmlib failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Fall back to mmdeploy if rtmlib fails
    if MMDEPLOY_AVAILABLE and not FORCE_RTMLIB:
        try:
            from mmdeploy_runtime import PoseTracker as MMDeployPoseTracker
            tracker = MMDeployPoseTracker(det_model_path, pose_model_path, device_name)
            _ = tracker.create_state()
            print("Using mmdeploy PoseTracker (TensorRT)")
            return tracker
        except Exception as e:
            print(f"mmdeploy PoseTracker failed: {e}")
    
    raise RuntimeError("No pose tracking backend available. Install rtmlib: pip install rtmlib")

# Platform detection for cross-platform executable handling
IS_WINDOWS = platform.system() == 'Windows'
IS_LINUX = platform.system() == 'Linux'

def get_executable_path(base_path, exe_name_without_ext):
    """Get the correct executable path based on platform."""
    if IS_WINDOWS:
        return os.path.join(base_path, f"{exe_name_without_ext}.exe")
    else:
        local_path = os.path.join(base_path, exe_name_without_ext)
        if os.path.exists(local_path) and os.access(local_path, os.X_OK):
            return local_path
        import shutil as sh
        system_exe = sh.which(exe_name_without_ext)
        if system_exe:
            return system_exe
        return local_path
####parameter
def rtm2json_rpjerror_with_calibrate_array(Video_path,out_video,rpj_all_dir,calibrate_array):
    halpe26_pose2sim_rpj_order = [16,-1,-1,-1,-1,20,17,21,18,22,19,8,2,9,3,10,4,-1,14,1,11,5,12,6,13,7]
    rpj_all = np.load(rpj_all_dir, allow_pickle=True)
    show_tr = 0.2
    camera_num = Video_path[-5:-4]

    
    cam_exclude = rpj_all['cam_choose']
    
    strongness = rpj_all['strongness_of_exclusion']

    temp_dir = os.getcwd()
    output_file = Path(out_video).parent.parent
    check_track = os.path.join(output_file,'pose-2d-tracked','pose_cam' + str(camera_num ) +'_json')
    output_file = os.path.join(output_file,'pose-2d-tracked','pose_cam' + str(camera_num ) +'_json')
    
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
    cam_num =int(camera_num)-1
    a = []
    
    # Check if calibrate_array is valid (2D array with proper shape)
    has_valid_calibration = (calibrate_array is not None and 
                             calibrate_array.ndim == 2 and 
                             not np.isnan(calibrate_array).any())
    
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
        
        if temp["people"][0] == {}:
            x = np.zeros(26)
            y = np.zeros(26)
            scores = np.zeros(26)
        else:
            x = np.array(temp["people"][0]['pose_keypoints_2d'])[dots]
            y = np.array(temp["people"][0]['pose_keypoints_2d'])[dots+1]
            scores =np.array(temp["people"][0]['pose_keypoints_2d'])[dots+2]
        
        keypointsx = np.append(keypointsx,[[x]],axis=0)
        keypointsy = np.append(keypointsy,[[y]],axis=0)
        keypoint_scores = np.append(keypoint_scores,[scores],axis=0)
        frame_temp = frame_temp+1

    keypointsx = np.delete(keypointsx,0,0)
    keypointsy = np.delete(keypointsy,0,0)
    keypoint_scores = np.delete(keypoint_scores,0,0)
    keypoints =np.concatenate((keypointsx, keypointsy), axis=1)
    track_state = np.delete(track_state,0,0)
    # import pdb;pdb.set_trace()

    while True:
        frame_count = count_frame
        if has_valid_calibration:
            cap.set(cv2.CAP_PROP_POS_FRAMES, calibrate_array[:,cam_num][frame_count])
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        ret, frame = cap.read()
        
       
        if (not ret) or (frame_count>=len(keypoint_scores)-1):
            break
        
        
        
        #第一個0代表第幾幀，第二個0代表畫面中的第幾+1位辨識體，第三個0代表第幾個節點
        count = 0
        for i in range(26):

            indicator = halpe26_pose2sim_rpj_order[i] 
            # import pdb;pdb.set_trace()
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

    
    
        # for i in range(len(rearranged_list)):
        #     print('a') # Update 'image_id' to match the index (starting from 1)
        #     #import pdb;pdb.set_trace()
    
    

    cap.release()
    out.release()
    clear_output(wait=False)


    os.chdir(temp_dir)
def timesync2rtm(video_folder,cam_num,opensim_folder,out_json):
    raw_video_path = video_folder
    calibrate_array = timesync_arrayoutput(video_folder,cam_num,opensim_folder)
    rtm_coord = []
    for i in range(cam_num):
        Video_path = os.path.join(raw_video_path,str(i+1)+'.mp4')
        out_dir = os.path.join(out_json,"pose_cam" + str(i+1) + "_json.json")
        rtm_coord.append(rtm2json_gpu_sync_calibrate(Video_path, out_dir, calibrate_array))
    #import pdb;pdb.set_trace()
    return rtm_coord,calibrate_array
def timesync_arrayoutput(video_folder,cam_num,opensim_folder):
    cap = []
    cap_array =[]
    end_record = []
    video_path=[]
    token =1
    for i in range(cam_num):
        cap.append(os.path.join(video_folder,str(i+1)+'_dates.npy'))
        video_path.append(os.path.join(video_folder,str(i+1)+'.mp4'))
        if os.path.isfile(os.path.join(video_folder,str(i+1)+'_dates.npy'))!=1:
            token =0
            break
        temp =np.load(os.path.join(video_folder,str(i+1)+'_dates.npy'))
        
        indices0 = np.where(temp[:,0] == 0)[0]
        indices1 = np.where(temp[:,1] == 0)[0]
        indices = np.intersect1d(indices0, indices1)
        temp =(temp[:,0]+temp[:,1])/2
        cap_array.append(temp*1000)
        end_record.append(min(indices)-1)
    TR = 50 ##ms
    diff_realworld_to_capread = 160 ##ms
    check =0
    calibrate = np.zeros((1,cam_num),int)
    mean_time  =[]
    if token ==1:
        while max(calibrate[0])<=min(end_record):
            aim = np.array([cap_array[0][calibrate[0][0]],cap_array[1][calibrate[0][1]],cap_array[2][calibrate[0][2]],cap_array[3][calibrate[0][3]]])
            mean_time.append(np.mean(aim)-diff_realworld_to_capread)
            try:
                calibrate_array = np.concatenate((calibrate_array, calibrate), axis=0)
            except NameError:
                calibrate_array = calibrate  # Initialize on the first loop
            for i in range(cam_num):
                print(max(aim)-aim[i])
                if max(aim)-aim[i]>TR:
                    calibrate[0][i] = calibrate[0][i]+1
                    check = check+1
            if check ==0:
                calibrate = calibrate+1
            
            
            check =0
        marker = np.array([-1])
        #import pdb;pdb.set_trace()
        if os.path.isfile(os.path.join(video_folder,'marker_stamp.npy')):
            marker = np.load(os.path.join(video_folder,'marker_stamp.npy'))*1000
            marker = (marker-min(mean_time))/1000+0.03333
            marker_path = Path(os.path.join(video_folder,'marker_stamp.npy'))
            marker_path.unlink()
        try:
            mean_time = (mean_time-min(mean_time))/1000+0.03333
        except:
            import pdb;pdb.set_trace()
        # Saving multiple arrays into a single file
        np.savez(os.path.join(opensim_folder ,'sync_time_marker.npz'), sync_timeline=np.array(mean_time), marker=marker)




    else:
        calibrate_array = np.array([np.nan])
        print('No Time Sync File')
    return calibrate_array

def timesync_video(video_folder,cam_num,opensim_folder):
    cap = []
    cap_array =[]
    end_record = []
    video_path=[]
    token =1
    for i in range(cam_num):
        cap.append(os.path.join(video_folder,str(i+1)+'_dates.npy'))
        video_path.append(os.path.join(video_folder,str(i+1)+'.mp4'))
        if os.path.isfile(os.path.join(video_folder,str(i+1)+'_dates.npy'))!=1:
            token =0
            break
        temp =np.load(os.path.join(video_folder,str(i+1)+'_dates.npy'))
        temp =(temp[:,0]+temp[:,1])/2
        indices = np.where(temp == 0)[0]
        cap_array.append(temp*1000)
        end_record.append(min(indices)-1)
    TR = 50 ##ms
    diff_realworld_to_capread = 160 ##ms
    check =0
    calibrate = np.zeros((1,cam_num),int)
    mean_time  =[]

    #calibrate_array =calibrate
    if token ==1:
        while max(calibrate[0])<=min(end_record):
            aim = np.array([cap_array[0][calibrate[0][0]],cap_array[1][calibrate[0][1]],cap_array[2][calibrate[0][2]],cap_array[3][calibrate[0][3]]])
            mean_time.append(np.mean(aim)-diff_realworld_to_capread)
            try:
                calibrate_array = np.concatenate((calibrate_array, calibrate), axis=0)
            except NameError:
                calibrate_array = calibrate  # Initialize on the first loop
            for i in range(cam_num):
                # print(max(aim)-aim[i])
                if max(aim)-aim[i]>TR:
                    calibrate[0][i] = calibrate[0][i]+1
                    check = check+1
            if check ==0:
                calibrate = calibrate+1
            check =0
        #import pdb; pdb.set_trace()
        #import pdb;pdb.set_trace()
        for i in range(cam_num):
        # Open the video
            # Get video properties
            cap = cv2.VideoCapture(video_path[i])
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            # Define the codec and create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec used to compress the frames
            out = cv2.VideoWriter(os.path.join(video_folder,str(i+1)+'sync.mp4'), fourcc, fps, (frame_width, frame_height))

            # Frame number you want to access
            for frame_index in range(np.shape(calibrate_array )[0]):
                frame_number =calibrate_array[frame_index][i]

                # Set the frame position
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

                # Read the frame
                ret, frame = cap.read()
                
                if ret:
                    out.write(frame)
                else:
                    print("Error: Could not read the frame.")

                # Release the video capture object
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            # Path to the file
            file_path = Path(video_path[i])
            cap_read_path = Path(os.path.join(video_folder,str(i+1)+'_dates.npy'))
            # Check if the file exists and delete it
            if file_path.exists():
                file_path.unlink()
                cap_read_path.unlink()
                print(f"File {file_path} has been deleted.")
            else:
                print(f"The file {file_path} does not exist.")
            os.rename(os.path.join(video_folder,str(i+1)+'sync.mp4'),video_path[i])

        marker = np.array([-1])
        #import pdb;pdb.set_trace()
        if os.path.isfile(os.path.join(video_folder,'marker_stamp.npy')):
            marker = np.load(os.path.join(video_folder,'marker_stamp.npy'))*1000
            marker = (marker-min(mean_time))/1000+0.03333
            marker_path = Path(os.path.join(video_folder,'marker_stamp.npy'))
            marker_path.unlink()

        mean_time = (mean_time-min(mean_time))/1000+0.03333

        array1 = np.array([1, 2, 3])
        array2 = np.array([4, 5, 6])

        # Saving multiple arrays into a single file
        np.savez(os.path.join(opensim_folder ,'sync_time_marker.npz'), sync_timeline=np.array(mean_time), marker=marker)
    else:
        print('No Time Sync File')
def rtm2json_gpu_sync_calibrate(Video_path, out_dir, calibrate_array):
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
    cam_num =int(os.path.splitext(os.path.basename(Video_path))[0])-1
    tracker = create_pose_tracker(det_model, pose_model, device_name)    
    sigmas = VISUALIZATION_CFG[skeleton_type]['sigmas']
    state = tracker.create_state(det_interval=1, det_min_bbox_size=100, keypoint_sigmas=sigmas)
    ###skeleton style
    data1 = []
    while True:
        success, frame = video.read()
        if not success:
            break
        results = tracker(state, frame, detect=-1)
        keypoints, bboxes, _ = results
        
        scores = keypoints[..., 2]
        keypoints = keypoints[..., :2]
        num_people = len(keypoints)
        
        if scores.size==0:
            temp = {"image_id" : frame_id, "people" : []}
            temp_keypoints = []
            for i in range(26):
                x = float(0)
                y = float(0)
                each_score = float(0)
                temp_keypoints.append(x)
                temp_keypoints.append(y)
                temp_keypoints.append(each_score)
            temp['people'].append({'person_id' : -1, "pose_keypoints_2d" : temp_keypoints})
            data1.append(temp)#存成相同格式
            
        else:
            temp = {"image_id" : frame_id, "people" : []}
            for person_id in range(num_people):
                temp_keypoints = []
                for i in range(26):
                    x = float(keypoints[person_id][i][0])
                    y = float(keypoints[person_id][i][1])
                    each_score = float(scores[person_id][i])
                    temp_keypoints.append(x)
                    temp_keypoints.append(y)
                    temp_keypoints.append(each_score)
                temp['people'].append({'person_id' : person_id, "pose_keypoints_2d" : temp_keypoints})
            data1.append(temp) #存成相同格式
        frame_id += 1
    
    # Check if calibrate_array is valid (2D array with proper shape)
    if (calibrate_array is not None and 
        calibrate_array.ndim == 2 and 
        not np.isnan(calibrate_array).any()):
        rearranged_list = [data1[i].copy() for i in calibrate_array[:,cam_num]]
        for i in range(len(rearranged_list)):
            rearranged_list[i]['image_id'] = i  # Update 'image_id' to match the index (starting from 1)
    else:
        rearranged_list = data1
    #import pdb;pdb.set_trace()
    with open(out_dir, "w") as save_file:
        json.dump(rearranged_list, save_file, indent = 6)  
    
    video.release()
    ### video output

    temp = rearranged_list
    os.chdir(AlphaPose_to_OpenPose )
    subprocess.run(['python', '-m','AlphaPose_to_OpenPose', '-i', out_dir])
    os.chdir(temp_dir)
    
    os.remove(out_dir)
    
    return rearranged_list
    
        
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
    tracker = create_pose_tracker(det_model, pose_model, device_name)    
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
        num_people = len(keypoints)
        
        if scores.size==0:
            temp = {"image_id" : frame_id, "people" : []}
            temp_keypoints = []
            for i in range(26):
                x = float(0)
                y = float(0)
                each_score = float(0)
                temp_keypoints.append(x)
                temp_keypoints.append(y)
                temp_keypoints.append(each_score)
            temp['people'].append({'person_id' : -1, "pose_keypoints_2d" : temp_keypoints})
            data1.append(temp)#存成相同格式
            
        else:
            temp = {"image_id" : frame_id, "people" : []}
            for person_id in range(num_people):
                temp_keypoints = []
                for i in range(26):
                    x = float(keypoints[person_id][i][0])
                    y = float(keypoints[person_id][i][1])
                    each_score = float(scores[person_id][i])
                    temp_keypoints.append(x)
                    temp_keypoints.append(y)
                    temp_keypoints.append(each_score)
                temp['people'].append({'person_id' : person_id, "pose_keypoints_2d" : temp_keypoints})
            data1.append(temp) #存成相同格式
        frame_id += 1
    
    ###將檔案放入json檔案中
    with open(out_dir, "w") as save_file:
        json.dump(data1, save_file, indent = 6)  
    
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
        
        num_people = len(temp[count_frame]['people'])
        
        for k in range(num_people):
            
            keypoints =[]
            keypoint_scores =[]
            
            if temp[count_frame]['people'][0]['person_id'] == -1:
                for i in range(26):
                    keypoints.append([0,0])
                    keypoint_scores.append(0) #呼叫該檔案中的keypoint_scores列表
            else:
                # print(len(temp), len(temp[count_frame]), len(temp[count_frame]['people']), k)
                # print(temp[count_frame]['people'])
                for i in range(26):
                    keypoints.append([temp[count_frame]['people'][k]["pose_keypoints_2d"][i*3],temp[count_frame]['people'][k]["pose_keypoints_2d"][i*3+1]])
                    keypoint_scores.append(temp[count_frame]['people'][k]["pose_keypoints_2d"][i*3+2]) #呼叫該檔案中的keypoint_scores列表
            

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
                    cv2.putText(frame, str(count_frame), (10, 10), font, fontScale, color, thickness, cv2.LINE_AA)
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

    # Convert unified JSON to per-frame OpenPose format
    os.chdir(AlphaPose_to_OpenPose)
    result = subprocess.run(['python', '-m','AlphaPose_to_OpenPose', '-i', out_dir], 
                           capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[WARNING] AlphaPose_to_OpenPose failed: {result.stderr}")
        print(f"[INFO] Keeping JSON file for debugging: {out_dir}")
    else:
        # Only remove the merged JSON if conversion succeeded
        try:
            os.remove(out_dir)
        except Exception as e:
            print(f"[WARNING] Could not remove temp JSON: {e}")
    os.chdir(temp_dir)
    



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

    #### json to openpose perframe - FIXED FORMAT to match AlphaPose_to_OpenPose expectations
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
        temp_keypoints = []
        # Handle case where instances might be empty
        if len(data[i].get('instances', [])) == 0:
            # No detection - create zero keypoints
            for k in range(26):
                temp_keypoints.append(0.0)
                temp_keypoints.append(0.0)
                temp_keypoints.append(0.0)
            data1.append({
                "image_id": i, 
                "people": [{"person_id": -1, "pose_keypoints_2d": temp_keypoints}]
            })
        else:
            # Process each detected person
            people_list = []
            for person_idx, instance in enumerate(data[i]['instances']):
                temp_keypoints = []
                for k in range(26):
                    score = instance['keypoint_scores'][k]
                    x = instance['keypoints'][k][0]
                    y = instance['keypoints'][k][1]
                    temp_keypoints.append(float(x))
                    temp_keypoints.append(float(y))
                    temp_keypoints.append(float(score))
                people_list.append({
                    "person_id": person_idx,
                    "pose_keypoints_2d": temp_keypoints
                })
            data1.append({"image_id": i, "people": people_list})

    result = data1

    save_file = open(out_dir, "w") 
    json.dump(result, save_file, indent = 6)  
    save_file.close() 
    
    os.chdir(AlphaPose_to_OpenPose)
    subprocess.run(['python', '-m','AlphaPose_to_OpenPose', '-i', out_dir])
    os.chdir(temp_dir)
    os.remove(json_dir)
    os.remove(out_dir)
    shutil.rmtree(output_file)
def rtm2json(Video_path,out_dir,out_video):
    try:
        rtm2json_gpu(Video_path,out_dir,out_video)
    except Exception as e:
        print(f"[WARNING] GPU pose estimation failed: {e}")
        print("[INFO] Falling back to CPU mode (MMPoseInferencer)...")
        traceback.print_exc()
        rtm2json_cpu(Video_path,out_dir,out_video)
    


def rtm2json_rpjerror(Video_path,out_video,rpj_all_dir):
    halpe26_pose2sim_rpj_order = [16,-1,-1,-1,-1,20,17,21,18,22,19,8,2,9,3,10,4,-1,14,1,11,5,12,6,13,7]
    rpj_all = np.load(rpj_all_dir, allow_pickle=True)
    show_tr = 0.2
    camera_num = Video_path[-5:-4]
    cam_exclude = rpj_all['cam_choose']
    
    strongness = rpj_all['strongness_of_exclusion']

    temp_dir = os.getcwd()
    output_file = Path(out_video).parent.parent
    check_track = os.path.join(output_file,'pose-2d-tracked','pose_cam' + str(camera_num ) +'_json')
    output_file = os.path.join(output_file,'pose-2d-tracked','pose_cam' + str(camera_num ) +'_json')
    
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
        
        if temp["people"][0] == {}:
            x = np.zeros(26)
            y = np.zeros(26)
            scores = np.zeros(26)
        else:
            x = np.array(temp["people"][0]['pose_keypoints_2d'])[dots]
            y = np.array(temp["people"][0]['pose_keypoints_2d'])[dots+1]
            scores =np.array(temp["people"][0]['pose_keypoints_2d'])[dots+2]
        
        keypointsx = np.append(keypointsx,[[x]],axis=0)
        keypointsy = np.append(keypointsy,[[y]],axis=0)
        keypoint_scores = np.append(keypoint_scores,[scores],axis=0)
        frame_temp = frame_temp+1

    keypointsx = np.delete(keypointsx,0,0)
    keypointsy = np.delete(keypointsy,0,0)
    keypoint_scores = np.delete(keypoint_scores,0,0)
    keypoints =np.concatenate((keypointsx, keypointsy), axis=1)
    track_state = np.delete(track_state,0,0)
    # import pdb;pdb.set_trace()

    while True:
        ret, frame = cap.read()
        frame_count = count_frame
        if (not ret) or (frame_count>=len(keypoint_scores)-1):
            break
        
        
        
        #第一個0代表第幾幀，第二個0代表畫面中的第幾+1位辨識體，第三個0代表第幾個節點
        count = 0
        for i in range(26):

            indicator = halpe26_pose2sim_rpj_order[i] 
            # import pdb;pdb.set_trace()
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
    openpose_exe = get_executable_path(os.path.join(openpose, 'bin'), 'OpenPoseDemo')
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

# video_folder=r'C:\Users\MyUser\Desktop\NTKCAP\Patient_data\v5\2025_02_13\raw_data\1\videos'
# cam_num = 4
# opensim_folder = r'C:\Users\MyUser\Desktop\NTKCAP\Patient_data\v5\2025_02_13\2025_02_13_17_19_calculated\1\opensim'
# out_json =r'C:\Users\MyUser\Desktop\NTKCAP\Patient_data\v5\2025_02_13\2025_02_13_17_19_calculated\1\pose-2d'
# rtm_coord = timesync2rtm(video_folder,cam_num,opensim_folder,out_json)
# with open(r"C:\Users\mauricetemp\Desktop\NTKCAP\Patient_data\multi_1p_exhibitiontest\2024_10_23\2024_11_28_17_54_calculated\outside4\my_list.json", "w") as f:
#     json.dump(rtm_coord, f)

# opensim_folder = r'C:\Users\MyUser\Desktop\NTKCAP\Patient_data\01\2025_02_11\2025_02_12_22_57_calculated\1\opensim'
# Video_path=r'C:\Users\MyUser\Desktop\NTKCAP\Patient_data\01\2025_02_11\raw_data\1\videos\4.mp4'
# Video_folder=r'C:\Users\MyUser\Desktop\NTKCAP\Patient_data\01\2025_02_11\raw_data\1\videos'
# cam_num = 4
# out_video=r'C:\Users\MyUser\Desktop\NTKCAP\Patient_data\01\2025_02_11\2025_02_12_22_57_calculated\1\videos_pose_estimation_repj_combine\4.mp4'
# rpj_all_dir=r'C:\Users\MyUser\Desktop\NTKCAP\Patient_data\01\2025_02_11\2025_02_12_22_57_calculated\1\User\reprojection_record.npz'
# calibrate_array = timesync_arrayoutput(Video_folder,cam_num,opensim_folder)
# rtm2json_rpjerror_with_calibrate_array(Video_path,out_video,rpj_all_dir,calibrate_array)