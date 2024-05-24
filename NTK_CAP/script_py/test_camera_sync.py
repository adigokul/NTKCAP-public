import numpy as np
import os
import  cv2
from pathlib import Path
video_folder =r'C:\Users\user\Desktop\NTKCAP\Patient_data\2d_test\2024_05_23\raw_data\walk9_TESTING\videos'
opensim_folder = video_folder
cam_num =4
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
calibrate_array =calibrate
if token ==1:
    while max(calibrate[0])<min(end_record):
        aim = np.array([cap_array[0][calibrate[0][0]],cap_array[1][calibrate[0][1]],cap_array[2][calibrate[0][2]],cap_array[3][calibrate[0][3]]])
        mean_time.append(np.mean(aim)-diff_realworld_to_capread)
        for i in range(cam_num):
            print(max(aim)-aim[i])
            if max(aim)-aim[i]>TR:
                calibrate[0][i] = calibrate[0][i]+1
                check = check+1
        if check ==0:
            calibrate = calibrate+1
        calibrate_array =np.concatenate((calibrate_array,calibrate), axis=0)
        check =0
else:
    print('No Time Sync File')
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
