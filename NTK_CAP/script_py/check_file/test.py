import import_ipynb
from full_process import *
import os

dir_temp = r'C:\Users\Hermes\Desktop\motion_temp\09_post_in'
openpose =  r'C:\Users\Hermes\Desktop\mmpose\py_file_yyver1\NTKCAP_1.1.1\NTKCAP_1.1.1\NTK_CAP\ThirdParty\openpose\openpose'

for walk  in os.listdir(dir_temp):
    video = os.listdir(os.path.join(dir_temp,walk,'Empty_project','videos'))
    if not os.path.exists(os.path.join(dir_temp,walk,'Empty_project','videos_output')):
        os.mkdir(os.path.join(dir_temp,walk,'Empty_project','videos_output'))
    if not os.path.exists(os.path.join(dir_temp,walk,'Empty_project','videos_json')):
        os.mkdir(os.path.join(dir_temp,walk,'Empty_project','videos_json'))
    for cam in range(4):
        if not os.path.exists(os.path.join(dir_temp,walk,'Empty_project','videos_json',str(cam))):
            os.mkdir(os.path.join(dir_temp,walk,'Empty_project','videos_json',str(cam)))
        json_s_folder = os.path.join(dir_temp,walk,'Empty_project','videos_json',str(cam))
        video_full_path = os.path.join(dir_temp,walk,'Empty_project','videos',video[cam])
        output_video = os.path.join(dir_temp,walk,'Empty_project','videos_output',video[cam])
        openpose2json_full(video_full_path,output_video,openpose,json_s_folder)

