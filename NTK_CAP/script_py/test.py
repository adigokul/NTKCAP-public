
r'''
import import_ipynb
from full_process import add_frame_from_video ,openpose2json_video

video_full_path=r'C:\Users\Hermes\Desktop\tsetinh\20231218post-in\walk1\Empty_project\videos\1_test.mp4'
output_video=r'C:\Users\Hermes\Desktop\tsetinh\20231218post-in\walk1\Empty_project\videos\1_test1.mp4'
add_frame_from_video(video_full_path,output_video)
'''
r'''
video_full_path =r'C:\Users\Hermes\Desktop\tsetinh\20231218post-in\walk1\Empty_project\videos\1.mp4'
output_video = r'C:\Users\Hermes\Desktop\tsetinh\20231218post-in\walk1\Empty_project\videos\1_test.mp4'
json_s_folder = r'C:\Users\Hermes\Desktop\tsetinh\20231218post-in\walk1\Empty_project\pose-2d\pose_cam1_json'
openpose2json_video(video_full_path,output_video,json_s_folder)
'''



from Pose2Sim import Pose2Sim


import os
os.chdir(r'F:\NTKCAP_1.1.1\NTKCAP_1.1.1\patient\testing\20230123\Apose\Empty_project')

Pose2Sim.personAssociation()
Pose2Sim.triangulation()
Pose2Sim.filtering()

