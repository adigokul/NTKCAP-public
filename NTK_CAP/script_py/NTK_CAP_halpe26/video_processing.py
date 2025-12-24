
try:
    from mmpose.apis import MMPoseInferencer
except ImportError:
    MMPoseInferencer = None
import cv2
import json
import numpy as np
import os
from pathlib import Path
try:
    from IPython.display import clear_output
except ImportError:
    clear_output = lambda: None

# 视频处理相关函数，例如添加帧编号等
def add_frame_from_video(video_full_path, output_video):
    # Implementation placeholder
    pass

def openpose2json_video(video_full_path, output_video, json_s_folder):
    # Implementation placeholder
    pass
