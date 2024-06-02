
import subprocess
import json
import os
from pathlib import Path
from IPython.display import clear_output

# 姿态估计相关函数
def rtm2json(Video_path, out_dir, out_video, dir_save):
    # Implementation from full_process.py
