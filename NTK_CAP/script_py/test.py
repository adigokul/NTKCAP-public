# get camera number andwrite config

import os
import sys
import cv2
import json
import sys
import numpy as np
import multiprocessing
import multiprocessing, threading, logging, sys, traceback
import time
import keyboard
import shutil
from datetime import datetime
import subprocess
import easymocap
import import_ipynb
from xml_update import *
os.chdir(r'C:\Users\mauricetemp\Desktop\NTKCAP')
from Pose2Sim import Pose2Sim1
import inspect;
import  serial


dir = r'C:\Users\mauricetemp\Desktop\NTKCAP\Patient_data\ANN_FAKE\2024_09_23\2024_11_12_17_05_calculated\Apose'
os.chdir(dir)
Pose2Sim.filtering()
print(inspect.getfile(Pose2Sim))