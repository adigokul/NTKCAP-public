#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import shutil
PWD = sys.executable
PWD = os.path.abspath(os.path.join(PWD, os.pardir))
source = os.path.join(PWD, "intri_constant",'intri.yml')
target = os.path.join(PWD,'calibration','Calibration','IntrinsicCalibration','output')
if not os.path.isdir(target):
    os.mkdir(target) 
target = os.path.join(PWD,'calibration','Calibration','IntrinsicCalibration','output','intri.yml')
if os.path.isfile(target):
    os. remove(target) 
shutil.copyfile(source, target)

print('successfully_copy_intristic_parameter')
input("Press enter to exit;")

