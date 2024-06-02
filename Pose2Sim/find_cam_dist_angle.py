## INIT
import os
import glob
import fnmatch
import numpy as np
import json
import itertools as it
import pandas as pd
import toml
from tqdm import tqdm
from scipy import interpolate
from collections import Counter
import logging


import cv2


R, _ = cv2.Rodrigues(np.array([ 2.146, 0.44, -0.076]))
T = np.array([-0.564, 0.12, 2.999])
H = np.block([[R,T.reshape(3,1)], [np.zeros(3), 1 ]])
R = H[0:3,0:3]
T =H[0:3,3]
R_t = np.transpose(R)
C1 = -R_t.dot(T)
A1 = R_t.dot(np.array([[0],[0],[1]]))

R, _ = cv2.Rodrigues(np.array([ 1.97, -0.577, 0.345]))
T = np.array([-0.385, 0.666, 2.898])
H = np.block([[R,T.reshape(3,1)], [np.zeros(3), 1 ]])
R = H[0:3,0:3]
T =H[0:3,3]
R_t = np.transpose(R)
C2 = -R_t.dot(T)
A2 = R_t.dot(np.array([[0],[0],[1]]))
R, _ = cv2.Rodrigues(np.array([ 0.387, -2.552, 1.275]))
T = np.array([0.109, 0.098, 3.102])
H = np.block([[R,T.reshape(3,1)], [np.zeros(3), 1 ]])
R = H[0:3,0:3]
T =H[0:3,3]
R_t = np.transpose(R)
C3 = -R_t.dot(T)
A3 = R_t.dot(np.array([[0],[0],[1]]))
R, _ = cv2.Rodrigues(np.array([ 0.81, 2.504, -1.317]))
T = np.array([-0.384, 0.045, 3.308])
H = np.block([[R,T.reshape(3,1)], [np.zeros(3), 1 ]])
R = H[0:3,0:3]
T =H[0:3,3]
R_t = np.transpose(R)
C4 = -R_t.dot(T)
A4 = R_t.dot(np.array([[0],[0],[1]]))


