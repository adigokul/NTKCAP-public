#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
    ###########################################################################
    ## OTHER SHARED UTILITIES                                                ##
    ###########################################################################
    
    Functions shared between modules, and other utilities
    
'''

## INIT
import toml
import numpy as np
import re
import cv2

import matplotlib as mpl
mpl.use('qt5agg')
mpl.rc('figure', max_open_warning=0)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PyQt6.QtWidgets import QMainWindow, QApplication, QWidget, QTabWidget, QVBoxLayout
import sys


## AUTHORSHIP INFORMATION
__author__ = "David Pagnon"
__copyright__ = "Copyright 2021, Maya-Mocap"
__credits__ = ["David Pagnon"]
__license__ = "BSD 3-Clause License"
__version__ = '0.4'
__maintainer__ = "David Pagnon"
__email__ = "contact@david-pagnon.com"
__status__ = "Development"

## FUNCTIONS from multi
def retrieve_calib_params(calib_file):
    '''
    Compute projection matrices from toml calibration file.
    
    INPUT:
    - calib_file: calibration .toml file.
    
    OUTPUT:
    - S: (h,w) vectors as list of 2x1 arrays
    - K: intrinsic matrices as list of 3x3 arrays
    - dist: distortion vectors as list of 4x1 arrays
    - inv_K: inverse intrinsic matrices as list of 3x3 arrays
    - optim_K: intrinsic matrices for undistorting points as list of 3x3 arrays
    - R: rotation rodrigue vectors as list of 3x1 arrays
    - T: translation vectors as list of 3x1 arrays
    '''
    
    calib = toml.load(calib_file)

    S, K, dist, optim_K, inv_K, R, R_mat, T = [], [], [], [], [], [], [], []
    for c, cam in enumerate(calib.keys()):
        if cam != 'metadata':
            S.append(np.array(calib[cam]['size']))
            K.append(np.array(calib[cam]['matrix']))
            dist.append(np.array(calib[cam]['distortions']))
            optim_K.append(cv2.getOptimalNewCameraMatrix(K[c], dist[c], [int(s) for s in S[c]], 1, [int(s) for s in S[c]])[0])
            inv_K.append(np.linalg.inv(K[c]))
            R.append(np.array(calib[cam]['rotation']))
            R_mat.append(cv2.Rodrigues(R[c])[0])
            T.append(np.array(calib[cam]['translation']))
    calib_params = {'S': S, 'K': K, 'dist': dist, 'inv_K': inv_K, 'optim_K': optim_K, 'R': R, 'R_mat': R_mat, 'T': T}
            
    return calib_params

## FUNCTIONS
def computeP(calib_file, undistort=False):
    '''
    Compute projection matrices from toml calibration file.
    
    INPUT:
    - calib_file: calibration .toml file.
    
    OUTPUT:
    - P: projection matrix as list of arrays
    '''
    
    K, R, T, Kh, H = [], [], [], [], []
    P = []
    
    calib = toml.load(calib_file)
    for cam in list(calib.keys()):
        if cam != 'metadata':
            K = np.array(calib[cam]['matrix'])
            if undistort:
                S = np.array(calib[cam]['size'])
                dist = np.array(calib[cam]['distortions'])
                optim_K = cv2.getOptimalNewCameraMatrix(K, dist, [int(s) for s in S], 1, [int(s) for s in S])[0]
                Kh = np.block([optim_K, np.zeros(3).reshape(3,1)])
            else:
                Kh = np.block([K, np.zeros(3).reshape(3,1)])
            
            R, _ = cv2.Rodrigues(np.array(calib[cam]['rotation'])) # 旋轉向量轉成旋轉矩陣
            T = np.array(calib[cam]['translation'])
            H = np.block([[R,T.reshape(3,1)], [np.zeros(3), 1 ]]) # 把旋轉矩陣跟平移矩陣合併
            
            P.append(Kh.dot(H)) # 相機變換矩陣，空間中的3D座標會在2D畫面的哪個位置
   
    return P # 長度為相機數，一個相機有一組相機變換矩陣


def weighted_triangulation(P_all,x_all,y_all,likelihood_all):
    '''
    Triangulation with direct linear transform,
    weighted with likelihood of joint pose estimation.
    
    INPUTS:
    - P_all: list of arrays. Projection matrices of all cameras
    - x_all,y_all: x, y 2D coordinates to triangulate
    - likelihood_all: likelihood of joint pose estimation
    
    OUTPUT:
    - Q: array of triangulated point (x,y,z,1.)
    '''
    A = np.empty((0,4))
    for c in range(len(x_all)):
        P_cam = P_all[c]
        A = np.vstack((A, (P_cam[0] - x_all[c]*P_cam[2]) * likelihood_all[c] ))
        #print(np.shape((P_cam[0] - x_all[c]*P_cam[2]) * likelihood_all[c] ))
        A = np.vstack((A, (P_cam[1] - y_all[c]*P_cam[2]) * likelihood_all[c] ))
        #import pdb;pdb.set_trace()
    
    if np.shape(A)[0] >= 4:
        S, U, Vt = cv2.SVDecomp(A)
        V = Vt.T
        Q = np.array([V[0][3]/V[3][3], V[1][3]/V[3][3], V[2][3]/V[3][3], 1])
    else: 
        Q = np.array([0.,0.,0.,1])
        
    return Q
def weighted_triangulation_R(P_all,x_all,y_all,likelihood_all):
    '''
    Triangulation with direct linear transform,
    weighted with likelihood of joint pose estimation.
    
    INPUTS:
    - P_all: list of arrays. Projection matrices of all cameras
    - x_all,y_all: x, y 2D coordinates to triangulate
    - likelihood_all: likelihood of joint pose estimation
    
    OUTPUT:
    - Q: array of triangulated point (x,y,z,1.)
    '''
    
    A = np.empty((0,4))
    for c in range(len(x_all)):
        ch = c
        P_cam = P_all[c]
        A = np.vstack((A, (P_cam[0] - x_all[c]*P_cam[2]) * likelihood_all[c] ))
        #print(np.shape((P_cam[0] - x_all[c]*P_cam[2]) * likelihood_all[c] ))
        A = np.vstack((A, (P_cam[1] - y_all[c]*P_cam[2]) * likelihood_all[c] ))
        #import pdb;pdb.set_trace()
    
    if np.shape(A)[0] >= 4:
        S, U, Vt = cv2.SVDecomp(A)
        V = Vt.T
        Q = np.array([V[0][3]/V[3][3], V[1][3]/V[3][3], V[2][3]/V[3][3], 1])
    else: 
        Q = np.array([0.,0.,0.,1])
        
    return np.max(A@Q)
def computemap(calib_file):
    # Only map one camera because four cameras has same intrinsic matrix and dist. coeff
    calib = toml.load(calib_file)
    mappingx = []
    mappingy = []
    for cam in list(calib.keys()):
        #import pdb;pdb.set_trace()
        if cam != 'metadata':
            K = np.array(calib[cam]['matrix'])
            Kh = np.block([K, np.zeros(3).reshape(3,1)])
            R, _ = cv2.Rodrigues(np.array(calib[cam]['rotation']))
            T = np.array(calib[cam]['translation'])
            H = np.block([[R,T.reshape(3,1)], [np.zeros(3), 1 ]])
            mappingx,mappingy = cv2.initUndistortRectifyMap(np.array(calib[cam]['matrix']), np.array(calib[cam]['distortions']), None, np.array(calib[cam]['matrix']),np.array([1920,1080]), cv2.CV_32FC1)
            break
            #import pdb;pdb.set_trace()
   
    return mappingx,mappingy

def bilinear_interpolate(map, x, y):
    # Get integer coordinates surrounding the point
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    # Ensure coordinates are within bounds
    x0 = np.clip(x0,0, map.shape[1] - 2)
    y0 = np.clip(y0,0, map.shape[0] - 2)
    
    x1 = x0 + 1
    
    y1 = y0 + 1
    
    # Bilinear interpolation
    Ia = map[y0, x0]
    Ib = map[y0, x1]
    Ic = map[y1, x0]
    Id = map[y1, x1]

    # Interpolation weights
    wa = (x1 - x) * (y1 - y)
    wb = (x - x0) * (y1 - y)
    wc = (x1 - x) * (y - y0)
    wd = (x - x0) * (y - y0)

    # Calculate the interpolated value
    return wa * Ia + wb * Ib + wc * Ic + wd * Id

def undistort_points1(mappingx,mappingy,coords_2D_kpt):
    x_undistorted = []
    y_undistorted = []
    
    for x_coor,y_coor,liklihood in list(zip(coords_2D_kpt[0],coords_2D_kpt[1],coords_2D_kpt[2])):
        
        x_undistorted.append(bilinear_interpolate(mappingx,x_coor,y_coor))
        y_undistorted.append(bilinear_interpolate(mappingy,x_coor,y_coor))
        
    coords_2D_kpt_undistorted = np.array([np.array(x_undistorted),np.array(y_undistorted),np.array(coords_2D_kpt[2])])

    return coords_2D_kpt_undistorted
def reprojection(P_all, Q):
    '''
    Reprojects 3D point on all cameras.
    
    INPUTS:
    - P_all: list of arrays. Projection matrix for all cameras
    - Q: array of triangulated point (x,y,z,1.)

    OUTPUTS:
    - x_calc, y_calc: list of coordinates of point reprojected on all cameras
    '''
    #import pdb;pdb.set_trace()
    x_calc, y_calc = [], []
    for c in range(len(P_all)):  
        P_cam = P_all[c]
        x_calc.append(P_cam[0].dot(Q) / P_cam[2].dot(Q))
        y_calc.append(P_cam[1].dot(Q) / P_cam[2].dot(Q))
        
    return x_calc, y_calc
        

def euclidean_distance(q1, q2):
    '''
    Euclidean distance between 2 points (N-dim).
    
    INPUTS:
    - q1: list of N_dimensional coordinates of point
    - q2: idem

    OUTPUTS:
    - euc_dist: float. Euclidian distance between q1 and q2
    
    '''
    
    q1 = np.array(q1)
    q2 = np.array(q2)
    dist = q2 - q1
    
    euc_dist = np.sqrt(np.sum( [d**2 for d in dist]))
    
    return euc_dist

####### Add by maurice
def euclidean_dist_with_multiplication(q1,q2,Q,calib_file):
    
    '''
    Euclidean distance between 2 points (N-dim) with focal point and dist calibration
    
    INPUTS:
    - q1: list of N_dimensional coordinates of point
    - q2: idem

    OUTPUTS:
    - euc_dist: float. Euclidian distance between q1 and q2
    
    '''
    q1 = np.array(q1)
    q2 = np.array(q2)
    dist = q2 - q1
    C = find_camera_coordinate(calib_file)
    D = euclidean_distance(C, Q)

    index = D
    euc_dist = np.sqrt(np.sum( [d**2 for d in dist]))
    euc_dist = euc_dist*index
    return euc_dist
def camera2point_dist(Q,calib_file):

    C = find_camera_coordinate(calib_file)
    D = euclidean_distance(C, Q)
    return D
def find_camera_coordinate(calib_file):
    calib_file['rotation']
    R, _ = cv2.Rodrigues(np.array(calib_file['rotation']))
    T = np.array(calib_file['translation'])    
    R_t = np.transpose(R)
    C = -R_t.dot(T)    
    return C

def RT_qca2cv(r, t):

    '''
    Converts rotation R and translation T 
    from Qualisys object centered perspective
    to OpenCV camera centered perspective
    and inversely.

    Qc = RQ+T --> Q = R-1.Qc - R-1.T
    '''

    r = r.T
    t = - r.dot(t) 

    return r, t


def rotate_cam(r, t, ang_x=0, ang_y=0, ang_z=0):
    '''
    Apply rotations around x, y, z in cameras coordinates
    Angle in radians
    '''

    r,t = np.array(r), np.array(t)
    if r.shape == (3,3):
        rt_h = np.block([[r,t.reshape(3,1)], [np.zeros(3), 1 ]]) 
    elif r.shape == (3,):
        rt_h = np.block([[cv2.Rodrigues(r)[0],t.reshape(3,1)], [np.zeros(3), 1 ]])
    
    r_ax_x = np.array([1,0,0, 0,np.cos(ang_x),-np.sin(ang_x), 0,np.sin(ang_x),np.cos(ang_x)]).reshape(3,3) 
    r_ax_y = np.array([np.cos(ang_y),0,np.sin(ang_y), 0,1,0, -np.sin(ang_y),0,np.cos(ang_y)]).reshape(3,3)
    r_ax_z = np.array([np.cos(ang_z),-np.sin(ang_z),0, np.sin(ang_z),np.cos(ang_z),0, 0,0,1]).reshape(3,3) 
    r_ax = r_ax_z.dot(r_ax_y).dot(r_ax_x)

    r_ax_h = np.block([[r_ax,np.zeros(3).reshape(3,1)], [np.zeros(3), 1]])
    r_ax_h__rt_h = r_ax_h.dot(rt_h)
    
    r = r_ax_h__rt_h[:3,:3]
    t = r_ax_h__rt_h[:3,3]

    return r, t


def quat2rod(quat, scalar_idx=0):
    '''
    Converts quaternion to Rodrigues vector

    INPUT:
    - quat: quaternion. np.array of size 4
    - scalar_idx: index of scalar part of quaternion. Default: 0, sometimes 3

    OUTPUT:
    - rod: Rodrigues vector. np.array of size 3
    '''

    if scalar_idx == 0:
        w, qx, qy, qz = np.array(quat)
    if scalar_idx == 3:
        qx, qy, qz, w = np.array(quat)
    else:
        print('Error: scalar_idx should be 0 or 3')

    rodx = qx * np.tan(w/2)
    rody = qy * np.tan(w/2)
    rodz = qz * np.tan(w/2)
    rod = np.array([rodx, rody, rodz])

    return rod


def quat2mat(quat, scalar_idx=0):
    '''
    Converts quaternion to rotation matrix

    INPUT:
    - quat: quaternion. np.array of size 4
    - scalar_idx: index of scalar part of quaternion. Default: 0, sometimes 3

    OUTPUT:
    - mat: 3x3 rotation matrix
    '''

    if scalar_idx == 0:
        w, qx, qy, qz = np.array(quat)
    elif scalar_idx == 3:
        qx, qy, qz, w = np.array(quat)
    else:
        print('Error: scalar_idx should be 0 or 3')

    r11 = 1 - 2 * (qy**2 + qz**2)
    r12 = 2 * (qx*qy - qz*w)
    r13 = 2 * (qx*qz + qy*w)
    r21 = 2 * (qx*qy + qz*w)
    r22 = 1 - 2 * (qx**2 + qz**2)
    r23 = 2 * (qy*qz - qx*w)
    r31 = 2 * (qx*qz - qy*w)
    r32 = 2 * (qy*qz + qx*w)
    r33 = 1 - 2 * (qx**2 + qy**2)
    mat = np.array([r11, r12, r13, r21, r22, r23, r31, r32, r33]).reshape(3,3).T

    return mat


def natural_sort(list): 
    '''
    Sorts list of strings with numbers in natural order
    Example: ['item_1', 'item_2', 'item_10']
    Taken from: https://stackoverflow.com/a/11150413/12196632
    '''

    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)] 
    
    return sorted(list, key=alphanum_key)


## CLASSES
class plotWindow():
    '''
    Display several figures in tabs
    Taken from https://github.com/superjax/plotWindow/blob/master/plotWindow.py

    USAGE:
    pw = plotWindow()
    f = plt.figure()
    plt.plot(x1, y1)
    pw.addPlot("1", f)
    f = plt.figure()
    plt.plot(x2, y2)
    pw.addPlot("2", f)
    '''

    def __init__(self, parent=None):
        self.app = QApplication(sys.argv)
        self.MainWindow = QMainWindow()
        self.MainWindow.__init__()
        self.MainWindow.setWindowTitle("Multitabs figure")
        self.canvases = []
        self.figure_handles = []
        self.toolbar_handles = []
        self.tab_handles = []
        self.current_window = -1
        self.tabs = QTabWidget()
        self.MainWindow.setCentralWidget(self.tabs)
        self.MainWindow.resize(1280, 720)
        self.MainWindow.show()

    def addPlot(self, title, figure):
        new_tab = QWidget()
        layout = QVBoxLayout()
        new_tab.setLayout(layout)

        figure.subplots_adjust(left=0.1, right=0.99, bottom=0.1, top=0.91, wspace=0.2, hspace=0.2)
        new_canvas = FigureCanvas(figure)
        new_toolbar = NavigationToolbar(new_canvas, new_tab)

        layout.addWidget(new_canvas)
        layout.addWidget(new_toolbar)
        self.tabs.addTab(new_tab, title)

        self.toolbar_handles.append(new_toolbar)
        self.canvases.append(new_canvas)
        self.figure_handles.append(figure)
        self.tab_handles.append(new_tab)

    def show(self):
        self.app.exec_() 