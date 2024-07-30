import toml
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

body_26_kps = [
    {0,  "Nose"},
    {1,  "LEye"},
    {2,  "REye"},
    {3,  "LEar"},
    {4,  "REar"},
    {5,  "LShoulder"},
    {6,  "RShoulder"},
    {7,  "LElbow"},
    {8,  "RElbow"},
    {9,  "LWrist"},
    {10, "RWrist"},
    {11, "LHip"},
    {12, "RHip"},
    {13, "LKnee"},
    {14, "Rknee"},
    {15, "LAnkle"},
    {16, "RAnkle"},
    {17,  "Head"},
    {18,  "Neck"},
    {19,  "Hip"},
    {20, "LBigToe"},
    {21, "RBigToe"},
    {22, "LSmallToe"},
    {23, "RSmallToe"},
    {24, "LHeel"},
    {25, "RHeel"}
]
# reference : https://github.com/Fang-Haoshu/Halpe-FullBody
kp_connections = [
    (0, 18),
    (4, 2),
    (2, 0),
    (0, 1),
    (1, 3),
    (17, 0),
    (18, 6),
    (6, 8),
    (8, 10),
    (18, 5),
    (5, 7),
    (7, 9),
    (18, 19),
    (19, 12),
    (12, 14),
    (14, 16),
    (16, 25),
    (16, 23), 
    (16, 21),
    (19, 11),
    (11, 13),
    (13, 15), 
    (15, 24),
    (15, 20),
    (15, 22),
]

# data : C:\Users\Brian\NTKCAP\Patient_data\Patient_ID\2024_05_07\raw_data
calib_path = r'C:\Users\Brian\NTKCAP\Patient_data\Patient_ID\2024_05_07\raw_data\calibration\Calib.toml'
with open(calib_path, 'r') as file:
    calib_para = toml.load(file)

camera_poses = []
cam_num = len(calib_para) - 1
for i in range(cam_num):
    cam = calib_para[f"cam_{i+1}"]
    K = np.array(cam['matrix'])
    dist = np.array(cam['distortions'])
    rvec = np.array(cam['rotation'])
    tvec = np.array(cam['translation']).reshape((3, 1))
    R, _ = cv2.Rodrigues(rvec)
    C = -np.linalg.inv(R).dot(tvec)
    camera_poses.append({'R': R, 't': tvec, 'C': C})
    
base_points = np.array([pose['C'].flatten() for pose in camera_poses])
height = 4
top_points = base_points + np.array([0, 0, height])


vertices = np.vstack([base_points, top_points])
faces = [
    [vertices[0], vertices[1], vertices[3], vertices[2]],  # base
    [vertices[4], vertices[5], vertices[7], vertices[6]],  # top
    [vertices[0], vertices[1], vertices[5], vertices[4]],  
    [vertices[1], vertices[2], vertices[6], vertices[5]],   
    [vertices[2], vertices[3], vertices[7], vertices[6]],  
    [vertices[3], vertices[0], vertices[4], vertices[7]],  
]
fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')

fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')

ax1.add_collection3d(Poly3DCollection(faces, facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25))
ax1.scatter(base_points[:, 0], base_points[:, 1], base_points[:, 2], c='r', marker='o')
for i, pose in enumerate(camera_poses):
    C = pose['C'].flatten()
    ax2.scatter(C[0], C[1], C[2], c='r', marker='o')
    ax2.text(C[0], C[1], C[2], f'Camera {i+1}', color='red')

ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')

ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.set_zlim(0, 7)

plt.show()