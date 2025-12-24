import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools as it
import json
import random
from matplotlib.animation import FuncAnimation
def euclidean_distance(q1, q2):
    if q1[0][0] is np.nan or q2[0][0] is np.nan:
        return np.nan
    else:
        q1 = np.array(q1)
        q2 = np.array(q2)
        dist = q2 - q1
        
        euc_dist = np.sqrt(np.sum( [d**2 for d in dist]))
        
        return euc_dist

def nb_person_each_frame(kp_frame):
    return len([nb for nb in kp_frame if nb is not np.nan]) 

# def init():
#     for line in lines_first_person:
#         line.set_data([], [])
#         line.set_3d_properties([])
#     return lines_first_person

# def update(frame):
#     # 更新第一个人的关节点
#     for i, line in enumerate(lines_first_person):
#         x, y, z = data_first_person[frame][i][:3]
#         line.set_data([x], [y])
#         line.set_3d_properties([z])
        
#     return lines_first_person

def update_tracker():
    pass
keypoints_p1 = np.array([[  0,   0, 180,   1],
                        [  0,   0, 160,   1],
                        [-10,   0, 150,   1],
                        [ 10,   0, 150,   1],
                        [-20,   0, 130,   1],
                        [ 20,   0, 130,   1],
                        [-30,   0, 110,   1],
                        [ 30,   0, 110,   1],
                        [  0,   0, 120,   1],
                        [-10,   0,  90,   1],
                        [ 10,   0,  90,   1],
                        [-10,   0,  60,   1],
                        [ 10,   0,  60,   1],
                        [-10,   0,  30,   1],
                        [ 10,   0,  30,   1],
                        [  0,   0, 140,   1],
                        [  0,   0, 110,   1],
                        [  0,   0,  80,   1],
                        [-10,   0,  75,   1],
                        [ 10,   0,  75,   1]])
keypoints_p1_step1 = np.array(
            [[  0,   0, 180,   1],
            [  0,   0, 160,   1],
            [-10,   0, 150,   1],
            [ 10,   0, 150,   1],
            [-20,   0, 130,   1],
            [ 20,   0, 130,   1],
            [-30,   0, 110,   1],
            [ 30,   0, 110,   1],
            [  0,   0, 120,   1],
            [-10,   0,  90,   1],
            [ 10,   0,  90,   1],
            [ -5,   0,  60,   1],  
            [  5,   0,  60,   1],  
            [  0,   0,  30,   1],  
            [  0,   0,  30,   1],  
            [  0,   0, 140,   1],
            [  0,   0, 110,   1],
            [  0,   0,  80,   1],
            [ -5,   0,  75,   1],  
            [  5,   0,  75,   1]])
keypoints_p2 = np.array(
            [[ 20,   0, 180,   1],
            [ 20,   0, 160,   1],
            [ 10,   0, 150,   1],
            [ 30,   0, 150,   1],
            [  0,   0, 130,   1],
            [ 40,   0, 130,   1],
            [-10,   0, 110,   1],
            [ 50,   0, 110,   1],
            [ 20,   0, 120,   1],
            [ 10,   0,  90,   1],
            [ 30,   0,  90,   1],
            [ 10,   0,  60,   1],
            [ 30,   0,  60,   1],
            [ 10,   0,  30,   1],
            [ 30,   0,  30,   1],
            [ 20,   0, 140,   1],
            [ 20,   0, 110,   1],
            [ 20,   0,  80,   1],
            [ 10,   0,  75,   1],
            [ 30,   0,  75,   1]]
)
keypoints_p2_step1 = np.array(
            [[ 20,   0, 180,   1],
            [ 20,   0, 160,   1],
            [ 10,   0, 150,   1],
            [ 30,   0, 150,   1],
            [  0,   0, 130,   1],
            [ 40,   0, 130,   1],
            [-10,   0, 110,   1],
            [ 50,   0, 110,   1],
            [ 20,   0, 120,   1],
            [ 10,   0,  90,   1],
            [ 30,   0,  90,   1],
            [ 15,   0,  60,   1],  
            [ 25,   0,  60,   1],  
            [ 20,   0,  30,   1],  
            [ 20,   0,  30,   1],  
            [ 20,   0, 140,   1],
            [ 20,   0, 110,   1],
            [ 20,   0,  80,   1],
            [ 15,   0,  75,   1],  
            [ 25,   0,  75,   1]])
keypoints_p3 = np.array(
            [[ 50, -20, 180,   1],
            [ 50, -20, 160,   1],
            [ 40, -20, 150,   1],
            [ 60, -20, 150,   1],
            [ 30, -20, 130,   1],
            [ 70, -20, 130,   1],
            [ 20, -20, 110,   1],
            [ 80, -20, 110,   1],
            [ 50, -20, 120,   1],
            [ 40, -20,  90,   1],
            [ 60, -20,  90,   1],
            [ 40, -20,  60,   1],
            [ 60, -20,  60,   1],
            [ 40, -20,  30,   1],
            [ 60, -20,  30,   1],
            [ 50, -20, 140,   1],
            [ 50, -20, 110,   1],
            [ 50, -20,  80,   1],
            [ 40, -20,  75,   1],
            [ 60, -20,  75,   1]])
# connections = [
#     (0, 1),  # head to neck
#     (1, 2),  # neck to left shoulder
#     (1, 3),  # neck to right shoulder
#     (2, 4),  # left shoulder to left elbow
#     (3, 5),  # right shoulder to right elbow
#     (4, 6),  # left elbow to left hand
#     (5, 7),  # right elbow to right hand
#     (1, 15), # neck to upper spine
#     (15, 16),# upper spine to mid spine
#     (16, 17),# mid spine to lower spine
#     (17, 9), # lower spine to left hip
#     (17, 10),# lower spine to right hip
#     (9, 11), # left hip to left knee
#     (10, 12),# right hip to right knee
#     (11, 13),# left knee to left foot
#     (12, 14) # right knee to right foot
# ]

# skeletons = [
#     (0, 17), (0, 18), (18, 6), (18, 5), (5, 7), (7, 9), (6, 8), (8, 10),
#     (18, 19), (19, 12), (12, 14), (14, 16), (16, 20), (16, 18), (16, 15),
#     (19, 11), (11, 13), (13, 15), (15, 20), (15, 18), (15, 17)
# ]
skeletons = [
    (0, 1), (1, 2), (2, 3), (3, 4), (3, 5), (3, 6), (0, 7), (7, 8),
    (8, 9), (9, 10), (9, 11), (9, 12), (13, 0),(13, 14), (14, 15), (13, 16),
    (16, 17), (17, 18), (13, 19), (19, 20), (20, 21)]

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# person1
# x1, y1, z1, _ = keypoints_p1.T
# x1_1, y1_1, z1_1, _ = keypoints_p1_step1.T

# ax.scatter(x1, y1, z1, c='red', marker='o')
# ax.scatter(x1_1, y1_1, z1_1, c='red', marker='o')

# person2
# x2, y2, z2, _ = keypoints_p2.T
# x2_1, y2_1, z2_1, _ = keypoints_p2_step1.T

# ax.scatter(x2, y2, z2, c='yellow', marker='o')
# ax.scatter(x2_1, y2_1, z2_1, c='yellow', marker='o')

# person3
x3, y3, z3, _ = keypoints_p3.T
# x2_1, y2_1, z2_1, _ = keypoints_p2_step1.T

# ax.scatter(x3, y3, z3, c='orange', marker='o')
# ax.scatter(x2_1, y2_1, z2_1, c='yellow', marker='o')

# for start, end in connections:
#     ax.plot([x1[start], x1[end]], [y1[start], y1[end]], [z1[start], z1[end]], 'blue')
    
# for start, end in connections:
#     ax.plot([x1_1[start], x1_1[end]], [y1_1[start], y1_1[end]], [z1_1[start], z1_1[end]], 'green')
    
# for start, end in connections:
#     ax.plot([x2[start], x2[end]], [y2[start], y2[end]], [z2[start], z2[end]], 'green')
    
# for start, end in connections:
#     ax.plot([x2_1[start], x2_1[end]], [y2_1[start], y2_1[end]], [z2_1[start], z2_1[end]], 'green')
    
# for start, end in connections:
#     ax.plot([x3[start], x3[end]], [y3[start], y3[end]], [z3[start], z3[end]], 'gray')
# ax.set_xlabel('X Coordinate')
# ax.set_ylabel('Y Coordinate')
# ax.set_zlabel('Z Coordinate')
# ax.set_title('3D Human Keypoints Visualization')
# ax.set_xlim([-100, 100])
# ax.set_ylim([-100, 100])
# ax.set_zlim([0, 200])
# plt.show()

# Real data
# real_person1_json_path = r"C:\Users\Brian\NTKCAP\Patient_data\0715MULTI_singletest\2024_07_15\2024_07_17_17_41_calculated\test1\person1.json"
# real_person2_json_path = r"C:\Users\Brian\NTKCAP\Patient_data\0715MULTI_singletest\2024_07_15\2024_07_17_17_41_calculated\test1\person2.json"
# with open(real_person1_json_path, 'r') as file:
#     real_data_1 = json.load(file)
# with open(real_person2_json_path, 'r') as file:
#     real_data_2 = json.load(file)

kp_frame1 = []
kp_frame1 += [keypoints_p1, keypoints_p2]
kp_frame1 = np.array(kp_frame1)
kp_frame2 = []
kp_frame2 += [keypoints_p1_step1, keypoints_p2_step1, keypoints_p3]
kp_frame2 = np.array(kp_frame2)
nb_keypoints = 20

json_name_list = ['movement_20_joint_frames_first_person_with_swing.json', 
                  'movement_20_joint_frames_second_person.json',
                  'movement_20_joint_frames_third_person_fixed.json',
                  'movement_20_joint_frames_fourth_person.json',
                  'movement_20_joint_frames_fifth_person.json']
# dist_threshold = 20
keypoints_raw = []
for idx, json_name in enumerate(json_name_list):
    keypoints_raw.append([])
    with open(json_name, 'r') as file:
        data = json.load(file)
    keypoints_raw[idx].append(data)

keypoints_raw = np.array(keypoints_raw).reshape((len(json_name_list), 30, 20, 4))

keypoint_by_frame = np.full((30, 5, 20, 4), np.nan)
for j in range(len(json_name_list)):
    for i in range(30):
        keypoint_by_frame[i, j, :, :] = keypoints_raw[j, i, :, :]
        
keypoints_raw = keypoint_by_frame
max_len = 0

for i in range(len(keypoints_raw)):
    max_len = max(max_len, len([k for k in keypoints_raw[i] if not np.any(np.isnan(k))]))
    
# np.random.seed(42)
keypoint_by_frame = np.full((len(keypoints_raw), max_len, 20, 4), np.nan)
for idx, frame in enumerate(keypoints_raw):
    np.random.shuffle(frame)
    indices = [i for i, value in enumerate([np.any(np.isnan(k)) for k in frame]) if not value]
    for j in range(min(max_len, len(indices))):
        keypoint_by_frame[idx, j, :, :] = keypoints_raw[idx, indices[j], :, :]
    np.random.shuffle(keypoint_by_frame[idx])

# for frame in keypoint_by_frame:
#     np.random.shuffle(frame)
    
#     for person in frame:
#         for joints in person:
#             nan_indices = np.isnan(joints)
#             sorted_indices = np.argsort(nan_indices)
#             joints[:] = joints[sorted_indices]
# for i in range(len(keypoint_by_frame)):
#     print([np.any(np.isnan(k)) for k in keypoint_by_frame[i]])            
# for i in range(len(keypoint_by_frame)):
    # print(keypoint_by_frame[i][4])
# personsIDs_comb = sorted(list(it.product(range(5),range(5))))
# keypoint_by_frame = keypoint_by_frame[:, :max_len, :, :]

tracker = {}
total_frame_num = len(keypoint_by_frame)

for f, kp_frame in enumerate(keypoint_by_frame): # frame
    nb_p_each_frame = nb_person_each_frame(kp_frame)
    
    if f != 0:
        personsIDs_comb = sorted(list(it.product(range(len(Q_old)),range(len(kp_frame))))) # 不事先知道這次trial總共有幾個人
        dist = []
        dist += [euclidean_distance(Q_old[comb[0]], kp_frame[comb[1]]) for comb in personsIDs_comb]
        
        personsIDs_comb_new = []
        for i in range(0, len(personsIDs_comb), int(len(personsIDs_comb) / nb_p_each_frame)):
            
            dist_unckeck = dist[i:i+4]
            
            if np.all(np.isnan(dist_unckeck)):
                continue
            
            personsIDs_comb_new.append(personsIDs_comb[dist_unckeck.index(min(dist_unckeck))+i])
        
        for idx, c in enumerate(personsIDs_comb_new):
            
            if f == 1: # first frame which has one previous frame
                
                tracker[f"person{idx+1}"] = {
                    'id' : c[1],
                    'matching status' : False,
                    'keypoints' : np.expand_dims(Q_old[c[0]], axis=0)
                }
                tracker[f"person{idx+1}"]['keypoints'] = np.append(tracker[f"person{idx+1}"]['keypoints'], np.expand_dims(kp_frame[c[1]], axis=0), axis=0)
                
            else:
                
                matched = False
                for idx_pre, (person, p_pre) in enumerate(tracker.items()):
                    if p_pre['id'] == -1 : continue
                        
                    if c[0] == p_pre['id'] and tracker[person]['matching status'] == True:
                        
                        matched = True
                        tracker[person]['id'] = c[1]
                        tracker[person]['matching status'] = False
                        tracker[person]['keypoints'] = np.append(tracker[person]['keypoints'], np.expand_dims(kp_frame[c[1]], axis=0), axis=0)
                        break
                if not matched: # 有新的人
                    nan_fill = np.full((f-1, 20, 4), np.nan)
                    new_name = f"person{len(tracker)+1}"
    
                    tracker[new_name] = {
                        'id':c[1],
                        'matching status' : False,
                        'keypoints':np.append(nan_fill, np.expand_dims(Q_old[c[0]], axis=0), axis=0)
                    }
                    tracker[new_name]['keypoints'] = np.append(tracker[new_name]['keypoints'], np.expand_dims(kp_frame[c[1]], axis=0), axis=0)
        
        for i in range(len(tracker)):
            if tracker[f"person{i+1}"]['matching status'] == True:
                nan_fill = np.full((total_frame_num - len(tracker[f"person{i+1}"]['keypoints']), 20, 4), np.nan)
                tracker[f"person{i+1}"]['id'] = -1
                tracker[f"person{i+1}"]['keypoints'] = np.append(tracker[f"person{i+1}"]['keypoints'], nan_fill, axis=0)
            tracker[f"person{i+1}"]['matching status'] = True
        comb_pre_frame = personsIDs_comb_new
        
    Q_old = kp_frame




# print(tracker['person1']['keypoints'].shape)
# print(tracker['person2']['keypoints'].shape)
# print(tracker['person3']['keypoints'].shape)
# print(tracker['person4']['keypoints'].shape)
# print(tracker['person5']['keypoints'].shape)
# print(len(tracker))

# real_data_1 = np.array(real_data_1).reshape(183,22,3)
# real_data_2 = np.array(real_data_2).reshape(183,22,3) 
for i in range(len(tracker)):
    tracker[f'person{i+1}']['keypoints'] = []

# real data
json_path = r"C:\Users\Brian\NTKCAP\Patient_data\0715MULTI_test6\2024_07_15\2024_07_23_17_59_calculated\test6\kp_data.json"
with open(json_path, 'r') as file:
    data = json.load(file)
    
import toml
import cv2
def calculate_camera_position(calib_file=r'C:\Users\Brian\NTKCAP\NTK_CAP\template\Empty_project\User\Config.toml'):
    calib = toml.load(calib_file)
    camera_positions = []

    for cam in calib.keys():
        if cam != 'metadata':
            rotation_vector = np.array(calib[cam]['rotation'])
            translation_vector = np.array(calib[cam]['translation'])

            # Skip cameras with failed calibration (zero extrinsic parameters)
            if np.allclose(rotation_vector, [0, 0, 0]) and np.allclose(translation_vector, [0, 0, 0]):
                print(f"[INFO] Skipping camera {cam} with failed calibration (zero extrinsic parameters)")
                continue

            R, _ = cv2.Rodrigues(rotation_vector)
            cam_center = -np.dot(R.T, translation_vector)
            camera_positions.append(cam_center)

    return camera_positions
# tracker[f'person1']['keypoints'] = real_data_1
# tracker[f'person2']['keypoints'] = real_data_2
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# lines = {}
# scatters = {}


# skeleton = [(i-1, j-1) for i, j in skeleton]

# 创建3D绘图
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# scatters = []
# for nb_per in data:
#     x_coords, y_coords, z_coords = [kp[0] for kp in nb_per], [kp[1] for kp in nb_per], [kp[2] for kp in nb_per]
#     scatter = ax.scatter(x_coords, y_coords, z_coords)
#     scatters.append(scatter)

# def update(frame):
#     for scatter, real_data in zip(scatters, data):
#         keypoints = real_data[frame]
#         x_coords, y_coords, z_coords = [kp[0] for kp in keypoints], [kp[1] for kp in keypoints], [kp[2] for kp in keypoints]
#         scatter._offsets3d = (x_coords, y_coords, z_coords)
#     return scatters
# ani = FuncAnimation(fig, update, frames=183, interval=1000/30, blit=True)
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_xlim(-5,5)
# ax.set_ylim(-5,5)
# ax.set_zlim(-5,5)
# plt.show()


import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 假设你已经有了 data 和 skeleton 变量
# data = [...your list of data...]  # 你的数据
# skeletons = [...your skeleton connections...]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

num_frames = len(data[0])
num_people = len(data)

scatters = {}
lines = {}
camera_positions = calculate_camera_position()
camera_scatter = ax.scatter(
    [pos[0] for pos in camera_positions],
    [pos[1] for pos in camera_positions],
    [pos[2] for pos in camera_positions],
    c='blue', marker='^', s=100, label='Camera'
)
# 初始化散点图和骨架线条
for person_id in range(num_people):
    kps = data[person_id][0]  # 第0帧的数据
    x, y, z = [kp[0] for kp in kps], [kp[1] for kp in kps], [kp[2] for kp in kps]
    scatters[person_id] = ax.scatter(x, y, z, 'o')  # 绘制关键点
    lines[person_id] = []
    for bone in skeletons:
        line, = ax.plot([kps[bone[0]][0], kps[bone[1]][0]],
                        [kps[bone[0]][1], kps[bone[1]][1]],
                        [kps[bone[0]][2], kps[bone[1]][2]], 'r-')
        lines[person_id].append(line)

def update(frame):
    for person_id in range(num_people):
        keypoints = data[person_id][frame]
        x, y, z = [kp[0] for kp in keypoints], [kp[1] for kp in keypoints], [kp[2] for kp in keypoints]
        scatters[person_id]._offsets3d = (x, y, z)
        for l, bone in zip(lines[person_id], skeletons):
            l.set_data([keypoints[bone[0]][0], keypoints[bone[1]][0]],
                       [keypoints[bone[0]][1], keypoints[bone[1]][1]])
            l.set_3d_properties([keypoints[bone[0]][2], keypoints[bone[1]][2]])
    return list(scatters.values()) + [l for sublist in lines.values() for l in sublist]

# 设置坐标轴标签和标题
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Z Coordinate')
ax.set_title('3D Human Keypoints Visualization')
ax.set_xlim([-5, 5])
ax.set_ylim([-5, 5])
ax.set_zlim([-5, 5])

ani = FuncAnimation(fig, update, frames=num_frames, interval=1000/30)
plt.show()







# # 初始化图形
# kps = real_data_1[0]
# x, y, z = kps[:, 0], kps[:, 1], kps[:, 2]
# scatter = ax.scatter(x, y, z)
# lines = []
# for bone in skeleton:
#     line, = ax.plot([kps[bone[0], 0], kps[bone[1], 0]],
#                     [kps[bone[0], 1], kps[bone[1], 1]],
#                     [kps[bone[0], 2], kps[bone[1], 2]], 'r-')
#     lines.append(line)

# # 更新函数
# def update(frame):
#     keypoints = real_data_1[frame]
#     x, y, z = keypoints[:, 0], keypoints[:, 1], keypoints[:, 2]
#     scatter._offsets3d = (x, y, z)
#     for l, bone in zip(lines, skeleton):
#         l.set_data([keypoints[bone[0], 0], keypoints[bone[1], 0]],
#                    [keypoints[bone[0], 1], keypoints[bone[1], 1]])
#         l.set_3d_properties([keypoints[bone[0], 2], keypoints[bone[1], 2]])
#     return [scatter] + lines

# # 创建动画
# frames_nb = real_data_1.shape[0]
# ani = FuncAnimation(fig, update, frames=frames_nb, interval=100, blit=True)

# # 显示动画
# plt.show()


