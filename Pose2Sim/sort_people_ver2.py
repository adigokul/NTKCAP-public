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

json_name_list = ['movement_20_joint_frames_first_person_with_swing.json', 
                  'movement_20_joint_frames_second_person.json',
                  'movement_20_joint_frames_third_person_fixed.json',
                  'movement_20_joint_frames_fourth_person.json',
                  'movement_20_joint_frames_fifth_person.json']

connections = [
    (0, 1),  # head to neck
    (1, 2),  # neck to left shoulder
    (1, 3),  # neck to right shoulder
    (2, 4),  # left shoulder to left elbow
    (3, 5),  # right shoulder to right elbow
    (4, 6),  # left elbow to left hand
    (5, 7),  # right elbow to right hand
    (1, 15), # neck to upper spine
    (15, 16),# upper spine to mid spine
    (16, 17),# mid spine to lower spine
    (17, 9), # lower spine to left hip
    (17, 10),# lower spine to right hip
    (9, 11), # left hip to left knee
    (10, 12),# right hip to right knee
    (11, 13),# left knee to left foot
    (12, 14) # right knee to right foot
]
# {19,  "Hip"},
# {12, "RHip"},
# {14, "Rknee"},
# {16, "RAnkle"},
# {21, "RBigToe"},
# {23, "RSmallToe"},
# {25, "RHeel"},
# {11, "LHip"},
# {13, "LKnee"},
# {15, "LAnkle"},
# {20, "LBigToe"},    
# {22, "LSmallToe"},
# {24, "LHeel"},
# {18,  "Neck"},
# {0,  "Nose"},
# {17,  "Head"},
# {6,  "RShoulder"},
# {8,  "RElbow"},
# {10, "RWrist"},
# {5,  "LShoulder"},
# {7,  "LElbow"},
# {9,  "LWrist"},
#     {1,  "LEye"},
#     {2,  "REye"},
#     {3,  "LEar"},
#     {4,  "REar"},
connections = [
    (0, 17),
    (0, 18),
    (18, 6),
    (18, 5),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10), 
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
    (15, 22)
] 
    
    
keypoints_raw = []
for idx, json_name in enumerate(json_name_list):
    keypoints_raw.append([])
    with open(json_name, 'r') as file:
        data = json.load(file)
    keypoints_raw[idx].append(data)

keypoints_raw = np.array(keypoints_raw).reshape((len(json_name_list), 30, 20, 4))
keypoints_raw = keypoints_raw.transpose(1, 0, 2, 3)
max_len = 0
for i in range(len(keypoints_raw)):
    max_len = max(max_len, len([k for k in keypoints_raw[i] if not np.all(k == None)]))

keypoint_by_frame = np.full((len(keypoints_raw), max_len, 20, 4), np.nan)
for idx, frame in enumerate(keypoints_raw):
    np.random.shuffle(frame)
    indices = [i for i, value in enumerate([np.all(k == None) for k in frame]) if not value]
    for j in range(min(max_len, len(indices))):
        keypoint_by_frame[idx, j, :, :] = keypoints_raw[idx, indices[j], :, :]
    np.random.shuffle(keypoint_by_frame[idx])

tracker = {}
total_frame_num = len(keypoint_by_frame)

for f, kp_frame in enumerate(keypoint_by_frame): # frame
    nb_p_each_frame = nb_person_each_frame(kp_frame)
    
    if f != 0:
        personsIDs_comb = sorted(list(it.product(range(len(Q_old)),range(len(kp_frame))))) 
        dist = []
        dist += [euclidean_distance(Q_old[comb[0]], kp_frame[comb[1]]) for comb in personsIDs_comb]
        
        personsIDs_comb_new = []
        for i in range(0, len(personsIDs_comb), int(len(personsIDs_comb) / nb_p_each_frame)):
            
            dist_uncheck = dist[i:i+4]
            
            if np.all(np.isnan(dist_uncheck)):
                continue
            
            personsIDs_comb_new.append(personsIDs_comb[dist_uncheck.index(min(dist_uncheck))+i])
        
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
                if not matched: # new person
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


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
lines = {}
scatters = {}
# print(len(tracker['person1']['keypoints']))
# print(len(tracker['person2']['keypoints']))
# print(len(tracker['person3']['keypoints']))
# print(len(tracker['person4']['keypoints']))
# print(len(tracker['person5']['keypoints']))
for person_id, data in tracker.items():
    kps = data['keypoints'][0]
    x, y, z = kps[:, 0], kps[:, 1], kps[:, 2]
    scatters[person_id] = ax.scatter(x, y, z, 'o')  # 绘制关键点
    lines[person_id] = []
    for bone in connections:
        line, = ax.plot([kps[bone[0], 0], kps[bone[1], 0]],
                        [kps[bone[0], 1], kps[bone[1], 1]],
                        [kps[bone[0], 2], kps[bone[1], 2]], 'r-')
        lines[person_id].append(line)
        
def update(frame):
    for person_id, data in tracker.items():
        keypoints = data['keypoints'][frame]
        x, y, z = keypoints[:, 0], keypoints[:, 1], keypoints[:, 2]
        scatters[person_id]._offsets3d = (x, y, z)
        for l, bone in zip(lines[person_id], connections):
            l.set_data([keypoints[bone[0], 0], keypoints[bone[1], 0]],
                       [keypoints[bone[0], 1], keypoints[bone[1], 1]])
            l.set_3d_properties([keypoints[bone[0], 2], keypoints[bone[1], 2]])
    return list(scatters.values()) + [l for sublist in lines.values() for l in sublist]
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Z Coordinate')
ax.set_title('3D Human Keypoints Visualization')
ax.set_xlim([-100, 250])
ax.set_ylim([-100, 100])
ax.set_zlim([0, 200])
ani = FuncAnimation(fig, update, frames=range(30), blit=False, interval=1000/30)
plt.show(block=True)
