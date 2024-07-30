import json
import numpy as np
import itertools as it
import toml
js_path = r"C:\Users\陳柏宏\pose2sim\Pose2Sim\S00_Demo_BatchSession\S00_P01_MultiParticipants\S00_P01_T02_Participants1-2\pose\cam01_json\cam01.0000.json"
with open(js_path, 'r') as json_f:
    data = json.load(json_f)
    json_data = []
for people in range(len(data['people'])):
    if len(data['people'][people]['pose_keypoints_2d']) < 3: continue
    else:
        json_data.append(data['people'][people]['pose_keypoints_2d'])
temp = []
for i in range(2):
    temp.append(json_data)

persons_per_view = [0] + [len(j) for j in temp]

#all_json_data_f : all_json_data_f.append(read_json(js_file)) # len=4, 每一個包含偵測到的點座標+信心
reconstruction_error_threshold = 0.1
distance = np.zeros((persons_per_view[-1], persons_per_view[-1])) + 2*reconstruction_error_threshold
# for compared_cam0, compared_cam1 in it.combinations(range(4), 2):
#     print(compared_cam0)
#     print(compared_cam1)
cum_persons_per_view = [0,2,4,6,8]
circ_constraint = np.identity(cum_persons_per_view[-1])
for i in range(len(cum_persons_per_view)-1):
    circ_constraint[cum_persons_per_view[i]:cum_persons_per_view[i+1], cum_persons_per_view[i+1]:cum_persons_per_view[-1]] = 1
    circ_constraint[cum_persons_per_view[i+1]:cum_persons_per_view[-1], cum_persons_per_view[i]:cum_persons_per_view[i+1]] = 1
def SVT(matrix, threshold):
    '''
    Find a low-rank approximation of the matrix using Singular Value Thresholding.

    INPUTS:
    - matrix: matrix to decompose
    - threshold: threshold for singular values

    OUTPUT:
    - matrix_thresh: low-rank approximation of the matrix
    '''
    
    U, s, Vt = np.linalg.svd(matrix) # decompose matrix
    s_thresh = np.maximum(s - threshold, 0) # set smallest singular values to zero
    matrix_thresh = U @ np.diag(s_thresh) @ Vt # recompose matrix

    return matrix_thresh

affinity = np.array([
    [0.25762177, 0, 0.01177014, 0.21563733, 0.32582933, 0.33939304, 0.46193657, 0.72774456],
    [0, 0.98914398, 0.15560496, 0.58599185, 0.56069029, 0.32317042, 0.45988597, 0.70719398],
    [0.86756702, 0.30926172, 0.14500206, 0, 0.24314083, 0.22962045, 0.14984241, 0.66602188],
    [0.61076104, 0.31000744, 0, 0.17285881, 0.82416331, 0.31078014, 0.96115138, 0.97671699],
    [0.07709458, 0.57433529, 0.82356954, 0.63607664, 0.76721703, 0, 0.78850917, 0.06715013],
    [0.24562359, 0.32044515, 0.85008281, 0.7899356, 0, 0.12551507, 0.26993709, 0.90684945],
    [0.73093103, 0.27604563, 0.71324689, 0.396357, 0.76350014, 0.08725839, 0.49456697, 0],
    [0.59087615, 0.04790774, 0.12320599, 0.41979629, 0.95485519, 0.48837971, 0, 0.9498026]
])
new_aff = affinity.copy()
N = new_aff.shape[0]
# print(N) = 8
index_diag = np.arange(N) 
# print(index_diag) [0 1 2 3 4 5 6 7]
new_aff[index_diag, index_diag] = 0.
# print(new_aff) 對角線數值也設為0

w_sparse = 0.1 
max_iter = 20
w_rank = 50
tol = 1e-4
Y = np.zeros_like(new_aff)
# print(Y) 大小跟aff一樣的零矩陣 
W = w_sparse - new_aff 
# print(W) 0.1 分別扣每個數值
mu = 64 
for iter in range(max_iter):
    # print(iter) 0~19
    new_aff0 = new_aff.copy()
    
    Q = new_aff + Y*1.0/mu
    Q = SVT(Q,w_rank/mu)
    new_aff = Q - (W + Y)/mu

    # Project X onto dimGroups
    for i in range(len(cum_persons_per_view) - 1):
        ind1, ind2 = cum_persons_per_view[i], cum_persons_per_view[i + 1]
        new_aff[ind1:ind2, ind1:ind2] = 0
        
    # Reset diagonal elements to one and ensure X is within valid range [0, 1]
    new_aff[index_diag, index_diag] = 1.
    new_aff[new_aff < 0] = 0
    new_aff[new_aff > 1] = 1
    
    # Enforce circular constraint
    new_aff = new_aff * circ_constraint
    new_aff = (new_aff + new_aff.T) / 2 # kept just in case X loses its symmetry during optimization 
    Y = Y + mu * (new_aff - Q)
    
    # Compute convergence criteria: break if new_aff is close enough to Q and no evolution anymore
    pRes = np.linalg.norm(new_aff - Q) / N # primal residual (diff between new_aff and SVT result)
    dRes = mu * np.linalg.norm(new_aff - new_aff0) / N # dual residual (diff between new_aff and previous new_aff)
    if pRes < tol and dRes < tol:
        break
    if pRes > 10 * dRes: mu = 2 * mu
    elif dRes > 10 * pRes: mu = mu / 2

    iter +=1
# print(new_aff)
proposals = []
# print(new_aff)
# for row in range(new_aff.shape[0]):
#     proposal_row = []
#     for cam in range(len(cum_persons_per_view)-1): # 除去起始0，四台相機
#         id_persons_per_view = new_aff[row, cum_persons_per_view[cam]:cum_persons_per_view[cam+1]]
#         # print(id_persons_per_view)
#         proposal_row += [np.argmax(id_persons_per_view) if (len(id_persons_per_view)>0 and max(id_persons_per_view)>0) else -1]
#         # print(proposal_row)
#     proposals.append(proposal_row)
#     # print(proposals)
# proposals = np.array(proposals, dtype=float)
# print(proposals)
# proposals, nb_detections = np.unique(proposals, axis=0, return_counts=True)
# proposals = proposals[np.argsort(nb_detections)[::-1]]
# print(proposals)
# print(nb_detections)
# proposals[proposals==-1] = np.nan
# print(proposals)
# mask = np.ones(proposals.shape[0], dtype=bool)
# print(mask)
# for i in range(1, len(proposals)):
#     print(i)
#     mask[i] = ~np.any(proposals[i] == proposals[:i], axis=0).any()
#     print(mask)
# proposals = proposals[mask]
# print(proposals)
# nb_cams_per_person = [np.count_nonzero(~np.isnan(p)) for p in proposals]
# proposals = np.array([p for (n,p) in zip(nb_cams_per_person, proposals) if n >= 2])
# print(nb_cams_per_person)
# print(proposals)
nb_persons_to_detect = 2
keypoints_nb = 26
Q = [[[np.nan]*3]*keypoints_nb for n in range(nb_persons_to_detect)]
Q_old = [[[np.nan]*3]*keypoints_nb for n in range(nb_persons_to_detect)]
error = [[] for n in range(nb_persons_to_detect)]
nb_cams_excluded = [[] for n in range(nb_persons_to_detect)]
id_excluded_cams = [[] for n in range(nb_persons_to_detect)]
Q_tot, error_tot, nb_cams_excluded_tot,id_excluded_cams_tot = [], [], [], []
nan_mask = np.isnan(Q)
print(nan_mask)
Q_old = np.where(nan_mask, Q_old, Q)
Q = [[] for n in range(nb_persons_to_detect)]
error = [[] for n in range(nb_persons_to_detect)]
nb_cams_excluded = [[] for n in range(nb_persons_to_detect)]
id_excluded_cams = [[] for n in range(nb_persons_to_detect)]
print(len(Q_old))
# print(Q)
# print(Q_old)
# print(Q_old == Q)
# def read_config_file(config):
#     '''
#     Read configation file.
#     '''

#     config_dict = toml.load(config)
#     return config_dict
# config = read_config_file(r"C:\Users\陳柏宏\NTKCAP\Patient_data\Patient_ID\2024_05_07\2024_06_02_14_47_calculated\Walk1\User\Config.toml")
# pose_model = config.get('pose').get('pose_model')
# model = DictImporter().import_(config.get('pose').get(pose_model))
# print(eval(pose_model))
frame_by_frame_dist = []
for i in range(5):
    frame_by_frame_dist += [1]
    # print(frame_by_frame_dist)
L = [  20,   27,  51,    33,   43,   23,   37,   24,   4,   68,   84,    3  ]
T = list(it.product(range(3),range(4)))
minL = [np.nanmin(L)]
argminL = [np.nanargmin(L)]
T_minL = [T[argminL[0]]]
# = [(0,0),(0,1),(0,2),(0,3),(1,0),(1,1),(1,2),(1,3),(2,0),(2,1),(2,2),(2,3)]

mask_tokeep = np.array([True for t in T])
i=0
while mask_tokeep.any()==True:
    mask_tokeep = mask_tokeep & np.array([t[0]!=T_minL[i][0] and t[1]!=T_minL[i][1] for t in T])
    if mask_tokeep.any()==True:
        indicesL_tokeep = np.where(mask_tokeep)[0]
        minL += [np.nanmin(np.array(L)[indicesL_tokeep]) if not np.isnan(np.array(L)[indicesL_tokeep]).all() else np.nan]
        argminL += [indicesL_tokeep[np.nanargmin(np.array(L)[indicesL_tokeep])] if not np.isnan(minL[-1]) else indicesL_tokeep[0]]
        T_minL += (T[argminL[i+1]],)
        i+=1
        
Q_kpt_new, personsIDs_sorted = [], []
associated_tuples = np.array(T_minL)
minL = np.array(minL)
print(associated_tuples)
# print(minL)
print(associated_tuples[:,1])
for i in range(6):
    id_in_old =  associated_tuples[:,1][associated_tuples[:,0] == i].tolist()
    # print(id_in_old)
    # if len(id_in_old) > 0:
    #     personsIDs_sorted += id_in_old
    #     Q_kpt_new += [Q_kpt[id_in_old[0]]]
    # else:
    #     personsIDs_sorted += [-1]
    #     Q_kpt_new += [Q_kpt_old[i]]

