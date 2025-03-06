from multiprocessing import Event, shared_memory, Manager, Queue, Array, Lock, Process
import time
import numpy as np
import cupy as cp
import json
import toml
import os
import cv2
class PreTri_Process(Process):
    def __init__(self, sync_tracker_shm_name0, sync_tracker_shm_name1, sync_tracker_shm_name2, sync_tracker_shm_name3, sync_tracker_queue_lst, start_evt, stop_evt, *args, **kwargs):
        super().__init__()
        self.sync_tracker_shm_name0 = sync_tracker_shm_name0
        self.sync_tracker_shm_name1 = sync_tracker_shm_name1
        self.sync_tracker_shm_name2 = sync_tracker_shm_name2
        self.sync_tracker_shm_name3 = sync_tracker_shm_name3
        self.sync_tracker_shm_lst = []
        self.sync_tracker_queue_lst = sync_tracker_queue_lst
        # self.keypoints_sync_shm_name = keypoints_sync_shm_name
        self.keypoints_sync_shm_shape = (4, 1, 26, 3)
        self.keypoints_shm_shape = (1, 26, 3)
        # self.sync_tracker_queue = sync_tracker_queue
        self.buffer_length = 20
        calib_file = r"C:\Users\MyUser\Desktop\NTKCAP\Patient_data\HealthCare020\2024_12_06\2025_02_18_00_02_calculated\1005_1\calib-2d\Calib_easymocap.toml"
        calib = toml.load(calib_file)
        self.cam_coord=cp.array([self.find_camera_coordinate(calib[list(calib.keys())[i]]) for i in range(4)])
        self.P = self.computeP(calib_file)
        mappingx, mappingy = self.computemap(calib_file)
        self.mappingx = cp.array(mappingx)
        self.mappingy = cp.array(mappingy)
        config=r"C:\Users\MyUser\Desktop\NTKCAP\NTK_CAP\template\Empty_project\User\Config.toml"
        cp.get_default_memory_pool().free_all_blocks()

        config_dict = toml.load(config)
        self.likelihood_threshold = config_dict.get('triangulation').get('likelihood_threshold')
        list_dynamic_mincam=  {'Hip':4,'RHip':4,'RKnee':3,'RAnkle':3,'RBigToe':3,'RSmallToe':3,'RHeel':3,'LHip':4,'LKnee':3,'LAnkle':3,'LBigToe':3,'LSmallToe':3,'LHeel':3,'Neck':3,'Head':2,'Nose':2,'RShoulder':3,'RElbow':3,'RWrist':3,'LShoulder':3,'LElbow':3,'LWrist':3}
        self.list_dynamic_mincam_prep = [self.map_to_listdynamic(value) for value in list_dynamic_mincam.values()]
        self.start_evt = start_evt
        self.stop_evt = stop_evt
    def run(self):
        self.sync_tracker_shm_lst.clear()
        # existing_shm_keypoints = shared_memory.SharedMemory(name=self.keypoints_sync_shm_name)
        # shared_array_keypoints = np.ndarray((self.buffer_length,) + self.keypoints_sync_shm_shape, dtype=np.float32, buffer=existing_shm_keypoints.buf)
        existing_shm_tracker0 = shared_memory.SharedMemory(name=self.sync_tracker_shm_name0)
        shared_array_tracker0 = np.ndarray((self.buffer_length,) + self.keypoints_shm_shape, dtype=np.float32, buffer=existing_shm_tracker0.buf)
        existing_shm_tracker1 = shared_memory.SharedMemory(name=self.sync_tracker_shm_name1)
        shared_array_tracker1 = np.ndarray((self.buffer_length,) + self.keypoints_shm_shape, dtype=np.float32, buffer=existing_shm_tracker1.buf)
        existing_shm_tracker2 = shared_memory.SharedMemory(name=self.sync_tracker_shm_name2)
        shared_array_tracker2 = np.ndarray((self.buffer_length,) + self.keypoints_shm_shape, dtype=np.float32, buffer=existing_shm_tracker2.buf)
        existing_shm_tracker3 = shared_memory.SharedMemory(name=self.sync_tracker_shm_name3)
        shared_array_tracker3 = np.ndarray((self.buffer_length,) + self.keypoints_shm_shape, dtype=np.float32, buffer=existing_shm_tracker3.buf)
        self.sync_tracker_shm_lst.append(shared_array_tracker0)
        self.sync_tracker_shm_lst.append(shared_array_tracker1)
        self.sync_tracker_shm_lst.append(shared_array_tracker2)
        self.sync_tracker_shm_lst.append(shared_array_tracker3)
        combinations_3 = [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]
        combinations_2 = [[2, 3], [1, 3], [1, 2], [0, 3], [0, 2], [0, 1]]
        coord1 = []
        coord2 = []
        coord3 = []
        coord4 = []
        cal_path = r"C:\Users\MyUser\Desktop\NTKCAP\Patient_data\HealthCare020\2024_12_06\2025_02_18_00_02_calculated\1005_1\pose-2d-tracked"
        template = ["pose_cam1_json", "pose_cam2_json", "pose_cam3_json", "pose_cam4_json"]
        path1 = os.path.join(cal_path, template[0])
        path2 = os.path.join(cal_path, template[1])
        path3 = os.path.join(cal_path, template[2])
        path4 = os.path.join(cal_path, template[3])
        for i in os.listdir(path1):
            path = os.path.join(path1, i)
            with open(path, 'r') as f:
                data1 = json.load(f)
                coord1.append(data1)
        for i in os.listdir(path2):
            path = os.path.join(path2, i)
            with open(path, 'r') as f:
                data2 = json.load(f)
                coord2.append(data2)
        for i in os.listdir(path3):
            path = os.path.join(path3, i)
            with open(path, 'r') as f:
                data3 = json.load(f)
                coord3.append(data3)
        for i in os.listdir(path4):
            path = os.path.join(path4, i)
            with open(path, 'r') as f:
                data4 = json.load(f)
                coord4.append(data4)
        c1 = np.array(coord1[0]['people'][0]['pose_keypoints_2d'])
        c2 = np.array(coord2[0]['people'][0]['pose_keypoints_2d'])
        c3 = np.array(coord3[0]['people'][0]['pose_keypoints_2d'])
        c4 = np.array(coord4[0]['people'][0]['pose_keypoints_2d'])
        arr1 = c1.reshape(26, 3)
        arr2 = c2.reshape(26, 3)
        arr3 = c3.reshape(26, 3)
        arr4 = c4.reshape(26, 3)
        stacked = np.stack([arr1, arr2, arr3, arr4], axis=0)
        keypoints_ids = [19, 12, 14, 16, 21, 23, 25, 11, 13, 15, 20, 22, 24, 18, 17, 0, 6, 8, 10, 5, 7, 9]
        P = cp.array(self.P)
        P_cam_0_comb4 = cp.concatenate([P[0][0].reshape(1, -1), P[1][0].reshape(1, -1), P[2][0].reshape(1, -1), P[3][0].reshape(1, -1)], axis=0)
        P_cam_1_comb4 = cp.concatenate([P[0][1].reshape(1, -1), P[1][1].reshape(1, -1), P[2][1].reshape(1, -1), P[3][1].reshape(1, -1)], axis=0)
        P_cam_2_comb4 = cp.concatenate([P[0][2].reshape(1, -1), P[1][2].reshape(1, -1), P[2][2].reshape(1, -1), P[3][2].reshape(1, -1)], axis=0)
        P_cam_comb4 = cp.stack([P_cam_0_comb4, P_cam_1_comb4, P_cam_2_comb4], axis=0)
        comb_3 = cp.array(combinations_3)
        P_cam_comb3 = cp.stack([
            cp.stack([cp.concatenate([P[d][j].reshape(1, -1) for d in combinations_3[i]], axis=0) for j in range(3)], axis=0)
            for i in range(4)
        ], axis=0)

        combinations_2 = [[2, 3], [1, 3], [1, 2], [0, 3], [0, 2], [0, 1]] 
        comb_2 = cp.array(combinations_2)
        P_cam_comb2 = cp.stack([
            cp.stack([cp.concatenate([P[d][j].reshape(1, -1) for d in combinations_2[i]], axis=0) for j in range(3)], axis=0)
            for i in range(6)
        ], axis=0)
        P0, P1, P2 = P[:, 0, :].reshape(1, 1, 1, 4, 4), P[:, 1, :].reshape(1, 1, 1, 4, 4), P[:, 2, :].reshape(1, 1, 1, 4, 4)
        P0_c3, P1_c3, P2_c3 = P[comb_3, 0, :][None, None, ...], P[comb_3, 1, :][None, None, ...], P[comb_3, 2, :][None, None, ...]
        P0_c2, P1_c2, P2_c2 = P[comb_2, 0, :][None, None, ...], P[comb_2, 1, :][None, None, ...], P[comb_2, 2, :][None, None, ...]
        
        result = cp.empty((22, 4, 3), dtype=cp.float32)
        prep_4 = cp.empty((1, 22, 1, 4, 3), dtype=cp.float32)
        prep_3 = cp.empty((1, 22, 4, 3, 3), dtype=cp.float32)
        prep_2 = cp.empty((1, 22, 6, 2, 3), dtype=cp.float32)

        prep_4like = cp.empty((1, 22, 1), dtype=cp.float32)
        prep_3like = cp.empty((1, 22, 4), dtype=cp.float32)
        prep_2like = cp.empty((1, 22, 6), dtype=cp.float32)

        A4 = cp.empty((1, 22, 1, 8, 4), dtype=cp.float32)
        A3 = cp.empty((1, 22, 4, 6, 4), dtype=cp.float32)
        A2 = cp.empty((1, 22, 6, 4, 4), dtype=cp.float32)

        Q4 = cp.empty((1, 22, 1, 4), dtype=cp.float32)
        Q3 = cp.empty((1, 22, 4, 4), dtype=cp.float32)
        Q2 = cp.empty((1, 22, 6, 4), dtype=cp.float32)

        real_dist4 = cp.empty((1, 22, 1), dtype=cp.float32)
        real_dist3 = cp.empty((1, 22, 4), dtype=cp.float32)
        real_dist2 = cp.empty((1, 22, 6), dtype=cp.float32)
        idx = 0
        count = 0
        stream_4 = cp.cuda.Stream(non_blocking=True)
        stream_3 = cp.cuda.Stream(non_blocking=True)
        stream_2 = cp.cuda.Stream(non_blocking=True)
        time.sleep(1)
        while self.start_evt.is_set():
            cap_s = time.time()
            try:
                idx_get = self.sync_tracker_queue_lst[0].get(timeout=0.01)
                
            except:
                continue
            # not_get = [0, 1, 2, 3]
            # keypoints_frame = [None] * 4
            
            # while not_get:
            #     i = not_get[0]
            #     try:
            #         idx_get = self.sync_tracker_queue_lst[i].get_nowait()
            #         not_get.pop(0)
            #         keypoints = self.sync_tracker_shm_lst[i][idx_get]
            #         keypoints_frame[i] = keypoints
            #     except Exception as e:
            #         not_get.append(not_get.pop(0))
            # np.copyto(shared_array_keypoints[idx, :], keypoints_frame) # 4 (1, 26, 3)
            
            # self.sync_tracker_queue.put(idx)
            # idx = (idx+1) % self.buffer_length
            
            prep = None
            prep = cp.array(stacked[:, keypoints_ids, :])
            
            result[:] = cp.transpose(prep, (1, 0, 2)) # (22, 4, 3)
            prep_4[:] = result[None, :, None, :] # (1, 22, 1, 4, 3)
            prep_3[:] = result[:, combinations_3, :][None, :] # (1, 22, 4, 3, 3)
            prep_2[:] = result[:, combinations_2, :][None, :] # (1, 22, 6, 2, 3)
            
            with stream_4:
                prep_4[:] = cp.stack([self.bilinear_interpolate_cupy(self.mappingx, prep_4[:, :, :, :, 0], prep_4[:, :, :, :, 1]), self.bilinear_interpolate_cupy(self.mappingy, prep_4[:, :, :, :, 0], prep_4[:, :, :, :, 1]), prep_4[:, :, :, :, 2]], axis=-1)  # Shape: (155, 22, 4, 3, 3)
                prep_4like[:] = cp.min(prep_4[:,:,:,:,2],axis=3) # (1, 22, 1)
                A4[:] = cp.stack([((P0 - prep_4[..., 0:1] * P2) * prep_4[..., 2:3]), ((P1 - prep_4[..., 1:2] * P2) * prep_4[..., 2:3])], axis=-2).reshape(1, 22, 1, 8, 4) # (1, 22, 1, 8, 4)
                f4 = cp.shape(A4)
                A4_flat = A4.reshape(-1, 8, 4)
                _, _, Vt4 = cp.linalg.svd(A4_flat, full_matrices=False)  
                V4 = Vt4.transpose(0, 2, 1)
                Q4[:] = cp.array([V4[:, 0, 3] / V4[:, 3, 3], V4[:, 1, 3] / V4[:, 3, 3], V4[:, 2, 3] / V4[:, 3, 3], cp.ones(V4.shape[0])]).T.reshape(f4[0], f4[1], f4[2], 4) # (1, 22, 1, 4)
                # real_dist4
                result_c4 = cp.einsum('cik,btk->cit', P_cam_comb4, Q4[:,:,0,:])
                rpj_coor_4 = cp.stack((cp.expand_dims((result_c4[0] / result_c4[2]).T, axis=(0, 2)), cp.expand_dims((result_c4[1] / result_c4[2]).T, axis=(0, 2))), axis=-1)
                rpj = cp.sqrt((rpj_coor_4[:, :, :, :, 0] - prep_4[:, :, :, :, 0]) ** 2 + (rpj_coor_4[:, :, :, :, 1] - prep_4[:, :, :, :, 1]) ** 2)
                real_dist4[:] = cp.max(cp.expand_dims(cp.sqrt(cp.sum((Q4[:, :, 0, 0:3] - cp.stack(self.cam_coord, axis=0)[:, None, :]) ** 2, axis=-1)).T, axis=(0, 2)) * rpj, axis=-1) # (1, 22, 1)
            with stream_3:
                prep_3[:] = cp.stack([self.bilinear_interpolate_cupy(self.mappingx, prep_3[:, :, :, :, 0], prep_3[:, :, :, :, 1]), self.bilinear_interpolate_cupy(self.mappingy, prep_3[:, :, :, :, 0], prep_3[:, :, :, :, 1]), prep_3[:, :, :, :, 2]], axis=-1)  # Shape: (155, 22, 4, 3, 3)
                prep_3like[:] = cp.min(prep_3[:,:,:,:,2],axis=3) # (1, 22, 4)
                A3[:] = cp.stack([((P0_c3 - prep_3[..., 0:1] * P2_c3) * prep_3[..., 2:3]), ((P1_c3 - prep_3[..., 1:2] * P2_c3) * prep_3[..., 2:3])], axis=-2).reshape(1, 22, 4, 6, 4) # (1, 22, 4, 6, 4)
                f3 = cp.shape(A3)
                A3_flat = A3.reshape(-1, 6, 4)
                _, _, Vt3 = cp.linalg.svd(A3_flat, full_matrices=False)
                V3 = Vt3.transpose(0, 2, 1)
                Q3[:] = cp.array([V3[:, 0, 3] / V3[:, 3, 3], V3[:, 1, 3] / V3[:, 3, 3], V3[:, 2, 3] / V3[:, 3, 3], cp.ones(V3.shape[0])]).T.reshape(f3[0], f3[1], f3[2], 4) # (1, 22, 4, 4)
                prep_3like[:] = cp.clip(prep_3like, self.likelihood_threshold, cp.inf)
                
                Q3_bug = Q3[cp.arange(Q3.shape[0])[:, None], cp.arange(Q3.shape[1])[None, :], cp.argmax(~cp.isinf(prep_3like), axis=2), :][:, :, cp.newaxis, :]
                # real_dist3
                result_c3 = cp.einsum('ncik,nbtk->ncit', P_cam_comb3, Q3.transpose(2, 0, 1, 3))
                rpj_coor_3 = cp.stack((cp.expand_dims((result_c3[:, 0] / result_c3[:, 2]).transpose(2, 0, 1), axis=0), cp.expand_dims((result_c3[:, 1] / result_c3[:, 2]).transpose(2, 0, 1), axis=0)),axis = -1)
                rpj = cp.sqrt((rpj_coor_3[:, :, :, :, 0] - prep_3[:, :, :, :, 0]) ** 2 + (rpj_coor_3[:, :, :, :, 1] - prep_3[:, :, :, :, 1]) ** 2)
                real_dist3[:] = cp.max(cp.expand_dims(cp.sqrt(cp.sum((Q3_bug.transpose(2, 0, 1, 3)[:, :, :, 0:3] - self.cam_coord[comb_3][:, :, None, :]) ** 2, axis=-1)).transpose(2, 0, 1), axis=0) * rpj, axis=-1) # (1, 22, 4)
            with stream_2:
                prep_2[:] = cp.stack([self.bilinear_interpolate_cupy(self.mappingx, prep_2[:, :, :, :, 0], prep_2[:, :, :, :, 1]), self.bilinear_interpolate_cupy(self.mappingy, prep_2[:, :, :, :, 0], prep_2[:, :, :, :, 1]), prep_2[:, :, :, :, 2]], axis=-1)  # Shape: (155, 22, 4, 3, 3)
                prep_2like[:] = cp.min(prep_2[:,:,:,:,2],axis=3) # (1, 22, 6)
                A2[:] = cp.stack([((P0_c2 - prep_2[..., 0:1] * P2_c2) * prep_2[..., 2:3]), ((P1_c2 - prep_2[..., 1:2] * P2_c2) * prep_2[..., 2:3])], axis=-2).reshape(1, 22, 6, 4, 4) # (1, 22, 6, 4, 4)
                f2 = cp.shape(A2)
                A2_flat = A2.reshape(-1, 4, 4)
                _, _, Vt2 = cp.linalg.svd(A2_flat, full_matrices=False)
                V2 = Vt2.transpose(0, 2, 1)
                Q2[:] = cp.array([V2[:, 0, 3] / V2[:, 3, 3], V2[:, 1, 3] / V2[:, 3, 3], V2[:, 2, 3] / V2[:, 3, 3], cp.ones(V2.shape[0])]).T.reshape(f2[0], f2[1], f2[2], 4) # (1, 22, 6, 4)
                prep_2like[:] = cp.clip(prep_2like, self.likelihood_threshold, cp.inf)
                Q2_bug = Q2[cp.arange(Q2.shape[0])[:, None], cp.arange(Q2.shape[1])[None, :], cp.argmax(~cp.isinf(prep_2like), axis=2), :][:, :, cp.newaxis, :]
                # real_dist2
                result_c2 = cp.einsum('ncik,nbtk->ncit', P_cam_comb2, Q2.transpose(2, 0, 1, 3))
                rpj_coor_2 = cp.stack((cp.expand_dims((result_c2[:, 0] / result_c2[:, 2]).transpose(2, 0, 1), axis=0), cp.expand_dims((result_c2[:, 1] / result_c2[:, 2]).transpose(2, 0, 1), axis=0)),axis = -1)
                rpj = cp.sqrt((rpj_coor_2[:, :, :, :, 0] - prep_2[:, :, :, :, 0]) ** 2 + (rpj_coor_2[:, :, :, :, 1] - prep_2[:, :, :, :, 1]) ** 2)
                real_dist2[:] = cp.max(cp.expand_dims(cp.sqrt(cp.sum((Q2_bug.transpose(2, 0, 1, 3)[:, :, :, 0:3] - self.cam_coord[comb_2][:, :, None, :]) ** 2, axis=-1)).transpose(2, 0, 1), axis=0) * rpj, axis=-1) # (1, 22, 6)
            
            stream_4.synchronize()
            stream_3.synchronize()
            stream_2.synchronize()
            # delete the liklelihoood vlue which is too low
            prep_like = cp.concatenate((prep_4like, prep_3like, prep_2like),axis =2)
            real_dist = cp.concatenate((real_dist4, real_dist3, real_dist2), axis=2)
            real_dist[cp.where(prep_like < self.likelihood_threshold)] = cp.inf
            non_inf_mask = ~cp.isinf(real_dist)
            min_locations_nan = cp.argmax(non_inf_mask, axis=2)
            real_dist_dynamic = cp.copy(real_dist)
            
            ## setting the list dynamic

            for i in range(22):
                real_dist_dynamic[:, i, self.list_dynamic_mincam_prep[i]] = cp.inf
            
            ## find the minimum combination
            Q = cp.concatenate((Q4, Q3, Q2), axis=2)
            temp_shape = cp.shape(Q)
            checkinf = cp.min(real_dist_dynamic,axis =2)
            min_locations = cp.argmin(real_dist_dynamic, axis=2)
            loc =cp.where(checkinf==cp.inf)
            min_locations[loc] = min_locations_nan[loc]
            batch_indices, time_indices = cp.meshgrid(cp.arange(temp_shape[0]), cp.arange(temp_shape[1]), indexing='ij')
            Q_selected = Q[batch_indices, time_indices, min_locations]
            Q_selected = Q_selected[:,:,0:3]
            Q_selected = cp.asnumpy(Q_selected)
            Q_tot_gpu = [Q_selected[i].ravel() for i in range(Q_selected.shape[0])]
            # print("tri:", count, time.time()-cap_s)
            # print("send to triangulation", count)
            count += 1
            
    def map_to_listdynamic(self, value):
        if value == 4:
            return [1,2,3,4,5,6,7,8,9,10]  # Map 4 to [0]
        elif value == 3:
            return [5,6,7,8,9,10]  # Map 3 to [1, 2, 3, 4]
        elif value == 2:
            return []  # Map 2 to [5, 6, 7, 8, 9, 10]
    def find_camera_coordinate(self, calib_file):
        calib_file['rotation']
        R, _ = cv2.Rodrigues(np.array(calib_file['rotation']))
        T = np.array(calib_file['translation'])
        R_t = np.transpose(R)
        C = -R_t.dot(T)
        return C
    def computemap(self, calib_file):
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
    
    def bilinear_interpolate_cupy(self, map, x, y):
    
        x0 = cp.clip(cp.floor(x).astype(cp.int32), 0, map.shape[1] - 2)
        y0 = cp.clip(cp.floor(y).astype(cp.int32), 0, map.shape[0] - 2)
        
        x1, y1 = x0 + 1, y0 + 1

        Ia = map[y0, x0]
        Ib = map[y0, x1]
        Ic = map[y1, x0]
        Id = map[y1, x1]

        wa = (x1 - x) * (y1 - y)
        wb = (x - x0) * (y1 - y)
        wc = (x1 - x) * (y - y0)
        wd = (x - x0) * (y - y0)

        return wa * Ia + wb * Ib + wc * Ic + wd * Id
    def computeP(self, calib_file):
        K, R, T, Kh, H = [], [], [], [], []
        P = []
        
        calib = toml.load(calib_file)
        for cam in list(calib.keys()):
            if cam != 'metadata':
                K = np.array(calib[cam]['matrix'])
                Kh = np.block([K, np.zeros(3).reshape(3,1)])
                R, _ = cv2.Rodrigues(np.array(calib[cam]['rotation']))
                T = np.array(calib[cam]['translation'])
                H = np.block([[R,T.reshape(3,1)], [np.zeros(3), 1 ]])
                
                P.append(Kh.dot(H))
    
        return P