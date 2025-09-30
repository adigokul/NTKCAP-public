from multiprocessing import Event, shared_memory, Manager, Queue, Array, Process
import os
import cv2
import time
import toml
import numpy as np
import cupy as cp
import json
class Triangulation(Process):
    def __init__(self, keypoints_sync_shm_name, sync_tracker_queue, start_evt, stop_evt, config, *args, **kwargs):
        super().__init__()
        self.keypoints_sync_shm_name = keypoints_sync_shm_name
        self.sync_tracker_queue = sync_tracker_queue
        self.buffer_length = 20
        self.keypoints_sync_shm_shape = (4, 1, 26, 3)
        self.start_evt = start_evt
        self.stop_evt = stop_evt
        self.config_dict = config
        calib_file = r"C:\Users\MyUser\Desktop\NTKCAP\Patient_data\HealthCare020\2024_12_06\2025_02_18_00_02_calculated\1005_1\calib-2d\Calib_easymocap.toml"
        config=r"C:\Users\MyUser\Desktop\NTKCAP\NTK_CAP\template\Empty_project\User\Config.toml"
    
        config_dict = toml.load(config)
        self.likelihood_threshold = config_dict.get('triangulation').get('likelihood_threshold')
        calib = toml.load(calib_file)
        self.cam_coord=cp.array([self.find_camera_coordinate(calib[list(calib.keys())[i]]) for i in range(4)])

        self.P = self.computeP(calib_file)
        mappingx, mappingy = self.computemap(calib_file)
        self.mappingx = cp.array(mappingx)
        self.mappingy = cp.array(mappingy)
    def run(self):
        combinations_4 = [[0, 1, 2, 3]]
        combinations_3 = [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]
        combinations_2 = [[2, 3], [1, 3], [1, 2], [0, 3], [0, 2], [0, 1]]
        existing_shm_keypoints = shared_memory.SharedMemory(name=self.keypoints_sync_shm_name)
        shared_array_keypoints = np.ndarray((self.buffer_length,) + self.keypoints_sync_shm_shape, dtype=np.float32, buffer=existing_shm_keypoints.buf)
        self.start_evt.wait()
        idx = 0
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
 
        prep = stacked[:, keypoints_ids, :]
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
        
        while self.start_evt.is_set():
            try:
                idx_get = self.sync_tracker_queue.get(timeout=0.01)
            except:
                
                continue
            keypoints = shared_array_keypoints[idx_get, : ] # shape = (4, 1, 26, 3)
            t1 = time.time()
            result = cp.array(np.transpose(prep, (1, 0, 2)))
            prep_4 = cp.expand_dims(result, axis=(0, 2))
            prep_3 = cp.expand_dims(result[:, combinations_3, :], axis=0)
            prep_2 = cp.expand_dims(result[:, combinations_2, :], axis=0)
            prep_4, prep_3, prep_2 = self.undistort_points_cupy(prep_4, prep_3, prep_2)
            
            prep_4like=cp.min(prep_4[:,:,:,:,2],axis=3)
            prep_3like=cp.min(prep_3[:,:,:,:,2],axis=3)
            prep_2like=cp.min(prep_2[:,:,:,:,2],axis=3)
            prep_like = cp.concatenate((prep_4like,prep_3like,prep_2like),axis =2)
            
            A4,A3,A2 = self.create_A(prep_4, prep_3, prep_2, P)
            
            Q4,Q3,Q2 = self.find_Q(A4,A3,A2)
            
            Q = cp.concatenate((Q4,Q3,Q2),axis = 2)
            Q3_bug,Q2_bug = self.create_QBug(self.likelihood_threshold,prep_3like,Q3,prep_2like,Q2)
            
            real_dist4, real_dist3, real_dist2 = self.find_real_dist_error(self.cam_coord, prep_4, Q4, prep_3, Q3, prep_2, Q2, Q3_bug, Q2_bug, P_cam_comb4, P_cam_comb3, comb_3, P_cam_comb2, comb_2)
            
            ## delete the liklelihoood vlue which is too low
            real_dist = cp.concatenate((real_dist4,real_dist3,real_dist2),axis = 2)
            loc = cp.where(prep_like < self.likelihood_threshold)
            #import pdb;pdb.set_trace()
            real_dist[loc] = cp.inf
            # Find the index of the first non-inf element along axis 2
            
            non_inf_mask = ~cp.isinf(real_dist)
            min_locations_nan = cp.argmax(non_inf_mask, axis=2)
            real_dist_dynamic = cp.copy(real_dist)
            list_dynamic_mincam=  {'Hip':4,'RHip':4,'RKnee':3,'RAnkle':3,'RBigToe':3,'RSmallToe':3,'RHeel':3,'LHip':4,'LKnee':3,'LAnkle':3,'LBigToe':3,'LSmallToe':3,'LHeel':3,'Neck':3,'Head':2,'Nose':2,'RShoulder':3,'RElbow':3,'RWrist':3,'LShoulder':3,'LElbow':3,'LWrist':3}
            list_dynamic_mincam_prep = [self.map_to_listdynamic(value) for value in list_dynamic_mincam.values()]
            ## setting the list dynamic
            
            for i in range(22):    
                real_dist_dynamic[:,i,list_dynamic_mincam_prep[i]] = cp.inf
            
            ## find the minimum combination
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
            print(time.time()-t1)
    
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

    def find_camera_coordinate(self, calib_file):
        calib_file['rotation']
        R, _ = cv2.Rodrigues(np.array(calib_file['rotation']))
        T = np.array(calib_file['translation'])
        H = np.block([[R,T.reshape(3,1)], [np.zeros(3), 1 ]])
        R_t = np.transpose(R)
        C = -R_t.dot(T)
        A = R_t.dot(np.array([[0],[0],[1]]))
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

    def create_QBug(self, likelihood_threshold,prep_3like,Q3,prep_2like,Q2):
        loc = cp.where(prep_3like < likelihood_threshold)
        prep_3like[loc] = cp.inf
        # Find the index of the first non-inf element along axis 2
        non_inf_mask3 = ~cp.isinf(prep_3like)
        min_locations_nan3 = cp.argmax(non_inf_mask3, axis=2)
        batch_indices = cp.arange(Q3.shape[0])[:, None]  # Shape: (184, 1)
        time_indices = cp.arange(Q3.shape[1])[None, :]   # Shape: (1, 22)

    # Use advanced indexing to extract the desired values
        selected_slices = Q3[batch_indices, time_indices, min_locations_nan3, :]
        
    # Add an additional dimension to match the shape (184, 22, 1, 4)
        Q3_bug = selected_slices[:, :, cp.newaxis, :]

        loc = cp.where(prep_2like < likelihood_threshold)
        prep_2like[loc] = cp.inf
        # Find the index of the first non-inf element along axis 2
        non_inf_mask2 = ~cp.isinf(prep_2like)
        min_locations_nan2 = cp.argmax(non_inf_mask2, axis=2)
        batch_indices = cp.arange(Q2.shape[0])[:, None]  # Shape: (184, 1)
        time_indices = cp.arange(Q2.shape[1])[None, :]   # Shape: (1, 22)

    # Use advanced indexing to extract the desired values
        selected_slices = Q2[batch_indices, time_indices, min_locations_nan2, :]
        
    # Add an additional dimension to match the shape (184, 22, 1, 4)
        Q2_bug = selected_slices[:, :, cp.newaxis, :]
        return Q3_bug,Q2_bug
    def find_Q(self, A4,A3,A2):
        f = cp.shape(A4)
        A_flat = A4.reshape(-1, 8, 4)  # Shape: (250 * 22 * 1, 8, 4)

        # Step 2: Perform SVD in batch using CuPy
        U, S, Vt = cp.linalg.svd(A_flat, full_matrices=False)  # Batched SVD

        # Step 3: Compute Q
        # Transpose Vt to get V
        V = Vt.transpose(0, 2, 1)  # Shape: (batch_size, 4, 4)

        # Extract and compute Q
        Q = cp.array([
            V[:, 0, 3] / V[:, 3, 3],
            V[:, 1, 3] / V[:, 3, 3],
            V[:, 2, 3] / V[:, 3, 3],
            cp.ones(V.shape[0])  # Add 1 as the last element of Q
        ]).T  # Shape: (batch_size, 4)

        # Step 4: Reshape Q back to (250, 22, 1, 4)
        Q4 = Q.reshape(f[0], f[1], f[2], 4)
        f = cp.shape(A3)
        A_flat = A3.reshape(-1, 6, 4)  # Shape: (250 * 22 * 1, 8, 4)

        # Step 2: Perform SVD in batch using CuPy
        U, S, Vt = cp.linalg.svd(A_flat, full_matrices=False)  # Batched SVD

        # Step 3: Compute Q
        # Transpose Vt to get V
        V = Vt.transpose(0, 2, 1)  # Shape: (batch_size, 4, 4)

        # Extract and compute Q
        Q = cp.array([
            V[:, 0, 3] / V[:, 3, 3],
            V[:, 1, 3] / V[:, 3, 3],
            V[:, 2, 3] / V[:, 3, 3],
            cp.ones(V.shape[0])  # Add 1 as the last element of Q
        ]).T  # Shape: (batch_size, 4)

        # Step 4: Reshape Q back to (250, 22, 1, 4)
        Q3 = Q.reshape(f[0], f[1], f[2], 4)
        
        f = cp.shape(A2)
        A_flat = A2.reshape(-1, 4, 4)  # Shape: (250 * 22 * 1, 8, 4)

        # Step 2: Perform SVD in batch using CuPy
        U, S, Vt = cp.linalg.svd(A_flat, full_matrices=False)  # Batched SVD
        # Step 3: Compute Q
        # Transpose Vt to get V
        V = Vt.transpose(0, 2, 1)  # Shape: (batch_size, 4, 4)

        # Extract and compute Q
        Q = cp.array([
            V[:, 0, 3] / V[:, 3, 3],
            V[:, 1, 3] / V[:, 3, 3],
            V[:, 2, 3] / V[:, 3, 3],
            cp.ones(V.shape[0])  # Add 1 as the last element of Q
        ]).T  # Shape: (batch_size, 4)

        # Step 4: Reshape Q back to (250, 22, 1, 4)
        Q2 = Q.reshape(f[0], f[1], f[2], 4)


        return Q4,Q3,Q2
    def create_A(self, prep_4,prep_3,prep_2,P):
        # Elements
        
        elements = [0, 1, 2, 3]

        # Total elements
        
        combinations_3 = [elements[:i] + elements[i+1:] for i in range(len(elements))]  # Delete one element
        combinations_2 = [elements[:i] + elements[i+1:j] + elements[j+1:] for i in range(len(elements)) for j in range(i+1, len(elements))]  # Delete two elements
        results = []

        # Iterate over the range 4 for c
        for c in range(4):
            # Compute the first part
            part1 = (P[c][0] - prep_4[:, :, :, c, 0:1] * P[c][2]) * prep_4[:, :, :, c, 2:3]
            # Compute the second part
            part2 = (P[c][1] - prep_4[:, :, :, c, 1:2] * P[c][2]) * prep_4[:, :, :, c, 2:3]
            
            # Append the results along the last axis
            results.append(part1)
            results.append(part2)
        # Concatenate all results along the new axis
        A4 = cp.stack(results, axis=3)  # Concatenate along the 4th dimension

        results1 = []
        final_results =[]
        # Iterate over the range 3 for c
        for i in range(4):
            results1 = []
            for c in range(3):
                camera_index = combinations_3[i]
                d = camera_index[c]
                
                # Compute the first part
                part1 = (P[d][0] - prep_3[:, :, i, c, 0:1] * P[d][2]) * prep_3[:, :, i, c, 2:3]
                # Compute the second part
                part2 = (P[d][1] - prep_3[:, :, i, c, 1:2] * P[d][2]) * prep_3[:, :, i, c, 2:3]
                #import pdb;pdb.set_trace()
                # Append the results along the last axis
                results1.append(part1)           
                results1.append(part2)
            inner_results_stacked = cp.stack(results1, axis=2)
            final_results.append(inner_results_stacked)
            
        A3 = cp.stack(final_results, axis=2) 
        
        results1 = []
        final_results =[]
        # Iterate over the range 3 for c
        for i in range(6):
            results1 = []
            for c in range(2):
                camera_index = combinations_2[i]
                d = camera_index[c]
                
                # Compute the first part
                part1 = (P[d][0] - prep_2[:, :, i, c, 0:1] * P[d][2]) * prep_2[:, :, i, c, 2:3]
                # Compute the second part
                part2 = (P[d][1] - prep_2[:, :, i, c, 1:2] * P[d][2]) * prep_2[:, :, i, c, 2:3]
                # Append the results along the last axis
                results1.append(part1)           
                results1.append(part2)
            inner_results_stacked = cp.stack(results1, axis=2) 
            final_results.append(inner_results_stacked)
            
        A2 = cp.stack(final_results, axis=2)

        return A4,A3,A2
    def find_real_dist_error(self, cam_coord, prep_4, Q4, prep_3, Q3, prep_2, Q2, Q3_bug, Q2_bug, P_cam_comb4, P_cam_comb3, comb_3, P_cam_comb2, comb_2):
    
        result_c4 = cp.einsum('cik,btk->cit', P_cam_comb4, Q4[:,:,0,:])
        
        x_c4 = result_c4[0] / result_c4[2]
        y_c4 = result_c4[1] / result_c4[2]
        dist_c4 = cp.sqrt(cp.sum((Q4[:, :, 0, 0:3] - cp.stack(cam_coord, axis=0)[:, None, :]) ** 2, axis=-1))
        
        X_c4 = cp.expand_dims(x_c4.T, axis=(0, 2))
        Y_c4 = cp.expand_dims(y_c4.T, axis=(0, 2))
        final_dist_c4 = cp.expand_dims(dist_c4.T, axis=(0, 2))

        rpj_coor_4 = cp.stack((X_c4,Y_c4), axis = -1)
        
        x1, y1 = prep_4[:, :, :, :, 0], prep_4[:, :, :, :, 1]
        x2, y2 = rpj_coor_4[:, :, :, :, 0], rpj_coor_4[:, :, :, :, 1]
        # Compute the Euclidean distance
        rpj = cp.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        real_dist  = final_dist_c4*rpj
        real_dist4 = cp.max(real_dist, axis=-1)

        # real_dist3
        

        result_c3 = cp.einsum('ncik,nbtk->ncit', P_cam_comb3, Q3.transpose(2, 0, 1, 3))
        x_c3 = result_c3[:, 0] / result_c3[:, 2]
        y_c3 = result_c3[:, 1] / result_c3[:, 2]

        dist_c3 = cp.sqrt(cp.sum((Q3_bug.transpose(2, 0, 1, 3)[:, :, :, 0:3] - cam_coord[comb_3][:, :, None, :]) ** 2, axis=-1))
        X_c3 = cp.expand_dims(x_c3.transpose(2, 0, 1), axis=0)
        Y_c3 = cp.expand_dims(y_c3.transpose(2, 0, 1), axis=0)
        final_dist_c3 = cp.expand_dims(dist_c3.transpose(2, 0, 1), axis=0)

        rpj_coor_3 = cp.stack((X_c3,Y_c3),axis = -1)
        
        x1, y1 = prep_3[:, :, :, :, 0], prep_3[:, :, :, :, 1]
        x2, y2 = rpj_coor_3[:, :, :, :, 0], rpj_coor_3[:, :, :, :, 1]
        # Compute the Euclidean distance
        rpj = cp.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        real_dist  = final_dist_c3*rpj
        real_dist3 = cp.max(real_dist, axis=-1)
        
        result_c2 = cp.einsum('ncik,nbtk->ncit', P_cam_comb2, Q2.transpose(2, 0, 1, 3))

        x_c2 = result_c2[:, 0] / result_c2[:, 2]
        y_c2 = result_c2[:, 1] / result_c2[:, 2]

        dist_c2 = cp.sqrt(cp.sum((Q2_bug.transpose(2, 0, 1, 3)[:, :, :, 0:3] - cam_coord[comb_2][:, :, None, :]) ** 2, axis=-1))
        X_c2 = cp.expand_dims(x_c2.transpose(2, 0, 1), axis=0)
        Y_c2 = cp.expand_dims(y_c2.transpose(2, 0, 1), axis=0)
        final_dist_c2 = cp.expand_dims(dist_c2.transpose(2, 0, 1), axis=0)  
        
        
        #[754.2223150392452, 1165.0827450392762]
        rpj_coor_2 = cp.stack((X_c2,Y_c2),axis = -1)
        
        x1, y1 = prep_2[:, :, :, :, 0], prep_2[:, :, :, :, 1]
        x2, y2 = rpj_coor_2[:, :, :, :, 0], rpj_coor_2[:, :, :, :, 1]
        # Compute the Euclidean distance
        rpj = cp.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        real_dist  = final_dist_c2*rpj
        real_dist2 = cp.max(real_dist, axis=-1)

        return real_dist4, real_dist3, real_dist2    
    def map_to_listdynamic(self, value):
        if value == 4:
            return [1,2,3,4,5,6,7,8,9,10]  # Map 4 to [0]
        elif value == 3:
            return [5,6,7,8,9,10]  # Map 3 to [1, 2, 3, 4]
        elif value == 2:
            return []  # Map 2 to [5, 6, 7, 8, 9, 10]      
            # triangulation
    def undistort_points_cupy(self, prep_4,prep_3,prep_2):
        
        x = prep_4[:, :, :, :, 0]  # Shape: (155, 22, 4, 3)
        y = prep_4[:, :, :, :, 1]  # Shape: (155, 22, 4, 3)
        likelihood = prep_4[:, :, :, :, 2]  # Shape: (155, 22, 4, 3)

        # Perform bilinear interpolation on x and y
        x_undistorted = self.bilinear_interpolate_cupy(self.mappingx, x, y)  # Shape: (155, 22, 4, 3)
        y_undistorted = self.bilinear_interpolate_cupy(self.mappingy, x, y)  # Shape: (155, 22, 4, 3)

        # Combine results into a single array
        prep_4 = cp.stack([x_undistorted, y_undistorted, likelihood], axis=-1)  # Shape: (155, 22, 4, 3, 3)

        # Extract x, y, and likelihood values
        x = prep_3[:, :, :, :, 0]  # Shape: (155, 22, 4, 3)
        y = prep_3[:, :, :, :, 1]  # Shape: (155, 22, 4, 3)
        likelihood = prep_3[:, :, :, :, 2]  # Shape: (155, 22, 4, 3)

        # Perform bilinear interpolation on x and y
        x_undistorted = self.bilinear_interpolate_cupy(self.mappingx, x, y)  # Shape: (155, 22, 4, 3)
        y_undistorted = self.bilinear_interpolate_cupy(self.mappingy, x, y)  # Shape: (155, 22, 4, 3)

        # Combine results into a single array
        prep_3 = cp.stack([x_undistorted, y_undistorted, likelihood], axis=-1)  # Shape: (155, 22, 4, 3, 3)


        x = prep_2[:, :, :, :, 0]  # Shape: (155, 22, 4, 3)
        y = prep_2[:, :, :, :, 1]  # Shape: (155, 22, 4, 3)
        likelihood = prep_2[:, :, :, :, 2]  # Shape: (155, 22, 4, 3)

        # Perform bilinear interpolation on x and y
        x_undistorted = self.bilinear_interpolate_cupy(self.mappingx, x, y)  # Shape: (155, 22, 4, 3)
        y_undistorted = self.bilinear_interpolate_cupy(self.mappingy, x, y)  # Shape: (155, 22, 4, 3)

        # Combine results into a single array
        prep_2 = cp.stack([x_undistorted, y_undistorted, likelihood], axis=-1)  # Shape: (155, 22, 4, 3, 3)
        return prep_4, prep_3, prep_2

    def bilinear_interpolate_cupy(self, map, x, y):
        
        # Get integer coordinates surrounding the point
        x0 = cp.floor(x).astype(cp.int32)
        y0 = cp.floor(y).astype(cp.int32)
        # Ensure coordinates are within bounds
        x0 = cp.clip(x0,0, map.shape[1] - 2)
        y0 = cp.clip(y0,0, map.shape[0] - 2)

        x1 = x0 + 1
        y1 = y0 + 1

        # Use cp.take_along_axis for advanced indexing
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