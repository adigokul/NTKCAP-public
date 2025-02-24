from multiprocessing import Event, shared_memory, Manager, Queue, Array, Lock, Process
import time
import numpy as np

class PreTri_Process(Process):
    def __init__(self, sync_tracker_shm_name0, sync_tracker_shm_name1, sync_tracker_shm_name2, sync_tracker_shm_name3, sync_tracker_queue_lst, keypoints_sync_shm_name, sync_tracker_queue, start_evt, stop_evt, *args, **kwargs):
        super().__init__()
        self.sync_tracker_shm_name0 = sync_tracker_shm_name0
        self.sync_tracker_shm_name1 = sync_tracker_shm_name1
        self.sync_tracker_shm_name2 = sync_tracker_shm_name2
        self.sync_tracker_shm_name3 = sync_tracker_shm_name3
        self.sync_tracker_shm_lst = []
        self.sync_tracker_queue_lst = sync_tracker_queue_lst
        self.keypoints_sync_shm_name = keypoints_sync_shm_name
        self.keypoints_sync_shm_shape = (4, 1, 26, 3)
        self.keypoints_shm_shape = (1, 26, 3)
        self.sync_tracker_queue = sync_tracker_queue
        self.buffer_length = 20
        self.start_evt = start_evt
        self.stop_evt = stop_evt
    def run(self):
        self.sync_tracker_shm_lst.clear()
        existing_shm_keypoints = shared_memory.SharedMemory(name=self.keypoints_sync_shm_name)
        shared_array_keypoints = np.ndarray((self.buffer_length,) + self.keypoints_sync_shm_shape, dtype=np.float32, buffer=existing_shm_keypoints.buf)
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
        
        idx = 0
        count = 0
        time.sleep(1)
        while self.start_evt.is_set():
            not_get = [0, 1, 2, 3]
            keypoints_frame = [None] * 4
            # cap_s = time.time()

            while not_get:
                i = not_get[0]
                try:
                    idx_get = self.sync_tracker_queue_lst[i].get_nowait()
                    not_get.pop(0)
                    keypoints = self.sync_tracker_shm_lst[i][idx_get]
                    keypoints_frame[i] = keypoints
                except Exception as e:
                    not_get.append(not_get.pop(0))
            np.copyto(shared_array_keypoints[idx, :], keypoints_frame)
            self.sync_tracker_queue.put(idx)
            idx = (idx+1) % self.buffer_length
            # print("send to triangulation", count)
            count += 1