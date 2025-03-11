from multiprocessing import Event, shared_memory, Manager, Queue, Array, Lock, Process
import time
import numpy as np
from collections import deque

class SyncProcess(Process):
    def __init__(self, raw_frame_shm_name0, raw_frame_shm_name1, raw_frame_shm_name2, raw_frame_shm_name3, sync_frame_shm_name0, sync_frame_shm_name1, sync_frame_shm_name2, sync_frame_shm_name3, time_stamp_array_lst, camera_queue_lst, sync_frame_queue_lst, *args, **kwargs):
        super().__init__()
        self.raw_frame_shm_name0 = raw_frame_shm_name0
        self.raw_frame_shm_name1 = raw_frame_shm_name1
        self.raw_frame_shm_name2 = raw_frame_shm_name2
        self.raw_frame_shm_name3 = raw_frame_shm_name3
        self.sync_frame_shm_name0 = sync_frame_shm_name0
        self.sync_frame_shm_name1 = sync_frame_shm_name1
        self.sync_frame_shm_name2 = sync_frame_shm_name2
        self.sync_frame_shm_name3 = sync_frame_shm_name3
        
        self.time_stamp_array_lst = time_stamp_array_lst
        self.camera_queue_lst = camera_queue_lst
        self.sync_frame_queue_lst = sync_frame_queue_lst
        # self.start_evt = start_evt
        # self.stop_evt = stop_evt
        self.buffer_length = 20
        self.frame_shm_shape = (1080, 1920, 3)
        self.shm_raw_frame_lst = []
        self.TR = 50 # ms
        self.frame_sync_buffer = deque(maxlen=30)
        self.time_stamp_sync_buffer = deque(maxlen=30)
    
    def run(self):
        self.shm_raw_frame_lst.clear()
        
        
        existing_shm_raw_frame0 = shared_memory.SharedMemory(name=self.raw_frame_shm_name0)
        shared_array_raw_frame0 = np.ndarray((self.buffer_length,) + self.frame_shm_shape, dtype=np.uint8, buffer=existing_shm_raw_frame0.buf)

        existing_shm_raw_frame1 = shared_memory.SharedMemory(name=self.raw_frame_shm_name1)
        shared_array_raw_frame1 = np.ndarray((self.buffer_length,) + self.frame_shm_shape, dtype=np.uint8, buffer=existing_shm_raw_frame1.buf)

        existing_shm_raw_frame2 = shared_memory.SharedMemory(name=self.raw_frame_shm_name2)
        shared_array_raw_frame2 = np.ndarray((self.buffer_length,) + self.frame_shm_shape, dtype=np.uint8, buffer=existing_shm_raw_frame2.buf)

        existing_shm_raw_frame3 = shared_memory.SharedMemory(name=self.raw_frame_shm_name3)
        shared_array_raw_frame3 = np.ndarray((self.buffer_length,) + self.frame_shm_shape, dtype=np.uint8, buffer=existing_shm_raw_frame3.buf)
        self.shm_raw_frame_lst.append(shared_array_raw_frame0)
        self.shm_raw_frame_lst.append(shared_array_raw_frame1)
        self.shm_raw_frame_lst.append(shared_array_raw_frame2)
        self.shm_raw_frame_lst.append(shared_array_raw_frame3)

        existing_shm_sync_frame0 = shared_memory.SharedMemory(name=self.sync_frame_shm_name0)
        shared_array_sync_frame0 = np.ndarray((self.buffer_length,) + self.frame_shm_shape, dtype=np.uint8, buffer=existing_shm_sync_frame0.buf)

        existing_shm_sync_frame1 = shared_memory.SharedMemory(name=self.sync_frame_shm_name1)
        shared_array_sync_frame1 = np.ndarray((self.buffer_length,) + self.frame_shm_shape, dtype=np.uint8, buffer=existing_shm_sync_frame1.buf)
        existing_shm_sync_frame2 = shared_memory.SharedMemory(name=self.sync_frame_shm_name2)
        shared_array_sync_frame2 = np.ndarray((self.buffer_length,) + self.frame_shm_shape, dtype=np.uint8, buffer=existing_shm_sync_frame2.buf)

        existing_shm_sync_frame3 = shared_memory.SharedMemory(name=self.sync_frame_shm_name3)
        shared_array_sync_frame3 = np.ndarray((self.buffer_length,) + self.frame_shm_shape, dtype=np.uint8, buffer=existing_shm_sync_frame3.buf)

        # self.start_evt.wait()
        time_stamp_frame = [None] * 4
        raw_frame_all = [None] * 4
        time_stamp_frame_sync = []
        raw_frame_sync = []
        next_queue = [0, 0, 0, 0]
        check = 0
        idx = 0
        count = 0
        time.sleep(1)
        while True:
            
            not_get = [0, 1, 2, 3]
            time_stamp_frame = [None] * 4
            raw_frame_all = [None] * 4
            time_stamp_frame_sync.clear()
            raw_frame_sync.clear()

            while not_get:
                i = not_get[0]
                try:
                    idx_get = self.camera_queue_lst[i].get_nowait()
                    not_get.pop(0)
                    time_stamp = self.time_stamp_array_lst[i][idx_get]
                    raw_frame = self.shm_raw_frame_lst[i][idx_get, :]
                    time_stamp_frame[i] = time_stamp
                    raw_frame_all[i] = raw_frame
                except Exception as e:
                    not_get.append(not_get.pop(0))
            if self.check_follow_up(time_stamp_frame):
                self.frame_sync_buffer.clear()
                self.time_stamp_sync_buffer.clear()
                next_queue = [0, 0, 0, 0]
                np.copyto(shared_array_sync_frame0[idx, :], raw_frame_all[0])
                np.copyto(shared_array_sync_frame1[idx, :], raw_frame_all[1])
                self.sync_frame_queue_lst[0].put(idx)
                np.copyto(shared_array_sync_frame2[idx, :], raw_frame_all[2])
                np.copyto(shared_array_sync_frame3[idx, :], raw_frame_all[3])                
                self.sync_frame_queue_lst[1].put(idx)
                
                
            else:
                self.frame_sync_buffer.append(raw_frame_all.copy())
                self.time_stamp_sync_buffer.append(time_stamp_frame.copy())
                for pos_idx, pos in enumerate(next_queue):
                    time_stamp_frame_sync.append(self.time_stamp_sync_buffer[pos][pos_idx])
                    raw_frame_sync.append(self.frame_sync_buffer[pos][pos_idx])
                np.copyto(shared_array_sync_frame0[idx, :], raw_frame_sync[0])
                np.copyto(shared_array_sync_frame1[idx, :], raw_frame_sync[1])
                self.sync_frame_queue_lst[0].put(idx)
                np.copyto(shared_array_sync_frame2[idx, :], raw_frame_sync[2])
                np.copyto(shared_array_sync_frame3[idx, :], raw_frame_sync[3])                
                self.sync_frame_queue_lst[1].put(idx)
                
                for i in range(4):
                    if max(time_stamp_frame_sync)-time_stamp_frame_sync[i] > self.TR:
                        next_queue[i] += 1
                        check += 1
                if check == 0:
                    self.frame_sync_buffer.popleft()
                    self.time_stamp_sync_buffer.popleft()
                else:
                    check = 0
                if max(time_stamp_frame) - min(time_stamp_frame) > self.TR:
                    try:
                        _ = self.camera_queue_lst[time_stamp_frame.index(min(time_stamp_frame))].get(timeout=0.05)
                    except:
                        continue
            idx = (idx+1) % self.buffer_length
            print("sync", count)
            count += 1
            
    def check_follow_up(self, time_stamp_frame):
        return all(max(time_stamp_frame) - ts <= self.TR for ts in time_stamp_frame)  