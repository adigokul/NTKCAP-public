import os
import cv2
import time
import numpy as np
from datetime import datetime
from multiprocessing import Event, shared_memory, Manager, Queue, Array, Lock, Process

class CameraProcess(Process):
    def __init__(self, shared_name, cam_id, start_evt, task_rec_evt, apose_rec_evt, calib_rec_evt, stop_evt, task_stop_rec_evt, calib_video_save_path, queue, address_base, record_task_name, cam_type='usb', cam_config=None, *args, **kwargs):
        super().__init__()
        ### define
        self.shared_name = shared_name
        self.cam_id = cam_id
        self.start_evt = start_evt
        self.task_rec_evt = task_rec_evt
        self.apose_rec_evt = apose_rec_evt
        self.calib_rec_evt = calib_rec_evt
        self.stop_evt = stop_evt
        self.task_stop_rec_evt = task_stop_rec_evt
        self.queue = queue
        self.buffer_length = 4
        self.delay_time = 0.2
        self.shared_dict_record = record_task_name
        self.record_date = datetime.now().strftime("%Y_%m_%d")
        self.calib_video_save_path = calib_video_save_path
        ### camera type
        self.cam_type = cam_type  # 'usb' or 'ae400'
        self.cam_config = cam_config or {}
        ### initialization
        self.recording = False
        self.start_time = None
        self.frame_width = 1920
        self.frame_height = 1080
        self.fps = 30
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.patientID_path = address_base
        self.time_stamp = None
        self.record_path = None

    def _init_usb_camera(self):
        """Initialize USB camera"""
        cap = cv2.VideoCapture(self.cam_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        print(f"[Cam {self.cam_id + 1}] USB camera initialized")
        return cap

    def _init_ae400_camera(self):
        """Initialize AE400 depth camera (RGB stream only)"""
        try:
            from openni import openni2
            import os

            openni2_path = self.cam_config.get('openni2_path')
            if not openni2_path:
                raise ValueError(f"[Cam {self.cam_id + 1}] Missing openni2_path in config")

            # Initialize OpenNI2
            openni2.initialize(openni2_path)
            dev = openni2.Device.open_any()

            # Create color stream
            color_stream = dev.create_color_stream()
            color_stream.start()

            print(f"[Cam {self.cam_id + 1}] AE400 camera initialized at {self.cam_config.get('ip', 'unknown IP')}")

            return {
                'type': 'ae400',
                'device': dev,
                'stream': color_stream,
                'openni2': openni2
            }
        except Exception as e:
            print(f"[Cam {self.cam_id + 1}] AE400 init error: {e}")
            raise

    def _read_frame(self, cap):
        """Unified frame reading interface"""
        if self.cam_type == 'usb':
            ret, frame = cap.read()
            return ret, frame

        elif self.cam_type == 'ae400':
            try:
                from openni import openni2
                stream = cap['stream']

                # Wait for frame with timeout
                idx = openni2.wait_for_any_stream([stream], timeout=2000)
                if idx is None:
                    return False, None

                # Read frame
                frame_data = stream.read_frame()
                buf = frame_data.get_buffer_as_uint8()
                vm = stream.get_video_mode()
                w, h = vm.resolutionX, vm.resolutionY

                # Convert RGB888 to BGR for OpenCV
                rgb = np.asarray(buf).reshape((h, w, 3))
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                # Resize to target resolution if needed
                if (w != self.frame_width) or (h != self.frame_height):
                    bgr = cv2.resize(bgr, (self.frame_width, self.frame_height), interpolation=cv2.INTER_LINEAR)

                return True, bgr
            except Exception as e:
                print(f"[Cam {self.cam_id + 1}] AE400 read error: {e}")
                return False, None

        return False, None

    def _release_camera(self, cap):
        """Unified camera release interface"""
        if self.cam_type == 'usb':
            try:
                cap.release()
                print(f"[Cam {self.cam_id + 1}] USB camera released")
            except Exception as e:
                print(f"[Cam {self.cam_id + 1}] USB release error: {e}")

        elif self.cam_type == 'ae400':
            try:
                cap['stream'].stop()
                cap['openni2'].unload()
                print(f"[Cam {self.cam_id + 1}] AE400 camera released")
            except Exception as e:
                print(f"[Cam {self.cam_id + 1}] AE400 release error: {e}")

    def run(self):
        shape = (1080, 1920, 3)
        # shared memory setup
        idx = 0 # the current index of frames in shared memory
        existing_shm = shared_memory.SharedMemory(name=self.shared_name)
        shared_array = np.ndarray((self.buffer_length,) + shape, dtype=np.uint8, buffer=existing_shm.buf)

        # Initialize camera based on type
        if self.cam_type == 'usb':
            cap = self._init_usb_camera()
        elif self.cam_type == 'ae400':
            cap = self._init_ae400_camera()
        else:
            print(f"[Cam {self.cam_id + 1}] Unknown camera type: {self.cam_type}")
            return

        (width, height) = (1920, 1080)
        self.frame_id_task = 0
        self.frame_id_apose = 0
        self.frame_counter = 0
        self.fps_start_time = time.time()
        self.fps_counter = 0
        self.current_fps = 0

        self.start_evt.wait()

        while True:
            cap_s = time.time()
            ret, frame = self._read_frame(cap)  # Use unified read interface
            cap_e = time.time()

            if not ret or self.stop_evt.is_set():
                break
            
            # Update frame counter and calculate FPS
            self.frame_counter += 1
            self.fps_counter += 1
            
            # Calculate FPS every 30 frames
            if self.fps_counter >= 30:
                current_time = time.time()
                elapsed_time = current_time - self.fps_start_time
                if elapsed_time > 0:
                    self.current_fps = self.fps_counter / elapsed_time
                self.fps_counter = 0
                self.fps_start_time = current_time
            
            # Add camera index and FPS overlay to frame
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.2
            color = (0, 255, 0)  # Green color
            thickness = 2
            
            # Camera index (top-left)
            cv2.putText(frame, f"Cam {self.cam_id + 1}", (20, 40), font, font_scale, color, thickness)
            
            # Frame counter (top-left, below camera index)
            cv2.putText(frame, f"Frame: {self.frame_counter}", (20, 80), font, font_scale, color, thickness)
            
            # FPS (top-right)
            fps_text = f"FPS: {self.current_fps:.1f}"
            text_size = cv2.getTextSize(fps_text, font, font_scale, thickness)[0]
            cv2.putText(frame, fps_text, (width - text_size[0] - 20, 40), font, font_scale, color, thickness)
            
            if self.task_stop_rec_evt.is_set() and self.recording:
                self.recording = False
                self.start_time = None
                self.frame_id_task = 0
                # Reset frame counter when stopping recording to prevent overflow
                self.frame_counter = 0
                self.out.release()
                np.save(os.path.join(self.record_path, f"{self.cam_id+1}_dates.npy"), self.time_stamp)
                
            if self.apose_rec_evt.is_set():
                
                if not self.recording:
                    self.record_path = None
                    self.recording = True
                    self.record_path = os.path.join(self.patientID_path, self.shared_dict_record['name'], self.record_date, 'raw_data', "Apose", "videos")
                    self.out = cv2.VideoWriter(os.path.join(self.record_path, f"{self.cam_id+1}.mp4"), self.fourcc, self.fps, (self.frame_width, self.frame_height))
                    self.frame_id_apose = 0
                self.out.write(frame)
                if self.frame_id_apose < 9:
                    
                    self.frame_id_apose += 1
                else:
                    self.recording = False
                    self.frame_id_apose = 0
                    self.out.release()
                    self.apose_rec_evt.clear()
                
            if self.calib_rec_evt.is_set():
                self.record_path = None
                self.out = cv2.VideoWriter(os.path.join(self.calib_video_save_path, f"{self.cam_id+1}.mp4"), self.fourcc, self.fps, (self.frame_width, self.frame_height))
                self.out.write(frame)
                self.out.release()
                self.calib_rec_evt.clear()
            if self.task_rec_evt.is_set():
                if not self.recording:
                    self.record_path = None
                    self.time_stamp = None
                    self.recording = True
                    self.record_path = os.path.join(self.patientID_path, self.shared_dict_record['name'], self.record_date, 'raw_data', self.shared_dict_record['task_name'], 'videos')
                    self.out = cv2.VideoWriter(os.path.join(self.record_path, f"{self.cam_id+1}.mp4"), self.fourcc, self.fps, (self.frame_width, self.frame_height))
                    self.frame_id_task = 0
                    self.start_time = self.shared_dict_record['start_time']
                    self.time_stamp = np.empty((1000000, 2))
                self.out.write(frame)
                dt_str1 = cap_s - self.start_time
                dt_str2 = cap_e - self.start_time
                self.time_stamp[self.frame_id_task] =[np.array(float( f"{dt_str1 :.3f}")),np.array(float( f"{dt_str2 :.3f}"))]
                self.frame_id_task += 1
            np.copyto(shared_array[idx,:], frame)
            
            # Prevent queue overflow by dropping old frames
            try:
                self.queue.put_nowait(idx)
            except:
                # Queue is full, skip old frames to prevent buffer overflow
                try:
                    while self.queue.qsize() > 2:  # Keep only 2 frames in queue
                        self.queue.get_nowait()
                    self.queue.put_nowait(idx)
                except:
                    pass  # Queue operations failed, continue
                    
            idx = (idx+1) % self.buffer_length

        self._release_camera(cap)