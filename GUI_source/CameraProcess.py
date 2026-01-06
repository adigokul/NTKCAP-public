import os
import sys
import cv2
import time
import numpy as np
from datetime import datetime
from multiprocessing import Event, shared_memory, Manager, Queue, Array, Lock, Process

class CameraProcess(Process):
    """
    Camera capture process supporting multiple camera types:
    - USB Webcam (DirectShow on Windows, V4L2 on Linux)
    - PoE Depth Cameras (LIPSedge AE400/AE450 via OpenNI2)
    """
    def __init__(self, shared_name, cam_id, start_evt, task_rec_evt, apose_rec_evt, calib_rec_evt, stop_evt, task_stop_rec_evt, calib_video_save_path, queue, address_base, record_task_name, cam_type='usb', cam_config=None, *args, **kwargs):
        super().__init__()
        ### define
        self.shared_name = shared_name
        self.cam_id = cam_id  # Logical camera index (0, 1, 2, 3)
        self.cam_type = cam_type  # 'usb', 'ae400', 'ae450', or 'poe'
        self.cam_config = cam_config or {}
        # Get actual device index from config (e.g., 0, 2, 4, 6 on Linux)
        self.device_index = self.cam_config.get('device_index', cam_id)
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

        # PoE camera specific
        self.openni2_initialized = False
        self.color_stream = None
        self.device = None

    # =========================================================================
    # USB Camera Methods
    # =========================================================================
    def _init_usb_camera(self):
        """Initialize USB webcam with platform-specific backend."""
        device_id = self.device_index

        if sys.platform.startswith('win'):
            cap = cv2.VideoCapture(device_id, cv2.CAP_DSHOW)
            print(f"[Cam {self.cam_id + 1}] Opening USB device {device_id} with DirectShow backend")
        else:
            cap = cv2.VideoCapture(device_id, cv2.CAP_V4L2)
            if not cap.isOpened():
                cap = cv2.VideoCapture(device_id)
            print(f"[Cam {self.cam_id + 1}] Opening USB device {device_id} with V4L2/default backend")

        if not cap.isOpened():
            print(f"[Cam {self.cam_id + 1}] ❌ USB camera device {device_id} failed to open")
            return None

        # Set camera resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 30)

        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[Cam {self.cam_id + 1}] USB camera: {actual_width}x{actual_height}")

        return cap

    def _read_usb_frame(self, cap):
        """Read a frame from USB camera. Returns (success, frame_bgr)."""
        ret, frame = cap.read()
        return ret, frame

    def _release_usb_camera(self, cap):
        """Release USB camera resources."""
        if cap is not None:
            cap.release()
        print(f"[Cam {self.cam_id + 1}] USB camera released")

    # =========================================================================
    # PoE Camera Methods (LIPSedge AE400/AE450 via OpenNI2)
    # =========================================================================
    def _init_poe_camera(self):
        """Initialize PoE camera (AE400/AE450) via OpenNI2."""
        try:
            from openni import openni2
        except ImportError:
            print(f"[Cam {self.cam_id + 1}] ❌ OpenNI2 Python bindings not installed")
            print(f"[Cam {self.cam_id + 1}] Install with: pip install openni")
            return None

        openni2_path = self.cam_config.get('openni2_path', '')
        camera_ip = self.cam_config.get('ip', 'unknown')

        if not openni2_path:
            print(f"[Cam {self.cam_id + 1}] ❌ OpenNI2 path not configured")
            return None

        if not os.path.exists(openni2_path):
            print(f"[Cam {self.cam_id + 1}] ❌ OpenNI2 path does not exist: {openni2_path}")
            return None

        try:
            print(f"[Cam {self.cam_id + 1}] Initializing PoE camera @ {camera_ip}")
            print(f"[Cam {self.cam_id + 1}] OpenNI2 path: {openni2_path}")

            # Initialize OpenNI2 with camera-specific path
            openni2.initialize(openni2_path)
            self.openni2_initialized = True
            time.sleep(0.5)  # Allow initialization to complete

            # Open device
            self.device = openni2.Device.open_any()
            device_info = self.device.get_device_info()
            print(f"[Cam {self.cam_id + 1}] Device: {device_info.name}")

            # Create and start color stream
            self.color_stream = self.device.create_color_stream()
            self.color_stream.start()

            # Get video mode info
            vm = self.color_stream.get_video_mode()
            print(f"[Cam {self.cam_id + 1}] PoE camera: {vm.resolutionX}x{vm.resolutionY} @ {vm.fps}fps")

            return self.color_stream

        except Exception as e:
            print(f"[Cam {self.cam_id + 1}] ❌ PoE camera initialization failed: {e}")
            self._release_poe_camera(None)
            return None

    def _read_poe_frame(self, stream):
        """Read a frame from PoE camera. Returns (success, frame_bgr)."""
        try:
            from openni import openni2

            # Wait for frame with timeout
            TIMEOUT_MS = 2000
            idx = openni2.wait_for_any_stream([stream], TIMEOUT_MS)

            if idx is None:
                return False, None

            # Read frame
            frame = stream.read_frame()
            buf = frame.get_buffer_as_uint8()
            vm = stream.get_video_mode()
            w, h = vm.resolutionX, vm.resolutionY

            # Convert RGB888 to BGR for OpenCV
            rgb = np.asarray(buf).reshape((h, w, 3))
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            return True, bgr

        except Exception as e:
            print(f"[Cam {self.cam_id + 1}] PoE frame read error: {e}")
            return False, None

    def _release_poe_camera(self, stream):
        """Release PoE camera resources."""
        try:
            from openni import openni2

            if self.color_stream is not None:
                try:
                    self.color_stream.stop()
                except:
                    pass
                self.color_stream = None

            if self.device is not None:
                try:
                    self.device.close()
                except:
                    pass
                self.device = None

            if self.openni2_initialized:
                try:
                    openni2.unload()
                except:
                    pass
                self.openni2_initialized = False
                time.sleep(1.0)  # Allow full resource release

            print(f"[Cam {self.cam_id + 1}] PoE camera released")

        except Exception as e:
            print(f"[Cam {self.cam_id + 1}] Error releasing PoE camera: {e}")

    # =========================================================================
    # Main Run Loop
    # =========================================================================
    def run(self):
        shape = (1080, 1920, 3)
        target_width, target_height = 1920, 1080

        # Shared memory setup
        idx = 0
        existing_shm = shared_memory.SharedMemory(name=self.shared_name)
        shared_array = np.ndarray((self.buffer_length,) + shape, dtype=np.uint8, buffer=existing_shm.buf)

        # Initialize camera based on type
        is_poe = self.cam_type in ('ae400', 'ae450', 'poe')

        if is_poe:
            cap = self._init_poe_camera()
            camera_label = f"PoE:{self.cam_config.get('ip', '?')}"
        else:
            cap = self._init_usb_camera()
            camera_label = f"USB:{self.device_index}"

        if cap is None:
            print(f"[Cam {self.cam_id + 1}] ❌ Camera initialization failed, exiting")
            return

        # Frame counters
        self.frame_id_task = 0
        self.frame_id_apose = 0
        self.frame_counter = 0
        self.fps_start_time = time.time()
        self.fps_counter = 0
        self.current_fps = 0
        consecutive_failures = 0
        MAX_FAILURES = 30  # Max consecutive read failures before giving up

        print(f"[Cam {self.cam_id + 1}] Camera initialized ({camera_label})")
        self.start_evt.wait()

        while True:
            cap_s = time.time()

            # Read frame based on camera type
            if is_poe:
                ret, frame = self._read_poe_frame(cap)
            else:
                ret, frame = self._read_usb_frame(cap)

            cap_e = time.time()

            # Handle stop event
            if self.stop_evt.is_set():
                break

            # Handle read failures
            if not ret or frame is None:
                consecutive_failures += 1
                if consecutive_failures >= MAX_FAILURES:
                    print(f"[Cam {self.cam_id + 1}] ❌ Too many consecutive failures, exiting")
                    break
                continue

            consecutive_failures = 0  # Reset on successful read

            # Resize frame to target resolution if needed
            if frame.shape[1] != target_width or frame.shape[0] != target_height:
                frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

            # Update frame counter and calculate FPS
            self.frame_counter += 1
            self.fps_counter += 1

            if self.fps_counter >= 30:
                current_time = time.time()
                elapsed_time = current_time - self.fps_start_time
                if elapsed_time > 0:
                    self.current_fps = self.fps_counter / elapsed_time
                self.fps_counter = 0
                self.fps_start_time = current_time

            # Add overlay to frame
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.2
            color = (0, 255, 0)  # Green
            thickness = 2

            # Camera info (top-left)
            cv2.putText(frame, f"Cam {self.cam_id + 1} ({camera_label})", (20, 40), font, font_scale, color, thickness)
            cv2.putText(frame, f"Frame: {self.frame_counter}", (20, 80), font, font_scale, color, thickness)

            # FPS (top-right)
            fps_text = f"FPS: {self.current_fps:.1f}"
            text_size = cv2.getTextSize(fps_text, font, font_scale, thickness)[0]
            cv2.putText(frame, fps_text, (target_width - text_size[0] - 20, 40), font, font_scale, color, thickness)

            # Handle recording events
            if self.task_stop_rec_evt.is_set() and self.recording:
                self.recording = False
                self.start_time = None
                self.frame_id_task = 0
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
                self.time_stamp[self.frame_id_task] = [np.array(float(f"{dt_str1:.3f}")), np.array(float(f"{dt_str2:.3f}"))]
                self.frame_id_task += 1

            # Copy frame to shared memory
            np.copyto(shared_array[idx, :], frame)

            # Queue management - prevent overflow
            try:
                self.queue.put_nowait(idx)
            except:
                try:
                    while self.queue.qsize() > 2:
                        self.queue.get_nowait()
                    self.queue.put_nowait(idx)
                except:
                    pass

            idx = (idx + 1) % self.buffer_length

        # Release camera
        if is_poe:
            self._release_poe_camera(cap)
        else:
            self._release_usb_camera(cap)
