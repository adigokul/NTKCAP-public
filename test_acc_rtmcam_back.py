import numpy as np
import cv2
import time
from multiprocessing import Process, Queue, shared_memory, Value, Lock
import os
from mmdeploy_runtime import PoseTracker
import os
# Define paths for the PoseTracker model
dir_mmdeploy = r'C:\Users\mauricetemp\Desktop\NTKCAP\NTK_CAP\ThirdParty\mmdeploy'
det_model_path = os.path.join(dir_mmdeploy, "rtmpose-trt", "rtmdet-m")
pose_model_path = os.path.join(dir_mmdeploy, "rtmpose-trt", "rtmpose-m")
device = "cuda"

# Visualization configuration
VISUALIZATION_CFG = dict(
    halpe26=dict(
        skeleton=[(15,13), (13,11), (11,19),(16,14), (14,12), (12,19),
                  (17,18), (18,19), (18,5), (5,7), (7,9), (18,6), (6,8),
                  (8,10), (1,2), (0,1), (0,2), (1,3), (2,4), (3,5), (4,6),
                  (15,20), (15,22), (15,24),(16,21),(16,23), (16,25)],
        palette=[(51,153,255), (0,255,0), (255,128,0)],
        link_color=[
            1, 1, 1, 2, 2, 2, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2
        ],
        point_color=[
            0, 0, 0, 0, 0, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 2, 2, 2, 1, 2, 1, 2, 1, 2
        ],
        sigmas=[
            0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072,
            0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089, 0.026,
            0.026, 0.066, 0.079, 0.079, 0.079, 0.079, 0.079, 0.079
        ]))
sigmas = VISUALIZATION_CFG['halpe26']['sigmas']
palette = VISUALIZATION_CFG['halpe26']['palette']
skeleton = VISUALIZATION_CFG['halpe26']['skeleton']
link_color = VISUALIZATION_CFG['halpe26']['link_color']
point_color = VISUALIZATION_CFG['halpe26']['point_color']

# Function to capture frames
def capture_frames(cam_index, shm_name, frame_shape, frame_size, buffer_length, capture_index, frame_queue):
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"Camera {cam_index} could not be opened.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_shape[1])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_shape[0])
    cap.set(cv2.CAP_PROP_FPS, 30)

    existing_shm = shared_memory.SharedMemory(name=shm_name)
    frame_data = np.ndarray((buffer_length * frame_size,), dtype=np.uint8, buffer=existing_shm.buf)

    prev_time = time.time()

    while True:
        current_time = time.time()

        ret, frame = cap.read()
        if not ret:
            print(f"Camera {cam_index}: Failed to capture frame")
            break

        frame_resized = cv2.resize(frame, (frame_shape[1], frame_shape[0]))

        buffer_index = capture_index.value % buffer_length
        start_index = buffer_index * frame_size

        frame_buffer = np.ndarray(frame_shape, dtype=np.uint8, buffer=frame_data[start_index:])
        frame_buffer[:] = frame_resized[:]

        with capture_index.get_lock():
            capture_index.value += 1

        frame_queue.put(buffer_index)

        prev_time = current_time
        time.sleep(0.01)

    cap.release()
    existing_shm.close()


# Function to process frames
def process_frames(cam_index, shm_name, shm_name_pose, frame_shape, frame_size, buffer_length, max_people, buffer_pose, frame_queue, display_queue, lock,num_cams):
    
    tracker = PoseTracker(det_model=det_model_path, pose_model=pose_model_path, device_name=device)
    state = tracker.create_state(det_interval=1, det_min_bbox_size=100, keypoint_sigmas=sigmas)
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    pose_shm = shared_memory.SharedMemory(name=shm_name_pose)
    
    frame_data = np.ndarray((buffer_length * frame_size,), dtype=np.uint8, buffer=existing_shm.buf)
    rtm_data = np.ndarray((buffer_pose * (max_people * 26 * 3 + 1) * num_cams), dtype=np.float16, buffer=pose_shm.buf)  # Adjust buffer size for 4 cameras

    prev_time = time.time()
    while True:
        try:
            buffer_index = frame_queue.get(timeout=5)
        except:
            print('No frame received.')
            continue

        buffer_index_pose = buffer_index % buffer_pose
        start_index = buffer_index * frame_size
        buffer_index_pose * (max_people * 26 * 3 + 1)
        start_index_pose = (buffer_index_pose + cam_index * buffer_pose) * (max_people * 26 * 3 + 1)  # Offset for camera index

        frame_buffer = np.ndarray(frame_shape, dtype=np.uint8, buffer=frame_data[start_index:])
        keypoints_buffer = np.ndarray((max_people * 26 * 3 + 1), dtype=np.float16, buffer=rtm_data[start_index_pose:])
        
        current_time = time.time()
        results = tracker(state, frame_buffer, detect=-1)
        keypoints, bboxes, _ = results
        keypoints_flattened = keypoints.flatten()
        people_count = len(results[0])

        arr = np.full((max_people * 26 * 3 + 1), -1.0)
        arr[0] = people_count
        arr[1:len(keypoints_flattened) + 1] = keypoints_flattened
        keypoints_buffer[:] = arr[:]

        display_queue.put(buffer_index)

        time.sleep(0.01)

    existing_shm.close()
    pose_shm.close()


# Function to display processed frames
def display_frames(cam_index, shm_name, shm_name_pose, frame_shape, frame_size, buffer_length, display_queue, palette, skeleton, link_color, point_color, max_people, buffer_pose,num_cams):
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    pose_shm = shared_memory.SharedMemory(name=shm_name_pose)

    frame_data = np.ndarray((buffer_length * frame_size,), dtype=np.uint8, buffer=existing_shm.buf)
    rtm_data = np.ndarray((buffer_pose * (max_people * 26 * 3 + 1) * num_cams), dtype=np.float16, buffer=pose_shm.buf)  # Adjust buffer size for 4 cameras

    last_time = time.time()

    while True:
        try:
            buffer_index = display_queue.get(timeout=5)
            start_index = buffer_index * frame_size
            start_index_pose = (buffer_index % buffer_pose + cam_index * buffer_pose) * (max_people * 26 * 3 + 1)  # Offset for camera index

            frame_buffer = np.ndarray(frame_shape, dtype=np.uint8, buffer=frame_data[start_index:])
            arr = np.ndarray((max_people * 26 * 3 + 1), dtype=np.float16, buffer=rtm_data[start_index_pose:])

            people_count = int(arr[0])
            fps = 1 / (time.time() - last_time)
            last_time = time.time()

            fps_text = f"FPS: {fps:.2f}"
            cv2.putText(frame_buffer, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            print(people_count)
            for i in range(people_count):
                arr_temp = arr[i * 78 + 1: i * 78 + 79]
                kpts = arr_temp.reshape(-1, 3)[:, :2]
                score = arr_temp.reshape(-1, 3)[:, 2]

                for (u, v), color in zip(skeleton, link_color):
                    if score[u] > 0.5 and score[v] > 0.5:
                        cv2.line(frame_buffer, tuple(map(int, kpts[u])), tuple(map(int, kpts[v])), palette[color], 2, cv2.LINE_AA)

                for kpt, show, color in zip(kpts, [1] * len(kpts), point_color):
                    if show:
                        cv2.circle(frame_buffer, tuple(map(int, kpt)), 1, palette[color], 2, cv2.LINE_AA)

            cv2.imshow(f"Camera {cam_index} Frame", frame_buffer)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except:
            continue

    existing_shm.close()
    pose_shm.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    from multiprocessing import Value, Lock

    num_cams = 1
    frame_shape = (833, 1250, 3)
    frame_size = np.prod(frame_shape)
    buffer_length = 4
    max_people = 15
    buffer_pose = 4

    shm_frame_data_list = []
    capture_index_list = []
    frame_queue_list = []
    display_queue_list = []

    shm_name_pose = "pose"
    shm_size_pose = int(buffer_pose * (max_people * 26 * 3 + 1) * num_cams * np.float16().nbytes)  # Single shared memory for poses
    shm_frame_data_pose = shared_memory.SharedMemory(create=True, size=shm_size_pose, name=shm_name_pose)

    # Create shared memory, queues, and values for each camera
    for cam_index in range(num_cams):
        shm_name = f"frame_{cam_index}"
        shm_size = int(buffer_length * frame_size)
        shm_frame_data = shared_memory.SharedMemory(create=True, size=shm_size, name=shm_name)
        shm_frame_data_list.append(shm_frame_data)

        capture_index = Value('i', 0, lock=True)
        capture_index_list.append(capture_index)

        frame_queue = Queue()
        frame_queue_list.append(frame_queue)

        display_queue = Queue()
        display_queue_list.append(display_queue)

    # Start capture processes
    capture_processes = []
    for cam_index in range(num_cams):
        p_capture = Process(target=capture_frames, args=(cam_index, f"frame_{cam_index}", frame_shape, frame_size, buffer_length, capture_index_list[cam_index], frame_queue_list[cam_index]))
        capture_processes.append(p_capture)
        p_capture.start()

    # Start processing processes
    processing_processes = []
    n_processes_per_camera = 2  # Number of processing processes per camera

    for cam_index in range(num_cams):
        for _ in range(n_processes_per_camera):
            p_process = Process(target=process_frames, args=(
                cam_index,  # Camera index
                f"frame_{cam_index}",  # Shared memory name for frames
                shm_name_pose,  # Shared memory for poses (same for all cameras)
                frame_shape,
                frame_size,
                buffer_length,
                max_people,
                buffer_pose,
                frame_queue_list[cam_index],  # Frame queue specific to the camera
                display_queue_list[cam_index],  # Display queue specific to the camera
                Lock(),  # Lock for synchronization
                num_cams  # Total number of cameras
            ))
            processing_processes.append(p_process)
            p_process.start()


    # Start display processes
    display_processes = []
    for cam_index in range(num_cams):
        p_display = Process(target=display_frames, args=(cam_index, f"frame_{cam_index}", shm_name_pose, frame_shape, frame_size, buffer_length, display_queue_list[cam_index], palette, skeleton, link_color, point_color, max_people, buffer_pose,num_cams))
        display_processes.append(p_display)
        p_display.start()

    # Join all processes
    for p_capture in capture_processes:
        p_capture.join()

    for p_process in processing_processes:
        p_process.join()

    for p_display in display_processes:
        p_display.join()

    # Clean up shared memory
    for shm_frame_data in shm_frame_data_list:
        shm_frame_data.close()
        shm_frame_data.unlink()

    shm_frame_data_pose.close()
    shm_frame_data_pose.unlink()

    cv2.destroyAllWindows()
