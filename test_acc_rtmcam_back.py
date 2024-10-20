import cv2
from multiprocessing import Process, shared_memory, Value
import numpy as np
import time
from mmdeploy_runtime import PoseTracker
import os

# Define paths for the PoseTracker model
dir_mmdeploy = r'C:\Users\mauricetemp\Desktop\NTKCAP\NTK_CAP\ThirdParty\mmdeploy'
det_model_path = os.path.join(dir_mmdeploy, "rtmpose-trt", "rtmdet-m")
pose_model_path = os.path.join(dir_mmdeploy, "rtmpose-trt", "rtmpose-m")
device = "cuda"
tracker = PoseTracker(det_model=det_model_path, pose_model=pose_model_path, device_name=device)

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

def capture_frames(cam_index, shm_name, frame_shape, frame_size, buffer_length, capture_index):
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"Camera {cam_index} could not be opened.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_shape[1])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_shape[0])
    cap.set(cv2.CAP_PROP_FPS, 30)

    existing_shm = shared_memory.SharedMemory(name=shm_name)
    frame_data = np.ndarray((buffer_length * frame_size,), dtype=np.uint8, buffer=existing_shm.buf)

    while True:
        print('CAP')
        ret, frame = cap.read()
        if not ret:
            #print(f"Camera {cam_index}: Failed to capture frame")
            break

        frame_resized = cv2.resize(frame, (frame_shape[1], frame_shape[0]))
        buffer_index = capture_index.value % buffer_length
        start_index = buffer_index * frame_size

        frame_buffer = np.ndarray(frame_shape, dtype=np.uint8, buffer=frame_data[start_index:])
        frame_buffer[:] = frame_resized[:]

        with capture_index.get_lock():
            capture_index.value += 1
            #print('capture:'+str(capture_index.value))

        time.sleep(0.01)
    
    cap.release()
    existing_shm.close()


def process_frames(shm_name, shm_name_pose, frame_shape, frame_size, buffer_length, capture_index, process_index, process_number, total_processes, max_people, buffer_pose):
    tracker = PoseTracker(det_model=det_model_path, pose_model=pose_model_path, device_name=device)
    state = tracker.create_state(det_interval=1, det_min_bbox_size=100, keypoint_sigmas=np.ones((17,)))

    existing_shm = shared_memory.SharedMemory(name=shm_name)
    pose_shm = shared_memory.SharedMemory(name=shm_name_pose)
    
    frame_data = np.ndarray((buffer_length * frame_size,), dtype=np.uint8, buffer=existing_shm.buf)
    rtm_data = np.ndarray((buffer_pose * (max_people * 26 * 3 + 1)), dtype=np.float16, buffer=pose_shm.buf)

    while True:
        if process_index.value >= capture_index.value:
            #print(f"Process {process_number}: No new frames to process. Waiting...")
            time.sleep(0.01)  # Sleep a bit to prevent busy-waiting
            continue
        
        # Ensure process handles only the frames assigned to it (e.g., odd/even)
        if process_index.value % total_processes != process_number:
            #print(f"Process {process_number}: Skipping frame {process_index.value} not assigned to this process.")
            time.sleep(0.01)  # Sleep to prevent busy-waiting
            continue

        # Frame processing logic
        buffer_index = process_index.value % buffer_length
        buffer_index_pose = process_index.value % buffer_pose
        start_index = buffer_index * frame_size
        start_index_pose = buffer_index_pose * (max_people * 26 * 3 + 1)

        frame_buffer = np.ndarray(frame_shape, dtype=np.uint8, buffer=frame_data[start_index:])
        keypoints_buffer = np.ndarray((max_people * 26 * 3 + 1), dtype=np.float16, buffer=rtm_data[start_index_pose:])

        #print(f"Process {process_number}: Processing frame {process_index.value}.")
        results = tracker(state, frame_buffer, detect=-1)
        keypoints, bboxes, _ = results
        keypoints_flattened = keypoints.flatten()
        people_count = len(results[0])

        arr = np.full((max_people * 26 * 3 + 1), -1.000)
        arr[0] = people_count
        arr[1:len(keypoints_flattened) + 1] = keypoints_flattened
        keypoints_buffer[:] = arr[:]

        # Update the unique process index after processing
        with process_index.get_lock():
            process_index.value += 1

    existing_shm.close()
    pose_shm.close()



def display_frames(shm_name, shm_name_pose, frame_shape, frame_size, buffer_length, capture_index, latest_processed_index, palette, skeleton, link_color, point_color, max_people, buffer_pose):
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    pose_shm = shared_memory.SharedMemory(name=shm_name_pose)
    
    frame_data = np.ndarray((buffer_length * frame_size,), dtype=np.uint8, buffer=existing_shm.buf)
    rtm_data = np.ndarray((buffer_pose * (max_people * 26 * 3 + 1)), dtype=np.float16, buffer=pose_shm.buf)

    last_time = time.time()
    
    while True:
        if latest_processed_index.value == 0:
            continue

        buffer_index = (latest_processed_index.value - 1) % buffer_length
        buffer_index_pose = (latest_processed_index.value - 1) % buffer_pose
        start_index_pose = buffer_index_pose * (max_people * 26 * 3 + 1)
        start_index = buffer_index * frame_size
        frame_buffer = np.ndarray(frame_shape, dtype=np.uint8, buffer=frame_data[start_index:])

        current_time = time.time()
        time_diff = current_time - last_time
        fps = 1 / time_diff if time_diff > 0 else 0
        last_time = current_time

        fps_text = f"FPS: {fps:.2f}"
        cv2.putText(frame_buffer, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        arr = np.ndarray((max_people * 26 * 3 + 1), dtype=np.float16, buffer=rtm_data[start_index_pose:])
        people_count = int(arr[0])

        for i in range(int(people_count)):
            arr_temp = arr[i * 79 + 1:i * 79 + 79]
            kpts = arr_temp.reshape(-1, 3)[:, :2]
            score = arr_temp.reshape(-1, 3)[:, 2]
            for (u, v), color in zip(skeleton, link_color):
                if score[u] > 0.5 and score[v] > 0.5:
                    cv2.line(frame_buffer, tuple(map(int, kpts[u])), tuple(map(int, kpts[v])), palette[color], 2, cv2.LINE_AA)
            for kpt, show, color in zip(kpts, [1] * len(kpts), point_color):
                if show:
                    cv2.circle(frame_buffer, tuple(map(int, kpt)), 1, palette[color], 2, cv2.LINE_AA)

        cv2.imshow(f"Frame", frame_buffer)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    existing_shm.close()
    pose_shm.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    num_cams = 1
    frame_shape = (720, 1280, 3)
    frame_size = np.prod(frame_shape)
    buffer_length = 400
    max_people = 10
    buffer_pose = 400

    shm_name = "frame"
    shm_size = (int(buffer_length * frame_size * np.uint8().nbytes)) * num_cams
    shm_frame_data = shared_memory.SharedMemory(create=True, size=shm_size, name=shm_name)

    shm_name_pose = "pose"
    shm_size = (int(buffer_pose * (max_people * 26 * 3 + 1) * np.float16().nbytes)) * num_cams
    shm_frame_data_pose = shared_memory.SharedMemory(create=True, size=shm_size, name=shm_name_pose)

    capture_index = Value('i', 0, lock=True)
    
    # Unique process index (Value) for each process
    process_indices = [Value('i', 0, lock=True) for _ in range(2)]  # Create a separate process index for each process

    total_processes = 2

    # Create capture process
    capture_processes = []
    for i in range(num_cams):
        p_capture = Process(target=capture_frames, args=(i, shm_name, frame_shape, frame_size, buffer_length, capture_index))
        capture_processes.append(p_capture)
        p_capture.start()

    # Create processing processes
    processing_processes = []
    for process_number in range(total_processes):
        # Pass each process its unique index from the process_indices list
        p_process = Process(target=process_frames, args=(shm_name, shm_name_pose, frame_shape, frame_size, buffer_length, capture_index, process_indices[process_number], process_number, total_processes, max_people, buffer_pose))
        processing_processes.append(p_process)
        p_process.start()

    # Create display process
    p_display = Process(target=display_frames, args=(shm_name, shm_name_pose, frame_shape, frame_size, buffer_length, capture_index, process_indices[0], palette, skeleton, link_color, point_color, max_people, buffer_pose))
    p_display.start()

    # Join all processes
    for p_capture in capture_processes:
        p_capture.join()

    for p_process in processing_processes:
        p_process.join()

    p_display.join()

    shm_frame_data.close()
    shm_frame_data.unlink()
    shm_frame_data_pose.close()
    shm_frame_data_pose.unlink()

    cv2.destroyAllWindows()
