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
        skeleton=[(15, 13), (13, 11), (11, 19), (16, 14), (14, 12), (12, 19),
                  (17, 18), (18, 19), (18, 5), (5, 7), (7, 9), (18, 6), (6, 8),
                  (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),
                  (15, 20), (15, 22), (15, 24), (16, 21), (16, 23), (16, 25)],
        palette=[(51, 153, 255), (0, 255, 0), (255, 128, 0)],
        link_color=[1, 1, 1, 2, 2, 2, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2],
        point_color=[0, 0, 0, 0, 0, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 2, 2, 2, 1, 2, 1, 2, 1, 2]
    ))

skeleton = VISUALIZATION_CFG['halpe26']['skeleton']
palette = VISUALIZATION_CFG['halpe26']['palette']
link_color = VISUALIZATION_CFG['halpe26']['link_color']
point_color = VISUALIZATION_CFG['halpe26']['point_color']


# Function to capture frames and store in shared memory
# def capture_frames(cam_index, shm_name, frame_shape, frame_size, buffer_length, capture_index, frame_queue):
#     cap = cv2.VideoCapture(cam_index)
#     if not cap.isOpened():
#         print(f"Camera {cam_index} could not be opened.")
#         return

#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_shape[1])
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_shape[0])
#     cap.set(cv2.CAP_PROP_FPS, 30)

#     existing_shm = shared_memory.SharedMemory(name=shm_name)
#     frame_data = np.ndarray((buffer_length * frame_size,), dtype=np.uint8, buffer=existing_shm.buf)

#     prev_time = time.time()  # Initialize the previous time

#     while True:
#         current_time = time.time()  # Get the current time at the start of the loop
        
#         ret, frame = cap.read()
#         if not ret:
#             print(f"Camera {cam_index}: Failed to capture frame")
#             break

#         frame_resized = cv2.resize(frame, (frame_shape[1], frame_shape[0]))

#         buffer_index = capture_index.value % buffer_length
#         start_index = buffer_index * frame_size

#         frame_buffer = np.ndarray(frame_shape, dtype=np.uint8, buffer=frame_data[start_index:])
#         frame_buffer[:] = frame_resized[:]

#         with capture_index.get_lock():
#             capture_index.value += 1

#         # Put the frame index in the queue for processing
#         frame_queue.put(buffer_index)

#         # Calculate time difference and print it
#         loop_duration = current_time - prev_time
#         #print(f"Time for the last loop: {loop_duration:.4f} seconds")

#         # Update the previous time for the next iteration
#         prev_time = current_time

#         time.sleep(0.01)

#     cap.release()
#     existing_shm.close()




def capture_frames(image_path, shm_name, frame_shape, frame_size, buffer_length, capture_index, frame_queue):
    # Load the image once
    image_path = r'C:\Users\mauricetemp\Downloads\Elderly-Day-Care-Facility.jpg'
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Failed to load image {image_path}")
        return

    # Resize the frame to the desired shape
    frame_resized = cv2.resize(frame, (frame_shape[1], frame_shape[0]))

    existing_shm = shared_memory.SharedMemory(name=shm_name)
    frame_data = np.ndarray((buffer_length * frame_size,), dtype=np.uint8, buffer=existing_shm.buf)

    prev_time = time.time()  # Initialize the previous time

    while True:
        current_time = time.time()  # Get the current time at the start of the loop

        buffer_index = capture_index.value % buffer_length
        start_index = buffer_index * frame_size

        # Put the resized frame into the shared memory buffer
        frame_buffer = np.ndarray(frame_shape, dtype=np.uint8, buffer=frame_data[start_index:])
        frame_buffer[:] = frame_resized[:]

        with capture_index.get_lock():
            capture_index.value += 1

        # Put the frame index in the queue for processing
        frame_queue.put(buffer_index)

        # Calculate time difference and print it
        loop_duration = current_time - prev_time
        # print(f"Time for the last loop: {loop_duration:.4f} seconds")

        # Update the previous time for the next iteration
        prev_time = current_time

        # Optional delay to simulate real-time behavior
        time.sleep(0.01)

    existing_shm.close()

# Function to process frames and mark them as ready for display
def process_frames(shm_name, shm_name_pose, frame_shape, frame_size, buffer_length, max_people, buffer_pose, frame_queue, display_queue, lock):
    tracker = PoseTracker(det_model=det_model_path, pose_model=pose_model_path, device_name=device)
    state = tracker.create_state(det_interval=1, det_min_bbox_size=100, keypoint_sigmas=np.ones((17,)))

    existing_shm = shared_memory.SharedMemory(name=shm_name)
    pose_shm = shared_memory.SharedMemory(name=shm_name_pose)
    frame_data = np.ndarray((buffer_length * frame_size,), dtype=np.uint8, buffer=existing_shm.buf)
    rtm_data = np.ndarray((buffer_pose * (max_people * 26 * 3 + 1)), dtype=np.float16, buffer=pose_shm.buf)
    prev_time = time.time()
    while True:
        
        try:
            # Get the next frame index from the queue
            buffer_index = frame_queue.get(timeout=5)  # Non-blocking get from the queue
        except:
            print('no frmae')
            continue  # If no frames are available, keep checking

        buffer_index_pose = buffer_index % buffer_pose
        start_index = buffer_index * frame_size
        start_index_pose = buffer_index_pose * (max_people * 26 * 3 + 1)

        frame_buffer = np.ndarray(frame_shape, dtype=np.uint8, buffer=frame_data[start_index:])
        keypoints_buffer = np.ndarray((max_people * 26 * 3 + 1), dtype=np.float16, buffer=rtm_data[start_index_pose:])
        current_time = time.time()
        results = tracker(state, frame_buffer, detect=-1)
        prev_time =time.time()
        loop_duration = prev_time-current_time 
        
        keypoints, bboxes, _ = results
        keypoints_flattened = keypoints.flatten()
        people_count = len(results[0])

        arr = np.full((max_people * 26 * 3 + 1), -1.000)
        arr[0] = people_count
        if people_count>1:
            print(people_count)
        arr[1:len(keypoints_flattened) + 1] = keypoints_flattened
        keypoints_buffer[:] = arr[:]
        #print(arr[0])
        #print(len(keypoints_flattened))
        #print(arr[26*3+2:26*3+10])

        # Put the processed frame index into the display queue
        display_queue.put(buffer_index)
        

        #print(f"Time for the last loop: {loop_duration:.4f} seconds")
        time.sleep(0.01)

    existing_shm.close()
    pose_shm.close()


# Function to display processed frames
def display_frames(shm_name, shm_name_pose, frame_shape, frame_size, buffer_length, display_queue, palette, skeleton, link_color, point_color, max_people, buffer_pose):
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    pose_shm = shared_memory.SharedMemory(name=shm_name_pose)

    frame_data = np.ndarray((buffer_length * frame_size,), dtype=np.uint8, buffer=existing_shm.buf)
    rtm_data = np.ndarray((buffer_pose * (max_people * 26 * 3 + 1)), dtype=np.float16, buffer=pose_shm.buf)
    last_time = time.time()

    while True:
        try:
            # Wait for the next processed frame index from the queue
            buffer_index = display_queue.get(timeout=5)  # Get the next frame index (with a timeout)

            start_index = buffer_index * frame_size
            
            start_index_pose = buffer_index % buffer_pose * (max_people * 26 * 3 + 1)

            frame_buffer = np.ndarray(frame_shape, dtype=np.uint8, buffer=frame_data[start_index:])
            arr = np.ndarray((max_people * 26 * 3 + 1), dtype=np.float16, buffer=rtm_data[start_index_pose:])

            people_count = int(arr[0])

            current_time = time.time()
            fps = 1 / (current_time - last_time) if current_time - last_time > 0 else 0
            last_time = current_time

            fps_text = f"FPS: {fps:.2f}"
            cv2.putText(frame_buffer, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            print('draw: '+ str(people_count))
            for i in range(people_count):
                
                arr_temp = arr[i * 79+1 :i * 79 + 79]
                kpts = arr_temp.reshape(-1, 3)[:, :2]
                score = arr_temp.reshape(-1, 3)[:, 2]
                if people_count>1 and i>0:
                    print(arr[26*3+2:26*3+10])
                    print('\noriginal:')
                    print(arr[0:10])
                    print(kpts)
                    #import pdb;pdb.set_trace()
                

                for (u, v), color in zip(skeleton, link_color):
                    if score[u] > 0.5 and score[v] > 0.5:
                        cv2.line(frame_buffer, tuple(map(int, kpts[u])), tuple(map(int, kpts[v])), palette[color], 2, cv2.LINE_AA)

                for kpt, show, color in zip(kpts, [1] * len(kpts), point_color):
                    if show:
                        cv2.circle(frame_buffer, tuple(map(int, kpt)), 1, palette[color], 2, cv2.LINE_AA)

            cv2.imshow("Frame", frame_buffer)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except:
            continue  # Continue if there's an issue with getting frames

    existing_shm.close()
    pose_shm.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    from multiprocessing import Value, Lock

    num_cams = 1
    frame_shape = (720, 1280, 3)  # Set the frame size to 720p
    #frame_shape = (1920, 1080, 3)
    frame_size = np.prod(frame_shape)
    buffer_length = 4  # Buffer length
    max_people = 10
    buffer_pose = 4

    shm_name = "frame"
    shm_size = int(buffer_length * frame_size)
    shm_frame_data = shared_memory.SharedMemory(create=True, size=shm_size, name=shm_name)

    shm_name_pose = "pose"
    shm_size_pose = int(buffer_pose * (max_people * 26 * 3 + 1) * np.float16().nbytes)
    shm_frame_data_pose = shared_memory.SharedMemory(create=True, size=shm_size_pose, name=shm_name_pose)

    capture_index = Value('i', 0, lock=True)
    frame_queue = Queue()  # Queue to hold frame indices for processing
    display_queue = Queue()  # Queue to hold frame indices for display

    # Capture process
    p_capture = Process(target=capture_frames, args=(0, shm_name, frame_shape, frame_size, buffer_length, capture_index, frame_queue))
    p_capture.start()

    # Multiple processing processes
    num_processes = 1  # Number of processing processes
    processing_processes = []
    for _ in range(num_processes):
        p_process = Process(target=process_frames, args=(shm_name, shm_name_pose, frame_shape, frame_size, buffer_length, max_people, buffer_pose, frame_queue, display_queue, Lock()))
        processing_processes.append(p_process)
        p_process.start()

    # Display process
    p_display = Process(target=display_frames, args=(shm_name, shm_name_pose, frame_shape, frame_size, buffer_length, display_queue, palette, skeleton, link_color, point_color, max_people, buffer_pose))
    p_display.start()

    # Join all capture and processing processes
    p_capture.join()
    for p_process in processing_processes:
        p_process.join()
    p_display.join()

    # Clean up shared memory
    shm_frame_data.close()
    shm_frame_data.unlink()
    shm_frame_data_pose.close()
    shm_frame_data_pose.unlink()

    cv2.destroyAllWindows()
