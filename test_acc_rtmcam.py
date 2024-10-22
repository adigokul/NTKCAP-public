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
def capture_frames(cam_index, num_cams, shm_name, frame_shape, frame_size, buffer_length, capture_index, process_index):
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"Camera {cam_index} could not be opened.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_shape[1])  # Set width
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_shape[0])  # Set height
    cap.set(cv2.CAP_PROP_FPS, 30)
    frame_count = 0

    # Access the shared memory by name
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    frame_data = np.ndarray((buffer_length * frame_size,), dtype=np.uint8, buffer=existing_shm.buf)

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Camera {cam_index}: Failed to capture frame")
            break

        # Resize the frame to match the expected resolution
        frame_resized = cv2.resize(frame, (frame_shape[1], frame_shape[0]))

        # Circular buffer logic: calculate index and store the frame
        buffer_index = capture_index.value % buffer_length
        start_index = buffer_index * frame_size

        # Store the frame in shared memory
        frame_buffer = np.ndarray(frame_shape, dtype=np.uint8, buffer=frame_data[start_index:])
        frame_buffer[:] = frame_resized[:]

        # Update the capture index (with lock)
        with capture_index.get_lock():
            capture_index.value += 1
        #print(f"Capture index updated to {capture_index.value}")
        frame_count += 1
        time.sleep(0.01)  # Simulate frame capture delay

    cap.release()
    existing_shm.close()

def process_frames(shm_name,shm_name_pose, frame_shape, frame_size, buffer_length, capture_index, process_index, process_number, total_processes, shm_name_rtm,max_people , buffer_pose):
    tracker = PoseTracker(det_model=det_model_path, pose_model=pose_model_path, device_name=device)
    state = tracker.create_state(det_interval=1, det_min_bbox_size=100, keypoint_sigmas=np.ones((17,)))  # Adjust keypoint_sigmas as needed
    # Access the shared memory by name
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    pose_shm = shared_memory.SharedMemory(name=shm_name_pose)
    frame_data = np.ndarray((buffer_length * frame_size,), dtype=np.uint8, buffer=existing_shm.buf)
    rtm_data = np.ndarray(( buffer_pose*(max_people *  26 * 3+1)), dtype= np.float16(), buffer=pose_shm.buf)
    
    while True:
        # Ensure there's always at least one frame to process (capture_index > process_index)
        if process_index.value >= capture_index.value:
            #print(f"Process waiting: PI={process_index.value}, CAP={capture_index.value}")  
            continue
        
        #print(process_index.value)
        # Circular buffer logic: calculate index and process the frame
        buffer_index = process_index.value % buffer_length
        buffer_index_pose = process_index.value % buffer_pose
        start_index = buffer_index * frame_size
        start_index_pose = buffer_index_pose*(max_people*26*3+1)
        #print('Processingindex:'+str(start_index_pose))
        frame_buffer = np.ndarray(frame_shape, dtype=np.uint8, buffer=frame_data[start_index:])
        keypoints_buffer = np.ndarray((max_people*26*3+1), dtype=np.float16(), buffer=rtm_data[start_index_pose:])
        #print('process:' +str(start_index_pose))
        results = tracker(state, frame_buffer, detect=-1)
        keypoints, bboxes, _ = results
        keypoints_flattened =keypoints.flatten()
        people_count = len(results[0])
        arr = np.full((max_people*26*3+1), -1.000)
        arr[0] = people_count 
        #import pdb;pdb.set_trace()
        arr[1:len(keypoints_flattened)+1] = keypoints_flattened
        #print('rtm')
        
        keypoints_buffer[:] = arr[:]
        #print(rtm_data[start_index_pose:start_index_pose+10])
        #print('prcoess_people: '+ str(keypoints_buffer[0]))
        
        # Process frames based on the modulo of the process number and total processes
        #if process_index.value % total_processes == process_number:
            #print(f"Process {process_number}: Processing frame {process_index.value} with modulo {process_index.value % total_processes}")

        # Update the process index after processing
        

        #print('Process----PI: ' +str(process_index.value)+'CAP: ' +str(capture_index.value))
        
        #print('PI: ' +str(process_index.value)+'DIS: ' +str(capture_index.value))
        with process_index.get_lock():
            process_index.value += 1
            print('Process PI:'+str(process_index.value))

    existing_shm.close()
    pose_shm.close()

def display_frames(shm_name, shm_name_pose,frame_shape, frame_size, buffer_length, capture_index, process_index, shm_name_rtm, palette, skeleton, link_color, point_color,max_people , buffer_pose):
    # Access the shared memory by name
    
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    pose_shm = shared_memory.SharedMemory(name=shm_name_pose)
    
    frame_data = np.ndarray((buffer_length * frame_size,), dtype=np.uint8, buffer=existing_shm.buf)
    rtm_data = np.ndarray((   buffer_pose*(max_people*26 * 3+1)), dtype= np.float16(), buffer=pose_shm.buf)
    last_time = time.time()  # Initialize last time to calculate FPS
    while True:
        #print('display')
        # Ensure that there's always a frame to display (capture_index > process_index)
        
        if process_index.value > capture_index.value or capture_index.value == 0 or process_index.value ==0:
            continue

        #print(f"Display: Capture={capture_index.value}, Process={process_index.value}")

        # Circular buffer logic: calculate buffer index and start index in shared memory
        buffer_index = (process_index.value - 1) % buffer_length
        buffer_index_pose = (process_index.value - 1) % buffer_pose
        start_index_pose = buffer_index_pose*(max_people*26*3+1)
        start_index = buffer_index * frame_size
        frame_buffer = np.ndarray(frame_shape, dtype=np.uint8, buffer=frame_data[start_index:])
        
        # Calculate FPS
        current_time = time.time()
        time_diff = current_time - last_time
        fps = 1 / time_diff if time_diff > 0 else 0
        last_time = current_time

        # Add FPS text on the frame using cv2.putText()
        fps_text = f"FPS: {fps:.2f}"
        cv2.putText(frame_buffer, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        # Display the frame with FPS
        #print('Display----PI: ' +str(process_index.value)+'CAP: ' +str(capture_index.value))
        # print('display_index:'+str(start_index_pose))
        arr = np.ndarray((max_people*26*3+1), dtype=np.float16(), buffer=rtm_data[start_index_pose:])
        #print('DIS:' +str(start_index_pose)+'arr'+str(arr[:3]))
        #print('PI: ' +str(process_index.value)+'DIS: ' +str(capture_index.value))
        people_count =int(arr[0])
        #print('people_count:' +str(people_count) )
        #print(arr[:10])

        for i in range(int(people_count)):
            arr_temp =arr[i*79+1:i*79+79]
            kpts = arr_temp.reshape(-1, 3)[:, :2]
            score = arr_temp.reshape(-1, 3)[:, 2]
            #import pdb;pdb.set_trace()
            show = [1] * len(kpts)
            for (u, v), color in zip(skeleton, link_color):
                if score[u] > 0.5 and score[v] > 0.5:
                    cv2.line(frame_buffer, tuple(map(int, kpts[u])), tuple(map(int, kpts[v])), palette[color], 2, cv2.LINE_AA)
                else:
                    show[u] = show[v] = 0
            for kpt, show, color in zip(kpts, show, point_color):
                if show:
                    cv2.circle(frame_buffer,tuple(map(int, kpt)), 1, palette[color], 2, cv2.LINE_AA)
                    print('circle')
        cv2.imshow(f"Frame", frame_buffer)
        print('Display PI:'+str(process_index.value))
        # Update process index
        # with process_index.get_lock():
        #     process_index.value += 1

        # Break the display loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    existing_shm.close()

    pose_shm.close()
    cv2.destroyAllWindows()
    time.sleep(0.01)

if __name__ == "__main__":
    from multiprocessing import Value

    num_cams = 1  # Number of webcams
    frame_shape = (720, 1280, 3)  # Set the frame size to 1920x1080
    # Set a max number of frames to process
    frame_size = np.prod(frame_shape)
    buffer_length = 400  # Increase the buffer size
    max_people = 10
    buffer_pose = 400
    # Create shared memory block for storing both frame counts and frame data
    shm_name = "frame"
    shm_size = (int(buffer_length * frame_size * np.uint8().nbytes) ) * num_cams
    shm_frame_data = shared_memory.SharedMemory(create=True, size=shm_size, name=shm_name)

    shm_name_pose = "pose"
    shm_size =  (int(buffer_pose*(max_people  * 26 * 3*max_people +1)* np.float16().nbytes)) * num_cams
    shm_frame_data_pose = shared_memory.SharedMemory(create=True, size=shm_size, name=shm_name_pose)
    
    # Shared indices for capture and processing with locks
    capture_index = Value('i', 0, lock=True)  # Shared capture index
    process_index = Value('i', 0, lock=True)  # Shared process index

    # Define how many processing tasks you want
    total_processes = 1  # You can change this to any number (e.g., 3, 4, etc.)

    # Create separate processes for capturing, processing, and displaying frames
    capture_processes = []
    for i in range(num_cams):
        p_capture = Process(target=capture_frames, args=(i, num_cams, shm_name, frame_shape, frame_size, buffer_length, capture_index, process_index))
        capture_processes.append(p_capture)
        p_capture.start()

    # Create the processing processes dynamically
    processing_processes = []
    for process_number in range(total_processes):
        p_process = Process(target=process_frames, args=(shm_name,shm_name_pose, frame_shape, frame_size, buffer_length, capture_index, process_index, process_number, total_processes,shm_name_pose,max_people , buffer_pose))
        processing_processes.append(p_process)
        p_process.start()

    # Display process remains the same
    p_display = Process(target=display_frames, args=(shm_name, shm_name_pose,frame_shape, frame_size, buffer_length, capture_index, process_index, shm_name_pose,palette, skeleton, link_color, point_color,max_people , buffer_pose))
    p_display.start()

    # Join all capture and processing processes
    for p_capture in capture_processes:
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