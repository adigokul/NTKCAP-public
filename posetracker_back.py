import cv2
import numpy as np
from mmdeploy_runtime import PoseTracker
import os
import threading
import time
import queue  # Import the queue module

# Model paths and device
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
        link_color=[
            1, 1, 1, 2, 2, 2, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2
        ],
        point_color=[
            0, 0, 0, 0, 0, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 2, 2, 2, 1, 2, 1, 2, 1, 2
        ],
        sigmas=[0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072,
                0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089, 0.026,
                0.026, 0.066, 0.079, 0.079, 0.079, 0.079, 0.079, 0.079]
    )
)

# Function to process frames using a PoseTracker
def pose_tracker_thread(name, tracker, state, frame_queue, results_queue):
    while True:
        frame = frame_queue.get()
        if frame is None:
            break
        # Perform pose tracking
        results = tracker(state, frame, detect=-1)
        results_queue.put((name, frame, results))  # Add frame to results queue

# Function to draw keypoints and skeleton on the frame
def draw_frame(frame, keypoints, scores, palette, skeleton, link_color, point_color):
    for kpts, score in zip(keypoints, scores):
        show = [1] * len(kpts)
        for (u, v), color in zip(skeleton, link_color):
            if score[u] > 0.5 and score[v] > 0.5:
                cv2.line(frame, tuple(map(int, kpts[u])), tuple(map(int, kpts[v])), palette[color], 2, cv2.LINE_AA)
            else:
                show[u] = show[v] = 0
        for kpt, show, color in zip(kpts, show, point_color):
            if show:
                cv2.circle(frame, tuple(map(int, kpt)), 1, palette[color], 2, cv2.LINE_AA)

# Main function to capture video and process with two trackers
def process_video_multithreaded(cam_id):
    cap = cv2.VideoCapture(cam_id)

    # Set up trackers and states for two threads
    tracker1 = PoseTracker(det_model=det_model_path, pose_model=pose_model_path, device_name=device)
    state1 = tracker1.create_state(det_interval=1, det_min_bbox_size=100, keypoint_sigmas=VISUALIZATION_CFG['halpe26']['sigmas'])
    
    tracker2 = PoseTracker(det_model=det_model_path, pose_model=pose_model_path, device_name=device)
    state2 = tracker2.create_state(det_interval=1, det_min_bbox_size=100, keypoint_sigmas=VISUALIZATION_CFG['halpe26']['sigmas'])

    # Queues for frame distribution and result collection
    frame_queue1 = queue.Queue()
    frame_queue2 = queue.Queue()
    results_queue = queue.Queue()

    # Buffer to store results temporarily
    results_buffer = []

    # Start threads for two pose trackers
    thread1 = threading.Thread(target=pose_tracker_thread, args=("Tracker1", tracker1, state1, frame_queue1, results_queue))
    thread2 = threading.Thread(target=pose_tracker_thread, args=("Tracker2", tracker2, state2, frame_queue2, results_queue))
    
    thread1.start()
    thread2.start()

    frame_count = 0

    # Capture loop
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame
        frame = cv2.resize(frame, (1920, 1080))

        # Send frames to trackers in alternating order
        if frame_count % 2 == 0:
            frame_queue1.put(frame)
        else:
            frame_queue2.put(frame)

        # Collect results into the buffer if available
        try:
            tracker_name, frame, results = results_queue.get_nowait()  # Non-blocking get
            results_buffer.append((tracker_name, frame, results))  # Add to buffer
        except queue.Empty:
            pass  # No results, continue

        # Process the buffer
        while results_buffer:
            tracker_name, frame, results = results_buffer.pop(0)  # Process the oldest result
            keypoints, bboxes, _ = results
            scores = keypoints[..., 2]
            keypoints = keypoints[..., :2]  # Extract the x, y coordinates

            # Draw keypoints and skeleton on the frame
            draw_frame(frame, keypoints, scores, VISUALIZATION_CFG['halpe26']['palette'], 
                       VISUALIZATION_CFG['halpe26']['skeleton'], VISUALIZATION_CFG['halpe26']['link_color'], 
                       VISUALIZATION_CFG['halpe26']['point_color'])
            
            print(f"{tracker_name} processed frame {frame_count}: {len(keypoints)} people detected.")
            
            # Display the frame
            cv2.imshow('camera', frame)

        frame_count += 1

        if cv2.waitKey(1) == ord('q'):
            break

    # Cleanup
    frame_queue1.put(None)
    frame_queue2.put(None)

    thread1.join()
    thread2.join()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_video_multithreaded(0)
