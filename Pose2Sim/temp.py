import cv2
import numpy as np

def extract_and_save_video(input_video_path, output_video_path, start_frame, end_frame):
    # Open the input video
    cap = cv2.VideoCapture(input_video_path)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Validate the frame range
    if start_frame < 0 or end_frame >= frame_count or start_frame >= end_frame:
        raise ValueError("Invalid frame range.")
    
    # Set up the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID' or 'MJPG' depending on your needs
    out = cv2.VideoWriter(output_video_path, fourcc, fps, 
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    
    # Skip to the start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Read and write frames between start_frame and end_frame
    current_frame = start_frame
    while current_frame <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        current_frame += 1
    
    # Release resources
    cap.release()
    out.release()
    print(f"Video saved to {output_video_path}")

# Example usage
input_video = "input.mp4"
output_video = "output.mp4"
start_frame = 100  # Choose the starting frame
end_frame = 500    # Choose the ending frame

a = r'D:\NTKCAP\Patient_data\0906_chen\2024_09_06\raw_data\1234\videos\4.mp4'
b = r'D:\NTKCAP\Patient_data\0906_chen\2024_09_06\raw_data\Apose\videos\4.mp4'
extract_and_save_video(a, b, 2,5)
