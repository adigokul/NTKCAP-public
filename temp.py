import cv2

def extract_video_frames(input_path, output_path, start_frame, end_frame, output_fps=None):
    # Open the video file
    cap = cv2.VideoCapture(input_path)

    # Get video properties
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 file

    # Use the specified output FPS or default to the original FPS
    fps = output_fps if output_fps is not None else input_fps

    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Initialize the video writer
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Loop through frames, saving the specified range to output
    current_frame = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or current_frame > end_frame:
            break
        if start_frame <= current_frame <= end_frame:
            out.write(frame)
        current_frame += 1

    # Release resources
    cap.release()
    out.release()
    print(f"Video saved to {output_path} with FPS: {fps}")

# Usage
input_video_path = "C:/Users/mauricetemp/Downloads/1.mp4"
output_video_path = "C:/Users/mauricetemp/Downloads/1t.mp4"
start_frame = 120 # specify start frame
end_frame = 800    # specify end frame
output_fps = 30   # specify desired output fps
extract_video_frames(input_video_path, output_video_path, start_frame, end_frame, output_fps)
