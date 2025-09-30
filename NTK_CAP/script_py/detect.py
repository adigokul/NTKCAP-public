import cv2

def export_first_frame(input_video_path, output_video_path):
    # Open the input video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    # Read the first frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        cap.release()
        return
    
    # Get the properties of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    codec = cv2.VideoWriter_fourcc(*'mp4v')  # You can change the codec if needed

    # Initialize the video writer
    out = cv2.VideoWriter(output_video_path, codec, fps, (width, height))
    
    # Write the first frame multiple times to generate a short video
    
    out.write(frame)
    
    # Release everything if job is finished
    cap.release()
    out.release()
    print("Export complete. The output video is saved as:", output_video_path)

# Example usage:
input_video_path = r'C:\Users\user\Desktop\NTKCAP\calibration\ExtrinsicCalibration\videos\1.mp4'
output_video_path = r'C:\Users\user\Desktop\NTKCAP\calibration\ExtrinsicCalibration\videos\1.mp4'
export_first_frame(input_video_path, output_video_path)
