import cv2
import numpy as np

# Input and output file paths
input_file = r"C:\Users\mauricetemp\Desktop\NTKCAP\Patient_data\33exhib_test1\2024_11_29\raw_data\1\videos\2.mp4"  # Path to the original video
output_file = r"C:\Users\mauricetemp\Desktop\NTKCAP\Patient_data\33exhib_test1\2024_11_29\raw_data\1\videos\22.mp4"  # Path to save the black video

# Open the original video
cap = cv2.VideoCapture(input_file)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Frame width
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Frame height
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total number of frames

# Define codec and create VideoWriter for the black video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

# Create a black frame with the same resolution
black_frame = np.zeros((height, width, 3), dtype=np.uint8)

# Write black frames to the output video
for _ in range(frame_count):
    out.write(black_frame)

# Release resources
cap.release()
out.release()

print(f"Black video created: {output_file}")
