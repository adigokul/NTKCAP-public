import cv2
import numpy as np
import matplotlib.pyplot as plt

def combine_videos(video_paths, window_title='Combined Video'):
    caps = [cv2.VideoCapture(path) for path in video_paths]

    if not all(cap.isOpened() for cap in caps):
        print("Could not open all videos")
        return

    plt.ion()  # Turn on interactive mode to update the plot
    fig, ax = plt.subplots()

    while True:
        frames = [cap.read()[1] for cap in caps]
        
        if any(f is None for f in frames):
            print("End of video stream reached")
            break

        frames = [cv2.resize(f, (300, 300)) if f is not None else np.zeros((300, 300, 3), dtype=np.uint8) for f in frames]
        top_row = np.hstack(frames[:2])
        bottom_row = np.hstack(frames[2:])
        combined_frame = np.vstack((top_row, bottom_row))
        combined_frame = cv2.cvtColor(combined_frame, cv2.COLOR_BGR2RGB)

        ax.clear()
        ax.imshow(combined_frame)
        plt.pause(0.001)

        # Exit condition check for 'q' is pressed
        if plt.waitforbuttonpress(0.001):
            break

    for cap in caps:
        cap.release()
    plt.close()

if __name__ == '__main__':
    video_paths = [
        r'C:\Users\user\Desktop\NTKCAP\Patient_data\1\2024_4_26\raw_data\task\videos\1.mp4',
        r'C:\Users\user\Desktop\NTKCAP\Patient_data\1\2024_4_26\raw_data\task\videos\2.mp4',
        r'C:\Users\user\Desktop\NTKCAP\Patient_data\1\2024_4_26\raw_data\task\videos\3.mp4',
        r'C:\Users\user\Desktop\NTKCAP\Patient_data\1\2024_4_26\raw_data\task\videos\4.mp4'
    ]
    combine_videos(video_paths)
