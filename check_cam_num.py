import cv2

def find_available_cameras(limit=10):
    """
    Finds available cameras up to a specified limit and returns their indices.

    :param limit: The maximum camera index to check.
    :return: A list of available camera indices.
    """
    available_cameras = []
    for i in range(limit):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
        else:
            break  # Stop if we fail to open a camera to avoid unnecessary checks.
    return available_cameras

# Set a reasonable limit to avoid checking too many indices.
# You can adjust the limit based on your expected number of connected cameras.
camera_indices = find_available_cameras(limit=10)
print(f"Available camera indices: {camera_indices}")
