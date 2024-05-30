# import cv2
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# import numpy as np
# from matplotlib.path import Path

# # Load an image
# img_path = r'C:\Users\user\Desktop\NTKCAP\calibration\ExtrinsicCalibration\1.jpg'
# img = mpimg.imread(img_path)  # This loads image in RGB

# fig, ax = plt.subplots()
# ax.imshow(img)
# points = []
# mask = None

# def onclick(event):
#     if len(points) < 4:  # Limit to four points
#         # Get and store the coordinates
#         ix, iy = event.xdata, event.ydata
#         points.append((ix, iy))
#         ax.plot(ix, iy, 'ro')
#         fig.canvas.draw()
#         print(f'Point {len(points)}: ({ix}, {iy})')
        
#         if len(points) == 4:
#             apply_green_mask()

# def apply_green_mask():
#     global mask
#     # Create a green mask
#     mask = np.zeros_like(img, dtype=np.uint8)
#     mask[..., 1] = 255  # Set green channel to 255

#     # Create a path from the points
#     path = Path(points)

#     # Apply the mask to the path area
#     for y in range(img.shape[0]):
#         for x in range(img.shape[1]):
#             if path.contains_point((x, y)):
#                 mask[y, x] = img[y, x]

#     # Update the image display
#     ax.imshow(mask)
#     fig.canvas.draw()
#     create_video()

# def create_video():
#     output_path = r'C:\Users\user\Desktop\NTKCAP\calibration\ExtrinsicCalibration\images\1\output_video.mp4'
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define the codec
#     out = cv2.VideoWriter(output_path, fourcc, 2.0, (img.shape[1], img.shape[0]))

#     # Convert mask from RGB to BGR for cv2
#     mask_bgr = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)

#     # Append the same masked image for 10 frames
#     for _ in range(10):
#         out.write(mask_bgr)

#     out.release()
#     print("Video saved successfully.")

# # Connect the click event
# cid = fig.canvas.mpl_connect('button_press_event', onclick)

# plt.show()



import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from matplotlib.path import Path

# Load an image
img_path = r'C:\Users\user\Desktop\NTKCAP\calibration\ExtrinsicCalibration\000009.jpg'
img = mpimg.imread(img_path)

fig, ax = plt.subplots()
ax.imshow(img)
points = []
mask = None

def onclick(event):
    if len(points) < 4:  # Limit to four points
        # Get and store the coordinates
        ix, iy = event.xdata, event.ydata
        points.append((ix, iy))
        ax.plot(ix, iy, 'ro')
        fig.canvas.draw()
        print(f'Point {len(points)}: ({ix}, {iy})')
        
        if len(points) == 4:
            apply_green_mask()

def apply_green_mask():
    global mask
    # Create a green mask
    mask = np.zeros_like(img)
    mask[..., 1] = 10  # Set green channel to 255

    # Create a path from the points
    path = Path(points)

    # Apply the mask to the path area
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if path.contains_point((x, y)):
                mask[y, x] = img[y, x]

    # Update the image display and save the image
    ax.imshow(mask)
    fig.canvas.draw()
    save_image()

def save_image():
    # Specify the path where you want to save the masked image
    save_path = r'C:\Users\user\Desktop\NTKCAP\calibration\ExtrinsicCalibration\images\1\masked_image.jpg'
    mpimg.imsave(save_path, mask)
    print("Masked image saved successfully.")

# Connect the click event
cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()
