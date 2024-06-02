import cv2
from tkinter import *
from PIL import Image, ImageTk
import numpy as np

def create_gui(img):
    global  img_gray, img_rgb, label, root,img_adapt,img_adapt_clean,display_type,button_toggle_image,button_subpix_decision,subpix_decision

    display_type = 'rgb'
    subpix_decision =True
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for displaying in Tkinter
    img_adapt = cv2.adaptiveThreshold(img_gray, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 21, 2)
    img_adapt_clean = cv2.adaptiveThreshold(img_gray, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 21, 2)
    points = []
    zoom_factor = 1.0
    center_x = img_rgb.shape[1] // 2
    center_y = img_rgb.shape[0] // 2
    results = {'corners': None}  # To store the corners
    def decide_subpix():
        global subpix_decision
        if subpix_decision ==True:
            subpix_decision = False
            button_subpix_decision.config(text="Subpix Off")
        else:
            subpix_decision = True
            button_subpix_decision.config(text="Subpix On")

        update_image()
    def toggle_image():
        global display_type
        if display_type == 'rgb':
            display_type = 'adapt'
            button_toggle_image.config(text="Adapted Imag")
        else:
            display_type = 'rgb'
            button_toggle_image.config(text="RGB  Imagee")
        update_image()

    def update_image():
        nonlocal zoom_factor, center_x, center_y
        height, width = img_rgb.shape[:2]
        new_width = int(width / zoom_factor)
        new_height = int(height / zoom_factor)

        center_x = np.clip(center_x, new_width // 2, width - new_width // 2)
        center_y = np.clip(center_y, new_height // 2, height - new_height // 2)

        x1 = center_x - new_width // 2
        y1 = center_y - new_height // 2
        x2 = center_x + new_width // 2
        y2 = center_y + new_height // 2

        # Use the current display type to determine which image to show
        current_img = img_rgb if display_type == 'rgb' else img_adapt
        cropped_img = current_img[y1:y2, x1:x2]
        resized_img = cv2.resize(cropped_img, (width, height), interpolation=cv2.INTER_LINEAR)

        photo = ImageTk.PhotoImage(image=Image.fromarray(resized_img))
        label.config(image=photo)
        label.image = photo  # Keep a reference to avoid garbage collection

    def click_event(event):
        nonlocal zoom_factor, center_x, center_y
        x = int(center_x + (event.x - label.winfo_width() / 2) / zoom_factor)
        y = int(center_y + (event.y - label.winfo_height() / 2) / zoom_factor)
        points.append((x, y))
        cv2.circle(img_rgb, (x, y), 5, (255, 0, 0), -1)  # Mark the point on the image
        cv2.circle(img_adapt, (x, y), 5, (255, 0, 0), -1)  # Mark the point on the image

        update_image()

    def save_points():
        nonlocal points
        global img_rgb
        if len(points) != 12:
            print("Please select exactly 12 points.")
            return None
        
        points_np = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)


# Assuming 'img_gray' is your grayscale image
# Apply corner refinement using a window size of (11, 11) and zero zone of (-1, -1)

        #refined_corners = cv2.cornerSubPix(img_adapt, points_np, (11, 11), (-1, -1), criteria)
        #img_with_corners = cv2.cvtColor(img_adapt, cv2.COLOR_GRAY2BGR)
        #refined_corners = cv2.cornerSubPix(img_gray, points_np, (11, 11), (-1, -1), criteria)
        #refined_corners = cv2.find4QuadCornerSubpix(img_gray, points_np, (11, 11), (-1, -1), criteria)
        if display_type =='rgb':
            if subpix_decision==False:
                refined_corners = cv2.find4QuadCornerSubpix(img_gray, points_np, (11, 11))
                refined_corners = refined_corners[1]
                img_with_corners = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
            elif subpix_decision==True:
                refined_corners = cv2.cornerSubPix(img_gray, points_np, (11, 11), (-1, -1), criteria)
                img_with_corners = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        elif display_type =='adapt':
            if subpix_decision==False:
                refined_corners = cv2.find4QuadCornerSubpix(img_adapt_clean, points_np, (11, 11))
                refined_corners = refined_corners[1]
                img_with_corners = cv2.cvtColor(img_adapt_clean, cv2.COLOR_GRAY2BGR)
            elif subpix_decision==True:
                refined_corners = cv2.cornerSubPix(img_adapt_clean, points_np, (11, 11), (-1, -1), criteria)  
                img_with_corners = cv2.cvtColor(img_adapt_clean, cv2.COLOR_GRAY2BGR)



        cv2.drawChessboardCorners(img_with_corners, (4, 3), refined_corners, True)

        corners = refined_corners.squeeze()
        global img_rgb,img_adapt
        img_rgb = cv2.cvtColor(img_with_corners, cv2.COLOR_BGR2RGB)
        img_adapt = cv2.adaptiveThreshold(img_gray, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 21, 2)
        update_image()

        return corners

    def confirm():
        corners = save_points()
        if corners is not None:
            results['corners'] = corners
            root.quit()

    def zoom(event):
        nonlocal zoom_factor, center_x, center_y
        global img_rgb, label
        # Calculate the relative mouse position to the image
        mouse_x = event.x
        mouse_y = event.y

        rel_x = (mouse_x - label.winfo_width() / 2) / zoom_factor + center_x
        rel_y = (mouse_y - label.winfo_height() / 2) / zoom_factor + center_y

        # Update zoom factor
        if event.delta > 0:  # Zoom in
            zoom_factor *= 1.1
        elif event.delta < 0:  # Zoom out
            zoom_factor /= 1.1

        # Adjust center based on the new zoom factor to keep the mouse position stable
        new_center_x = rel_x - (mouse_x - label.winfo_width() / 2) / zoom_factor
        new_center_y = rel_y - (mouse_y - label.winfo_height() / 2) / zoom_factor

        center_x = int(new_center_x)
        center_y = int(new_center_y)

        update_image()


    def reset_points():
        nonlocal points
        global img_rgb,img_gray,img_adapt
        points = []
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for displaying in Tkinter
        img_adapt = cv2.adaptiveThreshold(img_gray, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 21, 2)
        update_image()

    root = Tk()
    root.title("Chessboard Corner Labeler")

    frame = Frame(root)
    frame.pack(side="top", fill="x")

    button_save = Button(frame, text="Save Points", command=save_points)
    button_save.pack(side="left", padx=5, pady=10)

    button_reset = Button(frame, text="Reset Points", command=reset_points)
    button_reset.pack(side="left", padx=5, pady=10)

    button_confirm = Button(frame, text="Confirm", command=confirm)
    button_confirm.pack(side="left", padx=5, pady=10)

    button_toggle_image = Button(frame, text="Show Adapted Image", command=toggle_image)
    button_toggle_image.pack(side="left", padx=5, pady=10)

    button_subpix_decision= Button(frame, text="Subpix On", command=decide_subpix)
    button_subpix_decision.pack(side="left", padx=5, pady=10)
    photo = ImageTk.PhotoImage(image=Image.fromarray(img_rgb))
    label = Label(root, image=photo)
    label.pack()

    label.bind("<Button-1>", click_event)
    root.bind("<MouseWheel>", zoom)

    update_image()
    root.mainloop()
    root.destroy()  # Ensure the window is destroyed after loop ends
    return results['corners']

def Manual_GUI_detection(img):
   
    corners = create_gui(img)
    print("Corners obtained:", corners)
    return corners

# img_path = r'C:\Users\user\Desktop\NTKCAP\calibration\ExtrinsicCalibration\images\1\000000.jpg'
# a = cv2.imread(img_path)
# Manual_GUI_detection(a)
