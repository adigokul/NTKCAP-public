# Guidance : order starts with the red point -> orange -> yellow and from top to buttom
import cv2
import json
import argparse
import tkinter as tk
from tkinter import Canvas
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backend_bases import MouseButton
from easymocap.mytools.vis_base import plot_point
from easymocap.mytools import plot_cross, plot_line
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class ImageEditor:
    def __init__(self, master, image_path, json_file, colors, size):
        
        # all parameters
        self.master = master
        self.size = size
        self.num_points = size[0] * size[1]
        image, points = self.load_data(image_path, json_file)
        self.json_file = json_file
        self.all_points = points
        self.points = points['keypoints2d'] # keypoints2d
        self.colors = colors
        self.zoom_points = [20, 50, 20, 100]
        self.ifzomm_in = None
        self.image = image
        
        self.selected_point = None
        self.dragging = False
        
        # GUI window
        self.master.title('Image display')
        self.master.geometry('1200x780+75+10')
        self.master.resizable(False, False)
        
        # fig and ax
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        self.fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95)
        self.show()
        self.check_points()
        self.draw_points_lines()
        
        # canvas widget
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        
        # button
        button1 = tk.Button(master=self.master, text='save', command=self.update_json2d, width=20, height=2)
        button1.place(x=990, y=710)
        button2 = tk.Button(master=self.master, text='refresh', command=self.refresh_point, width=10, height=1)
        button2.place(x=600, y=720)
        button3 = tk.Button(master=self.master, text='zoom', command=self.zoom_in, width=20, height=2)
        button3.place(x=60, y=710)
        button4 = tk.Button(master=self.master, text='cancel', command=self.zoom_out, width=20, height=2)
        button4.place(x=250, y=710)
        
        # Mouse event
        self.fig.canvas.mpl_connect('pick_event', self.on_pick)
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_drag)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
    
    # rearrange to original fig size
    def zoom_out(self):
        self.ifzomm_in = False
        self.refresh_point()
    
    # load 2d points and pic without points
    def load_data(self, image_path, json_file):
        with open(json_file, 'r') as file:
            data = json.load(file)
            
        img = Image.open(image_path)
        
        return img, data
    
    # save update 2d points to json file
    def update_json2d(self):
        self.all_points['keypoints2d'] = self.points
        with open(self.json_file, 'w') as file:
            json.dump(self.all_points, file, indent=4)
        
        print('json successfully saved.')
        self.zoom_out()
        self.fig.savefig('output_image.jpg', bbox_inches='tight', pad_inches=0, transparent=True)
        
        
    # zoom in    
    def zoom_in(self):
        self.ifzomm_in = True
        x1 = self.zoom_points[0]
        x2 = self.zoom_points[2]
        y1 = self.zoom_points[1]
        y2 = self.zoom_points[3]
        self.ax.set_xlim(x1, x2)
        self.ax.set_ylim(y2, y1)
        self.fig.canvas.draw_idle()
    
    # redraw points after zooming in
    def refresh_point(self):
        self.ax.clear()
        self.show()
        self.draw_points_lines()
        self.fig.canvas.draw_idle()
    
    # create or delete point if 2d points are not matched
    def check_points(self):
        if len(self.points) < self.num_points:
            lack_number = self.num_points - len(self.points)
            for i in range(lack_number):
                create_point = [10.0, (i + 1) * 10.0, 1.0]
                self.points.append(create_point)
        elif len(self.points) > self.num_points:
            redundant_number = len(self.points) - self.num_points
            for i in range(redundant_number):
                self.points.pop(i+self.num_points)
        else:
            pass  
    
    # keep fig while dragging points
    def show(self):
        self.ax.imshow(self.image)
        self.ax.axis('off')
    
    # draw points function
    def draw_points_lines(self):
        zoom_left_pointx = self.zoom_points[0]
        zoom_left_pointy = self.zoom_points[1]
        zoom_right_pointx = self.zoom_points[2]
        zoom_right_pointy = self.zoom_points[3]
        point, = self.ax.plot(zoom_left_pointx, zoom_left_pointy, 'o', color=self.colors[6], markersize=6, picker=3)
        self.ax.text(zoom_left_pointx+1, zoom_left_pointy+1, 'L', color=self.colors[6], fontsize=8)
        point.set_gid(12)
        point, = self.ax.plot(zoom_right_pointx, zoom_right_pointy, 'o', color=self.colors[6], markersize=6, picker=3)
        self.ax.text(zoom_right_pointx+1, zoom_right_pointy+1, 'R', color=self.colors[6], fontsize=8)
        point.set_gid(13)
        column_num = len(self.points) / self.size[1]
        
        for idx, point in enumerate(self.points):
            x, y, _ = point
            labelx = round(x)
            labely = round(y)
            color_index = idx // self.size[0]
            point, = self.ax.plot(x, y, 'o', color=self.colors[color_index], markersize=4, picker=3)
            self.ax.text(labelx, labely, str(idx + 1), color=self.colors[color_index], fontsize=8)
            point.set_gid(idx) # 12points in total，starts from 0 -> red:0~3、org:4~7、yellow:8~11
            if idx != 0:
                self.ax.plot((x, previous_pointx), (y, previous_pointy), '-', color=self.colors[color_index], linewidth=1)
            previous_pointx = x
            previous_pointy = y
    
    # select a point
    def on_pick(self, event):
        self.selected_point = event.artist.get_gid()
        
    # start dragging    
    def on_press(self, event):
        if event.button == 1 and self.selected_point is not None:
            self.dragging = True
    
    # coordinates changing
    def on_drag(self, event):
        if self.selected_point is not None and self.dragging:
            if self.selected_point == self.num_points:
                self.zoom_points[0] = event.xdata
                self.zoom_points[1] = event.ydata
                
            elif self.selected_point == self.num_points+1:
                self.zoom_points[2] = event.xdata
                self.zoom_points[3] = event.ydata
                
            else:
                currentx = event.xdata
                currenty = event.ydata
                self.points[self.selected_point][0] = currentx
                self.points[self.selected_point][1] = currenty
                
            self.ax.clear()
            self.show()
            self.draw_points_lines()
            if self.ifzomm_in == True:
                self.zoom_in()
            
            self.fig.canvas.draw_idle()
    
    # release mouse button
    def on_release(self, event):
        if event.button == 1:
            self.dragging = False
            self.selected_point = None
            
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, help='Path to extrinsic images folder')
    parser.add_argument('--json_file', type=str, help='Path to chessboard corners')
    parser.add_argument('--size', type=str, default='4, 3', help='size of chessboard')
    args = parser.parse_args()
    
    size = []
    size.append(int(args.size.split(',')[0]))
    size.append(int(args.size.split(',')[1]))
    
    colors = [
        [1, 0, 0], # red
        [1, 166/255, 0], # orange
        [1, 1, 0], # yellow
        [0, 1, 0], # green
        [0, 0, 1], # blue
        [75/255, 0, 130/255], # indigo
        [1, 1, 1]  # white
    ]
    
    root = tk.Tk()
    app = ImageEditor(root, args.image_path, args.json_file, colors, size)
    root.mainloop()