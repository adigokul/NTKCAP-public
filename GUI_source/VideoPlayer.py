import os
import cv2
import time
import socket
import subprocess
import numpy as np
import pyqtgraph as pg
from PyQt6.QtGui import *
from PyQt6.QtCore import *
from PyQt6.QtWidgets import *
from PyQt6.QtWebEngineWidgets import QWebEngineView

class VideoPlayer(QThread):
    frame_changed = pyqtSignal(int)
    data_ready = pyqtSignal(np.ndarray, np.ndarray, int, int)
    def __init__(self, labels: list[QLabel], web_viewer: QWebEngineView, plot, parent=None):
        super().__init__(parent)
        self.video_path = None
        self.label1, self.label2, self.label3, self.label4 = labels[0], labels[1], labels[2], labels[3]
        self.frame_count = None
        self.timer = QTimer()
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer.timeout.connect(self.run)
        self.result_load_videos_allframes = []
        self.web_view_widget = web_viewer
        self.progress = 0
        self.start_server()

        self.plot = plot
    def load_video(self, cal_task_path: str):
        self.result_load_videos_allframes = []
        self.progress = 0
        for i in range(4):
            video_path = os.path.join(cal_task_path, 'videos_pose_estimation_repj_combine', f"{i+1}.mp4")
            cap = cv2.VideoCapture(video_path)
            self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.result_load_videos_allframes.append([])
            while True:
                ret, frame = cap.read()
                if not ret: break                
                self.result_load_videos_allframes[i].append(frame)
            cap.release()

    def np2qimage(self, img):
        resized_frame = cv2.cvtColor(cv2.resize(img, (self.label2.width(), self.label2.height()), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        height, width, channel = resized_frame.shape
        bytes_per_line = channel * width
        q_image = QImage(resized_frame, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        return pixmap
    def result_load_gait_figures(self, cal_task_path: str):
        self.plot.clear()
        result_post_analysis_imgs_data_path = os.path.join(cal_task_path, 'post_analysis', "raw_data.npz")
        data = np.load(result_post_analysis_imgs_data_path, allow_pickle=True)
        self.knee_data = data['Knee'].item()
        self.hip_data = data['Hip'].item()
        self.ankle_data = data['Ankle'].item()
        self.Stride  = data['Stride'].item()
        self.Speed = data['Speed'].item()
        R_knee = self.knee_data['R_knee']
        L_knee = self.knee_data['L_knee']
        
        self.plot.showGrid(x=True, y=True, alpha=0.2)  # Enable grid
        
    def set_slider_value(self, progress):
        self.progress = progress
        if self.progress !=0:
            if int(progress * 100) >= 100:
                self.progress = 0
                self.frame_changed.emit(0)
                self.web_view_widget.page().runJavaScript("resetAnimation();")
                
            else:
                self.frame_changed.emit(round(self.progress * 100))

    def run(self):        
        cap_b = time.perf_counter()      
        self.web_view_widget.page().runJavaScript(
            """
            (function() {
                if (typeof window.getAnimationProgress === 'function') {
                    return getAnimationProgress();
                } else {
                    return 0;
                }
            })();
            """,
            self.set_slider_value
        )
        
        self.label1.setPixmap(self.np2qimage(self.result_load_videos_allframes[0][int(self.progress * self.frame_count)]))
        self.label2.setPixmap(self.np2qimage(self.result_load_videos_allframes[1][int(self.progress * self.frame_count)]))
        self.label3.setPixmap(self.np2qimage(self.result_load_videos_allframes[2][int(self.progress * self.frame_count)]))
        self.label4.setPixmap(self.np2qimage(self.result_load_videos_allframes[3][int(self.progress * self.frame_count)]))
        
    def slider_changed(self):
        self.web_view_widget.page().runJavaScript(f"window.updateAnimationProgress({self.progress});")     
        self.label1.setPixmap(self.np2qimage(self.result_load_videos_allframes[0][int(self.progress * self.frame_count)]))
        self.label2.setPixmap(self.np2qimage(self.result_load_videos_allframes[1][int(self.progress * self.frame_count)]))
        self.label3.setPixmap(self.np2qimage(self.result_load_videos_allframes[2][int(self.progress * self.frame_count)]))
        self.label4.setPixmap(self.np2qimage(self.result_load_videos_allframes[3][int(self.progress * self.frame_count)]))
    def hip_flexion_plot(self):
        self.scatter_ball_R, self.scatter_ball_L, self.xline = None, None, None
        R_Hip = self.hip_data['R_Hip']
        L_Hip = self.hip_data['L_Hip']
        loc_max_finalR = self.hip_data['loc_max_finalR']
        loc_max_finalL = self.hip_data['loc_max_finalL']
        
        # Clear any existing plots
        self.plot.clear()
        self.plot.setYRange(min(min(np.concatenate([R_Hip,L_Hip])), min(np.concatenate([R_Hip,L_Hip]))) - 10, max(max(np.concatenate([R_Hip,L_Hip])), max(np.concatenate([R_Hip,L_Hip]))) + 10)
        self.plot.setXRange(0, len(R_Hip))  # Full data range for the plot
        #slider.valueChanged.disconnect()
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.addLegend()  # Re-add legend

        # Plot right and left Hip data
        self.plot.plot(R_Hip, pen=pg.mkPen('r', width=3), name="Right Hip")
        self.plot.plot(L_Hip, pen=pg.mkPen('b', width=3), name="Left Hip")

        # Plot max points
        self.plot.plot(loc_max_finalR, R_Hip[loc_max_finalR], pen=None, symbol='t', symbolSize=15, symbolBrush=(0, 255, 0))
        self.plot.plot(loc_max_finalL, L_Hip[loc_max_finalL], pen=None, symbol='t', symbolSize=15, symbolBrush=(0, 255, 255))

        # Plot midpoint stars
        midpoint_R = loc_max_finalR[len(loc_max_finalR) // 2]
        midpoint_L = loc_max_finalL[len(loc_max_finalL) // 2]
        self.plot.plot([midpoint_R], [R_Hip[midpoint_R]], pen=None, symbol='star', symbolSize=30, symbolBrush=(255, 0, 255))
        self.plot.plot([midpoint_L], [L_Hip[midpoint_L]], pen=None, symbol='star', symbolSize=30, symbolBrush=(255, 0, 255))

        # Create new scatter balls
        self.scatter_ball_R = self.plot.plot([0], [R_Hip[0]], pen=None, symbol='o', symbolSize=15, symbolBrush='r')
        self.scatter_ball_L = self.plot.plot([0], [L_Hip[0]], pen=None, symbol='o', symbolSize=15, symbolBrush='b')

        # Create a new vertical timeline (xline)
        self.xline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('k', width=2))
        self.plot.addItem(self.xline)

        # Update the initial plot
    def update_hip_flexion(self):
        frame_index = int(self.progress * self.frame_count)
        # Move the vertical timeline and scatter balls
        self.xline.setPos(frame_index)
        self.scatter_ball_R.setData([frame_index], [self.hip_data['R_Hip'][frame_index]])
        self.scatter_ball_L.setData([frame_index], [self.hip_data['L_Hip'][frame_index]])
    def knee_flexion_plot(self):
        self.scatter_ball_R, self.scatter_ball_L, self.xline = None, None, None
        R_knee = self.knee_data['R_knee']
        L_knee = self.knee_data['L_knee']
        loc_max_finalR = self.knee_data['loc_max_finalR']
        loc_max_finalL = self.knee_data['loc_max_finalL']
            
        
        # Clear any existing plots
        self.plot.clear()
        
        #slider.valueChanged.disconnect()
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.addLegend()  # Re-add legend
        self.plot.setYRange(min(min(np.concatenate([R_knee, L_knee])), min(np.concatenate([R_knee, L_knee]))) - 10, max(max(np.concatenate([R_knee, L_knee])), max(np.concatenate([R_knee, L_knee]))) + 10)
        self.plot.setXRange(0, len(R_knee))  # Full data range for the plot


        # Plot right and left knee data
        self.plot.plot(R_knee, pen=pg.mkPen('r', width=3), name="Right Knee")
        self.plot.plot(L_knee, pen=pg.mkPen('b', width=3), name="Left Knee")

        # Plot max points
        self.plot.plot(loc_max_finalR, R_knee[loc_max_finalR], pen=None, symbol='t', symbolSize=15, symbolBrush=(0, 255, 0))
        self.plot.plot(loc_max_finalL, L_knee[loc_max_finalL], pen=None, symbol='t', symbolSize=15, symbolBrush=(0, 255, 255))

        # Plot midpoint stars
        midpoint_R = loc_max_finalR[len(loc_max_finalR) // 2]
        midpoint_L = loc_max_finalL[len(loc_max_finalL) // 2]
        self.plot.plot([midpoint_R], [R_knee[midpoint_R]], pen=None, symbol='star', symbolSize=30, symbolBrush=(255, 0, 255))
        self.plot.plot([midpoint_L], [L_knee[midpoint_L]], pen=None, symbol='star', symbolSize=30, symbolBrush=(255, 0, 255))

        # Create new scatter balls
        self.scatter_ball_R = self.plot.plot([0], [R_knee[0]], pen=None, symbol='o', symbolSize=15, symbolBrush='r')
        self.scatter_ball_L = self.plot.plot([0], [L_knee[0]], pen=None, symbol='o', symbolSize=15, symbolBrush='b')

        # Create a new vertical timeline (xline)
        self.xline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('k', width=2))
        self.plot.addItem(self.xline)

    # Update the animation based on slider value
    def update_knee_flexion(self):
        frame_index = int(self.progress * self.frame_count)
        # Move the vertical timeline and scatter balls
        self.xline.setPos(frame_index)
        self.scatter_ball_R.setData([frame_index], [self.knee_data['R_knee'][frame_index]])
        self.scatter_ball_L.setData([frame_index], [self.knee_data['L_knee'][frame_index]])
    def ankle_flexion_plot(self):
        self.scatter_ball_R, self.scatter_ball_L, self.xline = None, None, None
        R_ankle = self.ankle_data['R_ankle']
        L_ankle = self.ankle_data['L_ankle']
        loc_max_finalR = self.ankle_data['loc_max_finalR']
        loc_max_finalL = self.ankle_data['loc_max_finalL']

        # Clear any existing plots
        self.plot.clear()
        #slider.valueChanged.disconnect()
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.addLegend()  # Re-add legend
        self.plot.setYRange(min(min(np.concatenate([R_ankle,L_ankle])), min(np.concatenate([R_ankle,L_ankle]))) - 10, max(max(np.concatenate([R_ankle,L_ankle])), max(np.concatenate([R_ankle,L_ankle]))) + 10)
        self.plot.setXRange(0, len(R_ankle))  # Full data range for the plot
        # Plot right and left ankle data
        self.plot.plot(R_ankle, pen=pg.mkPen('r', width=3), name="Right ankle")
        self.plot.plot(L_ankle, pen=pg.mkPen('b', width=3), name="Left ankle")

        # Plot max points
        self.plot.plot(loc_max_finalR, R_ankle[loc_max_finalR], pen=None, symbol='t', symbolSize=15, symbolBrush=(0, 255, 0))
        self.plot.plot(loc_max_finalL, L_ankle[loc_max_finalL], pen=None, symbol='t', symbolSize=15, symbolBrush=(0, 255, 255))

        # Plot midpoint stars
        midpoint_R = loc_max_finalR[len(loc_max_finalR) // 2]
        midpoint_L = loc_max_finalL[len(loc_max_finalL) // 2]
        self.plot.plot([midpoint_R], [R_ankle[midpoint_R]], pen=None, symbol='star', symbolSize=30, symbolBrush=(255, 0, 255))
        self.plot.plot([midpoint_L], [L_ankle[midpoint_L]], pen=None, symbol='star', symbolSize=30, symbolBrush=(255, 0, 255))

        # Create new scatter balls
        self.scatter_ball_R = self.plot.plot([0], [R_ankle[0]], pen=None, symbol='o', symbolSize=15, symbolBrush='r')
        self.scatter_ball_L = self.plot.plot([0], [L_ankle[0]], pen=None, symbol='o', symbolSize=15, symbolBrush='b')

        # Create a new vertical timeline (xline)
        self.xline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('k', width=2))
        self.plot.addItem(self.xline)

    def update_ankle_flexion(self):
        frame_index = int(self.progress * self.frame_count)
        # Move the vertical timeline and scatter balls
        self.xline.setPos(frame_index)
        self.scatter_ball_R.setData([frame_index], [self.ankle_data['R_ankle'][int(frame_index)]])
        self.scatter_ball_L.setData([frame_index], [self.ankle_data['L_ankle'][int(frame_index)]])
    def speed_plot(self):
        self.scatter_ball_B, self.scatter_ball_mean, self.xline, self.scatter_ball_flunc = None, None, None, None
        self.plot.clear()
        self.plot.addLegend(offset=(-10, 10)) 
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        
        B = self.Speed['B']
        mean_velocity = self.Speed['mean_velocity']
        flunc_velocity = self.Speed['flunc_velocity']
        interp_speed = self.Speed['interp_speed']
        bound = self.Speed['bound']
        bound_start_end = self.Speed['bound_start_end']
        rms_final_steady = self.Speed['rms_final_steady']
        rms_start_end = self.Speed['rms_start_end']
        rms_All = self.Speed['rms_All']
        self.plot.setYRange(min(min(np.concatenate([B,flunc_velocity])), min(np.concatenate([B,flunc_velocity]))) -0.1, max(max(np.concatenate([B,flunc_velocity])), max(np.concatenate([B,flunc_velocity]))) + 0.1)
        self.plot.setXRange(0, len(B))  # Full data range for the plot
        # Plot lines
        self.plot.plot(mean_velocity, pen=pg.mkPen('b', width=3), name="Mean Velocity")
        self.plot.plot(flunc_velocity, pen=pg.mkPen('orange', width=3), name="Fluctuation Velocity")
        self.plot.plot(B, pen=pg.mkPen('#CCCC00', width=3), name="Raw Data")

        self.xline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('k', width=2))
        self.plot.addItem(self.xline)

        self.scatter_ball_mean = self.plot.plot([0], [mean_velocity[0]], pen=None, symbol='o', symbolSize=15, symbolBrush='b')
        self.scatter_ball_B = self.plot.plot([0], [B[0]], pen=None, symbol='o', symbolSize=15, symbolBrush='#CCCC00')
        self.scatter_ball_flunc = self.plot.plot([0], [flunc_velocity[0]], pen=None, symbol='o', symbolSize=15, symbolBrush='orange')
        # Draw vertical lines (equivalent to axvline)
        for x in [min(bound), max(bound)]:
            self.plot.addItem(pg.InfiniteLine(pos=x, angle=90, pen=pg.mkPen('#00FF00', style=Qt.PenStyle.DashLine, width=2)))


        for x in [min(bound_start_end), max(bound_start_end)]:
            self.plot.addItem(pg.InfiniteLine(pos=x, angle=90, pen=pg.mkPen('dimgrey', style=Qt.PenStyle.DashLine, width=2)))

        # Text annotations
        text_1 = pg.TextItem(f'rms steady = {rms_final_steady:.5f}', color='k', anchor=(0, 0), border=None); text_1.setFont(pg.QtGui.QFont("Arial", 12))
        text_2 = pg.TextItem(f'rms start end = {rms_start_end:.5f}', color='k', anchor=(0, 0), border=None); text_2.setFont(pg.QtGui.QFont("Arial", 12))
        text_3 = pg.TextItem(f'rms All = {rms_All:.5f}', color='k', anchor=(0, 0), border=None); text_3.setFont(pg.QtGui.QFont("Arial", 12))
        text_4 = pg.TextItem(f'max speed = {np.max(mean_velocity):.5f}', color='k', anchor=(0, 0), border=None); text_4.setFont(pg.QtGui.QFont("Arial", 12))
        # Position text items on the plot
        self.plot.addItem(text_1, ignoreBounds=True)
        self.plot.addItem(text_2, ignoreBounds=True)
        self.plot.addItem(text_3, ignoreBounds=True)
        self.plot.addItem(text_4, ignoreBounds=True)

        text_1.setPos(5, np.max(mean_velocity) )
        text_2.setPos(5, np.max(mean_velocity) -0.1)
        text_3.setPos(5, np.max(mean_velocity) - 0.2)
        text_4.setPos(5, np.max(mean_velocity) - 0.3)

    def update_speed(self):
        frame_index = int(self.progress * self.frame_count)
        self.xline.setPos(frame_index)
        self.scatter_ball_B.setData([frame_index], [self.Speed['B'][frame_index]])
        self.scatter_ball_mean.setData([frame_index], [self.Speed['mean_velocity'][frame_index]])
        self.scatter_ball_flunc.setData([frame_index], [self.Speed['flunc_velocity'][frame_index]])
    def stride_plot(self):
        self.scatter_ball_heel_L, self.scatter_ball_heel_R = None, None
        self.plot.clear()
        self.plot.addLegend()  # Re-add legend
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        
        R_heel = self.Stride['R_heel']
        L_heel = self.Stride['L_heel']
        frame_R_heel_sground = self.Stride['frame_R_heel_sground']
        frame_L_heel_sground = self.Stride['frame_L_heel_sground']

        xmr = self.Stride['xmr']
        xml = self.Stride['xml']
        ymr = self.Stride['ymr']
        yml = self.Stride['yml']
        pace_r = self.Stride['pace_r']
        pace_l = self.Stride['pace_l']
        mid_r_index = self.Stride['mid_r_index']
        mid_l_index = self.Stride['mid_l_index']
        self.plot.setXRange(min(min(np.concatenate([R_heel[:, 0],L_heel[:, 0]])), min(np.concatenate([R_heel[:, 0],L_heel[:, 0]]))) -0.1, max(max(np.concatenate([R_heel[:, 0],L_heel[:, 0]])), max(np.concatenate([R_heel[:, 0],L_heel[:, 0]]))) + 0.1)
        self.plot.setYRange(min(min(np.concatenate([R_heel[:, 2],L_heel[:, 2]])), min(np.concatenate([R_heel[:, 2],L_heel[:, 2]]))) -0.1, max(max(np.concatenate([R_heel[:, 2],L_heel[:, 2]])), max(np.concatenate([R_heel[:, 2],L_heel[:, 2]]))) + 0.1)

        # Scatter plots for heel traces
        self.plot.plot(R_heel[:, 0], R_heel[:, 2], pen=None, symbol='o', symbolSize=10, symbolBrush='b', name='Right Heel Trace')
        self.plot.plot(L_heel[:, 0], L_heel[:, 2], pen=None, symbol='o', symbolSize=10, symbolBrush='orange', name='Left Heel Trace')

        # Scatter plots for heel strikes
        self.plot.plot(R_heel[frame_R_heel_sground, 0], R_heel[frame_R_heel_sground, 2], pen=None, symbol='o', symbolSize=15, symbolBrush='r', name='Right Heel Strikes')
        self.plot.plot(L_heel[frame_L_heel_sground, 0], L_heel[frame_L_heel_sground, 2], pen=None, symbol='o', symbolSize=15, symbolBrush='r', name='Left Heel Strikes')
        self.scatter_ball_heel_R = self.plot.plot([R_heel[0, 0]], [R_heel[0, 2]], pen=None, symbol='star', symbolSize=25, symbolBrush='k')
        self.scatter_ball_heel_L = self.plot.plot([L_heel[0, 0]], [L_heel[0, 2]], pen=None, symbol='star', symbolSize=25, symbolBrush='k')
        # Adding text annotations for pace_r and pace_l
        for i in range(len(xmr)):
            text_item = pg.TextItem(f'{pace_r[i]:.4f}', color='k')
            self.plot.addItem(text_item)
            text_item.setPos(xmr[i], ymr[i])

        for i in range(len(xml)):
            text_item = pg.TextItem(f'{pace_l[i]:.4f}', color='k')
            self.plot.addItem(text_item)
            text_item.setPos(xml[i], yml[i])

        # Highlighting specific points
        self.plot.plot([xmr[mid_r_index] - 0.01], [ymr[mid_r_index]], pen=None, symbol='t', symbolSize=20, symbolBrush='#00FF00')
        self.plot.plot([xml[mid_l_index] - 0.01], [yml[mid_l_index]], pen=None, symbol='t', symbolSize=20, symbolBrush='#00FF00')

    def update_stride(self):
        frame_index = int(self.progress * self.frame_count)
        self.scatter_ball_heel_R.setData([self.Stride['R_heel'][frame_index, 0]], [self.Stride['R_heel'][frame_index, 2]])
        self.scatter_ball_heel_L.setData([self.Stride['L_heel'][frame_index, 0]], [self.Stride['L_heel'][frame_index, 2]])
    def start_server(self):
        port = 8000
        if self.is_port_in_use(port):
            self.kill_process_using_port(port)
        print("Starting local server...")
        self.server_process = subprocess.Popen(["python", "-m", "http.server", str(port)],
                                                cwd=".", stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    def is_port_in_use(self, port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0
    def kill_process_using_port(self, port):
        try:
            result = subprocess.check_output(f"netstat -ano | findstr :{port}", shell=True, text=True)
            for line in result.splitlines():
                if "LISTENING" in line:
                    pid = line.split()[-1]
                    subprocess.run(f"taskkill /PID {pid} /F", shell=True)
                    
        except subprocess.CalledProcessError:
            pass
    
    def load_gltf_file_in_viewer(self, file_path):
        self.web_view_widget.setUrl(QUrl(f"http://localhost:8000/viewer.html?model={file_path}"))