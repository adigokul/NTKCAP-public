import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
import numpy as np
from PyQt6.QtCore import Qt

# Load data
data = np.load(r"C:\Users\MyUser\Desktop\NTKCAP\Patient_data\1006_justin\2024_12_06\2024_12_06_10_51_calculated\1006_justin_1\post_analysis\raw_data.npz", allow_pickle=True)
knee_data = data['Knee'].item()
hip_data = data['Hip'].item()
ankle_data = data['Ankle'].item()
Stride  = data['Stride'].item()
Speed = data['Speed'].item()

R_knee = knee_data['RKnee']
L_knee = knee_data['L_knee']

R_Hip = hip_data['R_Hip']
L_Hip = hip_data['L_Hip']

R_ankle = ankle_data['R_ankle']
L_ankle = ankle_data['L_ankle']

B =Speed['B']
mean_velocity=Speed['mean_velocity']
flunc_velocity =Speed['flunc_velocity']
interp_speed =Speed['interp_speed']
bound =Speed['bound']
bound_start_end =Speed['bound_start_end']
rms_final_steady =Speed['rms_final_steady']
rms_start_end = Speed['rms_start_end']
rms_All = Speed['rms_All']

R_heel =Stride['R_heel']
L_heel =Stride['L_heel']
frame_R_heel_sground =Stride['frame_R_heel_sground']
frame_L_heel_sground =Stride['frame_L_heel_sground']

xmr =Stride['xmr']
xml =Stride['xml']
ymr = Stride['ymr']
yml =Stride['yml']
pace_r = Stride['pace_r']
pace_l =Stride['pace_l']
mid_r_index =Stride['mid_r_index']
mid_l_index =Stride['mid_l_index']

# Extract knee data


# Set PyQtGraph background to white
pg.setConfigOption('background', 'w')  # White background
pg.setConfigOption('foreground', 'k')  # Set text and grid to black

# Initialize PyQtGraph application
app = QtWidgets.QApplication([])
win = QtWidgets.QWidget()
layout = QtWidgets.QVBoxLayout()
win.setLayout(layout)

# Create PyQtGraph plot
plot_widget = pg.GraphicsLayoutWidget()
layout.addWidget(plot_widget)

plot = plot_widget.addPlot(title="Knee Joint Animation with Slider Sync")
plot.setYRange(min(min(R_knee), min(L_knee)) - 10, max(max(R_knee), max(L_knee)) + 10)
plot.setXRange(0, len(R_knee))  # Full data range for the plot
plot.showGrid(x=True, y=True, alpha=0.2)  # Enable grid

# Initialize scatter balls and xline
scatter_ball_R = None
scatter_ball_L = None
xline = None

# Slider widget
slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
slider.setRange(0, len(R_knee) - 1)
slider.setValue(0)
layout.addWidget(slider)

# Checkbox widget for knee flexion
plot_combo = QtWidgets.QComboBox()
plot_combo.addItems(["None","Hip Flexion ","Knee Flexion","Ankle Flexion","Gait Speed","Stride"])
layout.addWidget(plot_combo)

# Play/Pause buttons
button_layout = QtWidgets.QHBoxLayout()
play_button = QtWidgets.QPushButton("Play")
pause_button = QtWidgets.QPushButton("Pause")
button_layout.addWidget(play_button)
button_layout.addWidget(pause_button)
layout.addLayout(button_layout)

# Timer for animation
fps = 30
playing = False

def hip_flexion_plot():
    global scatter_ball_R, scatter_ball_L, xline
    loc_max_finalR = hip_data['loc_max_finalR']
    loc_max_finalL = hip_data['loc_max_finalL']
        
    # Clear any existing plots
    plot.clear()
    #slider.valueChanged.disconnect()
    plot.showGrid(x=True, y=True, alpha=0.3)
    plot.addLegend()  # Re-add legend

    # Plot right and left Hip data
    plot.plot(R_Hip, pen=pg.mkPen('r', width=3), name="Right Hip")
    plot.plot(L_Hip, pen=pg.mkPen('b', width=3), name="Left Hip")

    # Plot max points
    plot.plot(loc_max_finalR, R_Hip[loc_max_finalR], pen=None, symbol='t', symbolSize=15, symbolBrush=(0, 255, 0))
    plot.plot(loc_max_finalL, L_Hip[loc_max_finalL], pen=None, symbol='t', symbolSize=15, symbolBrush=(0, 255, 255))

    # Plot midpoint stars
    midpoint_R = loc_max_finalR[len(loc_max_finalR) // 2]
    midpoint_L = loc_max_finalL[len(loc_max_finalL) // 2]
    plot.plot([midpoint_R], [R_Hip[midpoint_R]], pen=None, symbol='star', symbolSize=30, symbolBrush=(255, 0, 255))
    plot.plot([midpoint_L], [L_Hip[midpoint_L]], pen=None, symbol='star', symbolSize=30, symbolBrush=(255, 0, 255))

    # Create new scatter balls
    scatter_ball_R = plot.plot([0], [R_Hip[0]], pen=None, symbol='o', symbolSize=15, symbolBrush='r')
    scatter_ball_L = plot.plot([0], [L_Hip[0]], pen=None, symbol='o', symbolSize=15, symbolBrush='b')

    # Create a new vertical timeline (xline)
    xline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('k', width=2))
    plot.addItem(xline)

    # Update the initial plot
    
    slider.valueChanged.disconnect()
    
    slider.valueChanged.connect(update_hip_flexion)
    update_hip_flexion()
def knee_flexion_plot():
    global scatter_ball_R, scatter_ball_L, xline
    loc_max_finalR = knee_data['loc_max_finalR']
    loc_max_finalL = knee_data['loc_max_finalL']
        
    
    # Clear any existing plots
    plot.clear()
    #slider.valueChanged.disconnect()
    plot.showGrid(x=True, y=True, alpha=0.3)
    plot.addLegend()  # Re-add legend

    # Plot right and left knee data
    plot.plot(R_knee, pen=pg.mkPen('r', width=3), name="Right Knee")
    plot.plot(L_knee, pen=pg.mkPen('b', width=3), name="Left Knee")

    # Plot max points
    plot.plot(loc_max_finalR, R_knee[loc_max_finalR], pen=None, symbol='t', symbolSize=15, symbolBrush=(0, 255, 0))
    plot.plot(loc_max_finalL, L_knee[loc_max_finalL], pen=None, symbol='t', symbolSize=15, symbolBrush=(0, 255, 255))

    # Plot midpoint stars
    midpoint_R = loc_max_finalR[len(loc_max_finalR) // 2]
    midpoint_L = loc_max_finalL[len(loc_max_finalL) // 2]
    plot.plot([midpoint_R], [R_knee[midpoint_R]], pen=None, symbol='star', symbolSize=30, symbolBrush=(255, 0, 255))
    plot.plot([midpoint_L], [L_knee[midpoint_L]], pen=None, symbol='star', symbolSize=30, symbolBrush=(255, 0, 255))

    # Create new scatter balls
    scatter_ball_R = plot.plot([0], [R_knee[0]], pen=None, symbol='o', symbolSize=15, symbolBrush='r')
    scatter_ball_L = plot.plot([0], [L_knee[0]], pen=None, symbol='o', symbolSize=15, symbolBrush='b')

    # Create a new vertical timeline (xline)
    xline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('k', width=2))
    plot.addItem(xline)

    # Update the initial plot
    update_knee_flexion()
    slider.valueChanged.disconnect()
    slider.valueChanged.connect(update_knee_flexion)

def ankle_flexion_plot():
    global scatter_ball_R, scatter_ball_L, xline
    loc_max_finalR = ankle_data['loc_max_finalR']
    loc_max_finalL = ankle_data['loc_max_finalL']
        
    
    # Clear any existing plots
    plot.clear()
    #slider.valueChanged.disconnect()
    plot.showGrid(x=True, y=True, alpha=0.3)
    plot.addLegend()  # Re-add legend

    # Plot right and left ankle data
    plot.plot(R_ankle, pen=pg.mkPen('r', width=3), name="Right ankle")
    plot.plot(L_ankle, pen=pg.mkPen('b', width=3), name="Left ankle")

    # Plot max points
    plot.plot(loc_max_finalR, R_ankle[loc_max_finalR], pen=None, symbol='t', symbolSize=15, symbolBrush=(0, 255, 0))
    plot.plot(loc_max_finalL, L_ankle[loc_max_finalL], pen=None, symbol='t', symbolSize=15, symbolBrush=(0, 255, 255))

    # Plot midpoint stars
    midpoint_R = loc_max_finalR[len(loc_max_finalR) // 2]
    midpoint_L = loc_max_finalL[len(loc_max_finalL) // 2]
    plot.plot([midpoint_R], [R_ankle[midpoint_R]], pen=None, symbol='star', symbolSize=30, symbolBrush=(255, 0, 255))
    plot.plot([midpoint_L], [L_ankle[midpoint_L]], pen=None, symbol='star', symbolSize=30, symbolBrush=(255, 0, 255))

    # Create new scatter balls
    scatter_ball_R = plot.plot([0], [R_ankle[0]], pen=None, symbol='o', symbolSize=15, symbolBrush='r')
    scatter_ball_L = plot.plot([0], [L_ankle[0]], pen=None, symbol='o', symbolSize=15, symbolBrush='b')

    # Create a new vertical timeline (xline)
    xline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('k', width=2))
    plot.addItem(xline)

    # Update the initial plot
    update_ankle_flexion()
    slider.valueChanged.disconnect()
    slider.valueChanged.connect(update_ankle_flexion)
def speed_plot():
    global scatter_ball_B, scatter_ball_mean, xline,scatter_ball_flunc
    plot.clear()
    plot.addLegend()  # Re-add legend
    plot.showGrid(x=True, y=True, alpha=0.3)

    # Plot lines
    plot.plot(mean_velocity, pen=pg.mkPen('b', width=3), name="Mean Velocity")
    plot.plot(flunc_velocity, pen=pg.mkPen('orange', width=3), name="Fluctuation Velocity")
    plot.plot(B, pen=pg.mkPen('#CCCC00', width=3), name="Raw Data")

    xline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('k', width=2))
    plot.addItem(xline)

    scatter_ball_mean = plot.plot([0], [mean_velocity[0]], pen=None, symbol='o', symbolSize=15, symbolBrush='b')
    scatter_ball_B = plot.plot([0], [B[0]], pen=None, symbol='o', symbolSize=15, symbolBrush='#CCCC00')
    scatter_ball_flunc = plot.plot([0], [flunc_velocity[0]], pen=None, symbol='o', symbolSize=15, symbolBrush='orange')
    # Draw vertical lines (equivalent to axvline)
    for x in [min(bound), max(bound)]:
        plot.addItem(pg.InfiniteLine(pos=x, angle=90, pen=pg.mkPen('#00FF00', style=Qt.PenStyle.DashLine, width=2)))


    for x in [min(bound_start_end), max(bound_start_end)]:
        plot.addItem(pg.InfiniteLine(pos=x, angle=90, pen=pg.mkPen('dimgrey', style=Qt.PenStyle.DashLine, width=2)))

    # Text annotations
    text_1 = pg.TextItem(f'rms steady = {rms_final_steady:.5f}', color='k', anchor=(0, 0), border=None); text_1.setFont(pg.QtGui.QFont("Arial", 15))
    text_2 = pg.TextItem(f'rms start end = {rms_start_end:.5f}', color='k', anchor=(0, 0), border=None); text_2.setFont(pg.QtGui.QFont("Arial", 15))
    text_3 = pg.TextItem(f'rms All = {rms_All:.5f}', color='k', anchor=(0, 0), border=None); text_3.setFont(pg.QtGui.QFont("Arial", 15))
    text_4 = pg.TextItem(f'max speed = {np.max(mean_velocity):.5f}', color='k', anchor=(0, 0), border=None); text_4.setFont(pg.QtGui.QFont("Arial", 15))
    # Position text items on the plot
    plot.addItem(text_1, ignoreBounds=True)
    plot.addItem(text_2, ignoreBounds=True)
    plot.addItem(text_3, ignoreBounds=True)
    plot.addItem(text_4, ignoreBounds=True)

    text_1.setPos(5, np.max(mean_velocity) )
    text_2.setPos(5, np.max(mean_velocity) -0.1)
    text_3.setPos(5, np.max(mean_velocity) - 0.2)
    text_4.setPos(5, np.max(mean_velocity) - 0.3)
    update_speed()
    slider.valueChanged.disconnect()
    slider.valueChanged.connect(update_speed)
def stride_plot():
    global scatter_ball_heel_L,scatter_ball_heel_R
    plot.clear()
    plot.addLegend()  # Re-add legend
    plot.showGrid(x=True, y=True, alpha=0.3)

    # Scatter plots for heel traces
    plot.plot(R_heel[:, 0], R_heel[:, 2], pen=None, symbol='o', symbolSize=10, symbolBrush='b', name='Right Heel Trace')
    plot.plot(L_heel[:, 0], L_heel[:, 2], pen=None, symbol='o', symbolSize=10, symbolBrush='orange', name='Left Heel Trace')

    # Scatter plots for heel strikes
    plot.plot(R_heel[frame_R_heel_sground, 0], R_heel[frame_R_heel_sground, 2], pen=None, symbol='o', symbolSize=15, symbolBrush='r', name='Right Heel Strikes')
    plot.plot(L_heel[frame_L_heel_sground, 0], L_heel[frame_L_heel_sground, 2], pen=None, symbol='o', symbolSize=15, symbolBrush='r', name='Left Heel Strikes')
    scatter_ball_heel_R = plot.plot([R_heel[0, 0]], [R_heel[0, 2]], pen=None, symbol='star', symbolSize=25, symbolBrush='k')
    scatter_ball_heel_L = plot.plot([L_heel[0, 0]], [L_heel[0, 2]], pen=None, symbol='star', symbolSize=25, symbolBrush='k')
    # Adding text annotations for pace_r and pace_l
    for i in range(len(xmr)):
        text_item = pg.TextItem(f'{pace_r[i]:.4f}', color='k')
        plot.addItem(text_item)
        text_item.setPos(xmr[i], ymr[i])

    for i in range(len(xml)):
        text_item = pg.TextItem(f'{pace_l[i]:.4f}', color='k')
        plot.addItem(text_item)
        text_item.setPos(xml[i], yml[i])

    # Highlighting specific points
    plot.plot([xmr[mid_r_index] - 0.01], [ymr[mid_r_index]], pen=None, symbol='t', symbolSize=20, symbolBrush='#00FF00')
    plot.plot([xml[mid_l_index] - 0.01], [yml[mid_l_index]], pen=None, symbol='t', symbolSize=20, symbolBrush='#00FF00')
    update_stride()
    slider.valueChanged.disconnect()
    slider.valueChanged.connect(update_stride)
def update_hip_flexion():
    frame_index = slider.value()
    # Move the vertical timeline and scatter balls
    xline.setPos(frame_index)
    scatter_ball_R.setData([frame_index], [R_Hip[int(frame_index)]])
    scatter_ball_L.setData([frame_index], [L_Hip[int(frame_index)]])
# Update the animation based on slider value
def update_knee_flexion():
    frame_index = slider.value()
    # Move the vertical timeline and scatter balls
    xline.setPos(frame_index)
    scatter_ball_R.setData([frame_index], [R_knee[int(frame_index)]])
    scatter_ball_L.setData([frame_index], [L_knee[int(frame_index)]])
# Update the animation based on slider value
def update_ankle_flexion():
    frame_index = slider.value()
    # Move the vertical timeline and scatter balls
    xline.setPos(frame_index)
    scatter_ball_R.setData([frame_index], [R_ankle[int(frame_index)]])
    scatter_ball_L.setData([frame_index], [L_ankle[int(frame_index)]])
def update_speed():
    frame_index = slider.value()
    xline.setPos(frame_index)
    scatter_ball_B.setData([frame_index], [B[int(frame_index)]])
    scatter_ball_mean.setData([frame_index], [mean_velocity[int(frame_index)]])
    scatter_ball_flunc.setData([frame_index], [flunc_velocity[int(frame_index)]])
def update_stride():
    frame_index = slider.value()
    scatter_ball_heel_R.setData([R_heel[frame_index, 0]], [R_heel[frame_index, 2]])
    scatter_ball_heel_L.setData([L_heel[frame_index, 0]], [L_heel[frame_index, 2]])
    print('hi')
# Sync slider movement and animation update

# Increment frame and update slider value during play mode
def increment_frame():
    if playing:
        new_value = slider.value() + 1
        if new_value >= len(R_knee):
            new_value = 0  # Loop back to the start
        slider.setValue(new_value)

# Control animation play/pause
def play_animation():
    global playing
    playing = True

def pause_animation():
    global playing
    playing = False

# Clear and replot knee flexion data when checkbox is checked
def choose_plot(state):
    if state == 1:#knee flexion
        hip_flexion_plot()# Clear and replot everything
    elif state ==2:
        knee_flexion_plot()          
    elif state ==3:
        ankle_flexion_plot()
    elif state ==4:
        speed_plot()
    elif state ==5:
        stride_plot()
def update_fake():
    print('slider_connect')
# Timer configuration
timer = QtCore.QTimer()
timer.timeout.connect(increment_frame)
timer.start(int(1000 / fps))  # 30 FPS

# Connect slider, buttons, and checkbox
slider.valueChanged.connect(update_fake)
play_button.clicked.connect(play_animation)
pause_button.clicked.connect(pause_animation)
plot_combo.currentIndexChanged.connect(choose_plot)

# Initialize the initial plot


# Show the window
win.resize(1200, 800)
win.show()
app.exec()
