
import logging
import kivy
from kivy.app import App
from kivy.core.window import Window
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.core.window import Window
from kivy.properties import StringProperty
from kivy.clock import Clock
from kivy.uix.spinner import Spinner
from kivy.uix.popup import Popup
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.checkbox import CheckBox
import time
from datetime import datetime
import os

import cv2
from NTK_CAP.script_py.NTK_Cap import *

from check_extrinsic import *
import tkinter as tk
from tkinter import filedialog
from kivy.animation import Animation
from NTK_CAP.script_py.kivy_file_chooser import select_directories_and_return_list
import traceback
class NewPageScreen(Screen):
    def __init__(self, **kwargs):
        self.current_directory = os.getcwd()
        self.scriptpy_directory = os.path.join(self.current_directory,'NTK_CAP','script_py')
        self.config_path = os.path.join(self.current_directory, "config")
        self.time_file_path = os.path.join(self.current_directory, "cali_time.txt")
        
        self.record_path = self.current_directory
        self.calibra_path = os.path.join(self.current_directory, "calibration")
        self.patient_path = os.path.join(self.current_directory, "Patient_data")
        self.font_path = os.path.join(self.current_directory, "NTK_CAP", "ThirdParty", "Noto_Sans_HK", "NotoSansHK-Bold.otf")
        font_path = self.font_path
        btn_wide = 120
        btn_high = 50
        pos_ref_x =  [0.2,0.4,0.6,0.8,1.0]
        yindex = 0.08
        xindex = 0.02
        pos_ref_y =[1,1-yindex*1,1-yindex*2,1-yindex*3,1-yindex*4,yindex*5,1-yindex*6,0.37,0.20,0.07]
        
        pos_ref_y = [x - 0.1 for x in pos_ref_y]
        pos_ref_x = [x - 0.1 for x in pos_ref_x]
        super(NewPageScreen, self).__init__(**kwargs)
        layout = FloatLayout()
        btn1 = Button(text='Skeleton Demonstration', size_hint=(0.2, 0.1), pos_hint={'x': 0.1, 'y': 0.6})
        btn2 = Button(text='Live Camera Demonstration', size_hint=(0.2, 0.1), pos_hint={'x': 0.4, 'y': 0.6})
        btn3 = Button(text='Button 3', size_hint=(0.2, 0.1), pos_hint={'x': 0.7, 'y': 0.6})
        btn_back = Button(text='Main Page', size_hint=(0.2, 0.1), pos_hint={'center_x': 0.5, 'center_y': 0.1})
        btn_exit = Button(text='結束程式', size_hint=(0.15, 0.1), size=(btn_wide, 50), pos_hint={'center_x': pos_ref_x[4], 'center_y':pos_ref_y[8]}, on_release=self.button_exit, font_name=self.font_path)
        btn_exit.bind()
        btn_back.bind(on_release=self.go_back)
        btn1.bind(on_release=self.opensim_visual)
        btn2.bind(on_release=self.live_cam_demonstration)
        layout.add_widget(btn_exit)
        layout.add_widget(btn1)
        layout.add_widget(btn2)
        layout.add_widget(btn3)
        layout.add_widget(btn_back)
        self.add_widget(layout)

    def button_exit(self, instance):
        exit()
    def go_back(self, instance):
        self.manager.current = 'main'
    def opensim_visual(self,instance):
        root = tk.Tk()
        root.withdraw() # 隐藏根窗口

        initial_dir = self.patient_path if os.path.isdir(self.patient_path) else "/"
        cal_folder_path = tk.filedialog.askdirectory(initialdir=initial_dir) #選到要計算的ID下的日期
        root.destroy() # 關閉窗口
        opensim_vis_dir = os.path.join(self.scriptpy_directory,'opensim_visual_test.py')
        mot_dir =os.path.join(cal_folder_path,'opensim','Balancing_for_IK_BODY.mot')
        osim_dir = os.path.join(cal_folder_path,'opensim','Model_Pose2Sim_Halpe26_scaled.osim')
        vtp_dir = os.path.join(self.scriptpy_directory,'Opensim_visualize_python')
        
        subprocess.Popen(['python' , opensim_vis_dir, mot_dir, osim_dir,vtp_dir], shell=True)
    def live_cam_demonstration(self,instance):
        live_py_dir = os.path.join(self.scriptpy_directory,'pose_tracker_background_camera.py')
        mmdeploy_dir =os.path.join(self.current_directory, "NTK_CAP", "ThirdParty",'mmdeploy')
        det_dir =  os.path.join(mmdeploy_dir,'rtmpose-trt', 'rtmdet-m')
        pose_dir =  os.path.join(mmdeploy_dir,'rtmpose-trt' ,'rtmpose-m')
        for cam in range(4):
            
            subprocess.Popen(['python' , live_py_dir, 'cuda',det_dir, pose_dir,str(cam)], shell=True)



class NTK_CapApp(App):
    def build(self):##### build next page
        self.sm = ScreenManager()  # Now 'sm' is accessible throughout the app as 'self.sm'
        
        main_screen = Screen(name='main')
        new_page_screen = NewPageScreen(name='new_page')

        main_layout = self.setup_main_layout()  # Setup your main layout here
        main_screen.add_widget(main_layout)

        self.sm.add_widget(main_screen)
        self.sm.add_widget(new_page_screen)

        return self.sm


    def setup_main_layout(self):#5,9
        pos_ref_x =  [0.2,0.4,0.6,0.8,1.0]
        yindex = 0.08
        xindex = 0.02
        pos_ref_y =[1,1-yindex*1,1-yindex*2,1-yindex*3,1-yindex*4,yindex*5,1-yindex*6,0.37,0.20,0.07]
        
        pos_ref_y = [x - 0.1 for x in pos_ref_y]
        pos_ref_x = [x - 0.1 for x in pos_ref_x]
        # 設定整個視窗大小為730x660
        Window.size = (730, 660)
        # 指定視窗啟動位置
        Window.top = 50
        Window.left = 50
        # 視窗名稱
        Window.title = 'NTK_Cap'
        # 創建一個FloatLayout佈局

        # log file
        try:
            os.makedirs("log")
            print("創建log資料夾")
        except:
            print("log資料夾已存在")
        log_date = datetime.now()
        self.log_file = "log_" + str(log_date.year) + "_" + str(log_date.month) + "_" + str(log_date.day) + "_" + str(log_date.hour) + "_" + str(log_date.minute) + "_" + str(log_date.second) + ".txt"
        self.log_file = os.path.join("log", self.log_file)
        with open(self.log_file, 'a') as f:
            command = 'log history'
            f.write(command + '\n')
        print("Create log file : " + self.log_file)

        # path setting
        self.current_directory = os.getcwd()
        self.language = 'Chinese'
        self.config_path = os.path.join(self.current_directory, "config")
        self.time_file_path = os.path.join(self.current_directory, "cali_time.txt")
        
        self.record_path = self.current_directory
        self.calibra_path = os.path.join(self.current_directory, "calibration")
        self.patient_path = os.path.join(self.current_directory, "Patient_data")
        ###
        self.calib_toml_path = os.path.join(self.calibra_path, "Calib.toml")
        self.extrinsic_path = os.path.join(self.calibra_path,"ExtrinsicCalibration")
        ###
        # 字型設定
        self.font_path = os.path.join(self.current_directory, "NTK_CAP", "ThirdParty", "Noto_Sans_HK", "NotoSansHK-Bold.otf")
        font_path = self.font_path
        layout = FloatLayout()

        # 創建按鈕，並指定位置
        # 檢察系統檔案
        btn_wide = 120
        btn_high = 50
        btn_calibration_folder = Button(text='1-1建立新參數', size_hint=(0.19,0.1), size=(btn_wide, 50), pos_hint={'center_x': pos_ref_x[0], 'center_y':pos_ref_y[0]}, on_release=self.create_calibration_ask, font_name=self.font_path)
        layout.add_widget(btn_calibration_folder)
        btn_config = Button(text='1-2偵測相機', size_hint=(0.19,0.1), size=(btn_wide, 50), pos_hint={'center_x': pos_ref_x[1], 'center_y':pos_ref_y[0]}, on_release=self.button_config, font_name=self.font_path)
        layout.add_widget(btn_config)
        btn_check_cam = Button(text='1-3檢查相機', size_hint=(0.19,0.1), size=(btn_wide, 50), pos_hint={'center_x': pos_ref_x[2], 'center_y':pos_ref_y[0]},on_release=self.button_check_cam, font_name=self.font_path)
        layout.add_widget(btn_check_cam)

        # 相機校正
        # btn_intrinsic_record = Button(text='2-1拍攝內參', size_hint=(0.19,0.1), size=(btn_wide, 50), pos=(20, 500), on_press=self.button_intrinsic_record, font_name=self.font_path)
        # layout.add_widget(btn_intrinsic_record)
        # btn_intrinsic_calculate = Button(text='2-2計算內參', size_hint=(0.19,0.1), size=(btn_wide, 50), pos=(130, 500), on_press=self.button_intrinsic_calculate, font_name=self.font_path)
        # layout.add_widget(btn_intrinsic_calculate)
        # btn_intrinsic_check = Button(text='2-3檢查內參', size_hint=(0.19,0.1), size=(btn_wide, 50), pos=(20, 400), on_press=self.button_intrinsic_check, font_name=self.font_path)
        # layout.add_widget(btn_intrinsic_check)

        btn_extrinsic_record = Button(text='2-1拍攝外參', size_hint=(0.19,0.1), size=(btn_wide, 50),  pos_hint={'center_x': pos_ref_x[0], 'center_y':pos_ref_y[2]}, on_release=self.button_extrinsic_record, font_name=self.font_path)
        layout.add_widget(btn_extrinsic_record)
        btn_extrinsic_calculate = Button(text='2-2計算外參', size_hint=(0.19,0.1), size=(btn_wide, 50),  pos_hint={'center_x': pos_ref_x[1], 'center_y':pos_ref_y[2]}, on_release=self.button_extrinsic_calculate, font_name=self.font_path)
        layout.add_widget(btn_extrinsic_calculate)
        # btn_extrinsic_check = Button(text='2-3檢查外參', size_hint=(0.19,0.1), size=(btn_wide, 50),  pos_hint={'center_x': pos_ref_x[2], 'center_y':pos_ref_y[2]}, on_release=self.button_extrinsic_check, font_name=self.font_path)
        # layout.add_widget(btn_extrinsic_check)
        # btn_extrinsic_check = Button(text='3-3檢查外參', size_hint=(0.19,0.1), size=(btn_wide, 50), pos=(240, 400), on_press=self.button_extrinsic_check, font_name=self.font_path)
        # layout.add_widget(btn_extrinsic_check)

        # 拍攝人體動作
        btn_Apose_record = Button(text='3-1拍攝A-pose', size_hint=(0.19,0.1), size=(btn_wide, 50), pos_hint={'center_x': pos_ref_x[0], 'center_y':pos_ref_y[4]}, on_release=self.button_Apose_record, font_name=self.font_path)
        layout.add_widget(btn_Apose_record)
        btn_task_record = Button(text='3-2拍攝動作', size_hint=(0.19,0.1), size=(btn_wide, 50), pos_hint={'center_x': pos_ref_x[1], 'center_y':pos_ref_y[4]}, on_release=self.button_task_record, font_name=self.font_path)
        layout.add_widget(btn_task_record)

        # 計算Marker
        btn_calculate_Marker = Button(text='4計算Marker以及IK', size_hint=(0.4,0.1), size=(btn_wide + 60, 50), pos_hint={'center_x': 0.20, 'center_y':pos_ref_y[6]}, on_release=self.button_calculate_Marker, font_name=self.font_path)
        layout.add_widget(btn_calculate_Marker)

        # 計算IK
        # btn_calculate_IK = Button(text='5-1計算IK', size_hint=(0.19,0.1), size=(btn_wide, 50), pos=(240, 200), on_press=self.button_calculate_IK, font_name=self.font_path)
        # layout.add_widget(btn_calculate_IK)

        # 離開NTK_Cap
        btn_exit = Button(text='結束程式', size_hint=(0.15, 0.1), size=(btn_wide, 50), pos_hint={'center_x': pos_ref_x[4], 'center_y':pos_ref_y[8]}, on_release=self.button_exit, font_name=self.font_path)
        layout.add_widget(btn_exit)

        # 創建當前操作顯示
        self.label_log_hint = Label(text='目前執行操作', size_hint=(0.19,0.1), size=(400, 30),  pos_hint={'center_x': 0.20, 'center_y':pos_ref_y[7]}, font_name=self.font_path)
        layout.add_widget(self.label_log_hint)

        # 執行日期
        self.label_date = Label(text='', size_hint=(0.19,0.1), size=(400, 30),  pos_hint={'center_x': pos_ref_x[1], 'center_y':0.97}, font_name=self.font_path)
        layout.add_widget(self.label_date)
        Clock.schedule_interval(self.update_date, 1)
        
        # Patient ID
        self.patientID = "test"
        self.txt_patientID_real = TextInput(hint_text='Patient ID', multiline=False, size_hint=(0.19,0.1), size=(150, 50),  pos_hint={'center_x': pos_ref_x[4], 'center_y':pos_ref_y[0]}, font_name=self.font_path)
        Clock.schedule_interval(self.patient_ID_update, 0.1)
        layout.add_widget(self.txt_patientID_real)
        self.label_PatientID_real = Label(text=self.patientID, size_hint=(0.19,0.1), size=(400, 30), pos=(500, 570), font_name=self.font_path)
        layout.add_widget(self.label_PatientID_real)
        
        # 內參選擇相機
        # self.select_camID = 0
        # self.txt_cam_ID = TextInput(hint_text='choose cam ID(0~3)', multiline=False, size_hint=(0.19,0.1), size=(160, 40), pos=(20, 450), font_size=16, font_name=self.font_path)
        # layout.add_widget(self.txt_cam_ID)
        #Clock.schedule_interval(self.camID_update, 0.1)

        # Task Name
        self.task = "test"
        self.txt_task = TextInput(hint_text='Task name', multiline=False, size_hint=(0.19,0.1), size=(150, 50), pos_hint={'center_x': pos_ref_x[4], 'center_y':pos_ref_y[2]}, font_name=self.font_path)
        Clock.schedule_interval(self.task_update, 0.1)
        layout.add_widget(self.txt_task)
        self.label_task_real = Label(text=self.patientID, size_hint=(0.19,0.1), size=(400, 30), pos=(500, 470), font_name=self.font_path)
        layout.add_widget(self.label_task_real)

        self.label_log = Label(text=' ', size_hint=(0.19,0.1), size=(400, 50), pos_hint={'center_x': 0.20, 'center_y':pos_ref_y[7]-0.1}, font_name=self.font_path)
        layout.add_widget(self.label_log)

        ##### Button to next page
        btn_to_new_page = Button(text="Advanced Function", size_hint=(0.15, 0.1), size=(btn_wide, 50), pos_hint={'center_x': pos_ref_x[4], 'center_y':pos_ref_y[7]})
        btn_to_new_page.bind(on_release=lambda instance: setattr(self.sm, 'current', 'new_page'))
        layout.add_widget(btn_to_new_page)

        #spinner for camera ID
        # self.txt_camID_spinner = Spinner(text = 'cam ID', values = ("0","1","2","3"),size_hint=(0.19,0.1), size=(100, 30), pos=(20, 450), sync_height = True, font_size=16, font_name=self.font_path)
        # layout.add_widget(self.txt_camID_spinner)
        # self.txt_camID_spinner.bind(text = self.camID_update) 
        self.err_calib_extri = Label(text=read_err_calib_extri(self.current_directory), size_hint=(0.19,0.1), size=(400, 30),  pos_hint={'center_x': pos_ref_x[2], 'center_y':pos_ref_y[2]}, font_name=self.font_path)
        layout.add_widget(self.err_calib_extri)

        btn_toggle_language = Button(
            text='Switch Language',
            size_hint=(0.15, 0.1),
            pos_hint={'center_x': pos_ref_x[4], 'center_y': pos_ref_y[6]},
            on_release=lambda instance: self.toggle_language(btn_calibration_folder,btn_config, btn_check_cam, btn_extrinsic_record,btn_extrinsic_calculate,btn_Apose_record,btn_task_record,btn_calculate_Marker,btn_exit,instance)
        )
        layout.add_widget(btn_toggle_language)
        checkbox = CheckBox(size_hint=(None, None), size=(48, 48), pos_hint={'center_x': 0.3, 'center_y': 0.5})
        
        
        # Label for the CheckBox
        checkbox_label = Label(text="Enable feature", size_hint=(None, None), size=(200, 30), pos_hint={'center_x': 0.45, 'center_y': 0.5})
        checkbox.bind(active=self.on_checkbox_active)
        # Adding widgets to the layout
        layout.add_widget(checkbox)
        layout.add_widget(checkbox_label)
        self.COM_input = TextInput(hint_text='COM number', multiline=False, size_hint=(0.19,0.1), size=(150, 50), pos_hint={'center_x': pos_ref_x[4], 'center_y':pos_ref_y[4]}, font_name=self.font_path)
        Clock.schedule_interval(self.task_update, 0.1)
        layout.add_widget(self.COM_input)
        return layout
    
       
    def on_checkbox_active(self, checkbox, value):
        self.is_checkbox_viconsync_checked = value  # Store checkbox state in the app
        if value:
            print("Checkbox Checked")
            CP2102_output_signal(self.COM_input.text)
        else:
            print("Checkbox Unchecked")

    def toggle_language(self,btn_calibration_folder,btn_config,btn_check_cam,btn_extrinsic_record,btn_extrinsic_calculate,btn_Apose_record,btn_task_record,btn_calculate_Marker,btn_exit,instance):
        self.language = 'Chinese' if self.language == 'English' else 'English'
        self.update_ui_language(btn_calibration_folder,btn_config,btn_check_cam,btn_extrinsic_record,btn_extrinsic_calculate,btn_Apose_record,btn_task_record,btn_calculate_Marker,btn_exit)
        
    def update_ui_language(self,btn_calibration_folder,btn_config,btn_check_cam,btn_extrinsic_record,btn_extrinsic_calculate,btn_Apose_record,btn_task_record,btn_calculate_Marker,btn_exit):
        # Update all UI text based on current language
        if self.language == 'English':
            btn_calibration_folder.text = '1-1Create New \nParameters'
            btn_config.text = '1-2Detect camera'
            btn_check_cam.text = '1-3Check Camera'
            btn_extrinsic_record.text ='2-1Record Extri'
            btn_extrinsic_calculate.text = '2-2Calculate Extri'
            btn_Apose_record.text = '3-1Record A-pose'
            btn_task_record.text = '3-2Record task'
            btn_calculate_Marker.text = '4Calculate Marker & IK '
            btn_exit.text ='Exit'
            # Update all other widgets similarly
        else:
            btn_calibration_folder.text = '1-1建立新參數'
            btn_config.text = '1-2偵測相機'
            btn_check_cam.text = '1-3檢查相機'
            btn_extrinsic_record.text ='2-1拍攝外參'
            btn_extrinsic_calculate.text = '2-2計算外參'
            btn_Apose_record.text = '3-1拍攝A-pose'
            btn_task_record.text = '3-2拍攝動作'
            btn_calculate_Marker.text = '4計算Marker以及IK'
            btn_exit.text ='結束程式'
            # Update all other widgets similarly
    # log
    def add_log(self, message):
        print('hi')
        # with open(self.log_file, 'a') as f:
        #     date_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        #     command = date_time_str + " : " + message
        #     f.write(command + '\n')
    # button click
    def create_calibration_ask(self, instance):
            content = BoxLayout(orientation='vertical', padding=10, spacing=10)
            message = Label(text='Are you sure to reset calibration?')
            content.add_widget(message)
            button_layout = BoxLayout(size_hint_y=None, height='50dp', spacing=10)
            # No button
            no_btn = Button(text='No')
            button_layout.add_widget(no_btn)

            # Create the popup
            popup = Popup(title='Confirm Action', content=content, size_hint=(None, None), size=(400, 200))

            # Yes button
            yes_btn = Button(text='Yes')
            yes_btn.bind(on_release=lambda instance: self.create_calibration(instance, popup))
            button_layout.add_widget(yes_btn)

            no_btn.bind(on_release=lambda *args: popup.dismiss())

            content.add_widget(button_layout)

            popup.open()
        
    def create_calibration(self, instance,popup):
        popup.dismiss()
        create_calibration_folder(self.current_directory)
        self.label_log.text = "create new folder of calibration"
    # def button_create_new(self, instance):
    #     now = datetime.now()
    #     self.formatted_datetime = now.strftime("%Y_%m_%d_%H%M")
    #     with open(self.time_file_path, "w") as file:
    #         file.write(self.formatted_datetime)
    #     self.popup.dismiss()
    #     self.label_log.text = "Create new date of calibration:" + self.formatted_datetime
    #     create_calibration_folder(self.current_directory)
    #     self.add_log(self.label_log.text)
        
    # def button_use_recorded(self, instance):
    #     self.popup.dismiss()
    #     self.label_log.text = "Use recorded date of calibration:" + self.formatted_datetime
    #     self.add_log(self.label_log.text)
        
    def button_calibration_folder(self, instance):
        # self.label_log.text = "創建新的calibration資料夾"
        self.label_log.text = "create new folder of calibration"
        create_calibration_folder(self.current_directory)
        self.add_log(self.label_log.text)
    
    def button_config(self, instance):
        # self.label_log.text = "檢測Webcam ID並更新config"
        self.label_log.text = "detect Webcam ID and update config"
        camera_config_create(self.config_path)
        camera_config_update(self.config_path, 10)
        self.add_log(self.label_log.text)

    def button_check_cam(self, instance):
        # self.label_log.text = "開啟Webcam(按Q離開)"
        self.label_log.text = "open webcam(press Q to leave)"
        self.add_log(self.label_log.text)
        camera_config_open_camera(self.config_path)
        # self.label_log.text = "關閉Webcam"
        self.label_log.text = "close Webcam"
        self.add_log(self.label_log.text)

    
    def button_intrinsic_record(self, instance):
        # self.label_log.text = "拍攝內參"
        self.label_log.text = "create intrinsic"
        self.add_log(self.label_log.text)
        if int(self.select_camID) in (0, 1, 2, 3):
            camera_intrinsic_calibration(self.config_path, self.record_path, camera_ID=[int(self.select_camID)])
            # self.label_log.text = "拍攝完畢"
            self.label_log.text = "finished"
        else:
            # self.label_log.text = "輸入數值有誤，請輸入[0, 1, 2, 3]之一的數值"
            self.label_log.text = "please choose webcam"
        self.add_log(self.label_log.text)
    
    def button_intrinsic_calculate(self, instance):
        # self.label_log.text = '計算內參'
        self.label_log.text = 'caculating intrinsic'
        self.add_log(self.label_log.text)
        try:
            calib_intri(self.current_directory)
            # self.label_log.text = '內參計算完畢'
            self.label_log.text = 'caculate finished'
        except:
            # self.label_log.text = '檢查是否有拍攝內參'
            self.label_log.text = 'check intrinsic exist'
        self.add_log(self.label_log.text)
    
    # def button_intrinsic_check(self, instance):
    #     self.label_log.text = '檢查內參'
    #     self.add_log(self.label_log.text)
    
    def button_extrinsic_record(self, instance):
        # self.label_log.text = '拍攝外參'
        self.label_log.text = "create extrinsic"
        self.add_log(self.label_log.text)
        camera_extrinsicCalibration_record(self.config_path, self.record_path, button_capture=False, button_stop=False)
        # self.label_log.text = '拍攝完畢'
        self.label_log.text = "finished"
        self.add_log(self.label_log.text)
    
    def button_extrinsic_calculate(self, instance):
        self.label_log.text = '計算外參'
        self.label_log.text = 'caculating extrinsic'
        self.add_log(self.label_log.text)
        try:
            
            
            err_list =calib_extri(self.current_directory)
            self.label_log.text = 'calculate finished'
            
            self.err_calib_extri.text = err_list      
            # self.label_log.text = '外參計算完畢'

        except:
            # self.label_log.text = '檢查是否有拍攝以及計算內參，以及是否有拍攝外參'
            self.label_log.text = 'check intrinsic and extrinsic exist'
            self.err_calib_extri.text = 'no calibration file found'
        self.add_log(self.label_log.text)
        
 
    def button_extrinsic_check(self, instance):
        self.label_log.text = '檢查外參'
        self.add_log(self.label_log.text)
        #try:
        fintune_chessboard_v1(self.calib_toml_path, self.extrinsic_path)
        # self.label_log.text = '外參計算完畢'
        self.label_log.text = 'extrinsic check finished'
        # except:
        #     self.label_log.text = 'check extrinsic exist'
        self.add_log(self.label_log.text)
    

############### Apose 不能隔天重拍，要就要當下
    def button_Apose_record(self, instance):
        self.label_log.text = 'film A-pose'
        self.add_log(self.label_log.text)
        if self.label_PatientID_real.text == "":
            self.label_log.text = 'check Patient ID'
        elif os.path.isdir(os.path.join(self.record_path, "Patient_data",self.txt_patientID_real.text,datetime.now().strftime("%Y_%m_%d"),'raw_data'))==0: ## check if path exist
            camera_Apose_record(self.config_path,self.record_path,self.txt_patientID_real.text,datetime.now().strftime("%Y_%m_%d"),button_capture=False,button_stop=False) 
        else:
            content = BoxLayout(orientation='vertical', padding=10, spacing=10)
            message = Label(text='You did not change the Patient ID, Do you want to replace the original Apose?')
            content.add_widget(message)
            button_layout = BoxLayout(size_hint_y=None, height='50dp', spacing=10)
            # No button
            no_btn = Button(text='No')
            button_layout.add_widget(no_btn)

            # Create the popup
            popup = Popup(title='Confirm Action', content=content, size_hint=(None, None), size=(400, 200))

            # Yes button
            yes_btn = Button(text='Yes')
            yes_btn.bind(on_release=lambda instance: self.perform_Apose_recording(instance, popup))
            button_layout.add_widget(yes_btn)

            no_btn.bind(on_release=lambda *args: popup.dismiss())

            content.add_widget(button_layout)

            popup.open()
            self.label_log.text = self.label_PatientID_real.text + " film A-pose finished"
            self.add_log(self.label_log.text)

    def perform_Apose_recording(self, instance, popup):
        popup.dismiss()  # Dismiss the popup first
        shutil.rmtree(os.path.join(self.record_path, "Patient_data",self.txt_patientID_real.text,datetime.now().strftime("%Y_%m_%d"),'raw_data'))
        camera_Apose_record(self.config_path,self.record_path,self.txt_patientID_real.text,datetime.now().strftime("%Y_%m_%d"),button_capture=False,button_stop=False) 
        #import pdb;pdb.set_trace()
    def button_task_record(self, instance):
        # self.label_log.text = '拍攝動作'
        self.label_log.text = 'film motion'
        self.add_log(self.label_log.text)
        date = datetime.now().strftime("%Y_%m_%d")
        if self.label_PatientID_real.text == "":
            # self.label_log.text = '請輸入Patient ID'
            self.label_log.text = 'check Patient ID'
        elif self.label_task_real.text == "":
            # self.label_log.text = '請輸入task name'
            self.label_log.text = 'Enter task name'
        elif os.path.isdir(os.path.join(self.record_path, "Patient_data",self.txt_patientID_real.text,date,'raw_data',self.label_task_real.text))==0: ## check if path exist
            camera_Motion_record(self.config_path, self.record_path, self.label_PatientID_real.text, self.label_task_real.text, date, button_capture=False, button_stop=False)
            self.label_log.text = self.label_PatientID_real.text + " : " + self.label_task_real.text + ", film finished"
        else:
            content = BoxLayout(orientation='vertical', padding=10, spacing=10)
            message = Label(text='You did not change the Task Name, Do you want to replace the original Task?')
            content.add_widget(message)
            button_layout = BoxLayout(size_hint_y=None, height='50dp', spacing=10)
            # No button
            no_btn = Button(text='No')
            button_layout.add_widget(no_btn)

            # Create the popup
            popup = Popup(title='Confirm Action', content=content, size_hint=(None, None), size=(400, 200))

            # Yes button
            yes_btn = Button(text='Yes')
            yes_btn.bind(on_release=lambda instance: self.perform_Motion_recording(instance, popup,date))
            button_layout.add_widget(yes_btn)

            no_btn.bind(on_release=lambda *args: popup.dismiss())

            content.add_widget(button_layout)

            popup.open()
            self.label_log.text = self.label_PatientID_real.text + " : " + self.label_task_real.text + ", film finished"
        self.add_log(self.label_log.text)
    def perform_Motion_recording(self, instance, popup,date):
        popup.dismiss()  # Dismiss the popup first
        date = datetime.now().strftime("%Y_%m_%d")
        shutil.rmtree(os.path.join(self.record_path, "Patient_data",self.txt_patientID_real.text,date,'raw_data',self.label_task_real.text))
        if self.is_checkbox_viconsync_checked:
            camera_Motion_record_VICON_sync(self.config_path, self.record_path, self.label_PatientID_real.text, self.label_task_real.text, date,self.COM_input.text, button_capture=False, button_stop=False)
        else:
            camera_Motion_record(self.config_path, self.record_path, self.label_PatientID_real.text, self.label_task_real.text, date, button_capture=False, button_stop=False)
    
    def button_calculate_Marker(self, instance):
        # self.label_log.text = '計算Marker以及IK'
        try:
            self.label_log.text = 'calculating Marker and IK'

            # root = tk.Tk()
            # root.withdraw() # 隐藏根窗口

            #initial_dir = self.patient_path if os.path.isdir(self.patient_path) else "/"
            # cal_folder_path = tk.filedialog.askdirectory(initialdir=initial_dir) #選到要計算的ID下的日期
            # root.destroy() # 關閉窗口
            initial_dir = self.patient_path if os.path.isdir(self.patient_path) else "/"
            selected_directories =select_directories_and_return_list(initial_dir)
            # print("initial_dir:", self.patient_path)
            for dir_sel_loop in range(len(selected_directories)):
                cal_folder_path =selected_directories[dir_sel_loop]
                marker_caculate(self.current_directory , cal_folder_path)
            # self.label_log.text = 'Marker以及IK計算完畢'
            self.label_log.text = 'Marker and IK caculate finished'
            self.add_log(self.label_log.text)
        except Exception as e:
            self.label_log.text = 'Re-select the directory'
            self.add_log(self.label_log.text)
            
            print(f"An error occurred: {e}")
            traceback.print_exc()
    

    # def button_calculate_IK(self, instance):
    #     self.label_log.text = '計算IK'
    #     self.add_log(self.label_log.text)

    def button_exit(self, instance):
        # self.label_log.text = "離開NTK_Cap"
        self.label_log.text = "leave NTK_Cap"
        self.add_log(self.label_log.text)
        print("======================================================================================================")
        print("離開NTK_Cap")
        exit()

    # text
    def patient_ID_update(self, dt):
        self.label_PatientID_real.text = self.txt_patientID_real.text

    def task_update(self, dt):
        self.label_task_real.text = self.txt_task.text

    def camID_update(self, spinner, text):
        #self.select_camID = self.txt_cam_ID.text
        self.select_camID = text
        print(self.select_camID)

    def update_date(self, dt):
        now_date = datetime.now()
        #now_time = "Date : " + str(now_date.year) + "/" + str(now_date.month) + "/" + str(now_date.day)
        now_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.label_date.text = now_time



if __name__ == '__main__':
    NTK_CapApp().run()
