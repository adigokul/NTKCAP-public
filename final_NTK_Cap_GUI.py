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
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.scrollview import ScrollView
from kivy.uix.gridlayout import GridLayout
import time
from datetime import datetime
import os
import cv2
from NTK_CAP.script_py.NTK_Cap import *

from NTK_CAP.script_py.cloud_function import *
from check_extrinsic import *
import tkinter as tk
from tkinter import filedialog
from kivy.animation import Animation
from NTK_CAP.script_py.kivy_file_chooser import select_directories_and_return_list
import traceback
import requests
import sqlite3
from natsort import natsorted
SETTINGS_FILE = r'C:\Users\Hermes\Desktop\NTKCAP\Patient_data\settings.json'

FONT_PATH = os.path.join(os.getcwd(), "NTK_CAP", "ThirdParty", "Noto_Sans_HK", "NotoSansHK-Bold.otf")
# 连接到SQLite数据库
conn = sqlite3.connect(os.path.join(os.getcwd(),'icd10','icd10.db'))

# 定义模糊搜索函数
def fuzzy_search(query):
    cursor = conn.execute("""
        SELECT Code, CM2023_英文名稱, CM2023_中文名稱 
        FROM icd10_fts 
        WHERE CM2023_英文名稱 LIKE ? 
        OR CM2023_中文名稱 LIKE ? 
        OR Code LIKE ?
    """, (f'%{query}%', f'%{query}%', f'%{query}%'))
    results = cursor.fetchall()
    return results
class TaskEDITInputScreen(Screen):
    def __init__(self, btn= None,meetdir=None, actiondir=None,meetId=None,actionId = None, **kwargs):
        super(TaskEDITInputScreen, self).__init__(**kwargs)
        self.layout = BoxLayout(orientation='vertical')
        self.current_directory = os.getcwd()
        self.config_path = os.path.join(self.current_directory, "config")
        self.meetdir = meetdir
        self.actiondir = actiondir
        # Add "Choose Layout" button
        self.choose_layout_button = Button(text="Choose Layout", size_hint_y=None, height=40)
        self.choose_layout_button.bind(on_press=self.open_file_chooser)
        self.layout.add_widget(self.choose_layout_button)

        # Create the ScrollView to take up 80% of the layout

        self.results_layout = BoxLayout(orientation='horizontal', size_hint=(1, 0.8))
        
        self.layout.add_widget(self.results_layout)

        # Create a button at the bottom right
        bottom_right_button = Button(text="Print Task", size_hint=(0.3, 0.1), pos_hint={'right': 1, 'bottom': 1})
        bottom_right_button.bind(on_press=self.print_task_info)
        self.layout.add_widget(bottom_right_button)

        # Initialize instance variables for the task input boxes and spinner
        self.task_name_input = None
        self.task_number_input = None
        self.task_spinner = None
        self.layout_json_dir =os.path.join(self.config_path,'layout_temp.json')
        self.add_widget(self.layout)
        self.layouts = None 
        
        # Function to fetch data from the API
    def save_load_from_selected_action(self,meetId,actionId ):
        MeetActionID2json(meetId,actionId,self.layout_json_dir)
        self.load_layout(self.layout_json_dir)
    def fetch_layouts(self):
        if self.layouts is None:  # Only fetch if layouts have not been fetched yet
            host = "https://motion-service.yuyi-ocean.com"
            url = f"{host}/api/layouts"
            response = requests.get(url)
            if response.status_code == 200:
                self.layouts = response.json()['resources']  # Store the fetched layouts
        return self.layouts

    # Function to handle the selection of buttons
    def on_layout_selected(self, instance, column, layout_data):
        # Deselect all buttons in the same column and select only the current one
        if column == 'meet':
            for btn in self.meet_buttons:
                btn.background_color = (1, 1, 1, 1)  # Deselect (white background)
            self.selected_meet_data = layout_data  # Store full data for meet
        elif column == 'action':
            for btn in self.action_buttons:
                btn.background_color = (1, 1, 1, 1)  # Deselect (white background)
            self.selected_action_data = layout_data  # Store full data for action

        # Highlight the selected button
        instance.background_color = (0, 1, 0, 1)  # Selected (green background)

    # Function to create and show popup
    def show_layouts_popup(self, instance):
        layouts = self.fetch_layouts()  # Fetch layout data from the API or use cached data

        # Create separate GridLayouts for "meet" and "action"
        meet_layout = GridLayout(cols=1, padding=10, spacing=10, size_hint_y=None)
        action_layout = GridLayout(cols=1, padding=10, spacing=10, size_hint_y=None)

        # Bind layout heights to allow dynamic resizing
        meet_layout.bind(minimum_height=meet_layout.setter('height'))
        action_layout.bind(minimum_height=action_layout.setter('height'))

        # Add layout IDs as buttons to their respective layouts
        meet_layouts = [res for res in layouts if res.get('catalog') == 'meet']
        action_layouts = [res for res in layouts if res.get('catalog') == 'action']

        self.meet_buttons = []
        self.action_buttons = []

        for layout_data in meet_layouts:
            btn_meet = Button(text=layout_data['layoutId'], size_hint_y=None, height=40)
            btn_meet.bind(on_press=lambda x, ld=layout_data: self.on_layout_selected(x, 'meet', ld))
            self.meet_buttons.append(btn_meet)
            meet_layout.add_widget(btn_meet)

        for layout_data in action_layouts:
            btn_action = Button(text=layout_data['layoutId'], size_hint_y=None, height=40)
            btn_action.bind(on_press=lambda x, ld=layout_data: self.on_layout_selected(x, 'action', ld))
            self.action_buttons.append(btn_action)
            action_layout.add_widget(btn_action)
        
        # Create scrollable views for both layouts
        meet_scroll = ScrollView(size_hint=(0.5, None), size=(200, 300))
        meet_scroll.add_widget(meet_layout)

        action_scroll = ScrollView(size_hint=(0.5, None), size=(200, 300))
        action_scroll.add_widget(action_layout)
        
        # Create the main layout for the popup
        scroll_layout = BoxLayout(orientation='horizontal')

        # Add headers for the columns
        header_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=40)
        header_layout.add_widget(Label(text='Catalog: meet', size_hint=(0.5, 1)))
        header_layout.add_widget(Label(text='Catalog: action', size_hint=(0.5, 1)))
        
        # Add the scroll views to the main layout
        scroll_layout.add_widget(meet_scroll)
        scroll_layout.add_widget(action_scroll)

        # Create a Select button, always enabled
        self.btn_select_final = Button(text="Select", size_hint=(0.2, 0.1))
        self.btn_select_final.bind(on_release=self.final_selection_made)

        # Add the Select button at the bottom of the popup

        # Create the popup layout
        popup_layout = BoxLayout(orientation='vertical')
        popup_layout.add_widget(header_layout)
        popup_layout.add_widget(scroll_layout)  # Add the scrollable columns
        popup_layout.add_widget(self.btn_select_final)

        # Create the actual popup
        self.popup = Popup(title='Layout IDs',
                           content=popup_layout,
                           size_hint=(0.8, 0.8))

        # Open the popup
        self.popup.open()
    
    def final_selection_made(self, instance):
        
        def transform_fields(fields, catalog):
            transformed_fields = []

            for field in fields:
                # Create the basic transformed structure
                transformed_field = {
                    "type": field["elementType"],  # Map elementType to type
                    "title": field["title"],  # Title remains the same
                    "content": "",  # Default content is an empty string
                    "notetype": catalog  # Map catalog to notetype
                }

                # If the field has options, add the options and set default content
                if "options" in field and field["options"] is not None:
                    transformed_field["options"] = field["options"]
                    transformed_field["content"] = "Choose an option"
                
                # Append transformed field to the list
                transformed_fields.append(transformed_field)

            return transformed_fields

        # Transform and combine fields from both meet and action
        combined_data = {}  # Will store layoutId and fields together
        combined_fields = []

        # Transform and combine meet fields
        if self.selected_meet_data:
            transformed_meet_fields = transform_fields(self.selected_meet_data.get('fields', []), 'meet')
            print("Transformed Meet Fields:")
            print(transformed_meet_fields)
            combined_fields += transformed_meet_fields  # Add meet fields to combined list
            combined_data["meet_layoutId"] = self.selected_meet_data['layoutId']  # Save meet layoutId
        else:
            print("No Meet Layout selected.")

        # Transform and combine action fields
        if self.selected_action_data:
            transformed_action_fields = transform_fields(self.selected_action_data.get('fields', []), 'action')
            print("Transformed Action Fields:")
            print(transformed_action_fields)
            combined_fields += transformed_action_fields  # Add action fields to combined list
            combined_data["action_layoutId"] = self.selected_action_data['layoutId']  # Save action layoutId
        else:
            print("No Action Layout selected.")

        # Add combined fields to the final data
        combined_data["fields"] = combined_fields

        # Print the combined data
        print("Combined Data with IDs and Fields:")
        print(combined_data)
        # Save combined_data to a JSON file
        with open(self.layout_json_dir, "w") as json_file:
            json.dump(combined_data, json_file, indent=4)  # Save with indentation for readability
            
        # Add logic to close the popup after printing
        if hasattr(self, 'popup'):
            self.popup.dismiss()  # Close the popup if it's open

        self.load_layout(self.layout_json_dir)
    def open_file_chooser(self, instance):
        self.show_layouts_popup(self)
    def save_data(self, instance):
        data_to_save_meetnote = {}
        data_to_save_actionnote = {}
        
        # Iterate over all widgets in the results layout
        for box in self.left_half.children:
            
            if isinstance(box, BoxLayout):
                title_label = box.children[1]  # Assuming the Label is always the second widget
                title = title_label.text
                
                # Handle different input types
                input_widget = box.children[0]  # Assuming the input widget is always the first widget
                
                if isinstance(input_widget, TextInput):
                    data_to_save_meetnote[title] = input_widget.text
                elif isinstance(input_widget, Spinner):
                    data_to_save_meetnote[title] = input_widget.text  # Spinner's current selection
                elif isinstance(input_widget, BoxLayout):
                    data_to_save_meetnote[title] = input_widget.children[1].text
                print(data_to_save_meetnote)
        for box in self.right_half.children:
            
            if isinstance(box, BoxLayout):
                title_label = box.children[1]  # Assuming the Label is always the second widget
                title = title_label.text
                
                # Handle different input types
                input_widget = box.children[0]  # Assuming the input widget is always the first widget
                
                if isinstance(input_widget, TextInput):
                    data_to_save_actionnote[title] = input_widget.text
                elif isinstance(input_widget, Spinner):
                    data_to_save_actionnote[title] = input_widget.text  # Spinner's current selection
                elif isinstance(input_widget, BoxLayout):
                    data_to_save_actionnote[title] = input_widget.children[1].text
                print(data_to_save_actionnote)

        # import pdb;pdb.set_trace()
        # Now `data_to_save` contains all the titles and values
        # You can print it or save it to a file, database, etc.
        print(data_to_save_actionnote)

        with open(self.layout_json_dir, 'r', encoding='utf-8') as f:
            layout_data = json.load(f)
        layout_data = layout_data['fields']
        # Separate lists for 'meet' and 'action' items
        meetnote_data = []
        actionnote_data = []

        # Divide the items into the respective lists
        for item in layout_data:
            title = item['title']
            if item["notetype"] == 'meet':
                if title in data_to_save_meetnote:
                    item['content'] = data_to_save_meetnote[title]
                meetnote_data.append(item)
            elif item["notetype"] == 'action':
                if title in data_to_save_actionnote:
                    item['content'] = data_to_save_actionnote[title]
                actionnote_data.append(item)

        # Save 'meet' items to a separate JSON file
        self.meetnote_file_path = os.path.join(self.meetdir, 'Meet_note.json')
        with open(self.meetnote_file_path, 'w', encoding='utf-8') as json_file:
            json.dump(meetnote_data, json_file, indent=4, ensure_ascii=False)

        # Save 'action' items to a separate JSON file
        self.actionnote_file_path = os.path.join(self.actiondir, 'Action_note.json')
        with open(self.actionnote_file_path, 'w', encoding='utf-8') as json_file:
            json.dump(actionnote_data, json_file, indent=4, ensure_ascii=False)
        with open(self.layout_json_dir,'r',encoding='utf-8')as file:
            temp = json.load(file)
            temp['fields'] =meetnote_data+actionnote_data
        with open(self.layout_json_dir,'w',encoding='utf-8')as json_file:
            json.dump(temp, json_file, indent=4, ensure_ascii=False)
    def upload_cloud(self):
        Mstatus,Mmessage = meet_update(self.layout_json_dir,self.meetnote_file_path,self.meetId)
        Astatus,Amessage =action_update(self.layout_json_dir,self.actionnote_file_path,self.actionId)
        return Mstatus,Mmessage,Astatus,Amessage
    def load_layout(self, layout_file):
            self.layout_json_dir = layout_file
            with open(layout_file, 'r', encoding='utf-8') as f:
                layout_data = json.load(f)
            #import pdb;pdb.set_trace()
            # Clear the current layout
            layout_data = layout_data['fields']
            self.results_layout.clear_widgets()

            # Create left and right halves with size_hint
           
            left_scroll_view = ScrollView(size_hint=(0.5, 1))
            self.left_half = BoxLayout(orientation='vertical', size_hint_y=None, spacing=10, pos_hint={'top': 1})
            self.left_half.bind(minimum_height=self.left_half.setter('height'))  # Ensure scrolling works as expected
            left_scroll_view.add_widget(self.left_half)

            # Create scrollable right half
            right_scroll_view = ScrollView(size_hint=(0.5, 1))
            self.right_half = BoxLayout(orientation='vertical', size_hint_y=None, spacing=10, pos_hint={'top': 1})
            self.right_half.bind(minimum_height=self.right_half.setter('height'))  # Ensure scrolling works as expected
            right_scroll_view.add_widget(self.right_half)
            title_label = Label(text='Meet Note', size_hint=(0.5, None), height=40)
            self.left_half.add_widget(title_label)
            title_label = Label(text='Action Note', size_hint=(0.5, None), height=40)
            self.right_half.add_widget(title_label)
            for item in layout_data:
                # Determine whether to place in the left or right half
                target_layout = self.left_half if item.get('notetype') == 'meet' else self.right_half
                
                if item['type'] == 'input' and item['title'].lower() == 'symptoms':
                    new_box = BoxLayout(orientation='horizontal', size_hint_y=None, height=120)
                    title_label = Label(text=item['title'], size_hint=(0.5, None), height=120)  # 0.2 will make it take 20% of the available width
                    new_box.add_widget(title_label)

                    input_and_button_box = BoxLayout(orientation='horizontal', size_hint=(1, None), height=120)
                    self.idc_result_input = TextInput(text=item['content'], size_hint_y=None, height=120, multiline=True, font_name=FONT_PATH)
                    input_and_button_box.add_widget(self.idc_result_input)

                    search_button = Button(text="ICD\n-\n10", size_hint_x=None, width=30)
                    search_button.bind(on_press=self.open_search_popup)
                    input_and_button_box.add_widget(search_button)

                    new_box.add_widget(input_and_button_box)
                    target_layout.add_widget(new_box)

                elif item['type'] == 'input':
                    new_box = BoxLayout(orientation='horizontal', size_hint_y=None, height=40)
                    title_label = Label(text=item['title'], size_hint=(0.5, None), size_hint_y=None, height=40)
                    new_box.add_widget(title_label)
                    
                    string_input = TextInput(size_hint=(1, None), height=40, text=item['content'])
                                        # Check the title to assign to the correct instance variable
                    if item['title'].lower() == 'task type' and item.get('notetype') == 'action':
                        self.task_name_input = string_input
                    elif item['title'].lower() == 'task number' and item.get('notetype') == 'action':
                        self.task_number_input = string_input
                    new_box.add_widget(string_input)
                    
                    target_layout.add_widget(new_box)

                elif item['type'] == 'spinner':
                    new_box = BoxLayout(orientation='horizontal', size_hint_y=None, height=40)
                    title_label = Label(text=item['title'], size_hint=(0.5, None), size_hint_y=None, height=40)
                    new_box.add_widget(title_label)
                    
                    spinner = Spinner(
                        text=item['content'],
                        values=item['options'],
                        size_hint=(1, None),
                        height=40
                    )
                        # Check the title to assign to the correct instance variable
                    if item['title'].lower() == 'task type' and item.get('notetype') == 'action':
                        self.task_name_input = spinner
                    elif item['title'].lower() == 'task number' and item.get('notetype') == 'action':
                        self.task_number_input = spinner
                    new_box.add_widget(spinner)
                    
                    target_layout.add_widget(new_box)

            # Add the halves to the results layout
            self.results_layout.add_widget(left_scroll_view)
            self.results_layout.add_widget(right_scroll_view)


                    

    def open_search_popup(self, instance):
        # 创建弹出窗口
        popup_layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        
        self.popup_input = TextInput(size_hint_y=None, height=40, font_name=FONT_PATH)
        self.popup_input.bind(text=self.on_text)
        popup_layout.add_widget(self.popup_input)
        
        # 创建滚动视图来显示搜索结果
        self.results_view = ScrollView(size_hint=(1, None), size=(popup_layout.width, 300))
        self.result_layout = GridLayout(cols=1, size_hint_y=None)
        self.result_layout.bind(minimum_height=self.result_layout.setter('height'))
        self.results_view.add_widget(self.result_layout)
        popup_layout.add_widget(self.results_view)
        
        self.popup = Popup(title="Enter Search Query", content=popup_layout, size_hint=(0.9, 0.9))
        self.popup.open()
    
    def on_text(self, instance, value):
        self.result_layout.clear_widgets()
        if value.strip():  # 检查输入是否为空
            search_results = fuzzy_search(value)
            for code, en_description, cn_description in search_results[:25]:  # 显示前25个结果
                result_button = Button(text=f"{code}: {en_description} / {cn_description}", size_hint_y=None, height=40, font_name=FONT_PATH)
                result_button.bind(on_press=lambda x, c=code, e=en_description, cn=cn_description: self.select_result(c, e, cn))
                self.result_layout.add_widget(result_button)
    def rename_folder(self,old_folder_name, new_folder_name):
    # Check if the old folder exists
        
        if not os.path.exists(old_folder_name):
            print(f"Folder does not exist.")
            return f"Folder does not exist."
        
        # Check if the old and new folder names are the same
        if old_folder_name == new_folder_name:
            print(f"The old folder '{old_folder_name}' is the same as the new folder name. No renaming needed.")
            return 1
        
        # Check if the new folder already exists
        if os.path.exists(new_folder_name):
            print(f"Error: The target folder already exists.")
            # Optionally: handle the situation, like merging or deleting the existing folder
            # shutil.rmtree(new_folder_name)  # Use carefully to delete the existing folder
            return f"Error: The rename folder already exists."

        os.rename(old_folder_name, new_folder_name)
        return 1
    def select_result(self, code, en_description, cn_description):
        print(f"Selected: {code}: {en_description} / {cn_description}")
        
        # 在右侧布局的输入框中显示选择的IDC结果，结果用逗号分隔
        current_text = self.idc_result_input.text
        new_text = f"{code}: {en_description} / {cn_description}"
        if current_text:
            self.idc_result_input.text = f"{current_text}, {new_text}"
        else:
            self.idc_result_input.text = new_text
        
        self.popup.dismiss()
    def show_warning_popup(self,warning_text):
        # Layout for the popup
                layout = BoxLayout(orientation="vertical", padding=10, spacing=10)

                # Add a label with the message
                message_label = Label(text=warning_text)
                layout.add_widget(message_label)

                # Add the close button
                close_button = Button(text="Close", size_hint=(1, 0.5))
                close_button.bind(on_press=self.close_popup)
                layout.add_widget(close_button)

                # Create the popup window
                self.popup_window = Popup(title="Popup Example", content=layout, size_hint=(0.7, 0.4))
                self.popup_window.open()

    def close_popup(self, instance):
        # Close the popup window
        self.popup_window.dismiss()
    def print_task_info(self, instance):
        task_name = self.task_name_input.text
        task_number = self.task_number_input.text
        self.save_data(instance)
        Mstatus,Mmessage,Astatus,Amessage =self.upload_cloud()
        if Mstatus==200 and Astatus==200:
            rename_status =self.rename_folder(self.actiondir, os.path.join(self.meetdir,f"{task_name}_{task_number}"))
            if rename_status ==1:
                print(f"Task: {self.manager.parent_app.task_name}")
                self.actiondir = os.path.join(self.meetdir,f"{task_name}_{task_number}")
                
                self.btn.text = f"{task_name}_{task_number}"
                self.btn.unbind(on_press=None)
                self.btn.bind(on_press=lambda instance, btn=self.btn,meetdir =self.meetdir,actiondir =os.path.join(self.meetdir,f"{task_name}_{task_number}"),meetId=self.meetId,actionId=self.actionId: self.manager.parent_app.set_taskinput_screen_with_param('taskEDIT_input', btn,meetdir, actiondir,meetId,actionId))
                self.manager.current = 'main'  # Switch back to the main screen
            else:
                
                self.show_warning_popup(rename_status)
        else:
            self.show_warning_popup(Mmessage +' \n' + Amessage)
# class TaskEDITInputScreen(Screen):
#     def __init__(self, meetdir=None, actiondir=None, **kwargs):
#         super(TaskEDITInputScreen, self).__init__(**kwargs)
#         self.layout = BoxLayout(orientation='vertical')
#         self.current_directory = os.getcwd()
#         self.config_path = os.path.join(self.current_directory, "config")
#         self.meetdir = meetdir
#         self.actiondir = actiondir

#         # Add "Choose Layout" button
#         self.choose_layout_button = Button(text="Choose Layout", size_hint_y=None, height=40)
#         self.choose_layout_button.bind(on_press=self.open_file_chooser)
#         self.layout.add_widget(self.choose_layout_button)

#         # Create the ScrollView to take up 80% of the layout
#         self.results_layout = BoxLayout(orientation='horizontal', size_hint=(1, 0.8))
#         self.layout.add_widget(self.results_layout)

#         # Create a button at the bottom right
#         bottom_right_button = Button(text="Print Task", size_hint=(0.3, 0.1), pos_hint={'right': 1, 'bottom': 1})
#         bottom_right_button.bind(on_press=self.print_task_info)
#         self.layout.add_widget(bottom_right_button)

#         # Initialize instance variables for the task input boxes and spinner
#         self.task_name_input = None
#         self.task_number_input = None
#         self.task_spinner = None
#         self.layout_json_dir_meet = ''
#         self.layout_json_dir_action = ''
#         self.add_widget(self.layout)

#     def open_file_chooser(self, instance):
#         # 创建文件选择弹出窗口
#         file_chooser = FileChooserIconView(path='.', filters=['*.json'])
#         popup_layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
#         popup_layout.add_widget(file_chooser)
        
#         # 确认按钮
#         confirm_button = Button(text="Load Layout", size_hint_y=None, height=40)
#         confirm_button.bind(on_press=lambda x: self.load_layout(file_chooser.selection))
#         popup_layout.add_widget(confirm_button)
        
#         self.popup = Popup(title="Choose Layout File", content=popup_layout, size_hint=(0.9, 0.9))
#         self.popup.open()
#     def save_data(self, instance):
#         data_to_save_meetnote = {}
#         data_to_save_actionnote = {}
        
#         # Iterate over all widgets in the results layout
#         for box in self.left_half.children:
            
#             if isinstance(box, BoxLayout):
#                 title_label = box.children[1]  # Assuming the Label is always the second widget
#                 title = title_label.text
                
#                 # Handle different input types
#                 input_widget = box.children[0]  # Assuming the input widget is always the first widget
                
#                 if isinstance(input_widget, TextInput):
#                     data_to_save_meetnote[title] = input_widget.text
#                 elif isinstance(input_widget, Spinner):
#                     data_to_save_meetnote[title] = input_widget.text  # Spinner's current selection
#                 elif isinstance(input_widget, BoxLayout):
#                     data_to_save_meetnote[title] = input_widget.children[1].text
#                 print(data_to_save_meetnote)
#         for box in self.right_half.children:
            
#             if isinstance(box, BoxLayout):
#                 title_label = box.children[1]  # Assuming the Label is always the second widget
#                 title = title_label.text
                
#                 # Handle different input types
#                 input_widget = box.children[0]  # Assuming the input widget is always the first widget
                
#                 if isinstance(input_widget, TextInput):
#                     data_to_save_actionnote[title] = input_widget.text
#                 elif isinstance(input_widget, Spinner):
#                     data_to_save_actionnote[title] = input_widget.text  # Spinner's current selection
#                 elif isinstance(input_widget, BoxLayout):
#                     data_to_save_actionnote[title] = input_widget.children[1].text
#                 print(data_to_save_actionnote)

#         # import pdb;pdb.set_trace()
#         # Now `data_to_save` contains all the titles and values
#         # You can print it or save it to a file, database, etc.
#         print(data_to_save_actionnote)
        
#         with open(self.layout_json_dir_meet , 'r', encoding='utf-8') as f:
#             layout_data_meet = json.load(f)
        
#         with open(self.layout_json_dir_action, 'r', encoding='utf-8') as f:
#             layout_data_action = json.load(f)
#         # Clear the current layout
#         layout_data = layout_data_meet+layout_data_action

#         # Separate lists for 'meet' and 'action' items
#         meetnote_data = []
#         actionnote_data = []

#         # Divide the items into the respective lists
#         for item in layout_data:
#             title = item['title']
#             if item["notetype"] == 'meet':
#                 if title in data_to_save_meetnote:
#                     item['content'] = data_to_save_meetnote[title]
#                 meetnote_data.append(item)
#             elif item["notetype"] == 'action':
#                 if title in data_to_save_actionnote:
#                     item['content'] = data_to_save_actionnote[title]
#                 actionnote_data.append(item)

#         # Save 'meet' items to a separate JSON file
       
#         with open(self.layout_json_dir_meet, 'w', encoding='utf-8') as json_file:
#             json.dump(meetnote_data, json_file, indent=4, ensure_ascii=False)
#         # Save 'action' items to a separate JSON file
#         with open(self.layout_json_dir_action, 'w', encoding='utf-8') as json_file:
#             json.dump(actionnote_data, json_file, indent=4, ensure_ascii=False)
        
        
#     def load_layout(self, meetdir,actiondir):
       
            
#             # Read the selected layout file
#             self.layout_json_dir_meet = meetdir
#             with open( meetdir, 'r', encoding='utf-8') as f:
#                 layout_data_meet = json.load(f)
#             self.layout_json_dir_action = actiondir
#             with open( actiondir, 'r', encoding='utf-8') as f:
#                 layout_data_action = json.load(f)
#             # Clear the current layout
#             layout_data = layout_data_meet+layout_data_action
#             self.results_layout.clear_widgets()

#             # Create left and right halves with size_hint
           
#             left_scroll_view = ScrollView(size_hint=(0.5, 1))
#             self.left_half = BoxLayout(orientation='vertical', size_hint_y=None, spacing=10, pos_hint={'top': 1})
#             self.left_half.bind(minimum_height=self.left_half.setter('height'))  # Ensure scrolling works as expected
#             left_scroll_view.add_widget(self.left_half)

#             # Create scrollable right half
#             right_scroll_view = ScrollView(size_hint=(0.5, 1))
#             self.right_half = BoxLayout(orientation='vertical', size_hint_y=None, spacing=10, pos_hint={'top': 1})
#             self.right_half.bind(minimum_height=self.right_half.setter('height'))  # Ensure scrolling works as expected
#             right_scroll_view.add_widget(self.right_half)
#             title_label = Label(text='Meet Note', size_hint=(0.5, None), height=40)
#             self.left_half.add_widget(title_label)
#             title_label = Label(text='Action Note', size_hint=(0.5, None), height=40)
#             self.right_half.add_widget(title_label)
#             for item in layout_data:
#                 # Determine whether to place in the left or right half
#                 target_layout = self.left_half if item.get('notetype') == 'meet' else self.right_half
                
#                 if item['type'] == 'input' and item['title'].lower() == 'symptoms':
#                     new_box = BoxLayout(orientation='horizontal', size_hint_y=None, height=120)
#                     title_label = Label(text=item['title'], size_hint=(0.5, None), height=120)  # 0.2 will make it take 20% of the available width
#                     new_box.add_widget(title_label)

#                     input_and_button_box = BoxLayout(orientation='horizontal', size_hint=(1, None), height=120)
#                     self.idc_result_input = TextInput(text=item['content'], size_hint_y=None, height=120, multiline=True, font_name=FONT_PATH)
#                     input_and_button_box.add_widget(self.idc_result_input)

#                     search_button = Button(text="ICD\n-\n10", size_hint_x=None, width=30)
#                     search_button.bind(on_press=self.open_search_popup)
#                     input_and_button_box.add_widget(search_button)

#                     new_box.add_widget(input_and_button_box)
#                     target_layout.add_widget(new_box)

#                 elif item['type'] == 'input':
#                     new_box = BoxLayout(orientation='horizontal', size_hint_y=None, height=40)
#                     title_label = Label(text=item['title'], size_hint=(0.5, None), size_hint_y=None, height=40)
#                     new_box.add_widget(title_label)
                    
#                     string_input = TextInput(size_hint=(1, None), height=40, text=item['content'])
#                                         # Check the title to assign to the correct instance variable
#                     if item['title'].lower() == 'task name':
#                         self.task_name_input = string_input
#                     elif item['title'].lower() == 'task number':
#                         self.task_number_input = string_input
#                     new_box.add_widget(string_input)
                    
#                     target_layout.add_widget(new_box)

#                 elif item['type'] == 'spinner':
#                     new_box = BoxLayout(orientation='horizontal', size_hint_y=None, height=40)
#                     title_label = Label(text=item['title'], size_hint=(0.5, None), size_hint_y=None, height=40)
#                     new_box.add_widget(title_label)
                    
#                     spinner = Spinner(
#                         text=item['content'],
#                         values=item['options'],
#                         size_hint=(1, None),
#                         height=40
#                     )
#                         # Check the title to assign to the correct instance variable
#                     if item['title'].lower() == 'task name':
#                         self.task_name_input = spinner
#                     elif item['title'].lower() == 'task number':
#                         self.task_number_input = spinner
#                     new_box.add_widget(spinner)
                    
#                     target_layout.add_widget(new_box)

#             # Add the halves to the results layout
#             self.results_layout.add_widget(left_scroll_view)
#             self.results_layout.add_widget(right_scroll_view)


                    

#     def open_search_popup(self, instance):
#         # 创建弹出窗口
#         popup_layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        
#         self.popup_input = TextInput(size_hint_y=None, height=40, font_name=FONT_PATH)
#         self.popup_input.bind(text=self.on_text)
#         popup_layout.add_widget(self.popup_input)
        
#         # 创建滚动视图来显示搜索结果
#         self.results_view = ScrollView(size_hint=(1, None), size=(popup_layout.width, 300))
#         self.result_layout = GridLayout(cols=1, size_hint_y=None)
#         self.result_layout.bind(minimum_height=self.result_layout.setter('height'))
#         self.results_view.add_widget(self.result_layout)
#         popup_layout.add_widget(self.results_view)
        
#         self.popup = Popup(title="Enter Search Query", content=popup_layout, size_hint=(0.9, 0.9))
#         self.popup.open()
    
#     def on_text(self, instance, value):
#         self.result_layout.clear_widgets()
#         if value.strip():  # 检查输入是否为空
#             search_results = fuzzy_search(value)
#             for code, en_description, cn_description in search_results[:25]:  # 显示前25个结果
#                 result_button = Button(text=f"{code}: {en_description} / {cn_description}", size_hint_y=None, height=40, font_name=FONT_PATH)
#                 result_button.bind(on_press=lambda x, c=code, e=en_description, cn=cn_description: self.select_result(c, e, cn))
#                 self.result_layout.add_widget(result_button)
    
#     def select_result(self, code, en_description, cn_description):
#         print(f"Selected: {code}: {en_description} / {cn_description}")
        
#         # 在右侧布局的输入框中显示选择的IDC结果，结果用逗号分隔
#         current_text = self.idc_result_input.text
#         new_text = f"{code}: {en_description} / {cn_description}"
#         if current_text:
#             self.idc_result_input.text = f"{current_text}, {new_text}"
#         else:
#             self.idc_result_input.text = new_text
        
#         self.popup.dismiss()

#     def print_task_info(self, instance):
        
#         if self.task_name_input and self.task_number_input:
#             task_name = self.task_name_input.text
#             task_number = self.task_number_input.text
#             self.manager.parent_app.task_name = f"{task_name} {task_number}"
#             self.manager.parent_app.task_button.text =f"{task_name} {task_number}"
#             print(f"Task: {self.manager.parent_app.task_name}")
#             self.save_data(instance)
#             self.manager.current = 'main'  # Switch back to the main screen
#         else:
#             print("Task Name or Task Number input not found")
class TaskInputScreen(Screen):
    def __init__(self, layout_dir=None, **kwargs):
        super(TaskInputScreen, self).__init__(**kwargs)
        self.layout = BoxLayout(orientation='vertical')
        self.current_directory = os.getcwd()
        self.config_path = os.path.join(self.current_directory, "config")
        
        # Add "Choose Layout" button
        self.choose_layout_button = Button(text="Choose Layout", size_hint_y=None, height=40)
        self.choose_layout_button.bind(on_press=self.open_file_chooser)
        self.layout.add_widget(self.choose_layout_button)

        # Create the ScrollView to take up 80% of the layout

        self.results_layout = BoxLayout(orientation='horizontal', size_hint=(1, 0.8))
        
        self.layout.add_widget(self.results_layout)

        # Create a button at the bottom right
        bottom_right_button = Button(text="Print Task", size_hint=(0.3, 0.1), pos_hint={'right': 1, 'bottom': 1})
        bottom_right_button.bind(on_press=self.print_task_info)
        self.layout.add_widget(bottom_right_button)

        # Initialize instance variables for the task input boxes and spinner
        self.task_name_input = None
        self.task_number_input = None
        self.task_spinner = None
        self.layout_json_dir = ''
        self.add_widget(self.layout)
        self.layouts = None 
        self.load_layout(os.path.join(self.config_path,'layout.json'))
        # Function to fetch data from the API
    def fetch_layouts(self):
        if self.layouts is None:  # Only fetch if layouts have not been fetched yet
            host = "https://motion-service.yuyi-ocean.com"
            url = f"{host}/api/layouts"
            response = requests.get(url)
            if response.status_code == 200:
                self.layouts = response.json()['resources']  # Store the fetched layouts
        return self.layouts

    # Function to handle the selection of buttons
    def on_layout_selected(self, instance, column, layout_data):
        # Deselect all buttons in the same column and select only the current one
        if column == 'meet':
            for btn in self.meet_buttons:
                btn.background_color = (1, 1, 1, 1)  # Deselect (white background)
            self.selected_meet_data = layout_data  # Store full data for meet
        elif column == 'action':
            for btn in self.action_buttons:
                btn.background_color = (1, 1, 1, 1)  # Deselect (white background)
            self.selected_action_data = layout_data  # Store full data for action

        # Highlight the selected button
        instance.background_color = (0, 1, 0, 1)  # Selected (green background)

    # Function to create and show popup
    def show_layouts_popup(self, instance):
        layouts = self.fetch_layouts()  # Fetch layout data from the API or use cached data

        # Create separate GridLayouts for "meet" and "action"
        meet_layout = GridLayout(cols=1, padding=10, spacing=10, size_hint_y=None)
        action_layout = GridLayout(cols=1, padding=10, spacing=10, size_hint_y=None)

        # Bind layout heights to allow dynamic resizing
        meet_layout.bind(minimum_height=meet_layout.setter('height'))
        action_layout.bind(minimum_height=action_layout.setter('height'))

        # Add layout IDs as buttons to their respective layouts
        meet_layouts = [res for res in layouts if res.get('catalog') == 'meet']
        action_layouts = [res for res in layouts if res.get('catalog') == 'action']

        self.meet_buttons = []
        self.action_buttons = []

        for layout_data in meet_layouts:
            btn_meet = Button(text=layout_data['layoutId'], size_hint_y=None, height=40)
            btn_meet.bind(on_press=lambda x, ld=layout_data: self.on_layout_selected(x, 'meet', ld))
            self.meet_buttons.append(btn_meet)
            meet_layout.add_widget(btn_meet)

        for layout_data in action_layouts:
            btn_action = Button(text=layout_data['layoutId'], size_hint_y=None, height=40)
            btn_action.bind(on_press=lambda x, ld=layout_data: self.on_layout_selected(x, 'action', ld))
            self.action_buttons.append(btn_action)
            action_layout.add_widget(btn_action)
        
        # Create scrollable views for both layouts
        meet_scroll = ScrollView(size_hint=(0.5, None), size=(200, 300))
        meet_scroll.add_widget(meet_layout)

        action_scroll = ScrollView(size_hint=(0.5, None), size=(200, 300))
        action_scroll.add_widget(action_layout)
        
        # Create the main layout for the popup
        scroll_layout = BoxLayout(orientation='horizontal')

        # Add headers for the columns
        header_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=40)
        header_layout.add_widget(Label(text='Catalog: meet', size_hint=(0.5, 1)))
        header_layout.add_widget(Label(text='Catalog: action', size_hint=(0.5, 1)))
        
        # Add the scroll views to the main layout
        scroll_layout.add_widget(meet_scroll)
        scroll_layout.add_widget(action_scroll)

        # Create a Select button, always enabled
        self.btn_select_final = Button(text="Select", size_hint=(0.2, 0.1))
        self.btn_select_final.bind(on_release=self.final_selection_made)

        # Add the Select button at the bottom of the popup

        # Create the popup layout
        popup_layout = BoxLayout(orientation='vertical')
        popup_layout.add_widget(header_layout)
        popup_layout.add_widget(scroll_layout)  # Add the scrollable columns
        popup_layout.add_widget(self.btn_select_final)

        # Create the actual popup
        self.popup = Popup(title='Layout IDs',
                           content=popup_layout,
                           size_hint=(0.8, 0.8))

        # Open the popup
        self.popup.open()
    
    def final_selection_made(self, instance):
        
        def transform_fields(fields, catalog):
            transformed_fields = []

            for field in fields:
                # Create the basic transformed structure
                transformed_field = {
                    "type": field["elementType"],  # Map elementType to type
                    "title": field["title"],  # Title remains the same
                    "content": "",  # Default content is an empty string
                    "notetype": catalog  # Map catalog to notetype
                }

                # If the field has options, add the options and set default content
                if "options" in field and field["options"] is not None:
                    transformed_field["options"] = field["options"]
                    transformed_field["content"] = "Choose an option"
                
                # Append transformed field to the list
                transformed_fields.append(transformed_field)

            return transformed_fields

        # Transform and combine fields from both meet and action
        combined_data = {}  # Will store layoutId and fields together
        combined_fields = []

        # Transform and combine meet fields
        if self.selected_meet_data:
            transformed_meet_fields = transform_fields(self.selected_meet_data.get('fields', []), 'meet')
            print("Transformed Meet Fields:")
            print(transformed_meet_fields)
            combined_fields += transformed_meet_fields  # Add meet fields to combined list
            combined_data["meet_layoutId"] = self.selected_meet_data['layoutId']  # Save meet layoutId
        else:
            print("No Meet Layout selected.")

        # Transform and combine action fields
        if self.selected_action_data:
            transformed_action_fields = transform_fields(self.selected_action_data.get('fields', []), 'action')
            print("Transformed Action Fields:")
            print(transformed_action_fields)
            combined_fields += transformed_action_fields  # Add action fields to combined list
            combined_data["action_layoutId"] = self.selected_action_data['layoutId']  # Save action layoutId
        else:
            print("No Action Layout selected.")

        # Add combined fields to the final data
        combined_data["fields"] = combined_fields

        # Print the combined data
        print("Combined Data with IDs and Fields:")
        print(combined_data)
        # Save combined_data to a JSON file
        with open(os.path.join(self.config_path,'layout.json'), "w") as json_file:
            json.dump(combined_data, json_file, indent=4)  # Save with indentation for readability
        # Add logic to close the popup after printing
        if hasattr(self, 'popup'):
            self.popup.dismiss()  # Close the popup if it's open

        self.load_layout(os.path.join(self.config_path,'layout.json'))
    def open_file_chooser(self, instance):
        self.show_layouts_popup(self)
        
        # You can add additional logic here for what to do after selection
        ###############################
        # # 创建文件选择弹出窗口
        # file_chooser = FileChooserIconView(path='.', filters=['*.json'])
        # popup_layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        # popup_layout.add_widget(file_chooser)
        
        # # 确认按钮
        # confirm_button = Button(text="Load Layout", size_hint_y=None, height=40)
        # confirm_button.bind(on_press=lambda x: self.load_layout(file_chooser.selection))
        # popup_layout.add_widget(confirm_button)
        
        # self.popup = Popup(title="Choose Layout File", content=popup_layout, size_hint=(0.9, 0.9))
        # self.popup.open()
    def save_data(self, instance):
        data_to_save_meetnote = {}
        data_to_save_actionnote = {}
        
        # Iterate over all widgets in the results layout
        for box in self.left_half.children:
            
            if isinstance(box, BoxLayout):
                title_label = box.children[1]  # Assuming the Label is always the second widget
                title = title_label.text
                
                # Handle different input types
                input_widget = box.children[0]  # Assuming the input widget is always the first widget
                
                if isinstance(input_widget, TextInput):
                    data_to_save_meetnote[title] = input_widget.text
                elif isinstance(input_widget, Spinner):
                    data_to_save_meetnote[title] = input_widget.text  # Spinner's current selection
                elif isinstance(input_widget, BoxLayout):
                    data_to_save_meetnote[title] = input_widget.children[1].text
                print(data_to_save_meetnote)
        for box in self.right_half.children:
            
            if isinstance(box, BoxLayout):
                title_label = box.children[1]  # Assuming the Label is always the second widget
                title = title_label.text
                
                # Handle different input types
                input_widget = box.children[0]  # Assuming the input widget is always the first widget
                
                if isinstance(input_widget, TextInput):
                    data_to_save_actionnote[title] = input_widget.text
                elif isinstance(input_widget, Spinner):
                    data_to_save_actionnote[title] = input_widget.text  # Spinner's current selection
                elif isinstance(input_widget, BoxLayout):
                    data_to_save_actionnote[title] = input_widget.children[1].text
                print(data_to_save_actionnote)

        # import pdb;pdb.set_trace()
        # Now `data_to_save` contains all the titles and values
        # You can print it or save it to a file, database, etc.
        print(data_to_save_actionnote)

        with open(self.layout_json_dir, 'r', encoding='utf-8') as f:
            layout_data = json.load(f)
        layout_data = layout_data['fields']
        # Separate lists for 'meet' and 'action' items
        meetnote_data = []
        actionnote_data = []

        # Divide the items into the respective lists
        for item in layout_data:
            title = item['title']
            if item["notetype"] == 'meet':
                if title in data_to_save_meetnote:
                    item['content'] = data_to_save_meetnote[title]
                meetnote_data.append(item)
            elif item["notetype"] == 'action':
                if title in data_to_save_actionnote:
                    item['content'] = data_to_save_actionnote[title]
                actionnote_data.append(item)

        # Save 'meet' items to a separate JSON file
        meetnote_file_path = os.path.join(self.config_path, 'meetnote_layout.json')
        with open(meetnote_file_path, 'w', encoding='utf-8') as json_file:
            json.dump(meetnote_data, json_file, indent=4, ensure_ascii=False)

        # Save 'action' items to a separate JSON file
        actionnote_file_path = os.path.join(self.config_path, 'actionnote_layout.json')
        with open(actionnote_file_path, 'w', encoding='utf-8') as json_file:
            json.dump(actionnote_data, json_file, indent=4, ensure_ascii=False)
        with open(self.layout_json_dir,'r',encoding='utf-8')as file:
            temp = json.load(file)
            temp['fields'] =meetnote_data+actionnote_data
        with open(self.layout_json_dir,'w',encoding='utf-8')as json_file:
            json.dump(temp, json_file, indent=4, ensure_ascii=False)
        
    def load_layout(self, layout_file):
            self.layout_json_dir = layout_file
            with open(layout_file, 'r', encoding='utf-8') as f:
                layout_data = json.load(f)
            #import pdb;pdb.set_trace()
            # Clear the current layout
            layout_data = layout_data['fields']
            self.results_layout.clear_widgets()

            # Create left and right halves with size_hint
           
            left_scroll_view = ScrollView(size_hint=(0.5, 1))
            self.left_half = BoxLayout(orientation='vertical', size_hint_y=None, spacing=10, pos_hint={'top': 1})
            self.left_half.bind(minimum_height=self.left_half.setter('height'))  # Ensure scrolling works as expected
            left_scroll_view.add_widget(self.left_half)

            # Create scrollable right half
            right_scroll_view = ScrollView(size_hint=(0.5, 1))
            self.right_half = BoxLayout(orientation='vertical', size_hint_y=None, spacing=10, pos_hint={'top': 1})
            self.right_half.bind(minimum_height=self.right_half.setter('height'))  # Ensure scrolling works as expected
            right_scroll_view.add_widget(self.right_half)
            title_label = Label(text='Meet Note', size_hint=(0.5, None), height=40)
            self.left_half.add_widget(title_label)
            title_label = Label(text='Action Note', size_hint=(0.5, None), height=40)
            self.right_half.add_widget(title_label)
            for item in layout_data:
                # Determine whether to place in the left or right half
                target_layout = self.left_half if item.get('notetype') == 'meet' else self.right_half
                
                if item['type'] == 'input' and item['title'].lower() == 'symptoms':
                    new_box = BoxLayout(orientation='horizontal', size_hint_y=None, height=120)
                    title_label = Label(text=item['title'], size_hint=(0.5, None), height=120)  # 0.2 will make it take 20% of the available width
                    new_box.add_widget(title_label)

                    input_and_button_box = BoxLayout(orientation='horizontal', size_hint=(1, None), height=120)
                    self.idc_result_input = TextInput(text=item['content'], size_hint_y=None, height=120, multiline=True, font_name=FONT_PATH)
                    input_and_button_box.add_widget(self.idc_result_input)

                    search_button = Button(text="ICD\n-\n10", size_hint_x=None, width=30)
                    search_button.bind(on_press=self.open_search_popup)
                    input_and_button_box.add_widget(search_button)

                    new_box.add_widget(input_and_button_box)
                    target_layout.add_widget(new_box)

                elif item['type'] == 'input':
                    new_box = BoxLayout(orientation='horizontal', size_hint_y=None, height=40)
                    title_label = Label(text=item['title'], size_hint=(0.5, None), size_hint_y=None, height=40)
                    new_box.add_widget(title_label)
                    
                    string_input = TextInput(size_hint=(1, None), height=40, text=item['content'])
                                        # Check the title to assign to the correct instance variable
                    if item['title'].lower() == 'task type' and item.get('notetype') == 'action':
                        self.task_name_input = string_input
                    elif item['title'].lower() == 'task number' and item.get('notetype') == 'action':
                        self.task_number_input = string_input
                    new_box.add_widget(string_input)
                    
                    target_layout.add_widget(new_box)

                elif item['type'] == 'spinner':
                    new_box = BoxLayout(orientation='horizontal', size_hint_y=None, height=40)
                    title_label = Label(text=item['title'], size_hint=(0.5, None), size_hint_y=None, height=40)
                    new_box.add_widget(title_label)
                    
                    spinner = Spinner(
                        text=item['content'],
                        values=item['options'],
                        size_hint=(1, None),
                        height=40
                    )
                        # Check the title to assign to the correct instance variable
                    if item['title'].lower() == 'task type' and item.get('notetype') == 'action':
                        self.task_name_input = spinner
                    elif item['title'].lower() == 'task number' and item.get('notetype') == 'action':
                        self.task_number_input = spinner
                    new_box.add_widget(spinner)
                    
                    target_layout.add_widget(new_box)

            # Add the halves to the results layout
            self.results_layout.add_widget(left_scroll_view)
            self.results_layout.add_widget(right_scroll_view)


                    

    def open_search_popup(self, instance):
        # 创建弹出窗口
        popup_layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        
        self.popup_input = TextInput(size_hint_y=None, height=40, font_name=FONT_PATH)
        self.popup_input.bind(text=self.on_text)
        popup_layout.add_widget(self.popup_input)
        
        # 创建滚动视图来显示搜索结果
        self.results_view = ScrollView(size_hint=(1, None), size=(popup_layout.width, 300))
        self.result_layout = GridLayout(cols=1, size_hint_y=None)
        self.result_layout.bind(minimum_height=self.result_layout.setter('height'))
        self.results_view.add_widget(self.result_layout)
        popup_layout.add_widget(self.results_view)
        
        self.popup = Popup(title="Enter Search Query", content=popup_layout, size_hint=(0.9, 0.9))
        self.popup.open()
    
    def on_text(self, instance, value):
        self.result_layout.clear_widgets()
        if value.strip():  # 检查输入是否为空
            search_results = fuzzy_search(value)
            for code, en_description, cn_description in search_results[:25]:  # 显示前25个结果
                result_button = Button(text=f"{code}: {en_description} / {cn_description}", size_hint_y=None, height=40, font_name=FONT_PATH)
                result_button.bind(on_press=lambda x, c=code, e=en_description, cn=cn_description: self.select_result(c, e, cn))
                self.result_layout.add_widget(result_button)
    
    def select_result(self, code, en_description, cn_description):
        print(f"Selected: {code}: {en_description} / {cn_description}")
        
        # 在右侧布局的输入框中显示选择的IDC结果，结果用逗号分隔
        current_text = self.idc_result_input.text
        new_text = f"{code}: {en_description} / {cn_description}"
        if current_text:
            self.idc_result_input.text = f"{current_text}, {new_text}"
        else:
            self.idc_result_input.text = new_text
        
        self.popup.dismiss()

    def print_task_info(self, instance):
        
        if self.task_name_input and self.task_number_input:
            task_name = self.task_name_input.text
            task_number = self.task_number_input.text
            self.manager.parent_app.task_name = f"{task_name}_{task_number}"
            self.manager.parent_app.task_button.text =f"{task_name}_{task_number}"
            print(f"Task: {self.manager.parent_app.task_name}")
            
            self.save_data(instance)
            
            
            self.manager.current = 'main'  # Switch back to the main screen
        else:
            print("Task Name or Task Number input not found")
class ResultsPopup(Popup):
    
    def __init__(self, parent_app, **kwargs):
        super(ResultsPopup, self).__init__(**kwargs)
        self.parent_app = parent_app  # Save the parent app reference
        self.layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        self.current_directory = os.getcwd()
        

        input_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=30)
        self.spinner = Spinner(
            text='Name',
            values=('Name', 'Phone'),
            size_hint_x=None,
            width=100,
            font_name=FONT_PATH
        )
        input_layout.add_widget(self.spinner)

        self.search_input = TextInput(hint_text="Enter name or phone", multiline=False, font_name=FONT_PATH)
        self.search_input.bind(text=self.on_text)
        input_layout.add_widget(self.search_input)

        self.layout.add_widget(input_layout)

        self.scroll_view = ScrollView(size_hint=(1, 0.8))
        self.results_layout = GridLayout(cols=1, size_hint_y=None)
        self.results_layout.bind(minimum_height=self.results_layout.setter('height'))
        self.scroll_view.add_widget(self.results_layout)
        self.layout.add_widget(self.scroll_view)

        close_button = Button(text="Close", size_hint=(1, 0.1), font_name=FONT_PATH)
        close_button.bind(on_press=self.dismiss)
        self.layout.add_widget(close_button)

        self.add_widget(self.layout)  # Add the layout to the popup

    def on_text(self, instance, value):
        self.call_api(value)

    def call_api(self, query):
        if query.strip() == "":
            return

        host = "https://motion-service.yuyi-ocean.com"
        url = f"{host}/api/patients"
        search_type = 'name' if self.spinner.text == 'Name' else 'phone'
        params = {search_type: query}

        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                results = response.json()
                self.update_results(results['resources'])  # Update the results in the popup
            else:
                self.results_layout.clear_widgets()
                result_label = Label(text=f"Error: {response.status_code}", size_hint_y=None, height=40, font_name=FONT_PATH)
                self.results_layout.add_widget(result_label)
        except requests.exceptions.RequestException as e:
            self.results_layout.clear_widgets()
            result_label = Label(text=f"Request failed: {e}", size_hint_y=None, height=40, font_name=FONT_PATH)
            self.results_layout.add_widget(result_label)

    def update_results(self, results):
        self.results_layout.clear_widgets()
        
        for res in results:
            result_button = Button(text=f"Name: {res['name']}, Phone: {res['phone']}", size_hint_y=None, height=40, font_name=FONT_PATH)
            result_button.id = res['id']  # Store the id in the button's id property
            result_button.name = res['name']
            result_button.phone = res['phone']
            result_button.bind(on_press=self.on_result_button_press)
            self.results_layout.add_widget(result_button)

    def on_result_button_press(self, instance):
        print(f"Selected ID: {instance.id}")
        self.parent_app.patient_genID = instance.id  # Store the selected ID in the parent app's variable
        self.parent_app.patient_namephone = f"{instance.name} \n {instance.phone}"  # Store the selected name and phone in the parent app's variable
        self.dismiss()  # Close the popup after selection
        date = datetime.now().strftime("%Y_%m_%d")
        self.parent_app.update_tasklist(date)

class NumberedInputBox(BoxLayout):
    def __init__(self, text, parent_layout, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'horizontal'
        self.number_label = Label(text=text, size_hint_x=0.2)
        self.input_box = TextInput(multiline=False)
        self.delete_button = Button(text='Delete', size_hint_x=0.2)
        self.delete_button.bind(on_press=self.delete_self)

        self.add_widget(self.number_label)
        self.add_widget(self.input_box)
        self.add_widget(self.delete_button)

        self.parent_layout = parent_layout

    def delete_self(self, instance):
        self.parent_layout.remove_widget(self)


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
        self.pos_ref_x =  [0.2,0.4,0.6,0.8,1.0]
        yindex = 0.08
        xindex = 0.02
        self.pos_ref_y =[1,1-yindex*1,1-yindex*2,1-yindex*3,1-yindex*4,yindex*5,1-yindex*6,0.37,0.20,0.07]
        
        self.pos_ref_y = [x - 0.1 for x in self.pos_ref_y]
        self.pos_ref_x = [x - 0.1 for x in self.pos_ref_x]
        super(NewPageScreen, self).__init__(**kwargs)
        layout = FloatLayout()
        btn1 = Button(text='Skeleton Demonstration', size_hint=(0.2, 0.1), pos_hint={'x': 0.1, 'y': 0.6})
        btn2 = Button(text='Live Camera Demonstration', size_hint=(0.2, 0.1), pos_hint={'x': 0.4, 'y': 0.6})
        btn3 = Button(text='Button 3', size_hint=(0.2, 0.1), pos_hint={'x': 0.7, 'y': 0.6})
        btn_back = Button(text='Main Page', size_hint=(0.2, 0.1), pos_hint={'center_x': 0.5, 'center_y': 0.1})
        btn_exit = Button(text='結束程式', size_hint=(0.15, 0.1), size=(btn_wide, 50), pos_hint={'center_x': self.pos_ref_x[4], 'center_y':self.pos_ref_y[8]}, on_release=self.button_exit, font_name=self.font_path)
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
        self.sm.parent_app = self
        main_screen = Screen(name='main')
        new_page_screen = NewPageScreen(name='new_page')
        task_input_screen = TaskInputScreen(name='task_input')
        taskEDIT_input_screen = TaskEDITInputScreen(name='taskEDIT_input')
        main_layout = self.setup_main_layout()  # Setup your main self.layout here
        main_screen.add_widget(main_layout)

        self.sm.add_widget(main_screen)
        self.sm.add_widget(new_page_screen)
        self.sm.add_widget(task_input_screen) 
        self.sm.add_widget(taskEDIT_input_screen) 
        return self.sm


    def setup_main_layout(self):#5,9
        self.pos_ref_x =  [0.2,0.4,0.6,0.8,1.0]
        yindex = 0.08
        xindex = 0.02
        self.pos_ref_y =[1,1-yindex*1,1-yindex*2,1-yindex*3,1-yindex*4,yindex*5,1-yindex*6,0.37,0.20,0.07]
        
        self.pos_ref_y = [x - 0.1 for x in self.pos_ref_y]
        self.pos_ref_x = [x - 0.1 for x in self.pos_ref_x]
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
        self.task_name = ''

        self.mode_select = 'Recording'
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
        self.layout = FloatLayout()

        # 創建按鈕，並指定位置
        # 檢察系統檔案
        btn_wide = 120
        btn_high = 50
        btn_calibration_folder = Button(text='1-1建立新參數', size_hint=(0.19,0.1), size=(btn_wide, 50), pos_hint={'center_x': self.pos_ref_x[0], 'center_y':self.pos_ref_y[0]}, on_release=self.create_calibration_ask, font_name=self.font_path)
        self.layout.add_widget(btn_calibration_folder)
        btn_config = Button(text='1-2偵測相機', size_hint=(0.19,0.1), size=(btn_wide, 50), pos_hint={'center_x': self.pos_ref_x[1], 'center_y':self.pos_ref_y[0]}, on_release=self.button_config, font_name=self.font_path)
        self.layout.add_widget(btn_config)
        btn_check_cam = Button(text='1-3檢查相機', size_hint=(0.19,0.1), size=(btn_wide, 50), pos_hint={'center_x': self.pos_ref_x[2], 'center_y':self.pos_ref_y[0]},on_release=self.button_check_cam, font_name=self.font_path)
        self.layout.add_widget(btn_check_cam)
        self.meetId = ''
        # 相機校正
        # btn_intrinsic_record = Button(text='2-1拍攝內參', size_hint=(0.19,0.1), size=(btn_wide, 50), pos=(20, 500), on_press=self.button_intrinsic_record, font_name=self.font_path)
        # self.layout.add_widget(btn_intrinsic_record)
        # btn_intrinsic_calculate = Button(text='2-2計算內參', size_hint=(0.19,0.1), size=(btn_wide, 50), pos=(130, 500), on_press=self.button_intrinsic_calculate, font_name=self.font_path)
        # self.layout.add_widget(btn_intrinsic_calculate)
        # btn_intrinsic_check = Button(text='2-3檢查內參', size_hint=(0.19,0.1), size=(btn_wide, 50), pos=(20, 400), on_press=self.button_intrinsic_check, font_name=self.font_path)
        # self.layout.add_widget(btn_intrinsic_check)

        btn_extrinsic_record = Button(text='2-1拍攝外參', size_hint=(0.19,0.1), size=(btn_wide, 50),  pos_hint={'center_x': self.pos_ref_x[0], 'center_y':self.pos_ref_y[2]}, on_release=self.button_extrinsic_record, font_name=self.font_path)
        self.layout.add_widget(btn_extrinsic_record)
        btn_extrinsic_calculate = Button(text='2-2計算外參', size_hint=(0.19,0.1), size=(btn_wide, 50),  pos_hint={'center_x': self.pos_ref_x[1], 'center_y':self.pos_ref_y[2]}, on_release=self.button_extrinsic_calculate, font_name=self.font_path)
        self.layout.add_widget(btn_extrinsic_calculate)
        btn_extrinsic_manual_calculate = Button(text='2-2a人工外參', size_hint=(0.19,0.1), size=(btn_wide, 50),  pos_hint={'center_x': self.pos_ref_x[2], 'center_y':self.pos_ref_y[2]}, on_release=self.button_extrinsic_manual_calculate, font_name=self.font_path)
        self.layout.add_widget(btn_extrinsic_manual_calculate)
        # btn_extrinsic_check = Button(text='2-3檢查外參', size_hint=(0.19,0.1), size=(btn_wide, 50),  pos_hint={'center_x': self.pos_ref_x[2], 'center_y':self.pos_ref_y[2]}, on_release=self.button_extrinsic_check, font_name=self.font_path)
        # self.layout.add_widget(btn_extrinsic_check)
        # btn_extrinsic_check = Button(text='3-3檢查外參', size_hint=(0.19,0.1), size=(btn_wide, 50), pos=(240, 400), on_press=self.button_extrinsic_check, font_name=self.font_path)
        # self.layout.add_widget(btn_extrinsic_check)

        # 拍攝人體動作
        btn_Apose_record = Button(text='3-1拍攝A-pose', size_hint=(0.19,0.1), size=(btn_wide, 50), pos_hint={'center_x': self.pos_ref_x[0], 'center_y':self.pos_ref_y[4]}, on_release=self.button_Apose_record, font_name=self.font_path)
        self.layout.add_widget(btn_Apose_record)
        btn_task_record = Button(text='3-2拍攝動作', size_hint=(0.19,0.1), size=(btn_wide, 50), pos_hint={'center_x': self.pos_ref_x[1], 'center_y':self.pos_ref_y[4]}, on_release=lambda instance: self.button_task_record(instance), font_name=self.font_path)
        self.layout.add_widget(btn_task_record)

        # 計算Marker
        btn_calculate_Marker = Button(text='4計算Marker以及IK', size_hint=(0.4,0.1), size=(btn_wide + 60, 50), pos_hint={'center_x': 0.20, 'center_y':self.pos_ref_y[6]}, on_release=self.button_calculate_Marker, font_name=self.font_path)
        self.layout.add_widget(btn_calculate_Marker)

        # 計算IK
        # btn_calculate_IK = Button(text='5-1計算IK', size_hint=(0.19,0.1), size=(btn_wide, 50), pos=(240, 200), on_press=self.button_calculate_IK, font_name=self.font_path)
        # self.layout.add_widget(btn_calculate_IK)

        # 離開NTK_Cap
        btn_exit = Button(text='結束程式', size_hint=(0.15, 0.1), size=(btn_wide, 50), pos_hint={'center_x': self.pos_ref_x[4], 'center_y':self.pos_ref_y[8]}, on_release=self.button_exit, font_name=self.font_path)
        self.layout.add_widget(btn_exit)

        # 創建當前操作顯示
        self.label_log_hint = Label(text='目前執行操作', size_hint=(0.19,0.1), size=(400, 30),  pos_hint={'center_x': 0.20, 'center_y':self.pos_ref_y[7]}, font_name=self.font_path)
        self.layout.add_widget(self.label_log_hint)

        # 執行日期
        self.label_date = Label(text='', size_hint=(0.19,0.1), size=(400, 30),  pos_hint={'center_x': self.pos_ref_x[1], 'center_y':0.97}, font_name=self.font_path)
        self.layout.add_widget(self.label_date)
        Clock.schedule_interval(self.update_date, 1)
        
        # Patient ID
        self.patientID = "test"
        # Replace this line
        # self.txt_patientID_real = TextInput(hint_text='Patient ID', multiline=False, size_hint=(0.19,0.1), size=(150, 50), pos_hint={'center_x': self.pos_ref_x[4], 'center_y':self.pos_ref_y[0]}, font_name=self.font_path)
        # With this block
        self.patient_genID = ''  # Add this line to define the variable
        self.patient_namephone = '' # Add this line to define the variable
        self.task_name =''
        self.btn_patientID = Button(text='Enter Patient ID', size_hint=(0.19, 0.1), size=(150, 50), pos_hint={'center_x': self.pos_ref_x[4], 'center_y': self.pos_ref_y[0]}, font_name=self.font_path)
        # Assuming `self.layout` is an already defined variable
        self.btn_patientID.bind(on_press=lambda instance: self.show_input_popup( instance))

        self.layout.add_widget(self.btn_patientID)
        self.patient_id_event =Clock.schedule_interval(self.patient_ID_update_cloud, 0.1)

        self.label_PatientID_real = Label(text=self.patient_namephone, size_hint=(0.19,0.1), size=(400, 30), pos_hint={'center_x': self.pos_ref_x[3], 'center_y': self.pos_ref_y[0]}, font_name=self.font_path)
        self.layout.add_widget(self.label_PatientID_real)
        
        # 內參選擇相機
        # self.select_camID = 0
        # self.txt_cam_ID = TextInput(hint_text='choose cam ID(0~3)', multiline=False, size_hint=(0.19,0.1), size=(160, 40), pos=(20, 450), font_size=16, font_name=self.font_path)
        # self.layout.add_widget(self.txt_cam_ID)
        #Clock.schedule_interval(self.camID_update, 0.1)

        # Task Name
        self.task = "test"
        self.task_button = Button(text='Enter Task Name', size_hint=(0.19, 0.1), size=(150, 50), pos_hint={'center_x': self.pos_ref_x[4], 'center_y': self.pos_ref_y[2]}, font_name=self.font_path)
        self.task_button.bind(on_press=lambda instance: setattr(self.sm, 'current', 'task_input'))
        #self.task_button.bind(on_press=lambda instance: self.set_taskinput_screen_with_param('task_input', layout_dir=r'C:\Users\Hermes\Desktop\NTKCAP\config'))

        self.layout.add_widget(self.task_button)        
        self.patient_task_event =Clock.schedule_interval(self.task_update_cloud, 0.1)
        self.label_task_real = Label(text=self.task_name , size_hint=(0.19,0.1), size=(400, 30), pos=(500, 470), font_name=self.font_path)
        self.layout.add_widget(self.label_task_real)

        self.label_log = Label(text=' ', size_hint=(0.19,0.1), size=(400, 50), pos_hint={'center_x': 0.20, 'center_y':self.pos_ref_y[7]-0.1}, font_name=self.font_path)
        self.layout.add_widget(self.label_log)

        ##### Button to next page
        btn_to_new_page = Button(text="Advanced Function", size_hint=(0.15, 0.1), size=(btn_wide, 50), pos_hint={'center_x': self.pos_ref_x[4], 'center_y':self.pos_ref_y[7]})
        btn_to_new_page.bind(on_release=lambda instance: setattr(self.sm, 'current', 'new_page'))
        self.layout.add_widget(btn_to_new_page)

        #spinner for camera ID
        # self.txt_camID_spinner = Spinner(text = 'cam ID', values = ("0","1","2","3"),size_hint=(0.19,0.1), size=(100, 30), pos=(20, 450), sync_height = True, font_size=16, font_name=self.font_path)
        # self.layout.add_widget(self.txt_camID_spinner)
        # self.txt_camID_spinner.bind(text = self.camID_update) 
        self.err_calib_extri = Label(text=read_err_calib_extri(self.current_directory), size_hint=(0.19,0.1), size=(400, 30),  pos_hint={'center_x': self.pos_ref_x[3], 'center_y':self.pos_ref_y[2]}, font_name=self.font_path)
        self.layout.add_widget(self.err_calib_extri)

        btn_toggle_language = Button(
            text='Switch Language',
            size_hint=(0.15, 0.1),
            pos_hint={'center_x': self.pos_ref_x[4], 'center_y': self.pos_ref_y[6]},
            on_release=lambda instance: self.toggle_language(btn_calibration_folder,btn_config, btn_check_cam, btn_extrinsic_record,btn_extrinsic_calculate,btn_Apose_record,btn_task_record,btn_calculate_Marker,btn_exit,instance)
        )
        self.layout.add_widget(btn_toggle_language)
        checkbox = CheckBox(size_hint=(None, None), size=(48, 48), pos_hint={'center_x': 0.3, 'center_y': 0.5})
        
        
        # # # Label for the CheckBox
        # checkbox_label = Label(text="Enable feature", size_hint=(None, None), size=(200, 30), pos_hint={'center_x': 0.45, 'center_y': 0.5})
        # checkbox.bind(active=self.on_checkbox_active)
        # #Adding widgets to the self.layout
        # self.layout.add_widget(checkbox)
        # self.layout.add_widget(checkbox_label)
        self.btn_ttl = Button(text='ttl',size_hint=(0.09,0.05),size=(170, 30), pos_hint={'center_x': self.pos_ref_x[2]-0.05, 'center_y':self.pos_ref_y[4]-0.03}, on_release=self.on_checkbox_active, font_name=self.font_path,opacity=0)
        self.layout.add_widget(self.btn_ttl)
        #Spinner for feature selection
        self.feature_spinner = Spinner(
            text='Recording',
            values=('Recording', 'VICON Recording','Delay test'),
            size_hint=(0.19,0.05),
            size=(170, 30),
            pos_hint={'center_x': self.pos_ref_x[2], 'center_y':self.pos_ref_y[4]+0.03},
            font_name=self.font_path
        )
        self.feature_spinner.bind(text=self.on_spinner_select)
        self.layout.add_widget(self.feature_spinner)
        self.gait_anlaysis = Spinner(
            text='No Analysis',
            values=('No Analysis', 'Gait1'),
            size_hint=(0.19,0.05),
            size=(170, 30),
            pos_hint={ 'center_x': self.pos_ref_x[2], 'center_y':self.pos_ref_y[6]+0.03},
            font_name=self.font_path
        )
        
        self.layout.add_widget(self.gait_anlaysis)
        self.COM_input = TextInput(hint_text='COM', multiline=False, size_hint=(0.09,0.05),size=(170, 30),pos_hint={'center_x': self.pos_ref_x[2]+0.05, 'center_y':self.pos_ref_y[4]-0.03}, font_name=self.font_path,opacity=0)
        self.layout.add_widget(self.COM_input)

        date = datetime.now().strftime("%Y_%m_%d")
        self.btn_toggle_cloud_sinlge = Button(
            text='Cloud',
            size_hint=(0.15, 0.1),
            pos_hint={'center_x': self.pos_ref_x[4], 'center_y': self.pos_ref_y[4]},
            on_release=lambda instance: self.switch_cloud(date,instance)  # Pass the button instance
        )
        self.layout.add_widget(self.btn_toggle_cloud_sinlge)
        # Create a ScrollView
        # Create a ScrollView
        self.scroll_view = ScrollView(size_hint=(0.15, 0.4), size=(400, 300), pos_hint={'center_x': self.pos_ref_x[3], 'center_y': self.pos_ref_y[7]},)

        # Create a GridLayout to hold the buttons
        self.button_layout = GridLayout(cols=1, spacing=10, size_hint_y=None)
        self.button_layout.bind(minimum_height=self.button_layout.setter('height'))

        # Add buttons to the GridLayout
        # 

        # Add the GridLayout to the ScrollView
        self.scroll_view.add_widget(self.button_layout)

        # Add the ScrollView to your self.layout
        self.layout.add_widget(self.scroll_view)


        return self.layout
    def set_taskinput_screen_with_param(self, screen_name,btn,meetdir,actiondir,meetId,actionId ,**kwargs):
        # Get the screen instance
        screen = self.sm.get_screen(screen_name)
        screen.meetdir = meetdir
        screen.actiondir = actiondir
        screen.meetId = meetId
        screen.actionId = actionId
        screen.save_load_from_selected_action(meetId,actionId )
        #import pdb;pdb.set_trace()
        screen.btn=btn
        #screen.load_layout(meetdir,actiondir)
        # Pass the additional parameters to the screen
        for key, value in kwargs.items():
            setattr(screen, key, value)
        
        # Switch to the desired screen
        self.sm.current = screen_name

    def update_tasklist(self,date):
        self.layout.remove_widget(self.scroll_view)
        self.layout.remove_widget(self.button_layout)
        self.scroll_view = ScrollView(size_hint=(0.15, 0.4), size=(400, 300), pos_hint={'center_x': self.pos_ref_x[3], 'center_y': self.pos_ref_y[7]},)
        # Create a GridLayout to hold the buttons
        self.button_layout = GridLayout(cols=1, spacing=10, size_hint_y=None)
        self.button_layout.bind(minimum_height=self.button_layout.setter('height'))
        # Add buttons to the GridLayout
        # 
        # Add the GridLayout to the ScrollView
        self.scroll_view.add_widget(self.button_layout)

        # Add the ScrollView to your self.layout
        self.layout.add_widget(self.scroll_view)
        if self.btn_toggle_cloud_sinlge.text == 'Single':
            self.patient_genID = self.txt_patientID_real.text
        dir_list_tasks = os.path.join(self.record_path, 'Patient_data', self.patient_genID, date,'raw_data')
        if os.path.isdir(dir_list_tasks):
            all_folders = [name for name in os.listdir(dir_list_tasks) if os.path.isdir(os.path.join(dir_list_tasks, name))]
        
            # Filter out the "APose" folder
            filtered_folders = [folder for folder in all_folders if (folder.lower() != 'apose') and (folder.lower() != 'calibration')]
            filtered_folders = natsorted(filtered_folders, reverse=True)
            #mport pdb;pdb.set_trace()
            for taskname in range(len(filtered_folders)):  # Replace 20 with however many items you want
                btn = Button(text=filtered_folders[taskname], size_hint_y=None, height=40)
                
                # Bind a function to the button that will handle the selection
            
                actiondir = os.path.join(dir_list_tasks,filtered_folders[taskname],'Action_note.json')
                meetdir =os.path.join(dir_list_tasks,'Meet_note.json')
                with open(os.path.join(dir_list_tasks,filtered_folders[taskname],'actionId.json'),'r') as file:
                    temp= json.load(file)
                    actionId = temp['actionId']
                with open(os.path.join(dir_list_tasks,'meetId.json'),'r') as file:
                    temp= json.load(file)
                    meetId = temp['meetId']
                meetdir=dir_list_tasks
                actiondir =os.path.join(dir_list_tasks,filtered_folders[taskname])
                if os.path.exists(meetdir) and os.path.exists(actiondir):
                    
                    btn.bind(on_press=lambda instance, btn=btn,meetdir = meetdir,actiondir =actiondir,meetId=meetId,actionId=actionId: self.set_taskinput_screen_with_param('taskEDIT_input', btn,meetdir, actiondir,meetId,actionId))
                else:
                    name_dir = os.path.join(dir_list_tasks)
                    name = filtered_folders[taskname]
                    btn.bind(on_press=lambda instance, name_dir=name_dir, name=name, btn=btn: self.show_popup_rename_task(name_dir, name, btn))

                #btn.bind(on_release=self.on_button_select_tasklist)
                
                self.button_layout.add_widget(btn)
    def show_popup_rename_task(self, name_dir, name,btn):
        # Create a BoxLayout for the popup's content
        layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        #import pdb;pdb.set_trace()
        # Add a TextInput to the layout with larger font size and single-line input
        text_input = TextInput(text=btn.text, font_size=24, multiline=False)
        layout.add_widget(text_input)
    
        def rename_directory(old_name, new_name,btn):
            try:
                os.rename(old_name, new_name)
                print(f"Directory renamed from {old_name} to {new_name}")
                btn.text = text_input.text
                popup.dismiss()
            except FileNotFoundError:
                self.show_warning_popup(f"The directory was not found.")
            except FileExistsError:
                self.show_warning_popup(f"The directory already exists.")
            except Exception as e:
                self.show_warning_popup(f"An error occurred: {e}")

        # Function to be called when the Confirm button is pressed
        def on_confirm(instance):
            print("You entered:", text_input.text)
            
            rename_directory(os.path.join(name_dir, btn.text), os.path.join(name_dir, text_input.text),btn)
            
            

        # Add a Confirm button to the layout
        confirm_button = Button(text="Confirm", size_hint=(1, 0.2))
        confirm_button.bind(on_press=on_confirm)
        layout.add_widget(confirm_button)

        # Create the main popup for renaming
        popup = Popup(title="Rename Task",
                    content=layout,
                    size_hint=(0.6, 0.4),
                    auto_dismiss=False)

        # Open the main popup
        popup.open()

    def show_warning_popup(self,message):
        # Function to display a warning popup
        warning_layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        warning_label = Label(text=message, font_size=18)
        close_button = Button(text="Close", size_hint=(1, 0.2))

        # Function to close the warning popup
        def on_close(instance):
            warning_popup.dismiss()

        close_button.bind(on_press=on_close)
        warning_layout.add_widget(warning_label)
        warning_layout.add_widget(close_button)

        warning_popup = Popup(title="Warning",
                            content=warning_layout,
                            size_hint=(0.6, 0.3),
                            auto_dismiss=False)

        warning_popup.open()


    def switch_cloud(self,date,instance):
        if instance.text == 'Cloud':
            instance.text = 'Single'  # Update the button text
            self.layout.remove_widget(self.task_button)
            self.layout.remove_widget(self.btn_patientID)
            self.layout.remove_widget(self.label_PatientID_real)
            self.layout.remove_widget(self.label_task_real)
            Clock.unschedule(self.patient_id_event)
            Clock.unschedule(self.patient_task_event)
            
            self.patientID = "test"
            self.txt_patientID_real = TextInput(hint_text='Patient ID', multiline=False, size_hint=(0.19,0.1), size=(150, 50),  pos_hint={'center_x': self.pos_ref_x[4], 'center_y':self.pos_ref_y[0]}, font_name=self.font_path)
            self.txt_patientID_real.bind(on_text_validate=lambda instance: self.update_tasklist(date))
            self.patient_id_event =Clock.schedule_interval(self.patient_ID_update_single, 0.1)
            self.layout.add_widget(self.txt_patientID_real)
            self.label_PatientID_real = Label(text=self.patientID, size_hint=(0.19,0.1), size=(400, 30), pos=(500, 570), font_name=self.font_path)
            self.layout.add_widget(self.label_PatientID_real)
            self.task = "test"
            self.txt_task = TextInput(hint_text='Task name', multiline=False, size_hint=(0.19,0.1), size=(150, 50), pos_hint={'center_x': self.pos_ref_x[4], 'center_y':self.pos_ref_y[2]}, font_name=self.font_path)
            self.patient_task_event =Clock.schedule_interval(self.task_update_single, 0.1)
            self.layout.add_widget(self.txt_task)
            self.label_task_real = Label(text=self.patientID, size_hint=(0.19,0.1), size=(400, 30), pos=(500, 470), font_name=self.font_path)
            self.layout.add_widget(self.label_task_real)
            self.label_log = Label(text=' ', size_hint=(0.19,0.1), size=(400, 50), pos_hint={'center_x': 0.20, 'center_y':self.pos_ref_y[7]-0.1}, font_name=self.font_path)
            self.layout.add_widget(self.label_log)
        else:
            instance.text = 'Cloud'  # Update the button text
            self.task_button.text ='test'
            self.layout.remove_widget(self.txt_patientID_real)
            self.layout.remove_widget(self.txt_task)
            self.layout.remove_widget(self.label_PatientID_real)
            self.layout.remove_widget(self.label_task_real)
            Clock.unschedule(self.patient_id_event)
            Clock.unschedule(self.patient_task_event)
            self.patient_genID = ''  # Add this line to define the variable
            self.patient_namephone = '' # Add this line to define the variable
            self.task_name =''
            self.btn_patientID = Button(text='Enter Patient ID', size_hint=(0.19, 0.1), size=(150, 50), pos_hint={'center_x': self.pos_ref_x[4], 'center_y': self.pos_ref_y[0]}, font_name=self.font_path)
            # Assuming `self.layout` is an already defined variable
            self.btn_patientID.bind(on_press=lambda instance: self.show_input_popup( instance))

            self.layout.add_widget(self.btn_patientID)
            self.patient_id_event =Clock.schedule_interval(self.patient_ID_update_cloud, 0.1)
            self.label_PatientID_real = Label(text=self.patient_namephone, size_hint=(0.19,0.1), size=(400, 30), pos_hint={'center_x': self.pos_ref_x[3], 'center_y': self.pos_ref_y[0]}, font_name=self.font_path)
            self.layout.add_widget(self.label_PatientID_real)
            # Task Name
            self.task = "test"
            self.task_button = Button(text='Enter Task Name', size_hint=(0.19, 0.1), size=(150, 50), pos_hint={'center_x': self.pos_ref_x[4], 'center_y': self.pos_ref_y[2]}, font_name=self.font_path)
            self.task_button.bind(on_press=lambda instance: setattr(self.sm, 'current', 'task_input'))
            self.layout.add_widget(self.task_button)        
            self.patient_task_event =Clock.schedule_interval(self.task_update_cloud, 0.1)
            self.label_task_real = Label(text=self.task_name , size_hint=(0.19,0.1), size=(400, 30), pos=(500, 470), font_name=self.font_path)
            self.layout.add_widget(self.label_task_real)

            self.label_log = Label(text=' ', size_hint=(0.19,0.1), size=(400, 50), pos_hint={'center_x': 0.20, 'center_y':self.pos_ref_y[7]-0.1}, font_name=self.font_path)
            self.layout.add_widget(self.label_log)

    def show_input_popup(self, instance):
        # Create the popup with the parent app reference
        self.popup = ResultsPopup(title="Results", size_hint=(0.8, 0.8), parent_app=self)
        self.popup.open()
        date =datetime.now().strftime("%Y_%m_%d")


        

    def on_spinner_select(self, spinner, text):
        print(f'Selected feature: {text}')
        self.mode_select = text
        if text == 'VICON Recording':
            self.COM_input.opacity = 1  # Show COM input
            self.btn_ttl.opacity = 1  # Show TTL button
        else:
            self.COM_input.opacity = 0  # Hide COM input
            self.btn_ttl.opacity = 0  # Hide TTL button
    #Implement the logic for the selected feature here

    def on_checkbox_active(self, instance): # Store checkbox state in the app
        try:
            CP2102_output_signal(self.COM_input.text)
        except:
            print('COM input error')

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
            
            
            err_list =calib_extri(self.current_directory,0)
            self.label_log.text = 'calculate finished'
            
            self.err_calib_extri.text = err_list      
            # self.label_log.text = '外參計算完畢'

        except:
            # self.label_log.text = '檢查是否有拍攝以及計算內參，以及是否有拍攝外參'
            self.label_log.text = 'check intrinsic and extrinsic exist'
            self.err_calib_extri.text = 'no calibration file found'
        self.add_log(self.label_log.text)
    def button_extrinsic_manual_calculate(self, instance):
        self.label_log.text = '計算外參'
        self.label_log.text = 'caculating extrinsic'
        self.add_log(self.label_log.text)
        try:
            def remove_folder_with_contents(path):
                # Check if the directory exists
                if os.path.isdir(path):
                    # Recursively delete the directory and all its contents
                    shutil.rmtree(path)
            remove_folder_with_contents(os.path.join(self.extrinsic_path,'images'))
            remove_folder_with_contents(os.path.join(self.extrinsic_path,'yolo_backup'))
            remove_folder_with_contents(os.path.join(self.extrinsic_path,'chessboard'))
                
            err_list =calib_extri(self.current_directory,1)
            self.label_log.text = 'calculate finished'
            
            self.err_calib_extri.text = err_list      

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
    
    def update_Apose_note(self):
        olddir_meetnote = os.path.join(self.config_path, 'meetnote_layout.json')
        newdir_meetnote = os.path.join(self.record_path, "Patient_data",self.patient_genID,datetime.now().strftime("%Y_%m_%d"),'raw_data','Meet_note.json')
        shutil.copy2(olddir_meetnote, newdir_meetnote)
    def update_Task_note(self):
        olddir_meetnote = os.path.join(self.config_path, 'meetnote_layout.json')
        olddir_actionnote =os.path.join(self.config_path, 'actionnote_layout.json')
        newdir_meetnote = os.path.join(self.record_path, "Patient_data",self.patient_genID,datetime.now().strftime("%Y_%m_%d"),'raw_data','Meet_note.json')
        newdir_actionnote = os.path.join(self.record_path, "Patient_data",self.patient_genID,datetime.now().strftime("%Y_%m_%d"),'raw_data',self.label_task_real.text,'Action_note.json')
        shutil.copy2(olddir_meetnote,newdir_meetnote)
        shutil.copy2(olddir_actionnote, newdir_actionnote)
    
############### Apose 不能隔天重拍，要就要當下
    def meetnoteupload(self):
        date_str = datetime.now().strftime("%Y_%m_%d")
        date_str=datetime.strptime(date_str, "%Y_%m_%d").isoformat() + 'Z'
        status,message,meetId =meet_postupdate(os.path.join(self.config_path,'layout.json'),os.path.join(self.config_path, 'meetnote_layout.json'),os.path.join(self.config_path, 'location.json'),self.patient_genID,date_str)
        data = {
            "meetId" : meetId
        }
        if status!=200 and status!= 201:
            print(status)
            print('upload fail')
        else:
            with open(os.path.join(self.record_path, "Patient_data",self.patient_genID,datetime.now().strftime("%Y_%m_%d"),'raw_data','meetId.json'), 'w') as json_file:
                json.dump(data, json_file, indent=4)
        return meetId

    def actionnoteupload(self,taskname,meetId):
        date_str = datetime.now().strftime("%Y_%m_%d")
        date_str=datetime.strptime(date_str, "%Y_%m_%d").isoformat() + 'Z'
        
        status,message,actionId = action_postupdate(os.path.join(self.config_path,'layout.json'),os.path.join(self.config_path, 'actionnote_layout.json'),os.path.join(self.config_path, 'location.json'),self.patient_genID,date_str,taskname,meetId)
        data = {
            "actionId" : actionId
        }
        if status!=200 and status!= 201:
            print(status)
            print('upload fail')
        else:
            with open(os.path.join(self.record_path, "Patient_data",self.patient_genID,datetime.now().strftime("%Y_%m_%d"),'raw_data',taskname,'actionId.json'), 'w') as json_file:
                json.dump(data, json_file, indent=4)
    def button_Apose_record(self, instance):
        if self.btn_toggle_cloud_sinlge.text == 'Single':
            self.patient_genID =self.txt_patientID_real.text
        self.label_log.text = 'film A-pose'
        self.add_log(self.label_log.text)
        if self.label_PatientID_real.text == "":
            self.label_log.text = 'check Patient ID'
        elif os.path.isdir(os.path.join(self.record_path, "Patient_data",self.patient_genID,datetime.now().strftime("%Y_%m_%d"),'raw_data'))==0: ## check if path exist
            camera_Apose_record(self.config_path,self.record_path,self.patient_genID,datetime.now().strftime("%Y_%m_%d"),button_capture=False,button_stop=False) 
            self.update_Apose_note()
            if self.btn_toggle_cloud_sinlge.text == 'Cloud':
                self.meetId = self.meetnoteupload()
                self.actionnoteupload('Apose',self.meetId)
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
        if os.path.exists(os.path.join(self.record_path, "Patient_data",self.patient_genID,datetime.now().strftime("%Y_%m_%d"),'raw_data')):
        
            shutil.rmtree(os.path.join(self.record_path, "Patient_data",self.patient_genID,datetime.now().strftime("%Y_%m_%d"),'raw_data'))
        if popup:
            popup.dismiss()  # Dismiss the popup first
        camera_Apose_record(self.config_path,self.record_path,self.patient_genID,datetime.now().strftime("%Y_%m_%d"),button_capture=False,button_stop=False) 
        self.update_Apose_note()
        if self.btn_toggle_cloud_sinlge.text == 'Cloud':
            self.meetId = self.meetnoteupload()
            self.actionnoteupload('Apose',self.meetId)
        #import pdb;pdb.set_trace()
    def button_task_record(self,instance):
        
        if self.btn_toggle_cloud_sinlge.text == 'Single':
            self.patient_genID =self.txt_patientID_real.text
        # self.label_log.text = '拍攝動作'
        self.label_log.text = 'film motion'
        self.add_log(self.label_log.text)
        date = datetime.now().strftime("%Y_%m_%d")
        #import pdb;pdb.set_trace()
        if self.label_PatientID_real.text == "":
            # self.label_log.text = '請輸入Patient ID'
            self.label_log.text = 'check Patient ID'
        elif self.label_task_real.text == "":
            # self.label_log.text = '請輸入task name'
            self.label_log.text = 'Enter task name'
        elif os.path.isdir(os.path.join(self.record_path, "Patient_data",self.patient_genID,date,'raw_data','Apose'))==0:## check if Apose exist
            content = BoxLayout(orientation='vertical', padding=10, spacing=10)
            message = Label(text='You did not record Apose, are you sure to continue?')
            content.add_widget(message)
            button_layout = BoxLayout(size_hint_y=None, height='50dp', spacing=10)
            # No button
            no_btn = Button(text='No')
            button_layout.add_widget(no_btn)

            # Create the popup
            popup = Popup(title='Confirm Action', content=content, size_hint=(None, None), size=(400, 200))

            # Yes button
            yes_btn = Button(text='Yes')
            yes_btn.bind(on_release=lambda instance: self.perform_Motion_recording_no_Apose(instance, popup,date))
            button_layout.add_widget(yes_btn)

            no_btn.bind(on_release=lambda *args: popup.dismiss())

            content.add_widget(button_layout)

            popup.open()
            self.label_log.text = self.label_PatientID_real.text + " : " + self.label_task_real.text + ", film finished"
        elif os.path.isdir(os.path.join(self.record_path, "Patient_data",self.patient_genID,date,'raw_data',self.label_task_real.text))==0: ## check if path exist
            #camera_Motion_record(self.config_path, self.record_path, self.label_PatientID_real.text, self.label_task_real.text, date, button_capture=False, button_stop=False)
            self.camera_motion_final(self,date)
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
            yes_btn.bind(on_release=lambda instance: self.perform_Motion_recording_same_task(instance, popup,date))
            button_layout.add_widget(yes_btn)

            no_btn.bind(on_release=lambda *args: popup.dismiss())

            content.add_widget(button_layout)

            popup.open()
            self.label_log.text = self.label_PatientID_real.text + " : " + self.label_task_real.text + ", film finished"
        self.add_log(self.label_log.text)
    
    def perform_Motion_recording_same_task(self, instance, popup,date):
        popup.dismiss()  # Dismiss the popup first
        date = datetime.now().strftime("%Y_%m_%d")
        shutil.rmtree(os.path.join(self.record_path, "Patient_data",self.patient_genID,date,'raw_data',self.label_task_real.text))
        self.camera_motion_final(self,date)
    def perform_Motion_recording_no_Apose(self, instance, popup,date):
        popup.dismiss()  # Dismiss the popup first
        date = datetime.now().strftime("%Y_%m_%d")
        self.camera_motion_final(self, date)
        if self.btn_toggle_cloud_sinlge.text == 'Cloud':
            self.meetId = self.meetnoteupload()
            self.actionnoteupload(self.label_task_real.text,self.meetId)
    
    def camera_motion_final(self,instance, date):
        if self.btn_toggle_cloud_sinlge.text == 'Single':
            self.patient_genID =self.txt_patientID_real.text
        if self.mode_select == 'VICON Recording':
            camera_Motion_record_VICON_sync(self.config_path, self.record_path, self.patient_genID, self.label_task_real.text, date,self.COM_input.text, button_capture=False, button_stop=False)
        elif self.mode_select =='Recording':
            camera_Motion_record(self.config_path, self.record_path,  self.patient_genID, self.label_task_real.text, date, button_capture=False, button_stop=False)
        elif self.mode_select =='Delay test':
            camera_Motion_record_test_time_delay(self.config_path, self.record_path,  self.patient_genID, self.label_task_real.text, date, button_capture=False, button_stop=False)

        else:
            print('Error from mode select')
        self.update_tasklist(date)
        self.update_Task_note()
        if self.btn_toggle_cloud_sinlge.text == 'Cloud':
            if self.meetId=='':
                with open(os.path.join(self.record_path, "Patient_data",self.patient_genID,datetime.now().strftime("%Y_%m_%d"),'raw_data','meetId.json'),'r') as file:
                    temp= json.load(file)
                    self.meetId = temp['meetId']
            
            self.actionnoteupload(self.label_task_real.text,self.meetId)
    
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
                folder_calculated = marker_caculate(self.current_directory , cal_folder_path)
                if self.btn_toggle_cloud_sinlge.text == 'Cloud':
                    marker_calculate_upload(folder_calculated,os.path.join(self.current_directory,'config','location.json'))
                
                try:
                    if self.gait_anlaysis.text =='Gait1':
                        from NTK_CAP.script_py.gait_analysis import gait1
                        gait1(folder_calculated)
                except Exception as e:
                    print("An error occurred:")
                    traceback.print_exc()
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
    def patient_ID_update_cloud(self, dt):
        self.label_PatientID_real.text = self.patient_namephone
    def patient_ID_update_single(self, dt):
        self.label_PatientID_real.text = self.txt_patientID_real.text

    def task_update_cloud(self, dt):
        self.label_task_real.text = self.task_name
    def task_update_single(self, dt):
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
