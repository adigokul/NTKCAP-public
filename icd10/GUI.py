import sqlite3
import json
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.scrollview import ScrollView
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.popup import Popup
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.spinner import Spinner

# 连接到SQLite数据库
conn = sqlite3.connect('icd10.db')

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

class TaskInputScreen(App):
    def build(self):
        self.layout = BoxLayout(orientation='vertical')
        
        # 添加选择布局按钮
        self.choose_layout_button = Button(text="Choose Layout", size_hint_y=None, height=40)
        self.choose_layout_button.bind(on_press=self.open_file_chooser)
        self.layout.add_widget(self.choose_layout_button)

        # 创建滚动视图来显示布局内容
        self.scroll_view = ScrollView(size_hint=(1, 1))
        self.results_layout = GridLayout(cols=1, size_hint_y=None)
        self.results_layout.bind(minimum_height=self.results_layout.setter('height'))
        self.scroll_view.add_widget(self.results_layout)
        self.layout.add_widget(self.scroll_view)

        # Create a button at the bottom right
        bottom_right_button = Button(text="Print Task", size_hint=(0.3, 0.1), pos_hint={'right': 1, 'bottom': 1})
        bottom_right_button.bind(on_press=self.print_task_info)
        self.layout.add_widget(bottom_right_button)

        # Initialize instance variables for the task input boxes and spinner
        self.task_name_input = None
        self.task_number_input = None
        self.task_spinner = None

        return self.layout
    
    def open_file_chooser(self, instance):
        # 创建文件选择弹出窗口
        file_chooser = FileChooserIconView(path='.', filters=['*.json'])
        popup_layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        popup_layout.add_widget(file_chooser)
        
        # 确认按钮
        confirm_button = Button(text="Load Layout", size_hint_y=None, height=40)
        confirm_button.bind(on_press=lambda x: self.load_layout(file_chooser.selection))
        popup_layout.add_widget(confirm_button)
        
        self.popup = Popup(title="Choose Layout File", content=popup_layout, size_hint=(0.9, 0.9))
        self.popup.open()
    
    def load_layout(self, selection):
        if selection:
            layout_file = selection[0]
            self.popup.dismiss()
            
            # 读取选择的布局文件
            with open(layout_file, 'r') as f:
                layout_data = json.load(f)
            
            # 清空当前布局
            self.results_layout.clear_widgets()
            
            for item in layout_data:
                if item['type'] == 'input' and item['title'].lower() == 'symptoms':
                    # 替换为Search ICD-10按钮和显示选择结果的TextInput
                    new_box = BoxLayout(orientation='horizontal', size_hint_y=None, height=40)
                    title_label = Label(text=item['title'], size_hint_y=None, height=40)
                    new_box.add_widget(title_label)
                    
                    # Use a GridLayout to align TextInput and Button
                    grid_layout = GridLayout(cols=2, size_hint=(1, None), height=40)
                    
                    self.idc_result_input = TextInput(size_hint_y=None, height=40, multiline=False, font_name='NotoSansHK-Regular.otf')
                    grid_layout.add_widget(self.idc_result_input)

                    search_button = Button(text="Search ICD-10", size_hint_x=None, width=150)
                    search_button.bind(on_press=self.open_search_popup)
                    grid_layout.add_widget(search_button)
                    
                    new_box.add_widget(grid_layout)
                    self.results_layout.add_widget(new_box)
                    
                elif item['type'] == 'input':
                    new_box = BoxLayout(orientation='horizontal', size_hint_y=None, height=40)
                    title_label = Label(text=item['title'], size_hint_y=None, height=40)
                    new_box.add_widget(title_label)
                    
                    string_input = TextInput(size_hint=(1, None), height=40, text=item['content'])
                    new_box.add_widget(string_input)
                    
                    self.results_layout.add_widget(new_box)
                    
                    # Store references to task name and task number inputs
                    if item['title'].lower() == 'task name':
                        self.task_name_input = string_input
                    elif item['title'].lower() == 'task number':
                        self.task_number_input = string_input

                elif item['type'] == 'spinner':
                    new_box = BoxLayout(orientation='horizontal', size_hint_y=None, height=40)
                    title_label = Label(text=item['title'], size_hint_y=None, height=40)
                    new_box.add_widget(title_label)
                    
                    spinner = Spinner(
                        text='Choose an option',
                        values=item['options'],
                        size_hint=(1, None),
                        height=40
                    )
                    new_box.add_widget(spinner)
                    
                    self.results_layout.add_widget(new_box)
                    if item['title'].lower() == 'task name':
                        self.task_name_input = spinner
                    elif item['title'].lower() == 'task number':
                        self.task_number_input = spinner
                    # Store reference to the spinner
                    

    def open_search_popup(self, instance):
        # 创建弹出窗口
        popup_layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        
        self.popup_input = TextInput(size_hint_y=None, height=40, font_name='NotoSansHK-Regular.otf')
        self.popup_input.bind(text=self.on_text)
        popup_layout.add_widget(self.popup_input)
        
        # 创建滚动视图来显示搜索结果
        self.results_view = ScrollView(size_hint=(1, None), size=(popup_layout.width, 300))
        self.results_layout = GridLayout(cols=1, size_hint_y=None)
        self.results_layout.bind(minimum_height=self.results_layout.setter('height'))
        self.results_view.add_widget(self.results_layout)
        popup_layout.add_widget(self.results_view)
        
        self.popup = Popup(title="Enter Search Query", content=popup_layout, size_hint=(0.9, 0.9))
        self.popup.open()
    
    def on_text(self, instance, value):
        self.results_layout.clear_widgets()
        if value.strip():  # 检查输入是否为空
            search_results = fuzzy_search(value)
            for code, en_description, cn_description in search_results[:25]:  # 显示前25个结果
                result_button = Button(text=f"{code}: {en_description} / {cn_description}", size_hint_y=None, height=40, font_name='NotoSansHK-Regular.otf')
                result_button.bind(on_press=lambda x, c=code, e=en_description, cn=cn_description: self.select_result(c, e, cn))
                self.results_layout.add_widget(result_button)
    
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
        if self.task_name_input and self.task_number_input :
            task_name = self.task_name_input.text
            task_number = self.task_number_input.text
    
            print(f"{task_name}{task_number}")
        else:
            print("Task Name, Task Number, or Task Type input not found")

if __name__ == '__main__':
    TaskInputScreen().run()

# 关闭数据库连接
conn.close()
