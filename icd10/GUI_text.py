import os
import json
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.scrollview import ScrollView
from kivy.uix.gridlayout import GridLayout
from kivy.uix.spinner import Spinner
from kivy.uix.popup import Popup

class MyApp(App):
    def build(self):
        self.layout = BoxLayout(orientation='vertical')
        
        # 创建输入框和添加字符串输入框的按钮
        self.input_box = TextInput(size_hint_y=None, height=40, hint_text="Enter title here")
        self.layout.add_widget(self.input_box)
        
        self.add_string_input_button = Button(text="Add Input Box", size_hint_y=None, height=40)
        self.add_string_input_button.bind(on_press=self.add_string_input)
        self.layout.add_widget(self.add_string_input_button)
        
        self.add_spinner_button = Button(text="Add Spinner Box", size_hint_y=None, height=40)
        self.add_spinner_button.bind(on_press=self.open_spinner_popup)
        self.layout.add_widget(self.add_spinner_button)
        
        # 保存按钮
        self.save_button = Button(text="Save Layout", size_hint_y=None, height=40)
        self.save_button.bind(on_press=self.save_layout)
        self.layout.add_widget(self.save_button)
        
        # 选择布局按钮
        self.choose_layout_button = Button(text="Choose Layout", size_hint_y=None, height=40)
        self.choose_layout_button.bind(on_press=self.choose_layout)
        self.layout.add_widget(self.choose_layout_button)
        
        # 创建滚动视图来显示添加的盒子
        self.scroll_view = ScrollView(size_hint=(1, None), size=(self.layout.width, 300))
        self.box_layout = GridLayout(cols=1, size_hint_y=None)
        self.box_layout.bind(minimum_height=self.box_layout.setter('height'))
        self.scroll_view.add_widget(self.box_layout)
        self.layout.add_widget(self.scroll_view)
        
        return self.layout
    
    def add_string_input(self, instance):
        title = self.input_box.text.strip()
        if title:
            new_box = BoxLayout(orientation='horizontal', size_hint_y=None, height=40)
            title_label = Label(text=title, size_hint_y=None, height=40)
            new_box.add_widget(title_label)
            
            string_input = TextInput(size_hint_y=None, height=40)
            new_box.add_widget(string_input)
            
            delete_button = Button(text="Delete", size_hint_y=None, height=40)
            delete_button.bind(on_press=lambda x, box=new_box: self.delete_box(box))
            new_box.add_widget(delete_button)
            
            self.box_layout.add_widget(new_box)
            self.input_box.text = ""
        else:
            self.input_box.hint_text = "Please enter a title"
    
    def open_spinner_popup(self, instance):
        # 创建弹出窗口
        popup_layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        
        # 标题输入框
        self.spinner_title_input = TextInput(size_hint_y=None, height=40, hint_text="Enter spinner title here")
        popup_layout.add_widget(self.spinner_title_input)
        
        # 选项输入框
        self.spinner_option_input = TextInput(size_hint_y=None, height=40, hint_text="Enter options separated by commas")
        popup_layout.add_widget(self.spinner_option_input)
        
        # 确认按钮
        confirm_button = Button(text="Add Spinner Box", size_hint_y=None, height=40)
        confirm_button.bind(on_press=self.add_spinner_box)
        popup_layout.add_widget(confirm_button)
        
        # 创建弹出窗口
        self.popup = Popup(title="Add Spinner Options", content=popup_layout, size_hint=(0.9, 0.5))
        self.popup.open()
    
    def add_spinner_box(self, instance):
        title = self.spinner_title_input.text.strip()
        options = self.spinner_option_input.text.strip().split(',')
        
        if title and options:
            new_box = BoxLayout(orientation='horizontal', size_hint_y=None, height=40)
            title_label = Label(text=title, size_hint_y=None, height=40)
            new_box.add_widget(title_label)
            
            spinner = Spinner(
                text='Choose an option',
                values=options,
                size_hint=(None, None),
                size=(150, 40)
            )
            new_box.add_widget(spinner)
            
            delete_button = Button(text="Delete", size_hint_y=None, height=40)
            delete_button.bind(on_press=lambda x, box=new_box: self.delete_box(box))
            new_box.add_widget(delete_button)
            
            self.box_layout.add_widget(new_box)
            self.popup.dismiss()
        else:
            self.spinner_title_input.hint_text = "Please enter a title and options"
    
    def delete_box(self, box):
        self.box_layout.remove_widget(box)
    
    def save_layout(self, instance):
        layout_data = []
        for child in reversed(self.box_layout.children):  # 反转children列表
            widget_types = [type(w).__name__ for w in child.children]
            if 'TextInput' in widget_types:
                title_label = child.children[widget_types.index('Label')]
                string_input = child.children[widget_types.index('TextInput')]
                layout_data.append({
                    'type': 'input',
                    'title': title_label.text,
                    'content': string_input.text
                })
            elif 'Spinner' in widget_types:
                title_label = child.children[widget_types.index('Label')]
                spinner = child.children[widget_types.index('Spinner')]
                layout_data.append({
                    'type': 'spinner',
                    'title': title_label.text,
                    'options': spinner.values
                })
        
        with open('layout.json', 'w') as f:
            json.dump(layout_data, f)
    
    def choose_layout(self, instance):
        # 创建布局选择弹出窗口
        layout_files = [f for f in os.listdir() if f.endswith('.json')]
        if not layout_files:
            return

        popup_layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        
        # 添加标题
        popup_layout.add_widget(Label(text="Choose a Layout"))

        for layout_file in layout_files:
            layout_button = Button(text=layout_file, size_hint_y=None, height=40)
            layout_button.bind(on_press=lambda x, lf=layout_file: self.load_layout(lf))
            popup_layout.add_widget(layout_button)
        
        # 创建弹出窗口
        self.popup = Popup(title="Choose Layout", content=popup_layout, size_hint=(0.9, 0.5))
        self.popup.open()
    
    def load_layout(self, layout_file):
        self.popup.dismiss()
        # 读取选择的布局
        with open(layout_file, 'r') as f:
            layout_data = json.load(f)
        
        # 清空当前布局
        self.box_layout.clear_widgets()
        
        for item in layout_data:
            if item['type'] == 'input':
                new_box = BoxLayout(orientation='horizontal', size_hint_y=None, height=40)
                title_label = Label(text=item['title'], size_hint_y=None, height=40)
                new_box.add_widget(title_label)
                
                string_input = TextInput(size_hint_y=None, height=40, text=item['content'])
                new_box.add_widget(string_input)
                
                delete_button = Button(text="Delete", size_hint_y=None, height=40)
                delete_button.bind(on_press=lambda x, box=new_box: self.delete_box(box))
                new_box.add_widget(delete_button)
                
                self.box_layout.add_widget(new_box)
            elif item['type'] == 'spinner':
                new_box = BoxLayout(orientation='horizontal', size_hint_y=None, height=40)
                title_label = Label(text=item['title'], size_hint_y=None, height=40)
                new_box.add_widget(title_label)
                
                spinner = Spinner(
                    text='Choose an option',
                    values=item['options'],
                    size_hint=(None, None),
                    size=(150, 40)
                )
                new_box.add_widget(spinner)
                
                delete_button = Button(text="Delete", size_hint_y=None, height=40)
                delete_button.bind(on_press=lambda x, box=new_box: self.delete_box(box))
                new_box.add_widget(delete_button)
                
                self.box_layout.add_widget(new_box)

if __name__ == '__main__':
    MyApp().run()
