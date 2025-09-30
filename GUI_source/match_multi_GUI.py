import os
from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Match multiple person")
        self.resize(1200, 600)
        print(os.getcwd())
        self.patient_id_path = os.path.join(os.getcwd(), 'Patient_data', 'multi_person')

        self.unchcked_multi_task = self.generate_task_list()
        self.layout = QHBoxLayout()
        
        # Left: paths have not been checked
        self.left_list = QListWidget()
        self.left_list.addItems(self.unchcked_multi_task)  # add list
        self.left_list.clicked.connect(self.on_list_item_clicked)  # bind clicked event
        self.layout.addWidget(self.left_list)
        
        self.curtaskfolder = None
        self.dropdown_modified = False
        self.cur_apose = None
        self.cur_task = None
        self.task_folder_mod = None
        self.init_task_folder = None
        self.name_folder = None
        self.selected_task_path = None
        self.dropdown_apose = QComboBox()
        self.dropdown_apose.currentTextChanged.connect(lambda: self.on_dropdown_changed_apose())
        self.dropdown_task = QComboBox()
        self.dropdown_task.currentTextChanged.connect(lambda: self.on_dropdown_changed_task())
        # Area showing Apose cropped frames
        self.middle_widget = self.create_image_viewer_apose()
        self.layout.addWidget(self.middle_widget)

        # Area showing Tasks cropped frames
        self.right_widget = self.create_image_viewer_task()
        self.layout.addWidget(self.right_widget)

        self.folder_apose = None
        self.folder_task = None
        self.waring_msg = QMessageBox(self)
        self.waring_msg.setIcon(QMessageBox.Icon.Warning)
        self.waring_msg.setText("There is subject not matched!")
        self.waring_msg.setWindowTitle("Save match warning")  
        self.waring_msg.setStandardButtons(QMessageBox.StandardButton.Cancel)  
        
        # Main window setup
        container = QWidget()
        container.setLayout(self.layout)
        self.setCentralWidget(container)
        if self.left_list.count() > 0:
            self.left_list.setCurrentRow(0)
            self.on_list_item_clicked(self.left_list.model().index(0, 0))
    def generate_task_list(self):
        # All multi_person tasks which have not been checked 
        task_list = []
        for date in os.listdir(self.patient_id_path):
            unchcked_multi_date_path = os.path.join(self.patient_id_path, date, 'raw_data')
            for task in os.listdir(unchcked_multi_date_path):
                if task != 'calibration':
                    chcked_multi_task_name_path = os.path.join(unchcked_multi_date_path, task, 'name_checked')
                    if not os.path.exists(chcked_multi_task_name_path) and os.path.exists(os.path.join(unchcked_multi_date_path, task, 'name', 'name_cal.txt')):
                        task_list.append(os.path.join(date, 'raw_data', task))
        return task_list

    def on_list_item_clicked(self, index):
        self.dropdown_modified = False
        self.cur_apose = None
        self.cur_task = None
        self.task_folder_mod = None
        self.init_task_folder = None
        self.folder_apose = None
        self.folder_task = None
        # Select task
        selected_task = self.left_list.item(index.row()).text()
        self.selected_task_path = os.path.join(self.patient_id_path, selected_task)        
        # Load Apose cropped images
        self.name_folder = os.path.join(self.selected_task_path, "name")
        
        self.folder_apose = os.path.join(self.name_folder, 'Apose')
        self.cur_apose = os.listdir(self.folder_apose)[0]
        self.update_folder_dropdown_apose()
        
        # Load task cropped images
        
        self.folder_task = os.path.join(self.name_folder, 'task')
        self.cur_task = os.listdir(self.folder_task)[0]
        self.update_folder_dropdown_task()
        
    def on_dropdown_changed_apose(self):
        # drop down menu for swiching subject (Apose and task)
        selected_folder = self.dropdown_apose.currentText()
        
        self.cur_apose = selected_folder
        selected_path = os.path.join(self.folder_apose, selected_folder)
        
        self.load_images_to_viewer(selected_path, self.middle_widget)
    def on_dropdown_changed_task(self):
        selected_folder = self.dropdown_task.currentText()
        
        if not str.isdigit(selected_folder):
            if selected_folder:
                
                self.cur_task = self.init_task_folder[self.task_folder_mod.index(selected_folder)]
                self.curtaskfolder = self.folder_task
                selected_folder = self.cur_task
            else:                
                self.curtaskfolder = self.folder_task
                selected_folder = self.cur_task
        elif str.isdigit(selected_folder):
            self.cur_task = selected_folder
            self.curtaskfolder = self.folder_task
        selected_path = os.path.join(self.folder_task, selected_folder)        
        self.load_images_to_viewer(selected_path, self.right_widget)
    def create_image_viewer_apose(self):
        widget_apose = QWidget()
        layout = QVBoxLayout()
        label = QLabel()
        label.setFixedSize(400, 300)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)

        button_layout = QHBoxLayout()
        left_button = QPushButton("<")
        right_button = QPushButton(">")
        left_button.clicked.connect(lambda: self.change_image(-1, widget_apose))
        right_button.clicked.connect(lambda: self.change_image(1, widget_apose))
        button_layout.addWidget(left_button)
        button_layout.addWidget(right_button)
        layout.addLayout(button_layout)
        dropdown_settaskname = QHBoxLayout()
        
        btn_settotask = QPushButton("set to task")
        btn_settotask.clicked.connect(self.on_btn_settotask_clicked)
        dropdown_settaskname.addWidget(self.dropdown_apose)
        dropdown_settaskname.addWidget(btn_settotask)
        layout.addLayout(dropdown_settaskname)
        widget_apose.setLayout(layout)
        widget_apose.label = label
        
        widget_apose.image_list = []
        widget_apose.current_index = 0

        return widget_apose
    def on_btn_settotask_clicked(self):        

        self.task_folder_mod[self.init_task_folder.index(self.cur_task)] = f"{self.cur_task}({self.cur_apose})"
        self.dropdown_modified = True
        
        self.update_folder_dropdown_task()
        
        
    def update_folder_dropdown_task(self):        
        self.dropdown_task.clear()
        
        if not self.dropdown_modified:
            self.init_task_folder = [f for f in os.listdir(self.folder_task) if os.path.isdir(os.path.join(self.folder_task, f))]            
            self.task_folder_mod = self.init_task_folder.copy()
            self.dropdown_task.blockSignals(True)
            self.dropdown_task.addItems(self.task_folder_mod)
            self.dropdown_task.blockSignals(False)
            self.dropdown_task.setCurrentIndex(0)
        else:
            self.dropdown_task.blockSignals(True)
            self.dropdown_task.addItems(self.task_folder_mod)
            self.dropdown_task.blockSignals(False)
            self.dropdown_task.setCurrentIndex(self.init_task_folder.index(self.cur_task))
            
        if self.init_task_folder:
            first_folder = os.path.join(self.folder_task, self.cur_task)
            self.load_images_to_viewer(first_folder, self.right_widget)    
        
    def create_image_viewer_task(self):
        create_image_viewer_task = QWidget()
        layout = QVBoxLayout()

        label = QLabel()
        label.setFixedSize(400, 300)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)

        button_layout = QHBoxLayout()
        left_button = QPushButton("<")
        right_button = QPushButton(">")
        left_button.clicked.connect(lambda: self.change_image(-1, create_image_viewer_task))
        right_button.clicked.connect(lambda: self.change_image(1, create_image_viewer_task))
        button_layout.addWidget(left_button)
        button_layout.addWidget(right_button)
        layout.addLayout(button_layout)
        btn_save = QPushButton("save")
        btn_save.clicked.connect(self.on_btn_save_clicked)
        
        dropdown_savebtn_layout = QHBoxLayout()
        dropdown_savebtn_layout.addWidget(self.dropdown_task)
        dropdown_savebtn_layout.addWidget(btn_save)
        layout.addLayout(dropdown_savebtn_layout)
        
        create_image_viewer_task.setLayout(layout)
        create_image_viewer_task.label = label
        create_image_viewer_task.image_list = []
        create_image_viewer_task.current_index = 0
        return create_image_viewer_task
    def on_btn_save_clicked(self):
        for match_result in self.task_folder_mod:
            if str.isdigit(match_result):
                self.waring_msg.exec()
                return
        self.dropdown_modified = False
        self.init_task_folder = None
        
        checked_name_path = os.path.join(self.selected_task_path, 'name_checked')
        os.remove(os.path.join(self.name_folder, 'name_cal.txt'))
        self.left_list.setEnabled(False)
        os.rename(self.name_folder, checked_name_path)
        self.left_list.setEnabled(True)
        with open(os.path.join(checked_name_path, 'name.txt'), 'w') as file:
            for item in self.task_folder_mod:
                item_str = item[item.find('(')+1:item.find(')')]
                file.write(item_str + '\n')
        
        if self.left_list.count()-1 > 0:
            self.left_list.clear()
            self.unchcked_multi_task = self.generate_task_list()
            self.left_list.addItems(self.unchcked_multi_task)
            self.left_list.setCurrentRow(0)
            self.on_list_item_clicked(self.left_list.model().index(0, 0))
        else:
            QApplication.quit()
        
    def load_images_to_viewer(self, first_folder, viewer_widget):       
        # get the imgs(only .jpg and .png)        
        images = [os.path.join(first_folder, f) for f in os.listdir(first_folder) if f.endswith((".jpg", ".png"))]
        
        # update img view list
        viewer_widget.image_list = images
        viewer_widget.current_index = 0  # reset img index

        # display the first image
        self.display_image(viewer_widget, 0)
    def update_folder_dropdown_apose(self):        
        # Select task
        self.dropdown_apose.clear()
        apose_p_folders = [f for f in os.listdir(self.folder_apose) if os.path.isdir(os.path.join(self.folder_apose, f))]
        self.dropdown_apose.addItems(apose_p_folders)
        
        # default showing the first name
        if apose_p_folders:
            self.dropdown_apose.setCurrentIndex(0)
            first_folder = os.path.join(self.folder_apose, apose_p_folders[0])
            self.load_images_to_viewer(first_folder, self.middle_widget)       
    
    def display_image(self, viewer_widget, index):
        if viewer_widget.image_list:
            image_path = viewer_widget.image_list[index]
            pixmap = QPixmap(image_path).scaled(viewer_widget.label.size(), Qt.AspectRatioMode.KeepAspectRatio)
            viewer_widget.label.setPixmap(pixmap)
        else:
            viewer_widget.label.clear()

    def change_image(self, direction, viewer_widget):
        if viewer_widget.image_list:
            viewer_widget.current_index = (viewer_widget.current_index + direction) % len(viewer_widget.image_list)
            self.display_image(viewer_widget, viewer_widget.current_index)


if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()
