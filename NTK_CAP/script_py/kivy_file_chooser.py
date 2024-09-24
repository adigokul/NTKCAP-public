import os
import tkfilebrowser
from tkinter import Tk, Listbox, Button, messagebox, SINGLE, BOTH, LEFT, END
from datetime import datetime
from .cloud_function import recheck,getname

def select_directories_and_return_list(initial_dir):
    # Create the root window
    root = Tk()
    root.geometry('400x300')

    # List variable to store display names for the Listbox
    display_names = []
    # List variable to store full paths of selected directories
    full_paths = []

    def get_directories():
        selected_directories = tkfilebrowser.askopendirnames(initialdir=initial_dir)
        for directory in selected_directories:
            normalized_directory = os.path.normpath(directory)
            directory_name = os.path.basename(normalized_directory)
            parent_directory_path = os.path.dirname(normalized_directory)
            parent_directory_name = os.path.basename(parent_directory_path)
            display_name = f"{parent_directory_name} - {directory_name}"
            
            listbox.insert(END, display_name)
            display_names.append(display_name)
            full_paths.append(normalized_directory)

    def delete_selected():
        try:
            selection_index = listbox.curselection()[0]
            listbox.delete(selection_index)
            del display_names[selection_index]
            del full_paths[selection_index]
        except IndexError:
            messagebox.showinfo("Delete", "Please select an item to delete.")

    def confirm_completion():
        if messagebox.askyesno("Confirm Completion", "Are you sure you have completed your task?"):
            root.quit()

    # Add new buttons for Cloud Today and Cloud All actions
    def get_name_and_dates(initial_dir, date_condition=None):
        name_date_list = []  # List to store tuples of folder names and their corresponding date

        # Loop through the directories inside the initial_dir
        for name in os.listdir(initial_dir):
            name_path = os.path.join(initial_dir, name)
            
            # Check if it's a directory (the "name" folder)
            if os.path.isdir(name_path):
                
                # Loop through the subdirectories (the "date" folders) inside the "name" folder
                for date in os.listdir(name_path):
                    date_path = os.path.join(name_path, date)
                    
                    # Check if it's a directory (the "date" folder) and meets the condition
                    if os.path.isdir(date_path):
                        if (date_condition is None or date == date_condition) and os.path.exists(os.path.join(date_path, 'raw_data', 'meetId.json')):
                            # Store the folder name and date as a tuple
                            name_date_list.append((name, date))

        return name_date_list  # Return the list of tuples
    def cloud_today():
        date = datetime.now().strftime("%Y_%m_%d")
        name_date_dict = get_name_and_dates(initial_dir, date)
        name_date_dict_recheck = []
        for patientId, date in name_date_dict:
            if recheck(os.path.join(initial_dir, patientId, date)) == 0:
                
                name_date_dict_recheck.append((patientId, date,getname(patientId)))
        
        # Add name-date pairs to the listbox
        for patientId, date ,name in name_date_dict_recheck:
            display_name = f"{name} - {date}"
            listbox.insert(END, display_name)
            display_names.append(display_name)
            full_paths.append(os.path.join(initial_dir, patientId, date))
        if len(name_date_dict_recheck)==0:
            messagebox.showinfo("All files on "+ date + "are upload to cloud ")

    def cloud_all():
        name_date_dict = get_name_and_dates(initial_dir)
        name_date_dict_recheck = []
        for patientId, date in name_date_dict:
            if recheck(os.path.join(initial_dir, patientId, date)) == 0:
                name_date_dict_recheck.append((patientId, date,getname(patientId)))
        
        # Add name-date pairs to the listbox
        for patientId, date,name in name_date_dict_recheck:
            display_name = f"{name} - {date}"
            listbox.insert(END, display_name)
            display_names.append(display_name)
            full_paths.append(os.path.join(initial_dir, patientId, date))
        if len(name_date_dict_recheck)==0:
            messagebox.showinfo("All files are upload to cloud ")
    # Setup Listbox
    listbox = Listbox(root, selectmode=SINGLE)
    listbox.pack(fill=BOTH, expand=True)

    # Setup Buttons
    select_button = Button(root, text='Select directories...', command=get_directories)
    select_button.pack(side=LEFT, padx=5, pady=5)

    delete_button = Button(root, text='Delete Selected', command=delete_selected)
    delete_button.pack(side=LEFT, padx=5, pady=5)

    complete_button = Button(root, text='Confirm Completion', command=confirm_completion)
    complete_button.pack(side=LEFT, padx=5, pady=5)

    # New Buttons for Cloud Today and Cloud All
    cloud_today_button = Button(root, text='Cloud Today', command=cloud_today)
    cloud_today_button.pack(side=LEFT, padx=5, pady=5)

    cloud_all_button = Button(root, text='Cloud All', command=cloud_all)
    cloud_all_button.pack(side=LEFT, padx=5, pady=5)

    # Run the main loop
    root.mainloop()
    root.destroy()

    return full_paths
