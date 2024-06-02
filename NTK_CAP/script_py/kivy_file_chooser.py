import os
import tkfilebrowser
from tkinter import Tk, Listbox, Button, messagebox, SINGLE, BOTH, LEFT, END

def select_directories_and_return_list(initial_dir):
    # Create the root window
    root = Tk()
    root.geometry('400x300')

    # List variable to store display names for the Listbox
    display_names = []
    # List variable to store full paths of selected directories
    full_paths = []

    def get_directories():
        # Use the initial_dir provided to the main function
        selected_directories = tkfilebrowser.askopendirnames(initialdir=initial_dir)
        for directory in selected_directories:
            # Normalize the path to remove any redundant separators or up-level references
            normalized_directory = os.path.normpath(directory)
            
            # Get the directory name
            directory_name = os.path.basename(normalized_directory)
            
            # Get the parent directory path
            parent_directory_path = os.path.dirname(normalized_directory)
            
            # Get the parent directory name
            parent_directory_name = os.path.basename(parent_directory_path)
            
            # Format the display name as 'Parent Directory - Directory'
            display_name = f"{parent_directory_name} - {directory_name}"
            
            listbox.insert(END, display_name)
            display_names.append(display_name)
            full_paths.append(normalized_directory)  # Store the full path

    def delete_selected():
        try:
            selection_index = listbox.curselection()[0]
            listbox.delete(selection_index)
            del display_names[selection_index]
            del full_paths[selection_index]  # Keep the full_paths list in sync
        except IndexError:
            messagebox.showinfo("Delete", "Please select an item to delete.")

    def confirm_completion():
        if messagebox.askyesno("Confirm Completion", "Are you sure you have completed your task?"):
            root.quit()

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

    # Run the main loop
    root.mainloop()
    root.destroy()

    return full_paths  # Return the full paths of selected directories



