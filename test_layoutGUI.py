import requests
from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.popup import Popup
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.scrollview import ScrollView
import json
# Kivy App Class
class LayoutApp(App):
    def build(self):
        # Create the main layout
        main_layout = BoxLayout(orientation='vertical')

        # Button to trigger the popup
        btn_show_popup = Button(text="Show Layouts", size_hint=(1, 0.2))
        btn_show_popup.bind(on_press=self.show_layouts_popup)

        # Add button to main layout
        main_layout.add_widget(btn_show_popup)

        # Initialize layouts to None
        self.layouts = None  # This will store the fetched layouts
        self.selected_meet_data = None  # Store selected meet data
        self.selected_action_data = None  # Store selected action data

        return main_layout

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
        self.btn_select = Button(text="Select", size_hint=(0.2, 0.1), pos_hint={'right': 1})
        self.btn_select.bind(on_release=self.final_selection_made)

        # Add the Select button at the bottom of the popup
        button_layout = BoxLayout(size_hint_y=None, height=50, padding=[0, 10, 10, 10])
        button_layout.add_widget(self.btn_select)

        # Create the popup layout
        popup_layout = BoxLayout(orientation='vertical')
        popup_layout.add_widget(header_layout)
        popup_layout.add_widget(scroll_layout)  # Add the scrollable columns
        popup_layout.add_widget(button_layout)

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
        json_file  = r'C:\Users\mauricetemp\Desktop\NTKCAP\config\testlayout.json'
        # Save combined_data to a JSON file
        with open("combined_fields_with_id.json", "w") as json_file:
            json.dump(combined_data, json_file, indent=4)  # Save with indentation for readability
            print("Data saved to combined_fields_with_id.json")

        # Add logic to close the popup after printing
        if hasattr(self, 'popup'):
            self.popup.dismiss()  # Close the popup if it's open



# Run the app
if __name__ == '__main__':
    LayoutApp().run()
