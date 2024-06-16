import kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.popup import Popup
from kivy.uix.scrollview import ScrollView
from kivy.uix.gridlayout import GridLayout
from kivy.uix.spinner import Spinner
import requests

# Path to the font file that supports Chinese characters
FONT_PATH = r'C:\Users\Hermes\Desktop\NTKCAP\NTK_CAP\ThirdParty\Noto_Sans_HK\NotoSansHK-Regular.otf'  # Ensure this path is correct

class ResultsPopup(Popup):
    def __init__(self, **kwargs):
        super(ResultsPopup, self).__init__(**kwargs)
        self.layout = BoxLayout(orientation='vertical', padding=10, spacing=10)

        # Create a horizontal layout for the spinner and text input
        input_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=30)

        # Create a spinner for selecting search type
        self.spinner = Spinner(
            text='Name',
            values=('Name', 'Phone'),
            size_hint_x=None,
            width=100,
            font_name=FONT_PATH
        )
        input_layout.add_widget(self.spinner)

        # Create a text input for the search query
        self.search_input = TextInput(hint_text="Enter name or phone", multiline=False, font_name=FONT_PATH)
        self.search_input.bind(text=self.on_text)
        input_layout.add_widget(self.search_input)

        self.layout.add_widget(input_layout)

        # Create a scroll view to display the results
        self.scroll_view = ScrollView(size_hint=(1, 0.8))
        self.results_layout = GridLayout(cols=1, size_hint_y=None)
        self.results_layout.bind(minimum_height=self.results_layout.setter('height'))
        self.scroll_view.add_widget(self.results_layout)
        self.layout.add_widget(self.scroll_view)

        # Create a button to close the popup
        close_button = Button(text="Close", size_hint=(1, 0.1), font_name=FONT_PATH)
        close_button.bind(on_press=self.dismiss)
        self.layout.add_widget(close_button)

        self.add_widget(self.layout)  # Add the layout to the popup

    def on_text(self, instance, value):
        # Call the API whenever the text input changes
        self.call_api(value)

    def call_api(self, query):
        if query.strip() == "":
            return

        host = "https://motion-service.yuyi-ocean.com"
        url = f"{host}/api/patients"

        # Determine the search type
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
            result_button.bind(on_press=self.on_result_button_press)
            self.results_layout.add_widget(result_button)

    def on_result_button_press(self, instance):
        print(f"Selected ID: {instance.id}")
        # You can add additional functionality here to handle the selection
        self.dismiss()  # Close the popup after selection

class MyApp(App):
    def build(self):
        # Create the main layout
        self.layout = BoxLayout(orientation='vertical', padding=10, spacing=10)

        # Create a label
        self.label = Label(text="Press the button to enter a name or phone number:", font_name=FONT_PATH)
        self.layout.add_widget(self.label)

        # Create a button to trigger the popup
        self.button = Button(text="Enter Name or Phone Number", font_name=FONT_PATH)
        self.button.bind(on_press=self.show_input_popup)
        self.layout.add_widget(self.button)

        # Create a label to display results
        self.result_label = Label(text="", font_name=FONT_PATH)
        self.layout.add_widget(self.result_label)

        return self.layout

    def show_input_popup(self, instance):
        # Create the popup
        self.popup = ResultsPopup(title="Results", size_hint=(0.8, 0.8))
        self.popup.open()

# Run the application
if __name__ == '__main__':
    MyApp().run()
