import json

# Specify the path to your JSON file
json_file_path =r'C:\Users\mauricetemp\Desktop\kevin\2_processed.json'

# Open and read the JSON file
with open(json_file_path, 'r') as file:
    data = json.load(file)  # Load JSON data
import pdb;pdb.set_trace()
# Display the loaded JSON data
print(data)

# Access individual elements
print(data['name'])  # If JSON has a key 'name'
