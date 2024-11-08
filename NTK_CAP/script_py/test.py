import numpy as np
dir = r'C:\Users\mauricetemp\Desktop\NTKCAP\Patient_data\ANN_FAKE\2024_09_23\2024_11_08_13_34_calculated\1\opensim\sync_time_marker.npz'
# Load the .npz file
data = np.load(dir)

# Display all variables and their content
for variable_name in data:
    print(f"{variable_name}:")
    print(data[variable_name])
    print()  # Adds a newline for better readability

# Close the .npz file after reading (optional but good practice)
time = data["sync_timeline"]
data.close()

print(np.shape(time))
import pdb;pdb.set_trace()