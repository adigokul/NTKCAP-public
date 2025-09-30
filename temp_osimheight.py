import opensim as osim

# Load the model
model = osim.Model(r'C:\Users\mauricetemp\Desktop\NTKCAP\Patient_data\Kevin\2024_10_30\2024_10_30_18_01_calculated\1\opensim\Model_Pose2Sim_Halpe26_scaled.osim')

state = model.initSystem()

# Access the MarkerSet
marker_set = model.getMarkerSet()

# Example: Get the global coordinates of specific markers
# Replace 'marker_foot' and 'marker_head' with the actual names of your markers
marker_foot = marker_set.get('RBigToe')  # Replace with your actual foot marker name
marker_head = marker_set.get('Neck')  # Replace with your actual head marker name

# Get global positions of the markers
foot_global_position = marker_foot.getLocationInGround(state)
head_global_position = marker_head.getLocationInGround(state)

# Print the global coordinates of each marker
print(f"Global position of the foot marker: ({foot_global_position.get(0)}, {foot_global_position.get(1)}, {foot_global_position.get(2)})")
print(f"Global position of the head marker: ({head_global_position.get(0)}, {head_global_position.get(1)}, {head_global_position.get(2)})")

# Calculate the vertical distance (height) between the two markers
height = head_global_position.get(1) - foot_global_position.get(1)
print(f"Model height based on markers: {height:.2f} meters")
