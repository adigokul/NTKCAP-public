import subprocess
import os
def run_gltf_converter(osim_file_path, mot_file_paths, output=None):
    # Prepare the command with arguments
    cmd = [
        'python',  # Assuming you're running a Python script
        r'C:\Users\mauricetemp\Downloads\gltf-converter 1\gltf-converter\src\call.py',  # Replace this with the actual script name
        osim_file_path,  # First required argument (osim file path)
    ]
    
    # Add all mot file paths to the command
    cmd.extend(mot_file_paths)

    # Optionally add the output argument
    if output:
        cmd.extend(['--output', output])

    # Run the subprocess and capture output
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
        print("Subprocess output:", result.stdout)
        print("Subprocess errors:", result.stderr)
    except Exception as e:
        print(f"Error running subprocess: {e}")

# Example usage
osim_file = r'C:\Users\mauricetemp\Desktop\NTKCAP\Patient_data\667d29d0cfee0b0977061967\2024_09_24\2024_09_19_16_02_calculated\Walking_startend_1\opensim\Model_Pose2Sim_Halpe26_scaled.osim'
mot_files =[ r'C:\Users\mauricetemp\Desktop\NTKCAP\Patient_data\667d29d0cfee0b0977061967\2024_09_24\2024_09_19_16_02_calculated\Walking_startend_1\opensim\Balancing_for_IK_BODY.mot']
output_file = r'C:\Users\mauricetemp\Desktop\NTKCAP\Patient_data\667d29d0cfee0b0977061967\2024_09_24\2024_09_19_16_02_calculated\Walking_startend_1\opensim\temp1'

os.chdir(r'C:\Users\mauricetemp\Downloads\gltf-converter 1\gltf-converter\src')
run_gltf_converter(osim_file, mot_files, output_file)
# vtk pygltflib
#conda install -c conda-forge vtk
# pip install pygltflib