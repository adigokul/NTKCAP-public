import opensim as osim
import os
import numpy as np
import base64
from pygltflib import GLTF2, Scene, Node, Mesh, Animation, AnimationChannel, AnimationSampler, Accessor, Asset, Buffer, BufferView

def update_geometry_paths_in_memory(model, new_geometry_dir):
    model.initSystem()  # Initialize system to ensure model components are ready

    # Iterate over all bodies in the model
    for i in range(model.getBodySet().getSize()):
        body = model.getBodySet().get(i)
        # Check and update each attached geometry if any
        for component in body.getComponentsList():
            if isinstance(component, osim.Mesh):
                geom = osim.Mesh.safeDownCast(component)
                if geom:
                    old_path = geom.get_mesh_file()
                    new_path = os.path.join(new_geometry_dir, os.path.basename(old_path))
                    print(new_path)
                    geom.set_mesh_file(new_path)

    return model

def convert_opensim_to_gltf(model, motion, output_dir):
    gltf = GLTF2()

    # Create a basic GLTF scene with the OpenSim model's skeleton and motion
    scene = Scene()
    gltf.scenes.append(scene)

    nodes = []
    for body in model.getBodySet():
        node = Node(name=body.getName())
        gltf.nodes.append(node)
        scene.nodes.append(len(gltf.nodes) - 1)
        nodes.append(node)

        # Create placeholder mesh
        mesh = Mesh(name=body.getName())
        gltf.meshes.append(mesh)
        node.mesh = len(gltf.meshes) - 1

    # Prepare buffers and buffer views
    buffer_data = bytearray()

    # Add motion data as animations
    animation = Animation()
    gltf.animations.append(animation)

        # Extract motion data
    times = []
    translations = [[] for _ in range(model.getBodySet().getSize())]
    rotations = [[] for _ in range(model.getBodySet().getSize())]

    state = model.initSystem()
    for time_index in range(motion.getSize()):
        time = motion.getStateVector(time_index).getTime()
        times.append(time)
        model.realizePosition(state)

        for i, body in enumerate(model.getBodySet()):
            transform = body.getTransformInGround(state)
            translation = np.array(transform.p().to_numpy(), dtype=np.float32)
            rotation = np.array(transform.R().convertRotationToBodyFixedXYZ().to_numpy(), dtype=np.float32)

            translations[i].append(translation.tolist())
            rotations[i].append(rotation.tolist())


    #import pdb;pdb.set_trace()

    # Create buffers for time, translations, and rotations
    def create_buffer_view(data_array, component_type, type_str):
        nonlocal buffer_data
        byte_length = len(data_array.tobytes())
        buffer_data.extend(data_array.tobytes())
        buffer_view = BufferView(
            buffer=0,
            byteOffset=len(buffer_data) - byte_length,
            byteLength=byte_length,
        )
        gltf.bufferViews.append(buffer_view)
        accessor = Accessor(
            bufferView=len(gltf.bufferViews) - 1,
            byteOffset=0,
            componentType=component_type,
            count=len(data_array) // (3 if type_str == "VEC3" else 1),
            type=type_str,
            max=[float(x) for x in data_array.max(axis=0)] if type_str != "SCALAR" else [float(data_array.max())],
            min=[float(x) for x in data_array.min(axis=0)] if type_str != "SCALAR" else [float(data_array.min())]
        )
        gltf.accessors.append(accessor)
        return len(gltf.accessors) - 1

    times_array = np.array(times, dtype=np.float32)
    translations_array = np.array([item for sublist in translations for item in sublist], dtype=np.float32)
    rotations_array = np.array([item for sublist in rotations for item in sublist], dtype=np.float32)

    time_accessor_index = create_buffer_view(times_array, 5126, "SCALAR")
    translation_accessor_index = create_buffer_view(translations_array, 5126, "VEC3")
    rotation_accessor_index = create_buffer_view(rotations_array, 5126, "VEC3")

    # Add motion data to GLTF animations
    for i, node in enumerate(nodes):
        if not translations[i] or not rotations[i]:
            continue

        # Create translation and rotation channels
        translation_channel = AnimationChannel(
            sampler=len(gltf.samplers),
            target={"node": len(gltf.nodes) - len(nodes) + i, "path": "translation"}
        )
        rotation_channel = AnimationChannel(
            sampler=len(gltf.samplers) + 1,
            target={"node": len(gltf.nodes) - len(nodes) + i, "path": "rotation"}
        )
        animation.channels.append(translation_channel)
        animation.channels.append(rotation_channel)

        # Create samplers
        translation_sampler = AnimationSampler(
            input=time_accessor_index,
            output=translation_accessor_index
        )
        rotation_sampler = AnimationSampler(
            input=time_accessor_index,
            output=rotation_accessor_index
        )
        gltf.samplers.append(translation_sampler)
        gltf.samplers.append(rotation_sampler)

    # Add buffer to GLTF
    buffer = Buffer(
        uri="data:application/octet-stream;base64," + base64.b64encode(buffer_data).decode('utf-8'),
        byteLength=len(buffer_data)
    )
    gltf.buffers.append(buffer)

    # Set up GLTF asset information
    gltf.asset = Asset()
    gltf.asset.version = "2.0"

    # Write the GLTF file to the output directory
    output_path = os.path.join(output_dir, 'model.gltf')
    gltf.save(output_path)
    print(f"GLTF file saved to {output_path}")

class GltfConverter:
    def build(self, osim_model_path, motion_file_path, output_dir, new_geometry_dir):
        # Load the OpenSim model
        model = osim.Model(osim_model_path)

        # Update geometry paths in memory
        model = update_geometry_paths_in_memory(model, new_geometry_dir)

        # Initialize the system after updating paths
        state = model.initSystem()

        # Load the motion file
        motion = osim.Storage(motion_file_path)

        # Apply the motion to the model
        state = model.initSystem()

        # Remove or comment out the visualizer line
        # model.getVisualizer().show(state)

        # Convert the model with motion to GLTF
        convert_opensim_to_gltf(model, motion, output_dir)

# Example usage
if __name__ == "__main__":
    converter = GltfConverter()
    osim_model_path = r"C:/Users/Hermes/Desktop/NTKCAP/Patient_data/Patient_ID/2024_05_07/2024_06_02_14_47_calculated/Walk1/opensim/Model_Pose2Sim_Halpe26_scaled.osim"
    motion_file_path = r"C:/Users/Hermes/Desktop/NTKCAP/Patient_data/Patient_ID/2024_05_07/2024_06_02_14_47_calculated/Walk1/opensim/Balancing_for_IK_BODY.mot"
    output_dir = r"C:/Users/Hermes/Desktop/NTKCAP/Patient_data/Patient_ID/2024_05_07/2024_06_02_14_47_calculated/Walk1/opensim"
    new_geometry_dir = r"C:/Users/Hermes/Desktop/NTKCAP/NTK_CAP/script_py/Opensim_visualize_python"

    converter.build(osim_model_path, motion_file_path, output_dir, new_geometry_dir)
