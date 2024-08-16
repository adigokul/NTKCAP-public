
from pygltflib import GLTF2
import numpy as np
from pythreejs import *
from IPython.display import display

def gltf_to_threejs_mesh(gltf_file):
    gltf = GLTF2().load(gltf_file)
    
    if not gltf.meshes or not gltf.meshes[0].primitives:
        raise ValueError("The GLTF file does not contain any meshes or primitives.")
    
    mesh_data = gltf.meshes[0]
    primitive = mesh_data.primitives[0]
    
    position_accessor_index = primitive.attributes.get('POSITION')
    if position_accessor_index is None:
        raise ValueError("No POSITION attribute found in the primitive.")
    
    position_accessor = gltf.accessors[position_accessor_index]
    position_buffer_view = gltf.bufferViews[position_accessor.bufferView]
    position_buffer = gltf.buffers[position_buffer_view.buffer].uri
    positions = np.frombuffer(position_buffer, dtype=np.float32, count=position_accessor.count * 3, offset=position_buffer_view.byteOffset)
    positions = positions.reshape((position_accessor.count, 3))

    indices_accessor_index = primitive.indices
    if indices_accessor_index is None:
        raise ValueError("No indices attribute found in the primitive.")
    
    indices_accessor = gltf.accessors[indices_accessor_index]
    indices_buffer_view = gltf.bufferViews[indices_accessor.bufferView]
    indices_buffer = gltf.buffers[indices_buffer_view.buffer].uri
    indices = np.frombuffer(indices_buffer, dtype=np.uint16, count=indices_accessor.count, offset=indices_buffer_view.byteOffset)

    geometry = BufferGeometry(
        attributes={
            'position': BufferAttribute(positions, normalized=False),
            'index': BufferAttribute(indices, normalized=False)
        }
    )
    material = MeshStandardMaterial(color='grey', roughness=0.7, metalness=0.0)
    mesh = Mesh(geometry=geometry, material=material)
    
    return mesh

gltf_file_path = r'C:\Users\Hermes\Desktop\NTKCAP\Patient_data\Patient_ID\2024_05_07\2024_06_02_14_47_calculated\Walk1\opensim\model.gltf'

try:
    threejs_mesh = gltf_to_threejs_mesh(gltf_file_path)

    scene = Scene(children=[
        threejs_mesh,
        AmbientLight(intensity=0.5),
        DirectionalLight(position=[3, 5, 1], intensity=0.6)
    ])

    camera = PerspectiveCamera(position=[3, 3, 3], aspect=1.6)

    renderer = Renderer(camera=camera, scene=scene, controls=[OrbitControls(controlling=camera)])

    display(renderer)
except Exception as e:
    print(f"Error loading GLTF file: {e}")
