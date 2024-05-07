import opensim as osim
import time
import numpy as np
import argparse
import os
def update_geometry_paths_in_memory(model, new_geometry_dir):
    model.initSystem()  # Initialize system to ensure model components are ready

    # Iterate over all bodies in the model
    for i in range(model.getBodySet().getSize()):
        body = model.getBodySet().get(i)
        # Check and update each attached geometry if any
        for component in body.getComponentsList():
            # You can check if the component is a type of geometry you're interested in
            if isinstance(component, osim.Mesh):
                geom = osim.Mesh.safeDownCast(component)
                if geom:  # If it's indeed a Mesh and cast is successful
                    old_path = geom.get_mesh_file()
                    new_path = os.path.join(new_geometry_dir, os.path.basename(old_path))
                    geom.set_mesh_file(new_path)

    return model

def main(motfile, model_file,new_geometry_dir):
    time_data = osim.TimeSeriesTable(motfile)
    model = osim.Model(model_file)
    update_geometry_paths_in_memory(model, new_geometry_dir)
    osim.VisualizerUtilities_showMotion(model, time_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize motion data on a given model.")
    parser.add_argument("motfile", type=str, help="Path to the .mot file containing the motion data.")
    parser.add_argument("model_file", type=str, help="Path to the .osim model file.")
    parser.add_argument("new_geometry_dir", type=str, help="Path to the vtp folder.")

    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(args.motfile, args.model_file,args.new_geometry_dir)
