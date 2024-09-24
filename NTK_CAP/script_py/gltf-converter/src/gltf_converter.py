from pathlib import Path
from osimlib.viewport import osimViewport



def build(model_file, motions_file, output):
    # Set the desired width and height for the viewport (in pixels)
    viewport = osimViewport(800, 600)
    viewport.addModelAndMotionFiles(model_file, motions_file)
    viewport.show(output)