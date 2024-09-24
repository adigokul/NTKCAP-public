#!/bin/bash
SRC_DIR="./src"
DIST_DIR="./dist"

mkdir -p $DIST_DIR

cp -R $SRC_DIR"/Geometry" $DIST_DIR
cp $SRC_DIR"/basicShapes.gltf" $DIST_DIR
cp $SRC_DIR"/gltf_converter.py" $DIST_DIR

# Loop through all Python files in the source directory
for file in $DIST_DIR/*.py; do
    # Get the base name of the file (without directory and extension)
    base_name=$(basename $file .py)
    python3 -m nuitka --module --output-dir=$DIST_DIR $file
done

cp $SRC_DIR"/call.py" $DIST_DIR


# Set the source directory
LIB_DIR="$SRC_DIR/osimlib"
# Set the output directory
DIST_LIB_DIR="$DIST_DIR/osimlib"

mkdir -p $DIST_LIB_DIR

cp "$LIB_DIR/__init__.py" "$DIST_DIR/osimlib"
# Create the output directory if it doesn't exist


# Loop through all Python files in the source directory
for file in $LIB_DIR/*.py; do
    # Get the base name of the file (without directory and extension)
    base_name=$(basename $file .py)
    
    # Compile the file with Nuitka
    # python -m nuitka --follow-imports --include-plugin-directory=src --static-libpython=no --output-dir=$OUT_DIR $file
    python3 -m nuitka --module --output-dir=$DIST_LIB_DIR $file
done




echo "All files compiled successfully."