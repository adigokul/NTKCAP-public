#!/bin/bash

# Exit on any error
set -e

# --- CONFIGURATION ---
# We store weights in a central folder to keep the repo clean
WEIGHTS_DIR="models/weights"
RTMP_DIR="NTK_CAP/ThirdParty/mmdeploy/rtmpose-trt/rtmpose-m"
RTMD_DIR="NTK_CAP/ThirdParty/mmdeploy/rtmpose-trt/rtmdet-m"
DEPLOY_SCRIPT="NTK_CAP/ThirdParty/mmdeploy/tools/deploy.py"

mkdir -p $WEIGHTS_DIR
mkdir -p $RTMP_DIR
mkdir -p $RTMD_DIR

echo "--- Step 1: Downloading Model Weights (If missing) ---"

# RTMDet-m (Detector)
DET_PTH="$WEIGHTS_DIR/rtmdet_m_8xb32-300e_coco.pth"
# Check if weights exist locally
if [ ! -f "$DET_PTH" ]; then
    echo "Downloading RTMDet weights..."
    wget -O $DET_PTH https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_m_8xb32-300e_coco/rtmdet_m_8xb32-300e_coco_20220719_112220-529fce50.pth
fi

# RTMPose-m (Pose Estimator)
POSE_PTH="$WEIGHTS_DIR/rtmpose-m_simcc-aic-coco.pth"
if [ ! -f "$POSE_PTH" ]; then
    echo "Downloading RTMPose weights..."
    wget -O $POSE_PTH https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63e7ae62_20230126.pth
fi

echo "--- Step 2: Generating TensorRT Engines (Compiling for local GPU) ---"

# 1. Build Detector Engine (Static 320x320)
# This creates end2end.engine, deploy.json, and pipeline.json in the work-dir
python $DEPLOY_SCRIPT \
    NTK_CAP/ThirdParty/mmdeploy/configs/mmdet/detection/detection_tensorrt_static-320x320.py \
    NTK_CAP/ThirdParty/mmdeploy/ThirdParty/mmdetection/configs/rtmdet/rtmdet_m_8xb32-300e_coco.py \
    $DET_PTH \
    NTK_CAP/ThirdParty/mmdeploy/demo/resources/det.jpg \
    --work-dir $RTMD_DIR \
    --device cuda:0 \
    --show --dump-info

# 2. Build Pose Engine (Static 256x192)
python $DEPLOY_SCRIPT \
    NTK_CAP/ThirdParty/mmdeploy/configs/mmpose/pose-detection/pose-detection_simcc_tensorrt_static-256x192.py \
    NTK_CAP/ThirdParty/mmdeploy/ThirdParty/mmpose/projects/rtmpose/configs/rtmpose/body_2d_keypoint/simcc/coco/rtmpose-m_8xb64-420e_coco-256x192.py \
    $POSE_PTH \
    NTK_CAP/ThirdParty/mmdeploy/demo/resources/human.jpg \
    --work-dir $RTMP_DIR \
    --device cuda:0 \
    --show --dump-info

echo "--- BUILD SUCCESSFUL ---"
echo "Files created in: $RTMD_DIR and $RTMP_DIR"
