#!/bin/bash
set -e
WEIGHTS_DIR="models/weights"
RTMP_DIR="NTK_CAP/ThirdParty/mmdeploy/rtmpose-trt/rtmpose-m"
RTMD_DIR="NTK_CAP/ThirdParty/mmdeploy/rtmpose-trt/rtmdet-m"
DEPLOY_SCRIPT="NTK_CAP/ThirdParty/mmdeploy/tools/deploy.py"

mkdir -p $WEIGHTS_DIR $RTMP_DIR $RTMD_DIR

echo "Downloading weights..."
wget -nc -O $WEIGHTS_DIR/rtmdet_m.pth https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_m_8xb32-300e_coco/rtmdet_m_8xb32-300e_coco_20220719_112220-529fce50.pth
wget -nc -O $WEIGHTS_DIR/rtmpose_m.pth https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63e7ae62_20230126.pth

echo "Building Engines..."
python $DEPLOY_SCRIPT NTK_CAP/ThirdParty/mmdeploy/configs/mmdet/detection/detection_tensorrt_static-320x320.py NTK_CAP/ThirdParty/mmdeploy/ThirdParty/mmdetection/configs/rtmdet/rtmdet_m_8xb32-300e_coco.py $WEIGHTS_DIR/rtmdet_m.pth NTK_CAP/ThirdParty/mmdeploy/demo/resources/det.jpg --work-dir $RTMD_DIR --device cuda:0
python $DEPLOY_SCRIPT NTK_CAP/ThirdParty/mmdeploy/configs/mmpose/pose-detection/pose-detection_simcc_tensorrt_static-256x192.py NTK_CAP/ThirdParty/mmdeploy/ThirdParty/mmpose/projects/rtmpose/configs/rtmpose/body_2d_keypoint/simcc/coco/rtmpose-m_8xb64-420e_coco-256x192.py $WEIGHTS_DIR/rtmpose_m.pth NTK_CAP/ThirdParty/mmdeploy/demo/resources/human.jpg --work-dir $RTMP_DIR --device cuda:0
