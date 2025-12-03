@echo off
echo Setting up TensorRT environment...
set PATH=D:\NTKCAP\NTK_CAP\ThirdParty\TensorRT-8.6.1.6\lib;%PATH%

echo Activating conda environment...
call C:\ProgramData\Miniconda3\Scripts\activate.bat ntkcap_env

cd /d D:\NTKCAP\NTK_CAP\ThirdParty\mmdeploy

echo.
echo ========================================
echo Building RTMDet-m TensorRT model...
echo ========================================
python tools/deploy.py configs/mmdet/detection/detection_tensorrt_static-640x640.py ../mmpose/projects/rtmpose/rtmdet/person/rtmdet_m_640-8xb32_coco-person.py https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth demo/resources/human-pose.jpg --work-dir rtmpose-trt/rtmdet-m --device cuda:0

echo.
echo ========================================
echo Building RTMPose-m TensorRT model...
echo ========================================
python tools/deploy.py configs/mmpose/pose-detection_tensorrt_static-384x288.py ../mmpose/projects/rtmpose/rtmpose/body_2d_keypoint/rtmpose-m_8xb512-700e_body8-halpe26-384x288.py https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7-halpe26_700e-384x288-89e6428b_20230605.pth demo/resources/human-pose.jpg --work-dir rtmpose-trt/rtmpose-m --device cuda:0

echo.
echo ========================================
echo Done! Checking generated files...
echo ========================================
dir rtmpose-trt\rtmdet-m\end2end.engine
dir rtmpose-trt\rtmpose-m\end2end.engine

pause
