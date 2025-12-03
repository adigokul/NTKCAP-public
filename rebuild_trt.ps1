# Set TensorRT and cuDNN (from PyTorch) in PATH
$env:PATH = "D:\NTKCAP\NTK_CAP\ThirdParty\TensorRT-8.6.1.6\lib;C:\ProgramData\Miniconda3\envs\ntkcap_env\Lib\site-packages\torch\lib;" + $env:PATH

# Activate conda environment
& "C:\ProgramData\Miniconda3\shell\condabin\conda-hook.ps1"
conda activate ntkcap_env

# Change to mmdeploy directory
Set-Location "D:\NTKCAP\NTK_CAP\ThirdParty\mmdeploy"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Building RTMDet-m TensorRT model..." -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

python tools/deploy.py configs/mmdet/detection/detection_tensorrt_static-640x640.py `
    ../mmpose/projects/rtmpose/rtmdet/person/rtmdet_m_640-8xb32_coco-person.py `
    https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth `
    demo/resources/human-pose.jpg `
    --work-dir rtmpose-trt/rtmdet-m `
    --device cuda:0

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Building RTMPose-m TensorRT model..." -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

python tools/deploy.py configs/mmpose/pose-detection_tensorrt_static-384x288.py `
    ../mmpose/projects/rtmpose/rtmpose/body_2d_keypoint/rtmpose-m_8xb512-700e_body8-halpe26-384x288.py `
    https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7-halpe26_700e-384x288-89e6428b_20230605.pth `
    demo/resources/human-pose.jpg `
    --work-dir rtmpose-trt/rtmpose-m `
    --device cuda:0

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "Done! Checking generated files..." -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

Get-ChildItem "rtmpose-trt\rtmdet-m\end2end.engine" | Select-Object Name, Length, LastWriteTime
Get-ChildItem "rtmpose-trt\rtmpose-m\end2end.engine" | Select-Object Name, Length, LastWriteTime
