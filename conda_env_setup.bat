set root=C:\Users\user\anaconda3
call %root%\Scripts\activate.bat %root%

call conda activate pose2
call pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
call cd .\NTK_CAP\ThirdParty\EasyMocap
call python setup.py develop --user
call pip install -U openmim
call mim install mmengine
call mim install "mmcv>=2.0.1"
call mim install "mmdet>=3.3.0"
call cd mmpose
call pip install -r requirements.txt
call pip install -v -e .
call cd ..
call pip install bs4
call pip install multiprocess
call pip install keyboard
call pip install import_ipynb
call pip install kivy
call pip install "Pose2Sim==0.4"
call pip install numpy==1.22.4
call pip install ultralytics
call pip install tkfilebrowser