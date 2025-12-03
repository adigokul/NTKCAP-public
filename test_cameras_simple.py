"""
Simple camera test script - bypasses NVIDIA Broadcast issues
使用 Windows Media Foundation backend 代替 DirectShow
"""
import cv2
import os

# 設定環境變數來禁用 NVIDIA Broadcast
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '1'
os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'

print("=" * 50)
print("OpenCV Camera Detection Test")
print("=" * 50)
print(f"OpenCV version: {cv2.__version__}")
print()

# 測試不同的 backend
backends = [
    (cv2.CAP_MSMF, "Windows Media Foundation (MSMF)"),
    (cv2.CAP_DSHOW, "DirectShow"),
    (cv2.CAP_ANY, "Auto-detect")
]

for backend_id, backend_name in backends:
    print(f"\n嘗試使用 {backend_name}...")
    working_cameras = []
    
    for i in range(10):
        try:
            cap = cv2.VideoCapture(i, backend_id)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    working_cameras.append({
                        'index': i,
                        'resolution': f"{int(width)}x{int(height)}",
                        'fps': fps
                    })
                    print(f"  ✅ Camera {i}: {int(width)}x{int(height)} @ {fps:.1f} FPS")
                cap.release()
        except Exception as e:
            pass
    
    if working_cameras:
        print(f"\n使用 {backend_name} 找到 {len(working_cameras)} 個可用攝像頭")
        break
    else:
        print(f"  ❌ 沒有找到可用攝像頭")

print("\n" + "=" * 50)
if working_cameras:
    print(f"總結: 找到 {len(working_cameras)} 個可用攝像頭")
    print("建議使用索引:", [cam['index'] for cam in working_cameras])
else:
    print("錯誤: 沒有找到任何可用攝像頭")
    print("\n可能原因:")
    print("1. 攝像頭被其他程式佔用")
    print("2. NVIDIA Broadcast 正在運行")
    print("3. 驅動程式問題")
    print("4. USB 連接問題")
print("=" * 50)
