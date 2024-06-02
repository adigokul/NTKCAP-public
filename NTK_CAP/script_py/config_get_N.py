import wmi
import sys
import json

wmi_obj = wmi.WMI()

if len(sys.argv) > 1:
        file_path = sys.argv[1]
        print("config path: " + file_path)

# 查询所有USB裝置
video_devices = wmi_obj.Win32_PnPEntity(ClassGuid='{4d36e96c-e325-11ce-bfc1-08002be10318}')

# 打开第一个视频设备
cam_num = 0
for i in video_devices:
    video_device = i
    camera_name = video_device.Name
    if camera_name == "HD Camera":
        print(f"偵測到廣角相機: {camera_name}")
        cam_num = cam_num + 1

print("偵測到{0}台廣角相機相機".format(cam_num))

# 读取JSON文件
with open(file_path, 'r') as f:
    data = json.load(f)

# 更新字段值
data['cam']['number'] = cam_num

# 写入更新后的JSON文件
with open(file_path, 'w') as f:
    json.dump(data, f, indent=4)
