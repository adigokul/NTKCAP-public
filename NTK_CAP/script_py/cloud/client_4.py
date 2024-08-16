import os
import requests
import zipfile
from io import BytesIO
from tkinter import Tk, filedialog
import configparser


"""
能夠在條件 選擇 calculated ，包含upload.conf 檔案的目錄下
掃描屬於 action 的資料夾，並個別上傳這些檔案
"""

def find_files(apose_dir):
    # 初始化返回的物件
    trc_file = None
    mot_file = None
    scaled_osim_file = None

    last_name = os.path.basename(apose_dir)
    print(last_name)

    # 查找 opensim 目錄下的 .trc 和 .osim 和 .mot 文件
    opensim_dir = os.path.join(apose_dir, "opensim")
    for file_name in os.listdir(opensim_dir):
        if file_name.endswith(".trc"):
            trc_file = os.path.join(opensim_dir, file_name)
        elif file_name.endswith("scaled.osim"):
            scaled_osim_file = os.path.join(opensim_dir, file_name)
        elif file_name.endswith(".mot"):
            mot_file = os.path.join(opensim_dir, file_name)

    # 查找 pose-2d 目錄下的所有仔文件夾並壓縮
    pose_2d_dir = os.path.join(apose_dir, "pose-2d")
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        for root, dirs, files in os.walk(pose_2d_dir):
            for file in files:
                file_path = os.path.join(root, file)
                zip_file.write(file_path, os.path.relpath(file_path, pose_2d_dir))
    zip_buffer.seek(0)

    # 保存 zip 文件到指定目錄
    zip_file_path = os.path.join(apose_dir, "pose2d.zip")
    with open(zip_file_path, "wb") as f:
        f.write(zip_buffer.read())


    
    action_dict = {
        "actionName": last_name,
        "trc": trc_file,
        "mot": mot_file,
        "osim": scaled_osim_file,
        "zip": zip_file_path,
    }

    # Check for missing data in one condition
    # if not all(action_dict.values()):
    #     missing_data = [key for key, value in action_dict.items() if not value]
    #     message = f"Missing required data: {', '.join(missing_data)}"
    #     print(f"Warning: {message}")
    #     return None
    
    return action_dict


def choose_directory():
    # 建立一個隱形的 Tkinter 視窗
    root = Tk()
    root.withdraw()  # 隐藏主視窗
    root.call("wm", "attributes", ".", "-topmost", True)  # 把視窗置頂

    # 打開檔案選擇對話框
    selected_directory = filedialog.askdirectory(title="請選擇動作目錄")

    if selected_directory:
        conf_path = os.path.join(selected_directory, "upload.conf")
        if os.path.isfile(conf_path):
            print("選擇的目錄:", selected_directory)
            return selected_directory, conf_path
        else:
            print("選擇目錄中未找到上傳設定檔 upload.conf")
            return None
    else:
        print("未選擇任何目錄")
        return None


def post_meet_action(host, conf, action_dict):
    print(action_dict)
    url = f"{host}/api/meets/actions"

    # 建立表單數據
    files = {
        "patientId": (None, conf['PatientId'] ),  # None 表示這是一個普通的表單字段
        "location": (None, conf['Location']), 
        "datetime": (None, conf['Datetime']), 
        "actionName": (None, action_dict['actionName']),
        "osim": (
            "osim_file.osim",
            open(action_dict["osim"], "rb"),
            "application/octet-stream",
        ),
        "json2d": (
            "json2d.zip",
            open(action_dict["zip"], "rb"),
            "application/octet-stream",
        ),
        "mot": (
            "mot_file.mot",
            open(action_dict["mot"], "rb") if action_dict.get("mot") else None,
            "application/octet-stream",
        ),
        "trc": (
            "trc_file.trc",
            open(action_dict["trc"], "rb") if action_dict.get("trc") else None,
            "application/octet-stream",
        ),
    }
    #import pdb;pdb.set_trace()
    # 發送 POST 請求
    response = requests.post(url, files=files)

    if response.status_code == 200:
        print("Success:", response)
    else:
        print("Failed:", response.status_code, response.text)

def read_upload_conf(conf_path):
    config = configparser.ConfigParser()
    config.read(conf_path)
    # 解析配置文件中的内容
    upload_info = {
        'PatientId': config.get('DEFAULT', 'patientId'),
        'Location': config.get('DEFAULT', 'Location'),
        'Datetime': config.get('DEFAULT', 'Datetime')
    }
    return upload_info


if __name__ == "__main__":
    host = "https://motion-service.yuyi-ocean.com"
    # meet_id = "12345"

    base_dir, conf_path = choose_directory()

    conf = read_upload_conf(conf_path)
    print(conf)

    for action_name in os.listdir(base_dir):
        action_dir = os.path.join(base_dir, action_name)
        if os.path.isdir(action_dir):
            print(action_name)
            action_dict = find_files(action_dir)
            print("TRC Files:", action_dict['trc'])
            print("MOT Files:", action_dict['mot'])
            print("Scaled OSIM File:", action_dict['osim'])
            print("ZIP Buffer Size:", action_dict['zip'])                
            post_meet_action(host, conf, action_dict)