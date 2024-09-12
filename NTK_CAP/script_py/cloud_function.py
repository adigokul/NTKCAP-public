import os
import requests
import zipfile
from io import BytesIO
from tkinter import Tk, filedialog
import configparser
import os
from datetime import datetime
import json


"""
能夠在條件 選擇 calculated ，包含upload.conf 檔案的目錄下
掃描屬於 action 的資料夾，並個別上傳這些檔案
"""
host = "https://motion-service.yuyi-ocean.com"
def find_files(apose_dir,patientId,timestring):
   
    # 初始化返回的物件
    trc_file = None
    mot_file = None
    scaled_osim_file = None
    zip_file_path = None
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
    check_exist_ppose2d = 0
    pose_2d_dir = os.path.join(apose_dir, "pose-2d")
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        for root, dirs, files in os.walk(pose_2d_dir):
            for file in files:
                file_path = os.path.join(root, file)
                zip_file.write(file_path, os.path.relpath(file_path, pose_2d_dir))
                check_exist_ppose2d  = check_exist_ppose2d +1
    zip_buffer.seek(0)

    # 保存 zip 文件到指定目錄
    if check_exist_ppose2d>0:
        zip_file_path = os.path.join(apose_dir, "pose2d.zip")
        with open(zip_file_path, "wb") as f:
            f.write(zip_buffer.read())


    
    action_dict = {
        "patientId": patientId,  
        "datetime": timestring,  # Use the provided timestring
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


def choose_directory(calculated_filedir):
    # 建立一個隱形的 Tkinter 視窗
    
    selected_directory = calculated_filedir
    
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
        'PatientId': config.get('DEFAULT', 'PatientId'),
        'Location': config.get('DEFAULT', 'Location'),
        'Datetime': config.get('DEFAULT', 'Datetime')
    }
    return upload_info

def extract_info(dir_path):
    # Split the path into components
    path_components = dir_path.split(os.sep)
    
    # Extract Maurice123 and 2024_05_21 from the path
    patient_name = path_components[-3]
    date_str = path_components[-2]
    
    # Convert date from YYYY_MM_DD to ISO 8601 format with time and timezone
    date_iso = datetime.strptime(date_str, "%Y_%m_%d").isoformat() + 'Z'
    
    # Print the extracted values
    print("Patient Name:", patient_name)
    print("Date:", date_iso)
    return patient_name, date_iso

def extract_location(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # Extract and print the location
    location = data["location"][0]
    print(location)
    return location

def create_config(patient_name, date_iso, location, config_file_path):
    config = configparser.ConfigParser()
    
    config['DEFAULT'] = {
        'PatientId': patient_name,
        'Location': location,
        'Datetime': date_iso
    }
    
    with open(config_file_path, 'w') as configfile:
        config.write(configfile)
    print(f"Configuration file created at {config_file_path}")



def marker_calculate_upload(calculated_filedir, json_file_path):
    
    
    try:
        patient_name, date_iso = extract_info(calculated_filedir)
    except Exception as e:
        print(f"Error extracting info from {calculated_filedir}: {e}")
        return

    try:
        location = extract_location(json_file_path)
    except Exception as e:
        print(f"Error extracting location from {json_file_path}: {e}")
        return

    try:
        create_config(patient_name, date_iso, location, os.path.join(calculated_filedir, 'upload.conf'))
    except Exception as e:
        print(f"Error creating config file in {calculated_filedir}: {e}")
        return

    try:
        base_dir, conf_path = choose_directory(calculated_filedir)
    except Exception as e:
        print(f"Error choosing directory from {calculated_filedir}: {e}")
        return

    try:
        conf = read_upload_conf(conf_path)
        print(conf)
    except Exception as e:
        print(f"Error reading upload config from {conf_path}: {e}")
        return

    try:
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
    except Exception as e:
        print(f"Error uploading in {base_dir}: {e}")


def meet_postupdate(dir_layout,dir_notevalue,dir_location,patientId,timestring):
    message = []
    with open(dir_layout , 'r', encoding='utf-8') as f:
        layout = json.load(f)
    with open(dir_notevalue , 'r', encoding='utf-8') as f:
        value = json.load(f)
    with open(dir_location , 'r', encoding='utf-8') as f:
        location = json.load(f)
    url =f"{host}/api/layouts/layoutId/"+ layout["meet_layoutId"]
    response = requests.get(url)
    layout = response.json()
    value = [{'title': item['title'], 'content': item['content']} for item in value]
    location = location['location'][0]
    patientId
    output = {
    "datetime": timestring,  # Use the provided timestring
    "location": location,  # Use the provided location
    "patientId": patientId,  # Use the provided patientId
    "prescription": None,  # Set to None as per your example
    "layout": layout,  # Make sure layout is a properly structured variable
    "notes": [
        {
            "title": item["title"],  # Keep the title as is from value
            "content": [] if ((item["content"] == "") or (item["content"] == "Choose an option")) else [item["content"]]  # Handle empty and non-empty content
        } for item in value  # Iterate through the value list to generate notes
    ]
    }
    output['layout']['fields'][8]['title']
    output["notes"][8]['title']
    # Print the JSON output for debugging
    print(json.dumps(output, indent=4))
    #import pdb;pdb.set_trace()
    url = f"{host}/api/meets"
    response = requests.post(url, json=output)
    #if response.text["message"]
    if response.status_code ==201:
        meetId = response.headers["Location"].split('/')[-1]
        print('Successfully create a meet')
        #import pdb;pdb.set_trace()
    while response.status_code == 400:
        message =  json.loads(response.text)
        message = message['message']
        if message == 'The datetime with location is already exist.':
            findmeetIdtemp = requests.get(url+'?patientId='+str(patientId))
            findmeetIdtemp= findmeetIdtemp.json()
            findmeetIdtemp['resources'][0]
            matching_records = [item for item in findmeetIdtemp['resources'] if item['datetime'] == timestring] 
            meetId = matching_records[0]['id']
            del output['datetime']
            del output['location']
            del output['patientId']
            output["description"] = None
            output["meetId"] = str(meetId)
            output = {
                "id":output["meetId"],
                "prescription": output['prescription'],
                "description": output['description'],
                "layout": output['layout'],
                "notes": output['notes']
            }           
            response = requests.put(url,json = output)
            print(response.status_code)
            response.text
            print('Successfully create a meet')
            #update
        else:
            #error 
            print("fail to create meet note " +message)
            break

    return response.status_code,message,meetId
def action_postupdate(dir_layout,dir_notevalue,dir_location,patientId,timestring,actionname,meetId):
    message = []
    if actionname!='Apose':
        
        with open(dir_layout , 'r', encoding='utf-8') as f:
            layout = json.load(f)
        with open(dir_notevalue , 'r', encoding='utf-8') as f:
            value = json.load(f)
        with open(dir_location , 'r', encoding='utf-8') as f:
            location = json.load(f)
        url =f"{host}/api/layouts/layoutId/"+ layout["action_layoutId"]
        response = requests.get(url)
        layout = response.json()
        value = [{'title': item['title'], 'content': item['content']} for item in value]
        location = location['location'][0]
        patientId
        output = {
        "patientId": patientId,
        "datetime": timestring,  # Use the provided timestring
        "location": location,  # Use the provided location
        "actionName": actionname,
        "layout": layout,  # Make sure layout is a properly structured variable
        "notes": [
            {
                "title": item["title"],  # Keep the title as is from value
                "content": [] if ((item["content"] == "") or (item["content"] == "Choose an option")) else [item["content"]]  # Handle empty and non-empty content
            } for item in value  # Iterate through the value list to generate notes
        ]
    }
    else:
        with open(dir_layout , 'r', encoding='utf-8') as f:
            layout = json.load(f)
        with open(dir_notevalue , 'r', encoding='utf-8') as f:
            value = json.load(f)
        with open(dir_location , 'r', encoding='utf-8') as f:
            location = json.load(f)
        url =f"{host}/api/layouts/layoutId/"+ layout["action_layoutId"]
        response = requests.get(url)
        layout = response.json()
        value = [{'title': item['title'], 'content': item['content']} for item in value]
        location = location['location'][0]
        patientId
        output = {
        "patientId": patientId,
        "datetime": timestring,  # Use the provided timestring
        "location": location,  # Use the provided location
        "actionName": actionname,
        "layout": layout,  # Make sure layout is a properly structured variable
        "notes": [
            {
                "title": item["title"],  # Keep the title as is from value
                "content": [] # Handle empty and non-empty content
            } for item in value  # Iterate through the value list to generate notes
        ]
    }
    #import pdb;pdb.set_trace()
    #### check if calculated folder exist
    
    #import pdb;pdb.set_trace()
    url = f"{host}/api/actions"
    response = requests.post(url,json=output)
    response.status_code
    response.text
    if response.status_code ==201:
        actionId = response.headers["Location"].split('/')[-1]
        print('Successfully create an action')
        #import pdb;pdb.set_trace()
    
    while response.status_code == 400:
        message =  json.loads(response.text)
        message = message['message']
        
        if message == "action is created.":
            findmeetIdtemp = requests.get(f"{host}/api/meets/" +str(meetId) +'/actions')
            
            findmeetIdtemp= findmeetIdtemp.json()
            findmeetIdtemp['resources'][0]
            matching_records = [item for item in findmeetIdtemp['resources'] if item['datetime'] == timestring] 
            actionId = matching_records[0]['id']
            del output['datetime']
            del output['location']
            del output['patientId']
            del output['actionName']
            
            output["actionId"] = actionId
            output = {
                "actionId":output["actionId"],
                "layout": output['layout'],
                "notes": output['notes']
            }           
            response = requests.put(f"{host}/api/actions",json = output)
            print(response.status_code)
            response.text
            print('Successfully create a action')
            #update
        else:
            #error 
            print("fail to create action note " +message)
            break

    return response.status_code,message,actionId
def meet_update(dir_layout,dir_notevalue,meetId):
    with open(dir_layout , 'r', encoding='utf-8') as f:
        layout = json.load(f)
    with open(dir_notevalue , 'r', encoding='utf-8') as f:
        value = json.load(f)
    url =f"{host}/api/layouts/layoutId/"+ layout["meet_layoutId"]
    response = requests.get(url)
    layout = response.json()
    value = [{'title': item['title'], 'content': item['content']} for item in value]
    location = location['location'][0]
    output = {
    "id" : str(meetId),
    "prescription": None,  # Set to None as per your example
    "description" : None,
    "layout": layout,  # Make sure layout is a properly structured variable
    "notes": [
        {
            "title": item["title"],  # Keep the title as is from value
            "content": [] if ((item["content"] == "") or (item["content"] == "Choose an option")) else [item["content"]]  # Handle empty and non-empty content
        } for item in value  # Iterate through the value list to generate notes
    ]
    }
    url = f"{host}/api/meets"
    response = requests.put(url,json = output)
    if response.status_code == 200:
        print('Successfully update a meet')
    else:
        message =  json.loads(response.text)
        message = message['message']
        print("fail to update meet note " +message)
def action_update(dir_layout,dir_notevalue,dir_location,timestring,meetId):
    with open(dir_layout , 'r', encoding='utf-8') as f:
        layout = json.load(f)
    with open(dir_notevalue , 'r', encoding='utf-8') as f:
        value = json.load(f)
    with open(dir_location , 'r', encoding='utf-8') as f:
        location = json.load(f)
    url =f"{host}/api/layouts/layoutId/"+ layout["action_layoutId"]
    response = requests.get(url)
    layout = response.json()
    value = [{'title': item['title'], 'content': item['content']} for item in value]
    location = location['location'][0]
    findmeetIdtemp = requests.get(f"{host}/api/meets/" +str(meetId) +'/actions')
    findmeetIdtemp= findmeetIdtemp.json()
    findmeetIdtemp['resources'][0]
    matching_records = [item for item in findmeetIdtemp['resources'] if item['datetime'] == timestring] 
    actionId = matching_records[0]['id']
    output = {
    "actionId":actionId ,
    "layout": layout,  # Make sure layout is a properly structured variable
    "notes": [
        {
            "title": item["title"],  # Keep the title as is from value
            "content": [] if ((item["content"] == "") or (item["content"] == "Choose an option")) else [item["content"]]  # Handle empty and non-empty content
        } for item in value  # Iterate through the value list to generate notes
    ]
}   
    response = requests.put(f"{host}/api/actions",json = output)
    print(response.status_code)
    response.text
    print('Successfully update a action')
    return response.status_code,actionId

dir_layout=r'C:\Users\mauricetemp\Desktop\NTKCAP\config\layout.json'
dir_notevalue=r'C:\Users\mauricetemp\Desktop\NTKCAP\config\meetnote_layout.json'
dir_location=r'C:\Users\mauricetemp\Desktop\NTKCAP\config\location.json'
patientId = "66d94d6626a882267fa69252"
date_str = datetime.now().strftime("%Y_%m_%d")
date_str='2024-09-15T00:00:00Z'
#datetime.strptime(date_str, "%Y_%m_%d").isoformat() + 'Z'
#timestring = datetime.now().strftime("%Y-%m-%dT%H:%M:%S") + 'Z'
#meet_postupdate(dir_layout,dir_notevalue,dir_location,patientId,date_str)
#import pdb;pdb.set_trace()
# meet_id = "12345"
# Example usage
# calculated_filedir = r'C:\Users\mauricetemp\Desktop\NTKCAP\Patient_data\66d94d6626a882267fa69252\2024_05_07\2024_09_03_16_47_calculated'
# json_file_path = r'C:\Users\mauricetemp\Desktop\NTKCAP\config\location.json'
# marker_calculate_upload(calculated_filedir, json_file_path)



# Replace 'path_to_json_file' with the actual path to your JSON file
dir_layout=r'C:\Users\mauricetemp\Desktop\NTKCAP\config\layout.json'
dir_notevalue=r'C:\Users\mauricetemp\Desktop\NTKCAP\config\actionnote_layout.json'
dir_location=r'C:\Users\mauricetemp\Desktop\NTKCAP\config\location.json'
meetId ='66e15635bd48a32ea268b0f2'
#action_postupdate(dir_layout,dir_notevalue,dir_location,patientId,date_str,'walk1',meetId)

#action_update(dir_layout,dir_notevalue,dir_location,date_str,meetId)
#marker_calculate_upload