import os
import requests
import zipfile
from io import BytesIO
from tkinter import Tk, filedialog
import configparser
import os
from datetime import datetime
import json
from natsort import natsorted

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

def action_postupdate(dir_layout,dir_notevalue,dir_location,patientId,timestring,actionname,tasktype,meetId):
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
        "taskType": tasktype,
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
        "taskType": tasktype,
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
    #url = f"{host}/api/meets/{meetId}"
    #response = requests.get(url)
    response.status_code
    response.text
    r''''{"id":"66ea45f6f43367445fa25231","datetime":"2024-09-18T00:00:00Z","location":"NYCU402-1","patient":{"id":"667d29d0cfee0b0977061967","phone":"0910101010","name":"黃亮維","sex":"F","birthday":"2024-05-26T16:00:00Z","height":0.0,"weight":0.0,"careNumber":null,"note":null},"actions":[],"prescription":null,"description":null,"layout":{"id":"66bdd8045421d11d1f523e15","layoutId":"common_01","catalog":"meet","createTime":"2024-08-15T10:27:16.431Z","fields":[{"catalog":"meet","elementType":"input","title":"Task number","options":null},{"catalog":"meet","elementType":"spinner","title":"Task outin","options":["Inside camera","walk from outside"]},{"catalog":"meet","elementType":"spinner","title":"Facing","options":["Door","Window"]},{"catalog":"meet","elementType":"input","title":"Symptoms","options":null},{"catalog":"meet","elementType":"input","title":"Fall Risk Level","options":null},{"catalog":"meet","elementType":"input","title":"Temperature(C)","options":null},{"catalog":"meet","elementType":"input","title":"Level of Mood","options":null},{"catalog":"meet","elementType":"spinner","title":"Treatment Phase","options":["Pre phase","Post-injection","Post-trainging"]},{"catalog":"meet","elementType":"spinner","title":"Cloth Color","options":["Red","Orange","Yellow","Green","Blue","Purple"]}]},"notes":[{"title":"Task number","content":[]},{"title":"Task outin","content":[]},{"title":"Facing","content":[]},{"title":"Symptoms","content":["I23: Certain current complications following ST elevation (STEMI) and non-ST elevation (NSTEMI) myocardial infarction (within the 28 day period) / ST段上升之心肌梗塞 (STEMI）與非ST段上升之心肌梗塞（NSTEMI）後造成之併發症（28天內）"]},{"title":"Fall Risk Level","content":["12"]},{"title":"Temperature(C)","content":[]},{"title":"Level of Mood","content":[]},{"title":"Treatment Phase","content":["Post-trainging"]},{"title":"Cloth Color","content":[]}]}'
    '''
    #import pdb;pdb.set_trace()
    #import pdb;pdb.set_trace()
    if response.status_code ==201:
        actionId = response.headers["Location"].split('/')[-1]
        print('Successfully create an action')
        #import pdb;pdb.set_trace()
    while response.status_code == 400:
        message =  json.loads(response.text)
        message = message['message']
        #import pdb;pdb.set_trace()
        if message == "action is created.":
            findmeetIdtemp = requests.get(f"{host}/api/meets/" +str(meetId) +'/actions')
            #import pdb;pdb.set_trace()
            findmeetIdtemp= findmeetIdtemp.json()
            findmeetIdtemp['resources'][0]
            matching_records = [item for item in findmeetIdtemp['resources'] if item['actionName'] == actionname] 
            actionId = matching_records[0]['id']
            del output['datetime']
            del output['location']
            del output['patientId']
            
            output["actionId"] = actionId
            output = {
                "actionId":output["actionId"],
                "layout": output['layout'],
                "taskType": tasktype,
                "notes": output['notes']
            }           
            response = requests.put(f"{host}/api/actions",json = output)
            print(response.status_code)
            response.text
            print('Successfully create a action')
            #pdate
        else:
            #error 
            print("fail to create action note " +message)
            break

    return response.status_code,message,actionId
def meet_update(dir_layout,dir_notevalue,meetId):
    message =''
    with open(dir_layout , 'r', encoding='utf-8') as f:
        layout = json.load(f)
    with open(dir_notevalue , 'r', encoding='utf-8') as f:
        value = json.load(f)
    url =f"{host}/api/layouts/layoutId/"+ layout["meet_layoutId"]
    response = requests.get(url)
    layout = response.json()
    value = [{'title': item['title'], 'content': item['content']} for item in value]
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
    return response.status_code,message
def action_update(dir_layout,dir_notevalue,actionId,tasktype,actionname,samefoldercheck):
    
    message = ''
    with open(dir_layout , 'r', encoding='utf-8') as f:
        layout = json.load(f)
    with open(dir_notevalue , 'r', encoding='utf-8') as f:
        value = json.load(f)
   
    url =f"{host}/api/layouts/layoutId/"+ layout["action_layoutId"]
    response = requests.get(url)
    layout = response.json()
    value = [{'title': item['title'], 'content': item['content']} for item in value]
    actionId
    if samefoldercheck == False:
        output = {
        "actionId":actionId ,
        "layout": layout,  # Make sure layout is a properly structured variable
        "taskType": tasktype,
        "actionName":actionname,
        "notes": [
            {
                "title": item["title"],  # Keep the title as is from value
                "content": [] if ((item["content"] == "") or (item["content"] == "Choose an option")) else [item["content"]]  # Handle empty and non-empty content
            } for item in value  # Iterate through the value list to generate notes
        ]
    }   
    else:
        output = {
        "actionId":actionId ,
        "layout": layout,  # Make sure layout is a properly structured variable
        "taskType": tasktype,
        "notes": [
            {
                "title": item["title"],  # Keep the title as is from value
                "content": [] if ((item["content"] == "") or (item["content"] == "Choose an option")) else [item["content"]]  # Handle empty and non-empty content
            } for item in value  # Iterate through the value list to generate notes
        ]
    }   
    response = requests.put(f"{host}/api/actions",json = output)
    print(response.status_code)
    
    if response.status_code == 200:
        print('Successfully update a action')
    else:
        message =  json.loads(response.text)
        message = message['message']
        print("fail to update action note " +message)
    return response.status_code,message 

def MeetActionID2json(meetId, actionId, outputdir):
    # Fetch action and meet data from API
    response = requests.get(f"{host}/api/actions/" + actionId)
    actionoutput = response.json()
    
    response = requests.get(f"{host}/api/meets/" + meetId)
    meetoutput = response.json()

    # Function to get content from notes
    def get_content_from_notes(notes, title, element_type):
        for note in notes:
            if note['title'] == title:
                # Return "Choose an option" if it's a spinner, or an empty string if it's an input
                if element_type == "spinner":
                    return note['content'][0] if note['content'] else "Choose an option"
                elif element_type == "input":
                    return note['content'][0] if note['content'] else ""
        return "Choose an option" if element_type == "spinner" else ""

    # Transform the meet and action data into the desired format
    def transform_data(meet_data, action_data):
        # Initial structure
        new_data = {
            "meet_layoutId": meet_data['layout']['layoutId'],
            "action_layoutId": action_data['layout']['layoutId'],
            "fields": []
        }

        # Process meet data fields
        for field in meet_data['layout']['fields']:
            # Get content from notes based on title and element type
            content = get_content_from_notes(meet_data['notes'], field['title'], field['elementType'])
            
            new_field = {
                "type": field['elementType'],
                "title": field['title'],
                "content": content,  # Set content from notes
                "notetype": "meet"
            }

            # Add options if they exist
            if field.get('options'):
                new_field['options'] = field['options']

            new_data["fields"].append(new_field)

        # Process action data fields
        for field in action_data['layout']['fields']:
            # Get content from notes based on title and element type
            content = get_content_from_notes(action_data['notes'], field['title'], field['elementType'])
            
            new_field = {
                "type": field['elementType'],
                "title": field['title'],
                "content": content,  # Set content from notes
                "notetype": "action"
            }

            # Add options if they exist
            if field.get('options'):
                new_field['options'] = field['options']

            new_data["fields"].append(new_field)

        return new_data

    # Transform the data and write to output file
    new_json_data = transform_data(meetoutput, actionoutput)
    
    with open(outputdir, 'w', encoding='utf-8') as json_file:
        json.dump(new_json_data, json_file, indent=4, ensure_ascii=False)
def getTasktype(actionId=None):
    content = []
    url =f"{host}/api/taskTypes"
    response = requests.get(url)
    response = response.json()
    task_types = [item['taskType'] for item in response['resources'] if item!='Apose']
    #import pdb;pdb.set_trace()
    if actionId:
        url = f"{host}/api/actions/"+ actionId
        response =requests.get(url)
        response = response.json()
        content = response['taskType']
    return task_types,content
    #import pdb;pdb.set_trace()
def getTasknumber(actionId):
    url =f"{host}/api/actions/{actionId}"
    response =requests.get(url)
    response = response.json()
    #import pdb;pdb.set_trace()
    task_number = response['actionName'].split('_')[-1]

    return task_number
def upload_calculated_file(actions_Id,dir_calculated):
    for action_name,id in actions_Id:

        files = {
            "actionId": (None,id ),  # None 表示這是一個普通的表單字段
            "osim": (
                "osim_file.osim",
                open(os.path.join(dir_calculated,action_name,'opensim','Model_Pose2Sim_Halpe26_scaled.osim'), "rb"),
                "application/octet-stream",
            ),
            "json2d": (
                "json2d.zip",
                open(os.path.join(dir_calculated,action_name,'pose-2d','pose2d.zip'), "rb"),
                "application/octet-stream",
            ),
            
            "trc": (
                "trc_file.trc",
                open(os.path.join(dir_calculated,action_name,'opensim','Empty_project_filt_0-30.trc'), "rb") ,
                "application/octet-stream",
            ),
        }
        if os.path.exists(os.path.join(dir_calculated,action_name,'opensim','Balancing_for_IK_BODY.mot')):
            files["mot"] = (
                "mot_file.mot",
                open(os.path.join(dir_calculated,action_name,'opensim','Balancing_for_IK_BODY.mot'), "rb") ,
                "application/octet-stream",
            ),

        response = requests.put(f"{host}/api/actions/files", files=files)
        import pdb;pdb.set_trace()

def check_zip_pose2d(dir_action):
    token = 0
    # 查找 pose-2d 目錄下的所有仔文件夾並壓縮
    check_exist_ppose2d = 0
    pose_2d_dir = os.path.join(dir_action, "pose-2d")
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        for root, dirs, files in os.walk(pose_2d_dir):
            for file in files:
                file_path = os.path.join(root, file)
                zip_file.write(file_path, os.path.relpath(file_path, pose_2d_dir))
                check_exist_ppose2d  = check_exist_ppose2d +1
    zip_buffer.seek(0)
    #import pdb;pdb.set_trace()
    # 保存 zip 文件到指定目錄
    if check_exist_ppose2d>0:
        zip_file_path = os.path.join(pose_2d_dir, "pose2d.zip")
        token = 1
        with open(zip_file_path, "wb") as f:
            f.write(zip_buffer.read())
    return token
def check_calculated_file(response,calculated_folders,dir_date):
    files_notcalculated = []
    # List to store actionNames where any of 'mot', 'json2D', 'osim', or 'trc' is None
    incomplete_actions = []
    
    # Iterate through each action in the response
    for action in response['actions']:
        # Check if any of 'mot', 'json2D', 'osim', or 'trc' is None
        if action =='Apose':
            if action['json2D'] is None or action['osim'] is None or action['trc'] is None:
                incomplete_actions.append([action['actionName'],action['id']])
        else:
            if action['mot'] is None or action['json2D'] is None or action['osim'] is None or action['trc'] is None:
                incomplete_actions.append([action['actionName'],action['id']])
    
    for calculated_dir in calculated_folders:
        temp = os.path.join(dir_date,calculated_dir)
        token = 1
        #import pdb;pdb.set_trace()
        for action ,id in incomplete_actions:
            if action =='Apose':
                if os.path.exists(os.path.join(temp,action))==0 or check_zip_pose2d(os.path.join(temp,action))==0 or os.path.exists(os.path.join(temp,action,'opensim','Empty_project_filt_0-30.trc'))==0 or os.path.exists(os.path.join(temp,action,'opensim','Model_Pose2Sim_Halpe26_scaled.osim'))==0:
                    token=0
            else:
                if os.path.exists(os.path.join(temp,action))==0 or check_zip_pose2d(os.path.join(temp,action))==0 or os.path.exists(os.path.join(temp,action,'opensim','Balancing_for_IK_BODY.mot'))==0 or os.path.exists(os.path.join(temp,action,'opensim','Empty_project_filt_0-30.trc'))==0 or os.path.exists(os.path.join(temp,action,'opensim','Model_Pose2Sim_Halpe26_scaled.osim'))==0:
                    token=0
            #import pdb;pdb.set_trace()    
        if token==1:
            upload_calculated_file(incomplete_actions,temp)
            break
        
    return files_notcalculated
# def check_actionname(actionId,actionname):
def checkactioname(raw_data_dir,response,filtered_folders,filtered_actionID):
    actions = response['actions']
    id_action_pairs = [(item['id'], item['actionName']) for item in actions]
    # List of tuples (id, actionName)

    # Check correspondence
    non_corresponding_pairs = []

    for action_id, folder in zip(filtered_actionID, filtered_folders):
        if (action_id, folder) not in id_action_pairs:
            # Save the action_id and folder if they don't correspond
            output ={
                'actionName':folder
            }
            non_corresponding_pairs.append((action_id, folder))
    non_corresponding_pairs_with_correct = []

    # Step 1: Rename folders to correct names with 'temp' suffix
    for action_id, folder in non_corresponding_pairs:
        # Find the correct folder name for the action_id in id_action_pairs
        correct_folder = next((correct_folder for id, correct_folder in id_action_pairs if id == action_id), None)
        
        if correct_folder:
            # Save the action_id, incorrect folder, and correct folder in the new list
            non_corresponding_pairs_with_correct.append((action_id, folder, correct_folder))
            old_folder_path = os.path.join(raw_data_dir, folder)
            new_folder_path_temp = os.path.join(raw_data_dir, correct_folder + 'temp')
            
            # Rename the folder if it exists
            if os.path.exists(old_folder_path):
                os.rename(old_folder_path, new_folder_path_temp)
                print(f"Renamed folder '{folder}' to '{correct_folder}temp'")

    # Step 2: Remove 'temp' suffix to ensure correct folder name
    for action_id, wrong_folder, correct_folder in non_corresponding_pairs_with_correct:
        new_folder_path_temp = os.path.join(raw_data_dir, correct_folder + 'temp')
        final_folder_path = os.path.join(raw_data_dir, correct_folder)
        
        # Rename temp folder to final correct name
        if os.path.exists(new_folder_path_temp):
            os.rename(new_folder_path_temp, final_folder_path)
            print(f"Renamed folder '{correct_folder}temp' to '{correct_folder}'")
    return non_corresponding_pairs_with_correct
def rename_actionname(dir_marker_calculated,non_corresponding_pairs_with_correct):
    for action_id, wrong_folder, correct_folder in non_corresponding_pairs_with_correct:
        old_folder_path = os.path.join(dir_marker_calculated, wrong_folder )
        new_folder_path_temp = os.path.join(dir_marker_calculated, correct_folder+ 'temp')
        
        # Rename temp folder to final correct name
        if os.path.exists(old_folder_path):
            os.rename(old_folder_path, new_folder_path_temp)
            print(f"Renamed folder '{correct_folder}temp' to '{correct_folder}'")
    for action_id, wrong_folder, correct_folder in non_corresponding_pairs_with_correct:
        new_folder_path_temp = os.path.join(dir_marker_calculated, correct_folder + 'temp')
        final_folder_path = os.path.join(dir_marker_calculated, correct_folder)
        
        # Rename temp folder to final correct name
        if os.path.exists(new_folder_path_temp):
            os.rename(new_folder_path_temp, final_folder_path)
            print(f"Renamed folder '{correct_folder}temp' to '{correct_folder}'")
def recheck(dir_date):
    raw_data_dir = os.path.join(dir_date,'raw_data')
    meetId_json = os.path.join(raw_data_dir,'meetId.json')
    
    if os.path.exists(meetId_json):
        response =requests.get(f"{host}/api/meets/{ json.load(open(meetId_json,'r'))['meetId']}").json()
        import pdb;pdb.set_trace()
        #import pdb;pdb.set_trace()
        items = os.listdir(raw_data_dir)
        filtered_folders = [
        item for item in items
        if not item.endswith('.json') and os.path.isdir(os.path.join(raw_data_dir, item)) and item.lower() != 'calibration'
    ]
        print(filtered_folders)
        filtered_actionID = [
            json.load(open(os.path.join(raw_data_dir,dir_action,'actionId.json'), 'r'))['actionId'] for  dir_action in filtered_folders
        ]

        non_corresponding_pairs_with_correct = checkactioname(raw_data_dir,response,filtered_folders,filtered_actionID)
        calculated_folders = [folder for folder in os.listdir(dir_date) 
                      if os.path.isdir(os.path.join(dir_date, folder)) and folder.endswith('calculated')]
        calculated_folders = natsorted(calculated_folders, reverse=True)
        for dir_marker_calculated in calculated_folders:
            rename_actionname(os.path.join(dir_date,dir_marker_calculated),non_corresponding_pairs_with_correct)
        #import pdb;pdb.set_trace()
        check_calculated_file(response,calculated_folders,dir_date)

    
    # checkmeet(dir_meetlocal)

    # checkaction(dir_actionlocal)
#getTasktype()  
dir_layout=r'C:\Users\mauricetemp\Desktop\NTKCAP\config\layout.json'
dir_notevalue=r'C:\Users\mauricetemp\Desktop\NTKCAP\config\meetnote_layout.json'
dir_location=r'C:\Users\mauricetemp\Desktop\NTKCAP\config\location.json'
patientId = "66d94d6626a882267fa69252"
date_str = datetime.now().strftime("%Y_%m_%d")
date_str='2024-09-18T00:00:00Z'
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
patientId  ='667d29d0cfee0b0977061967'
date_str='2024-09-18T00:00:00Z'
dir_layout=r'C:\Users\mauricetemp\Desktop\NTKCAP\config\layout.json'
dir_notevalue=r'C:\Users\mauricetemp\Desktop\NTKCAP\config\actionnote_layout.json'
dir_location=r'C:\Users\mauricetemp\Desktop\NTKCAP\config\location.json'
meetId="66ea45f6f43367445fa25231"
task_type ='Walking_start'
#action_postupdate(dir_layout,dir_notevalue,dir_location,patientId,date_str,'walk1',task_type,meetId)

#action_update(dir_layout,dir_notevalue,dir_location,date_str,meetId)
#marker_calculate_upload

meetId ="66e26cb4bd48a32ea268b0f5"
actionId = "66e2959abd48a32ea268b0fa"
outputdir = r'C:\Users\mauricetemp\Desktop\NTKCAP\config\layout_temp.json'
#MeetActionID2json(meetId,actionId,outputdir)

actionId = "66eb05d3bd94767b715d512e"
#getTasknumber(actionId)
dir_date = r'C:\Users\mauricetemp\Desktop\NTKCAP\Patient_data\667d29d0cfee0b0977061967\2024_09_24'
#recheck(dir_date)