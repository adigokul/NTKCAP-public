import sys
import json
import numpy as np
import torch
import pickle
from torchreid.utils import FeatureExtractor
import torchreid
import cv2
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
from torchvision import transforms
import os

def extract_features(image, extractor):
    shape = image.shape
    
    img = np.array(image, dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # transform = transforms.Compose([
        # transforms.CenterCrop((256, 128)),  
        # transforms.Resize((256, 128)),
        # transforms.ToTensor(),  
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])
    
    # processed_image = transform(img).numpy()   
    # restored_image = processed_image.transpose(1, 2, 0)
    
    feature = extractor([img])
    return feature[0]

def reid_matching(Apose_cropped_frame, task_cropped_frame):
    extractor = FeatureExtractor(
        model_name='resnet50',
        model_path="C:/Users/MyUser/Desktop/NTKCAP_thirdparty/deep-person-reid/log/resnet50/model/model.pth.tar-60", 
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    similarities = []
    for p_task in range(len(task_cropped_frame)):
        similarities.append([])
        for p_apose in range(len(Apose_cropped_frame)):       
            similarities[p_task].append([])
            for cam_id_apose in range(len(Apose_cropped_frame[0])):
                feat1 = extract_features(Apose_cropped_frame[p_apose][cam_id_apose], extractor)
                feat2 = extract_features(task_cropped_frame[p_task][cam_id_apose], extractor)
                feat1_cpu = feat1.cpu().numpy()
                feat2_cpu = feat2.cpu().numpy()
                similarity = cosine_similarity(feat1_cpu.reshape(1, -1), feat2_cpu.reshape(1, -1))[0][0]
                similarities[p_task][p_apose].append(float(similarity))
    return similarities

if __name__ == "__main__":
    json_path = sys.argv[1]
    with open(json_path, 'r') as f:
        cropped_frame = json.load(f)
    Apose_cropped_frame = [[np.array(cam) for cam in person] for person in cropped_frame["data1"]]
    task_cropped_frame = [[np.array(cam) for cam in person] for person in cropped_frame["data2"]]
    result = reid_matching(Apose_cropped_frame, task_cropped_frame)
    
    data_list = np.array(result).tolist()
    with open(json_path, 'w') as file:        
        json.dump(data_list, file)   