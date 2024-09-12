import json
dir = r'D:\NTKCAP\Patient_data\667d29d0cfee0b0977061967\2024_09_12\raw_data\meetId.json'
with open(dir,'r')as file:
    data= json.load(file)['meedId']
import pdb;pdb.set_trace()
