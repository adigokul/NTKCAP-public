import json
import sys

if len(sys.argv) > 1:
        file_path = sys.argv[1]
        print("config path: " + file_path)

data = {
    "cam": {
        "number": 0,
        "list": [],
        "resolution": [1920, 1080],
        "name": "HD camera"
    }
}

with open(file_path, 'w') as f:
    json.dump(data, f, indent=4)