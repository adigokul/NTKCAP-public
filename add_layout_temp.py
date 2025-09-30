import requests

# Your host and URL
host = "https://motion-service.yuyi-ocean.com"
url = f"{host}/api/layouts"

# Payload (ensure 'catalog' is valid)
files ={
  "catalog": "meet",
  "layoutId": "1235",
  "fields": [
    {
      "id"         : "66b5e0f874ce7144ee2b6ed8",
      "catalog"    : "meet"                    ,
      "elementType": "input"                   ,
      "title"      : "Task number"             ,
      "options"    :None
    }
  ]
}
# Send the POST request
response = requests.post(url, json=files)

# Check if the request was successful
if response.status_code == 200:
    print("Layout successfully posted.")
else:
    print(f"Failed to post layout. Status code: {response.status_code}")
    print("Response details:", response.text)  # Print detailed server error message
