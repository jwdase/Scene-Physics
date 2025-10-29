import requests

url = "https://raw.githubusercontent.com/mikedh/trimesh/main/models/featuretype.STL"
file_name = "downloaded_model.stl"

try:
    response = requests.get(url)
    response.raise_for_status()  # Check if the request was successful
    
    with open(file_name, 'wb') as f:
        f.write(response.content)
        
    print(f"Successfully downloaded mesh to {file_name}")

except requests.exceptions.RequestException as e:
    print(f"Error downloading file: {e}")