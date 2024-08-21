import os
import requests
import zipfile
from tqdm import tqdm
import yaml

# Sample YAML content
yaml_content = """
datasets_links:
  coco:
    train:
      images: ""
      annotation: ""
    validation: 
      images: "http://images.cocodataset.org/zips/val2017.zip"
      annotation: "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
"""

# Load the YAML content into a Python dictionary
data = yaml.safe_load(yaml_content)
datasets_dict = data.get('datasets_links', {})

# Base destination folder where the structure will be created
base_dest_folder = "./downloads"

def ensure_directory_exists(directory: str):
    """Check if a directory exists, and if not, create it."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' created.")

def download_file(url: str, dest_folder: str) -> str:
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    # Get the file name from the URL
    file_name = os.path.join(dest_folder, url.split("/")[-1])

    # Send a GET request to the URL to start downloading the file
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    # Open a local file with write-binary mode
    with open(file_name, 'wb') as file:
        # Use tqdm to display the download progress
        for data in tqdm(response.iter_content(1024), total=total_size//1024, unit='KB'):
            file.write(data)

    print(f"Downloaded {file_name}")
    return file_name

def extract_zip(file_path: str, dest_folder: str):
    try:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(dest_folder)
        print(f"Extracted {file_path} to {dest_folder}")
        return True
    except zipfile.BadZipFile:
        print(f"Failed to extract {file_path}: Bad zip file")
        return False

def download_and_extract_files(file_urls: dict, base_dest_folder: str):
    for dataset_name, dataset_content in file_urls.items():
        for data_type, links in dataset_content.items():
            for category, url in links.items():
                # Define the folder structure based on the YAML structure
                dest_folder = os.path.join(base_dest_folder, dataset_name, data_type, category)
                
                # Ensure the directory exists
                ensure_directory_exists(dest_folder)
                
                # Download and extract the files if URL is provided
                if url:
                    zip_file_path = download_file(url, dest_folder)
                    if zip_file_path.endswith('.zip'):
                        success = extract_zip(zip_file_path, dest_folder)
                        if success:
                            os.remove(zip_file_path)
                            print(f"Removed {zip_file_path}")

if __name__ == "__main__":
    # Run the function with the provided YAML data
    import os
import requests
import zipfile
from tqdm import tqdm
import yaml

# Sample YAML content
yaml_content = """
datasets_links:
  coco:
    train:
      images: ""
      annotation: ""
    validation: 
      images: "http://images.cocodataset.org/zips/val2017.zip"
      annotation: "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
"""

# Load the YAML content into a Python dictionary
data = yaml.safe_load(yaml_content)
datasets_dict = data.get('datasets_links', {})

# Base destination folder where the structure will be created
base_dest_folder = "./downloads"

def ensure_directory_exists(directory: str):
    """Check if a directory exists, and if not, create it."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' created.")

def download_file(url: str, dest_folder: str) -> str:
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    # Get the file name from the URL
    file_name = os.path.join(dest_folder, url.split("/")[-1])

    # Send a GET request to the URL to start downloading the file
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    # Open a local file with write-binary mode
    with open(file_name, 'wb') as file:
        # Use tqdm to display the download progress
        for data in tqdm(response.iter_content(1024), total=total_size//1024, unit='KB'):
            file.write(data)

    print(f"Downloaded {file_name}")
    return file_name

def extract_zip(file_path: str, dest_folder: str):
    try:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(dest_folder)
        print(f"Extracted {file_path} to {dest_folder}")
        return True
    except zipfile.BadZipFile:
        print(f"Failed to extract {file_path}: Bad zip file")
        return False

def download_and_extract_files(file_urls: dict, base_dest_folder: str):
    for dataset_name, dataset_content in file_urls.items():
        for data_type, links in dataset_content.items():
            for category, url in links.items():
                # Define the folder structure based on the YAML structure
                dest_folder = os.path.join(base_dest_folder, dataset_name, data_type, category)
                
                # Ensure the directory exists
                ensure_directory_exists(dest_folder)
                
                # Download and extract the files if URL is provided
                if url:
                    zip_file_path = download_file(url, dest_folder)
                    if zip_file_path.endswith('.zip'):
                        success = extract_zip(zip_file_path, dest_folder)
                        if success:
                            os.remove(zip_file_path)
                            print(f"Removed {zip_file_path}")

if __name__ == "__main__":
    # Run the function with the provided YAML data
    download_and_extract_files(datasets_dict, base_dest_folder)
