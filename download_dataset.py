import os
import requests
import zipfile
from tqdm import tqdm



_links = [
        "http://images.cocodataset.org/zips/val2017.zip",
        "https://example.com/file2.zip",
        "https://example.com/file3.zip"
    ]
dest_folder = "./downloads"

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

def download_and_extract_files(file_urls: list, dest_folder: str):
    for url in file_urls:
        zip_file_path = download_file(url, dest_folder)
        if zip_file_path.endswith('.zip'):
            success = extract_zip(zip_file_path, dest_folder)
            if success:
                os.remove(zip_file_path)
                print(f"Removed {zip_file_path}")

if __name__ == "__main__":
    # Example usage
    download_and_extract_files(file_urls, dest_folder)
