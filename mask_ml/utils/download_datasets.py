import requests
from tqdm import tqdm
import os
import zipfile
import hashlib
import json
import logging

logger = logging.getLogger('datasets_downloader')
logging.basicConfig(
    level=logging.INFO,  # Adjust this to DEBUG for more detailed logs
    format='%(message)s'
)


coco_datasets = [

    {
    '2014_train_images': 'http://images.cocodataset.org/zips/train2014.zip', 
    '2014_train_val_annotations':'http://images.cocodataset.org/annotations/annotations_trainval2014.zip'},
    {
    '2014_val_images': 'http://images.cocodataset.org/zips/val2014.zip',
    '2014_train_val_annotations':'http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
    },
    {
    '2015_test_images':'http://images.cocodataset.org/zips/test2014.zip',
    '2015_testing_image_info': ""
    },
    {
    '2017_train_images': 'http://images.cocodataset.org/zips/train2017.zip',
    '2017_train_val_annotations':'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
    },
    {
    '2017_val_images': 'http://images.cocodataset.org/zips/val2017.zip',
    '2017_train_val_annotations':'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
    },
    {
    '2017_test_images': 'http://images.cocodataset.org/zips/test2017.zip',
    '2017_testing_image_info': 'http://images.cocodataset.org/annotations/image_info_test2017.zip'
    },
    {
        '2017_unlabeled_images': 'http://images.cocodataset.org/zips/unlabeled2017.zip',
        '2017_unlabeled_image_info': 'http://images.cocodataset.org/annotations/image_info_unlabeled2017.zip'
    }

]


def unzip_file(zip_path, extract_to):
    logger.info(f"Unzipping {zip_path} to {extract_to}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        logger.info(f"Unzipped {zip_path} successfully!")
        logger.info('Removing the zip file after Unzipping')
        os.remove(zip_path)
    except zipfile.BadZipFile:
        logger.info(f"Error: {zip_path} is not a valid ZIP file.")


def calculate_and_update_hashes(dataset_dir, dataset_name, hashes, json_file_path):
    """
    Calculate and update the hash for a dataset if it's missing in the hash record.

    Args:
        dataset_dir (str): Directory where the dataset is located.
        dataset_name (str): Name of the dataset.
        hashes (dict): Existing hash records.
        json_file_path (str): Path to the hashes.json file.
    """
    dataset_path = os.path.join(dataset_dir, f"{dataset_name}.zip")
    
    # Check if the dataset folder or zip exists
    if os.path.exists(dataset_path):
        logger.info(f"Found {dataset_name}.zip, calculating its hash...")
        with open(dataset_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
    else:
        logger.warning(f"Zip file for {dataset_name} does not exist. Skipping...")
        return
    
    # Check if the hash is already recorded
    if dataset_name in hashes and hashes[dataset_name] == file_hash:
        logger.info(f"Hash for {dataset_name} already exists and matches.")
        return

    # Update the hash record
    logger.info(f"Updating hash for {dataset_name} in {json_file_path}...")
    hashes[dataset_name] = file_hash

    # Save updated hashes to JSON
    with open(json_file_path, 'w') as f:
        json.dump(hashes, f, indent=4)
    logger.info(f"Hash for {dataset_name} updated successfully!")

def download_coco_dataset():
    coco_dataset_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'datasets', 'coco')
    os.makedirs(coco_dataset_dir, exist_ok=True)

    # SELECT WHICH DATASET YOU WANT TO DOWNLOAD
    logger.info("Choose what dataset to download:")
    for i, dataset in enumerate(coco_datasets):
        dataset_name = list(dataset.keys())[0]
        logger.info(f"{i}: {dataset_name}")
    option = int(input("Choose an option: "))
    assert 0 <= option < len(coco_datasets), f"{option} is not a valid option"
    selected_dataset = coco_datasets[option]

    dataset_names_info = list(selected_dataset.keys())

    # Load or create a hash record file
    json_file_path = os.path.join(coco_dataset_dir, 'hashes.json')
    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as f:
            hashes = json.load(f)
    else:
        hashes = {}

    # Check and update hashes for existing datasets
    for dataset_name in dataset_names_info:
        calculate_and_update_hashes(coco_dataset_dir, dataset_name, hashes, json_file_path)

    # Download and process datasets
    for dataset_name in dataset_names_info:
        dataset_link = selected_dataset[dataset_name]
        if not dataset_link:
            logger.info(f"Skipping {dataset_name} (no link provided).")
            continue

        # File paths
        download_path = os.path.join(coco_dataset_dir, f"{dataset_name}.zip")
        extract_to = os.path.join(coco_dataset_dir, dataset_name)

        # If folder exists and hash is already recorded, skip download
        if os.path.exists(extract_to) and dataset_name in hashes:
            logger.info(f"{dataset_name} is already extracted and hash is verified. Skipping download.")
            continue

        # Download file
        logger.info(f"Downloading {dataset_name}...")
        response = requests.get(dataset_link, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        with open(download_path, 'wb') as f:
            for chunk in tqdm(response.iter_content(chunk_size=1024), desc=f"Downloading {dataset_name}", total=total_size // 1024):
                f.write(chunk)
        logger.info(f"{dataset_name} downloaded successfully!")

        # Extract file
        logger.info(f"Extracting {dataset_name}...")
        unzip_file(download_path, extract_to)
        logger.info(f"{dataset_name} extracted successfully!")

        # Calculate and update hash after download
        calculate_and_update_hashes(coco_dataset_dir, dataset_name, hashes, json_file_path)

    logger.info("All operations complete!")


# def download_sa1b_dataset():
#     sa1b_links_file = os.path('..','..','datasets','sa1b.txt')
#     with open(sa1b_links_file, 'r',) as f:
#         data = f.read()
#         for link in data.split():
#             result = requests.get(link)



# Run the script
if __name__ == "__main__":
    download_coco_dataset()