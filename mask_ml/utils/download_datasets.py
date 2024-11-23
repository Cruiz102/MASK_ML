import requests
from tqdm import tqdm
import os
import zipfile
import asyncio
import tarfile
import hashlib
import json
import logging

logger = logging.getLogger('datasets_downloader')
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
        logger.info(f'Removing the zip file after Unzipping')
        os.remove(zip_path)
    except zipfile.BadZipFile:
        logger.info(f"Error: {zip_path} is not a valid ZIP file.")

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
    images_name_dir = dataset_names_info[0]
    annotations_name_dir = dataset_names_info[1]

    # Check if the dataset has been already downloaded
    if os.path.exists(os.path.join(coco_dataset_dir, images_name_dir)) or os.path.exists(os.path.join(coco_dataset_dir, annotations_name_dir)):
        logger.info("This dataset is already downloaded. Skipping...")
        return

    # Load or create a hash record file
    json_file_path = os.path.join(coco_dataset_dir, 'hashes.json')
    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as f:
            hashes = json.load(f)
    else:
        hashes = {}

    # Verify integrity if hashes are available
    for dataset_name in [images_name_dir, annotations_name_dir]:
        dataset_path = os.path.join(coco_dataset_dir, dataset_name)
        if dataset_name in hashes and os.path.exists(dataset_path):
            logger.info(f"Verifying integrity for {dataset_name}...")
            current_hash = hashlib.md5(open(dataset_path, 'rb').read()).hexdigest()
            if hashes[dataset_name] != current_hash:
                raise RuntimeError(f"The hash for {dataset_name} does not match the expected value. Data may be corrupted.")
            else:
                logger.info(f"{dataset_name} integrity verified.")
        else:
            logger.info(f"No hash record found for {dataset_name}, skipping verification.")

    # Download and process datasets
    for dataset_name, dataset_link in selected_dataset.items():
        if not dataset_link:
            logger.info(f"Skipping {dataset_name} (no link provided).")
            continue
        download_path = os.path.join(coco_dataset_dir, f"{dataset_name}.zip")
        extract_to = os.path.join(coco_dataset_dir, dataset_name)
        if not os.path.exists(download_path):
            logger.info(f"Downloading {dataset_name}...")
            response = requests.get(dataset_link, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            with open(download_path, 'wb') as f:
                for chunk in tqdm(response.iter_content(chunk_size=1024), desc=f"Downloading {dataset_name}", total=total_size // 1024):
                    f.write(chunk)
            logger.info(f"{dataset_name} downloaded successfully!")
        else:
            logger.info(f"{dataset_name} already exists, skipping download.")

        # Extract the file
        if not os.path.exists(extract_to):
            unzip_file(download_path, extract_to)
            logger.info(f"{dataset_name} extracted successfully!")
        else:
            logger.info(f"{dataset_name} already extracted, skipping unzipping.")

        # Calculate and store the hash
        logger.info(f"Calculating hash for {dataset_name}...")
        with open(os.path.join(extract_to, dataset_name), 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        hashes[dataset_name] = file_hash

    # Save the updated hashes
    with open(json_file_path, 'w') as f:
        json.dump(hashes, f, indent=4)

    logger.info("All operations complete!")


def download_sa1b_dataset():
    sa1b_links_file = os.path('..','..','datasets','sa1b.txt')
    with open(sa1b_links_file, 'r',) as f:
        data = f.read()
        for link in data.split():
            result = requests.get(link)



# Run the script
if __name__ == "__main__":
    download_coco_dataset()