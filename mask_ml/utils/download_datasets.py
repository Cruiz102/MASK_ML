import requests
from tqdm import tqdm
import os
import zipfile
import hashlib
import json
import logging
import kagglehub

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

# Update kaggle_datasets to separate competitions and datasets
kaggle_competitions = [
    {
        'name': 'imagenet-object-localization-challenge',
        'description': 'ImageNet Object Localization Challenge'
    },
]

kaggle_datasets = [
    {
        'name': 'sautkin/imagenet1kvalid',
        'description': 'Imagenet Validation dataset'
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


def download_kaggle_competition(competition_name, save_dir):
    """
    Download a competition dataset from Kaggle using kagglehub.
    
    Args:
        competition_name (str): Name of the competition
        save_dir (str): Directory to save the dataset
    """
    try:
        logger.info(f"Downloading competition dataset: {competition_name}")
        path = kagglehub.model_download(f"competitions/{competition_name}")
        logger.info(f"Successfully downloaded competition data to: {path}")
        
    except Exception as e:
        logger.error(f"Error downloading from Kaggle: {str(e)}")

def select_and_download_dataset():
    """
    Main function to select and download datasets from different sources
    """
    logger.info("\nSelect dataset source:")
    logger.info("1: COCO Datasets")
    logger.info("2: Kaggle Competitions")
    logger.info("3: Kaggle Datasets")
    
    while True:
        try:
            source = int(input("\nEnter your choice (1, 2, or 3): "))
            if source in [1, 2, 3]:
                break
            logger.info("Please enter 1, 2, or 3")
        except ValueError:
            logger.info("Please enter a valid number")

    if source == 1:
        download_coco_dataset()
    elif source == 2:
        select_and_download_kaggle_competition()
    else:
        select_and_download_kaggle_dataset()

def select_and_download_kaggle_competition():
    """
    Function to select and download a specific Kaggle competition dataset
    """
    logger.info("\nAvailable Kaggle competitions:")
    for i, competition in enumerate(kaggle_competitions, 1):
        logger.info(f"{i}: {competition['description']} ({competition['name']})")
    
    while True:
        try:
            choice = int(input("\nEnter the number of the competition dataset you want to download: "))
            if 1 <= choice <= len(kaggle_competitions):
                break
            logger.info(f"Please enter a number between 1 and {len(kaggle_competitions)}")
        except ValueError:
            logger.info("Please enter a valid number")

    selected_competition = kaggle_competitions[choice - 1]
    
    # Create datasets directory if it doesn't exist
    dataset_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'datasets', 'kaggle_competitions')
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Create specific directory for the selected competition
    competition_dir = os.path.join(dataset_dir, selected_competition['name'])
    os.makedirs(competition_dir, exist_ok=True)
    
    # Download the competition dataset
    logger.info(f"\nDownloading {selected_competition['description']}...")
    download_kaggle_competition(selected_competition['name'], competition_dir)

def select_and_download_kaggle_dataset():
    """
    Function to select and download a specific Kaggle dataset
    """
    logger.info("\nAvailable Kaggle datasets:")
    for i, dataset in enumerate(kaggle_datasets, 1):
        logger.info(f"{i}: {dataset['description']} ({dataset['name']})")
    
    while True:
        try:
            choice = int(input("\nEnter the number of the dataset you want to download: "))
            if 1 <= choice <= len(kaggle_datasets):
                break
            logger.info(f"Please enter a number between 1 and {len(kaggle_datasets)}")
        except ValueError:
            logger.info("Please enter a valid number")

    selected_dataset = kaggle_datasets[choice - 1]
    
    # Download the dataset
    logger.info(f"\nDownloading {selected_dataset['description']}...")
    download_kaggle_dataset(selected_dataset['name'], None)  # Pass None since save_dir is not used

def download_kaggle_dataset(dataset_name, save_dir):
    """
    Download a dataset from Kaggle using kagglehub.
    
    Args:
        dataset_name (str): Name of the dataset in format 'owner/dataset-name'
        save_dir (str): Directory to save the dataset
    """
    try:
        logger.info(f"Downloading {dataset_name} from Kaggle...")
        logger.info(kagglehub.dataset_download(dataset_name))
        logger.info("Successfully downloaded dataset")
        
    except Exception as e:
        logger.error(f"Error downloading from Kaggle: {str(e)}")

# Update the main block to use the new selection function
if __name__ == "__main__":
    select_and_download_dataset()