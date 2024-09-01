import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision import transforms
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
import os
from torchvision.io import read_image
import numpy as np
import torch.functional as F
from torchvision.transforms import Resize
import json
from torch.utils.data import DataLoader
import torchvision
from mask_ml.utils.download_dataset import download_and_extract_files
import os
from omegaconf import DictConfig, OmegaConf
import torchvision

# Assuming you have these dataset classes defined somewhere
# from your_datasets_module import CocoSegmentationDataset, SA1BImageDataset

def create_dataloader(cfg: DictConfig) -> DataLoader:
    dataset_name = cfg['dataset']
    
    # Get the paths from the config
    image_dir = cfg['datasets'][dataset_name]['base_dir']
    batch_size = cfg['batch_size']
    shuffle = cfg.get('shuffle', False)
    download = cfg.get('download', False)
    image_size = cfg.get("image_size", 256)
    
    # Select the appropriate dataset
    if dataset_name == "coco":
        dataset = CocoSegmentationDataset("annotation_dir", image_dir, download=download)
    elif dataset_name == "sa1b":
        dataset = SA1BImageDataset(image_dir, download=download)
    elif dataset_name == 'cifar100':
        # Define the transformation to convert images to tensors and normalize
        transform = transforms.Compose([
            transforms.ToTensor(),  # Convert PIL Image to Tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Resize((256, 256)),  # Resize the image to 64x64 pixels
        ])
        dataset = torchvision.datasets.CIFAR100(image_dir, download=True, transform= transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    print(dataset_name, "This is the dataset name")

    return dataloader


class CocoSegmentationDataset(Dataset):
    def __init__(self, 
                 annotation_dir: str ,
                 img_dir: str ,
                 image_size: tuple = (256, 256),
                 objects_num: int = 1,
                 download = False
                 ):
        self.download = download
        self.image_size = image_size
        self.coco_tool = COCO(annotation_dir)
        self.img_dir = img_dir
        self.img_ids = self.coco_tool.getImgIds()
        self.resize = Resize(self.image_size)
        self.objects_num = objects_num

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco_tool.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        
        # Read the image
        image = read_image(img_path)
        # Resize the image
        image = self.resize(image)
        # Load annotations
        ann_ids = self.coco_tool.getAnnIds(imgIds=img_id)
        anns = self.coco_tool.loadAnns(ann_ids)
        height, width = image.shape[1], image.shape[2]


        # Initialize an empty mask
        mask = torch.zeros((height, width))
        masks = torch.tensor([])
        target_objects = list(range(self.objects_num))     
        target_objects = np.random.choice(target_objects, size=self.objects_num, replace=False)

        if self.download:
        #Check if the directory exists
            if not os.path.exists(self.img_dir):
                # If it does not exist, create it
                download_and_extract_files()
                print(f"Directory '{self.img_dir}' created.")
            else:
                print(f"Directory '{self.img_dir}' already exists.")

        #FIXME: There is going to be an index access error if the number of annotations is less than self.objects_num
        for i in target_objects:
            if 'segmentation' in anns[i]:
                if isinstance(anns[i]['segmentation'], list):
                    # Polygon format
                    rle = maskUtils.frPyObjects(anns[i]['segmentation'], height, width)
                    m = maskUtils.decode(rle)
                    m = torch.tensor(m.transpose(2,0,1)[0], dtype=torch.float32).unsqueeze(0)

                    # Combine masks
                    mask = torch.maximum(mask, m)

                    if masks.numel() == 0:
                        masks = mask
                    else:
                        masks = torch.cat((masks, mask), dim=0)
            if 'bbox' in anns[i]:
                x, y, w, h = map(int, anns[i]['bbox'])
                box_data  = torch.tensor([x,y,w,h])

        return image, masks, box_data
    


class SA1BImageDataset(Dataset):
    def __init__(self, root_dir, transforms=None, download = False):
        """
        Args:
            root_dir (string): Directory with all the images and JSON files.
            transforms (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transforms = transforms
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)
        json_path = os.path.join(self.root_dir, img_name.replace('.jpg', '.json'))

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Load JSON metadata
        with open(json_path, 'r') as f:
            metadata = json.load(f)

        # Convert relevant data to tensors
        bbox = torch.tensor(metadata['bbox'])
        segmentation = torch.tensor(metadata['segmentation']['counts'])
        area = torch.tensor(metadata['area'])
        point_coords = torch.tensor(metadata['point_coords'])
        crop_box = torch.tensor(metadata['crop_box'])
        predicted_iou = torch.tensor(metadata['predicted_iou'])
        stability_score = torch.tensor(metadata['stability_score'])

        sample = {
            'image': image,
            'bbox': bbox,
            'segmentation': segmentation,
            'area': area,
            'point_coords': point_coords,
            'crop_box': crop_box,
            'predicted_iou': predicted_iou,
            'stability_score': stability_score
        }

        if self.transforms:
            sample['image'] = self.transforms(sample['image'])

        return sample


if __name__ == "__main__":
    dataset = CocoSegmentationDataset(img_dir="/home/cesarruiz/Downloads/val2017/", annotation_dir='/home/cesarruiz/Downloads/panoptic_annotations_trainval2017/instances_val2017.json')
    print(dataset[0][1].shape)
