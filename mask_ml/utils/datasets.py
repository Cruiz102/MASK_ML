import torch
from torch.utils.data import Dataset
from torchvision import transforms
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
import os
from torchvision.io import read_image
import numpy as np
from torchvision.transforms import Resize
import json
from torch.utils.data import DataLoader
import torchvision
from omegaconf import DictConfig
from typing import Union, Tuple

# This function is to use in th train.py and eval.py of the project.
def create_dataloader(cfg: DictConfig) -> Union[Tuple[DataLoader, DataLoader],DataLoader]:
    dataset_name = cfg.get('datasets', {}).get('name')
    batch_size = cfg.get('batch_size', 32) 
    shuffle = cfg.get('shuffle', False)
    # Select the appropriate dataset
    if dataset_name == "coco_segmentation":
        image_size = cfg.get('datasets').get('image_size')
        annotation_train_dir = cfg.get('datasets').get('annotation_train_dir')
        annotation_test_dir = cfg.get('datasets').get('annotation_test_dir')
        image_train_dir = cfg.get('datasets').get('image_train_dir')
        image_test_dir = cfg.get('datasets').get('image_test_dir')
        dataset_train = CocoSegmentationDataset(annotation_dir =annotation_train_dir , img_dir=image_train_dir,
                                                image_size=(image_size,image_size))
        dataset_test = CocoSegmentationDataset(annotation_dir =annotation_test_dir , img_dir=image_test_dir,
                                                        image_size=(image_size,image_size))
    elif dataset_name == "sa1b":
        dataset_train = SA1BImageDataset(image_dir)
    elif dataset_name =='imagenet_classification':
        dataset_train = torchvision.datasets.ImageNet(image_dir, split='val')
        
    elif dataset_name == 'cifar100_classification':
        base_dir = cfg.get('datasets').get('base_dir')
        transform = transforms.Compose([
            transforms.ToTensor(),  # Convert PIL Image to Tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Resize((image_size,image_size)),  # Resize the image to 64x64 pixels
        ])
        dataset_train = torchvision.datasets.CIFAR100(base_dir, download=True, transform= transform)
        dataset_test = torchvision.datasets.CIFAR100(base_dir, download=True,train=False,  transform= transform)

    elif dataset_name == 'mnist_classification':
        transform = transforms.Compose([
            transforms.ToTensor(),  
            transforms.Normalize(mean=[0.485], std=[0.229]),
            transforms.Resize((image_size,image_size)),  
        ])
        dataset_train = torchvision.datasets.MNIST(image_dir, train=True, download=True, transform=transform)
        dataset_test = torchvision.datasets.MNIST(image_dir, train=False,transform=transform)

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=shuffle)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=shuffle)
    print(dataset_name, "This is the dataset name")
    return (dataloader_train, dataloader_test)


class CocoSegmentationDataset(Dataset):
    def __init__(self, 
                 annotation_dir: str ,
                 img_dir: str ,
                 image_size: tuple = (256, 256),
                 objects_num: int = 1,
                 ):
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
