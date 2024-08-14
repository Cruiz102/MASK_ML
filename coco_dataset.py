import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
import os
from torchvision.io import read_image
import numpy as np
from torchvision.transforms.functional import to_tensor
from utils import PreProcessor
import torch.functional as F
from torchvision.transforms import Resize


class CocoSegmentationDataset(Dataset):
    def __init__(self, 
                 annotation_dir: str = "/home/cesarruiz/Downloads/panoptic_annotations_trainval2017/instances_val2017.json",
                 img_dir: str = "/home/cesarruiz/Downloads/val2017/",
                 resize_to: tuple = (256, 256)):
        self.coco_tool = COCO(annotation_dir)
        self.img_dir = img_dir
        self.img_ids = self.coco_tool.getImgIds()
        self.resize = Resize(resize_to)

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

        for ann in anns:
            if 'segmentation' in ann:
                if isinstance(ann['segmentation'], list):
                    # Polygon format
                    rle = maskUtils.frPyObjects(ann['segmentation'], height, width)
                    m = maskUtils.decode(rle)
                    m = torch.tensor(m.transpose(2,0,1)[0], dtype=torch.float32).unsqueeze(0)

                    # Combine masks
                    mask = torch.maximum(mask, m)

                    if masks.numel() == 0:
                        masks = mask
                    else:
                        masks = torch.cat((masks, mask), dim=0)

        return image, masks

if __name__ == "__main__":
    dataset = CocoSegmentationDataset(img_dir="/home/cesarruiz/Downloads/val2017/", annotation_dir='/home/cesarruiz/Downloads/panoptic_annotations_trainval2017/instances_val2017.json')
    print(dataset[0][1].shape)
