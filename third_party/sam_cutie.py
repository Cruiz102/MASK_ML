from third_party.Cutie.cutie.inference.inference_core import InferenceCore
from third_party.Cutie.cutie.utils.get_default_model import get_default_model
from typing import Optional, List,Union, Tuple
import torch
from torch import Tensor
import numpy as np
from ultralytics import SAM
import cv2
class SamCutiePipeline:
    def __init__(self):

        self.sam_model = SAM("mobile_sam.pt")
        self.cutie = get_default_model()
        self.processor = InferenceCore(self.cutie, cfg=self.cutie.cfg)
        self.objects = None
        self.masks = torch.Tensor()
        self.objects_names = []
        self.objects_images = []
        super().__init__()
    def save_object(self,object_name: str, image: np.ndarray, points: Union[np.ndarray, List, None] = None,point_labels: Optional[List] = None,
                     mask: Optional[np.ndarray] = None) ->Tuple[Tensor, Tensor]:
        if mask is not None and points is not None:
            raise Exception("Both points and mask were specified but only one must be set.")
        self.objects_names.append(object_name)
        self.objects_images.append(image)
        self.objects = list(range(1, len(self.objects_names) +1))
        if points is not None:    
            ultra_sam_prediciton = self.sam_model.predict(image, points=points, labels=point_labels)
            mask = ultra_sam_prediciton[0].masks.data
            print('mask', mask)

        if len(self.masks) == 0:
            print(self.masks)
            self.masks =mask[0]
            self.objects_images = torch.from_numpy(image)
        else:
            mask = np.array(mask[0]).astype(np.uint8)
            mask[mask==1] = len(self.objects)
            self.masks = mask 
        img  = torch.from_numpy(image).cuda().float()
        mask = self.processor.step(img, self.masks, objects=self.objects)
        return img, mask

    def _create_mask_channels(self,output_mask: torch.Tensor) -> torch.Tensor:
        # Create a tensor to hold the mask channels
        num_ids = (torch.max(output_mask)).item()
        mask_channels = torch.zeros(num_ids, output_mask.shape[0],output_mask.shape[1])
        # Assign values of 1 to corresponding channels for each ID
        for id in range(num_ids):
            mask_channels[id] = (output_mask == id +1).float()
        return mask_channels

    def predict_mask(self, img):
        img_tensors = torch.from_numpy(img).cuda().float()
        output_probs = self.processor.step(img_tensors)
        masks = self.processor.output_prob_to_mask(output_probs)
        masks = self._create_mask_channels(self.masks).cpu().numpy()
        return masks
    