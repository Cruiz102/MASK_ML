
from third_party.Cutie.cutie.inference.inference_core import InferenceCore
from third_party.Cutie.cutie.utils.get_default_model import get_default_model
from third_party.MobileSAM.mobile_sam import sam_model_registry, SamPredictor
from torchvision.transforms.functional import to_tensor
from torch import from_numpy
import torch
import numpy as np
from typing import Optional, List
import cv2
from dataclasses import dataclass


@dataclass
class BBoxOutput:
    x1: float
    y1: float
    x2: float
    y2: float
    iou_condifence: float
    mask: Optional[List] = None
    track_id : Optional[str] = None
import random
def generate_random_rgb():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
def get_bbox_from_mask(mask):
    """Applies morphological transformations to the mask, displays it, and if with_contours is True, draws
    contours.
    """
    mask = mask.astype(np.uint8)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return []
    x1, y1, w, h = cv2.boundingRect(contours[0])
    x2, y2 = x1 + w, y1 + h
    if len(contours) > 1:
        for b in contours:
            x_t, y_t, w_t, h_t = cv2.boundingRect(b)
            x1 = min(x1, x_t)
            y1 = min(y1, y_t)
            x2 = max(x2, x_t + w_t)
            y2 = max(y2, y_t + h_t)
    return [x1, y1, x2, y2]

class SamCutiePipeline:
    def __init__(self):
        sam_checkpoint = "third_party/weights/mobile_sam.pt"
        model_type = "vit_t"
        device = "cuda"
        self.model = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.model.to(device=device)
        self.predictor = SamPredictor(self.model)
        self.cutie = get_default_model()
        self.processor = InferenceCore(self.cutie, cfg=self.cutie.cfg)
        self.objects = None
        self.masks = torch.Tensor()
        self.objects_names = []
        self.objects_images = []
        super().__init__()
    def save_object(self,object_name: str, image: np.ndarray, points: Optional[np.ndarray] = None,point_labels: Optional[List] = None,
                     mask: Optional[np.ndarray] = None):
        if mask is not None and points is not None:
            raise Exception("Both points and mask were specified but only one must be set.")
        
        self.predictor.set_image(image)
        self.objects_names.append(object_name)
        self.objects_images.append(image)
        self.objects = list(range(1, len(self.objects_names) +1))
        if points is not None:    
            mask, scores, logits = self.predictor.predict(
                point_coords=points,
                point_labels=torch.tensor(point_labels),
                num_multimask_outputs=1,
                use_stability_score=True)
            

        if len(self.masks) == 0:
            print(self.masks)
            self.masks = from_numpy(mask[0]).cuda()
        else:
            mask = np.array(mask[0]).astype(np.uint8)
            mask[mask==1] = len(self.objects)
            self.masks = from_numpy(mask).cuda() 
    
   
        # objects = np.unique(self.masks.cpu().numpy())
        # self.objects = objects[objects != 0].tolist()
        img  = to_tensor(image).cuda().float()
        self.processor.step(img, self.masks, objects=self.objects)


    def create_mask_channels(self,output_mask: torch.Tensor) -> torch.Tensor:
        # Create a tensor to hold the mask channels
        num_ids = (torch.max(output_mask)).item()
        mask_channels = torch.zeros(num_ids, output_mask.shape[0],output_mask.shape[1])
        # Assign values of 1 to corresponding channels for each ID
        for id in range(num_ids):
            mask_channels[id] = (output_mask == id +1).float()

        return mask_channels

    def get_bounding_box(self, img):
        results = []
        img_tensors = to_tensor(img).cuda().float()
        output_probs = self.processor.step(img_tensors)
        
        self.masks = self.processor.output_prob_to_mask(output_probs)
        for idx, mask in enumerate(self.create_mask_channels(self.masks).cpu().numpy()):
            box = get_bbox_from_mask(mask)
            if len(box) > 1:
                results.append(
                    BBoxOutput(
                        x1=box[0],
                        y1=box[1],
                        x2=box[2],
                        y2=box[3],
                        confidence=1,
                        mask=mask,
                        track_id=self.objects_names[idx]
                    )
                )
        
        
        return results
    