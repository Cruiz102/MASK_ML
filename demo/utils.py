
from dataclasses import dataclass
from typing import Optional, List
from torch import Tensor
import numpy as np
import cv2
# Load the model

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


def to_bbox(mask: Tensor):
        results = []
        for idx, mask in enumerate(mask.cpu().numpy()):
            box = get_bbox_from_mask(mask)
            if len(box) > 1:
                results.append(
                    BBoxOutput(
                        x1=box[0],
                        y1=box[1],
                        x2=box[2],
                        y2=box[3],
                        iou_condifence=1,
                        mask=mask,
                        track_id=str(idx)
                    )
                )

        return results