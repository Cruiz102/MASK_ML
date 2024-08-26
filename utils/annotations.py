import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
from typing import Tuple, List, Dict, Optional
import cv2
import copy
from dataclasses import dataclass
from enum import Enum



@dataclass
class BBoxOutput:
    x1: float
    y1: float
    x2: float
    y2: float
    iou_condifence: float
    mask: Optional[List] = None
    track_id : Optional[str] = None
class DrawAnnotator:
    def __init__(self, enable_depth = False) -> None:
        self.color = (255,0,0)
        self.line_width = 2
        self.enable_depth = enable_depth
        self.font_scale = 0.5
        self.tf = 1
        

    def draw_bbox(self, img, bbox: List[BBoxOutput],draw_mask=True,
                   object_color:Optional[Dict[str,Tuple]] = None, box_color=(255, 0, 0)):
        self.color = box_color
        output = copy.deepcopy(img)
        for i, box in enumerate(bbox):
            p1, p2 = ((int(box.x1), int(box.y1)), (int(box.x2), int(box.y2)))
            output = cv2.rectangle(output, p1, p2, self.color, thickness=self.line_width, lineType=cv2.LINE_AA)


            if draw_mask and object_color and box.track_id:
                alpha = 0.5
                color_overlay = np.zeros_like(img)
                color_overlay[box.mask == 1] = object_color[box.track_id]
                mask_overlay = cv2.addWeighted(output, 1 - alpha, color_overlay, alpha, 0)
                output = cv2.add(output, cv2.subtract(mask_overlay, output))

        return output