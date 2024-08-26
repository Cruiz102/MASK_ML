
from dataclasses import dataclass
from typing import Optional, List

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
