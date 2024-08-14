from typing import Optional, Any, Union, List, Tuple
import numpy as np
import torch
import yaml
from PIL import Image

def read_yaml(yaml_file: str) -> Optional[Any] :
    if not yaml_file:
        print("No YAML file provided or file not found.")
        return None
    try:
        with open(yaml_file, 'r') as f:
            return yaml.safe_load(f)
    except (FileNotFoundError, PermissionError, yaml.YAMLError) as e:
        print(f"An error occurred: {e}")
        return None
    

class PreProcessorConfig:
    do_resize: bool
    augmentation: bool

class PreProcessor:
    def __init__(self):
        self.new_size = (255,255)

    def resize(self, img: Union[np.ndarray, torch.tensor, Image.Image]):

        resized_image = img.resize(self.new_size)
        return resized_image


# AI GENERATED
def calculate_conv2d_output_dimensions(H_in, W_in, K, S, P=0, D=1):
    """
    Calculate the output height and width for a 2D convolutional layer.

    Parameters:
    H_in (int): Input height
    W_in (int): Input width
    K (int): Kernel size (assuming square kernel)
    S (int): Stride
    P (int): Padding (default is 0)
    D (int): Dilation (default is 1)

    Returns:
    H_out (int): Output height
    W_out (int): Output width
    """
    H_out = (H_in + 2 * P - D * (K - 1) - 1) // S + 1
    W_out = (W_in + 2 * P - D * (K - 1) - 1) // S + 1
    return H_out, W_out
