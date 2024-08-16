from typing import Optional, Any, Union, List, Tuple
import numpy as np
import torch
import yaml
from PIL import Image
import math
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



def sinusoidal_positional_encoding(seq_len, d_model, device=None):
    """
    Generate a sinusoidal positional encoding matrix.

    Parameters:
    seq_len: Length of the input sequence.
    d_model: Dimensionality of the embedding vector.
    device: The device (CPU/GPU) where the tensor should be created.

    Returns:
    A tensor of shape (seq_len, d_model) containing the positional encodings.
    """
    position = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float, device=device) * (-math.log(10000.0) / d_model))
    
    pe = torch.zeros(seq_len, d_model, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    return pe

