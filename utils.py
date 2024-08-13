from typing import Optional, Any
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

    def resize(self, img):
        resized_image = img.resize(self.new_size)
        return resized_image


