import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image


from sam2.build_sam import build_sam2_video_predictor
class Sam2Predictor:
    def __init__(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        print(f"using device: {device}")

        if device.type == "cuda":
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        sam2_checkpoint = "/home/cesar/Projects/MASK_ML/sam2_hiera_large.pt"
        model_cfg = "sam2_hiera_l.yaml"

        predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

        self.masks = torch.Tensor()
        self.objects_names = []
        self.objects_images = []
        self.last_mask = torch.zeros(1).cuda()
        super().__init__()
    def predict_mask(self, img):
        img_tensors = torch.from_numpy(img).cuda().float()
        
        img_tensors = img_tensors / 255.0
        img_tensors =img_tensors.permute(2, 0, 1) 
        # print(img_tensors.dtype, "img")
        output_probs = self.processor.step(img_tensors)
        masks = self.processor.output_prob_to_mask(output_probs)
        masks = self._create_mask_channels(masks)
        # print("output", output_probs)
        # print(output_probs.device, self.last_mask.device)
        if torch.equal(output_probs, self.last_mask):
            print("Something Wrong in here>: Same Image??")
        self.last_mask = output_probs
        return masks
    