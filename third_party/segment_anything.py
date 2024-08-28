# This is a implementation of the segment everything idea from the SamMobilev2 paper
#https://arxiv.org/abs/2312.09579

from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")  # load an official model
# Predict with the model
results = model("https://ultralytics.com/images/bus.jpg", iou=1.0, show = True)  # predict on an image
import time
time.sleep(10)